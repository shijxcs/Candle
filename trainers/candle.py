import time
import numpy as np
import os.path as osp
import os
import datetime
import pickle
from collections import defaultdict
from functools import reduce
from operator import mul
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import AverageMeter, load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms import build_transform

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .losses import *

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "Places": "a photo of a {}.",
    "Places_LT": "a photo of a {}.",
    "iNaturalist2018": "a photo of a {}.",
}

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class Fusion(nn.Module):

    def __init__(self, c_in, num_classes, mask, num_heads=8, dtype=None):
        super(Fusion, self).__init__()
        self.dtype = dtype
        self.attn = nn.MultiheadAttention(c_in, num_heads=num_heads, dropout=0.1)
        self.ratio = nn.Parameter(torch.tensor([0.1])) 
        self.norm = nn.LayerNorm(c_in, dtype=self.dtype)
        self.n_cls = num_classes
        self.mask_type = mask
    
    def generate_mask(self, features):
        device = features.device
        batchsize = features.shape[0] - 2 * self.n_cls
        mask = torch.zeros((batchsize + 2 * self.n_cls, batchsize + 2 * self.n_cls), dtype=torch.bool)
        if(self.mask_type == "in_visual"):
            mask[0: batchsize + self.n_cls, 0: batchsize + self.n_cls] = 1
            for i in range(batchsize + self.n_cls):
                mask[i, i] = 0
    
        elif(self.mask_type == "in_text"):
            mask[batchsize + self.n_cls:, batchsize + self.n_cls:] = 1
            for i in range(self.n_cls):
                mask[batchsize + i, batchsize + i] = 0

        elif(self.mask_type == "between_visual_text"):
             mask[0: batchsize + self.n_cls, batchsize + self.n_cls:] = 1
             mask[batchsize + self.n_cls:, 0: batchsize + self.n_cls] = 1   
                
        else:
            mask[:, :] = 0
                     
        mask = mask.to(device)
        return mask
            
    def forward(self, features, phase="train"):
        features = self.norm(features)
        if self.mask_type is not None:
            x, _ = self.attn(features, features, features, need_weights=False, attn_mask=self.generate_mask(features))
        else:
            x, _ = self.attn(features, features, features, need_weights=False)
        features = self.ratio * x + features
        return features


class Head(nn.Module):

    def __init__(self, visual_prototypes, text_features, logit_scale, scale_factor, mask, vir_init):
        super(Head, self).__init__()
        hidden_dim = text_features.shape[1]
        dtype = text_features.dtype

        self.n_cls = text_features.shape[0] # all classes
        
        self.text_features = text_features.data
        self.visual_prototypes = visual_prototypes.data
        
        self.logit_scale = logit_scale.exp()
        self.logit_scale_visual = logit_scale.exp() * scale_factor 
        
        proj_image = torch.eye(hidden_dim, hidden_dim, dtype=dtype)       
        proj_text = torch.eye(hidden_dim, hidden_dim, dtype=dtype)
        self.proj_image = nn.Parameter(proj_image)
        self.proj_text = nn.Parameter(proj_text)

          
        self.fusion = Fusion(hidden_dim, self.n_cls, mask, dtype=dtype)
        
        # initialization from text 
        if(vir_init == "text"):
            virtual_prototypes = text_features[visual_prototypes.shape[0]:, ...] 
            
        # initialization from visual prototypes
        elif(vir_init == "visual"):
            mean_visual_prototypes = torch.mean(visual_prototypes, dim=0).unsqueeze(0).repeat(self.n_cls - visual_prototypes.shape[0], 1)

            rand_turb = torch.empty(mean_visual_prototypes.shape).to(text_features.device)
            nn.init.normal_(rand_turb, std=0.05)
            virtual_prototypes = mean_visual_prototypes + rand_turb 
        
        self.virtual_prototypes = nn.Parameter(virtual_prototypes)
        

    def forward(self, image_features, phase="train"):
        image_features_proj = image_features @ self.proj_image
        text_features_proj = self.text_features @ self.proj_text

        visual_prototypes = torch.concat((self.visual_prototypes, self.virtual_prototypes), dim=0)
        visual_prototypes_proj = visual_prototypes @ self.proj_image
        
        image_features_proj_norm = image_features_proj / image_features_proj.norm(dim=-1, keepdim=True)
        text_features_proj_norm = text_features_proj / text_features_proj.norm(dim=-1, keepdim=True)
        visual_prototypes_proj_norm = visual_prototypes_proj / visual_prototypes_proj.norm(dim=-1, keepdim=True)
        
        if (phase == "test"):
            logits = self.logit_scale * image_features_proj_norm @ text_features_proj_norm.t()
        else:
            batch_x = torch.cat((image_features_proj_norm, visual_prototypes_proj_norm), dim=0)
            logits = self.logit_scale * batch_x @ text_features_proj_norm.t()
            
        all_features = torch.cat((image_features_proj, visual_prototypes_proj, text_features_proj), dim=0)
        all_features = self.fusion(all_features, phase)
        
        image_features_proj = all_features[:-2*self.n_cls, ...]
        visual_prototypes_proj = all_features[-2*self.n_cls:-self.n_cls, ...]
        text_features_proj = all_features[-self.n_cls:, ...]

        image_features_proj_norm = image_features_proj / image_features_proj.norm(dim=-1, keepdim=True)
        text_features_proj_norm = text_features_proj / text_features_proj.norm(dim=-1, keepdim=True)
        visual_prototypes_proj_norm = visual_prototypes_proj / visual_prototypes_proj.norm(dim=-1, keepdim=True)
        
        if (phase == "test"):
            logits_new_vt = self.logit_scale * image_features_proj_norm @ text_features_proj_norm.t()
            logits_new_vv = self.logit_scale_visual * image_features_proj_norm @ visual_prototypes_proj_norm.t()
        
        else:
            batch_x = torch.cat((image_features_proj_norm, visual_prototypes_proj_norm), dim=0)
            logits_new_vt = self.logit_scale * batch_x @ text_features_proj_norm.t()   
            logits_new_vv = self.logit_scale_visual * batch_x @ visual_prototypes_proj_norm.t()
        
        return logits, logits_new_vt, logits_new_vv
      

class ProtoManager(nn.Module):

    def __init__(self, cfg, clip_model, text_features, visual_prototypes):
        super().__init__()        
            
        self.register_buffer("text_features", text_features)
        self.register_buffer("visual_prototypes", visual_prototypes)
        
        dtype = clip_model.dtype
        logit_scale = clip_model.logit_scale.data
        scale_factor = cfg.TRAINER.SCALE
        mask = cfg.TRAINER.MASK
        vir_init = cfg.TRAINER.VIR_INIT

        head = Head(self.visual_prototypes, self.text_features, logit_scale, scale_factor, mask, vir_init).type(clip_model.dtype)

        self.head = head
        

class CustomCLIP(nn.Module):

    def __init__(self, cfg, clip_model, text_features, visual_prototypes):
        super().__init__()
        self.clip_model = clip_model     
        self.proto_manager = ProtoManager(cfg, clip_model, text_features, visual_prototypes)
        
    def forward(self, image, phase="train"):
        head = self.proto_manager.head
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)

        logits, logits_new_vt, logits_new_vv = head(image_features, phase)

        return logits, logits_new_vt, logits_new_vv     
        
@TRAINER_REGISTRY.register()
class Candle(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
    
    def save_variable(self, v, fp): 
        f = open(fp,'wb')        
        pickle.dump(v, f, 0)               
        f.close()  
        return
        
    def load_variable(self, fp):
        try:
            f = open(fp,'rb')
            r = pickle.load(f)
            f.close()
            return r
  
        except EOFError:  
            return "Error: empty file!"


    def get_features(self):
        data_loader = deepcopy(self.train_loader_x)
        tfm_test = build_transform(self.cfg, is_train=False)
        data_loader.dataset.transform = tfm_test 
        
        features = []
        labels = []
        
        def parse_batch_train(batch):
            input = batch["img"]
            label = batch["label"]
            input = input.to(self.device)
            label = label.to(self.device)
            return input, label 

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = parse_batch_train(batch)
            with torch.no_grad():
                feature = self.clip_model.encode_image(input)
            features.append(feature)
            labels.append(label)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        return features, labels

    def get_prototypes(self, features, labels, num_classes):
        prototypes = torch.empty(num_classes,
                                 features.shape[-1],
                                 dtype=features.dtype)
        for i in range(num_classes):
            prototypes[i] = torch.mean(features[(labels == i).nonzero(), :].squeeze(dim=1), axis=0, keepdim=True)

        return prototypes

    def build_model(self):
        cfg = self.cfg
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        self.clip_model = clip_model.to(self.device)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        
        phase = cfg.TRAINER.PHASE
        task = cfg.TRAINER.TASK
        self.task = task + phase
        if (phase == "train"):
            cachepath = osp.join(cfg.OUTPUT_DIR, "cache.txt")
            if(task == "B2N"):
                classnames_all = self.dm.dataset.all_classnames
                num_classes_all = len(classnames_all)
                num_classes_train = self.dm.dataset.num_classes 
                self.num_classes_all = num_classes_all
                self.num_classes_train = num_classes_train
            
            elif(task == "XD"):
                classnames_train = self.dm.dataset.all_classnames        
                num_classes_train = len(classnames_train) 
                self.num_classes_train = num_classes_train
                class_names_new = self.dm.dataset_new.all_classnames
                classnames_all = classnames_train
                classnames_all.extend(class_names_new)
                self.num_classes_all = len(classnames_all)

            v = [self.num_classes_all, self.num_classes_train]
            self.save_variable(v, cachepath)
        
            template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            class_tokens = [template.format(c.replace("_", " ")) for c in classnames_all]
            print(f"Prompts: {class_tokens}")
            class_tokens = torch.cat([clip.tokenize(p) for p in class_tokens])
            class_tokens = class_tokens.to(self.device)

            print(f"Calculating text features")

            with torch.no_grad():
                text_features = self.clip_model.encode_text(class_tokens)

            print(f"Calculating visual features")

            visual_features, labels = self.get_features()

            print(f"Calculating visual prototypes")
            visual_prototypes = self.get_prototypes(visual_features, labels, num_classes_train) 
        
            visual_prototypes = visual_prototypes.to(self.device)
        
        elif (phase == "test"):
            cachepath = osp.join(os.getcwd(), cfg.MODEL_DIR, "cache.txt")
            subsample = cfg.DATASET.SUBSAMPLE_CLASSES 
            self.task += subsample  
            v = self.load_variable(cachepath)
            self.num_classes_all, self.num_classes_train = v[0], v[1]
            
            text_features = torch.empty((self.num_classes_all, 512), dtype=clip_model.dtype).to(self.device)
            visual_prototypes = torch.empty((self.num_classes_train, 512), dtype=clip_model.dtype).to(self.device)
        
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, clip_model, text_features, visual_prototypes)
        
        cls_num_list = self.dm.dataset.get_cls_num_list()
        cls_num_list_new = [0 for i in range(self.num_classes_all - self.num_classes_train)]
        cls_num_list.extend(cls_num_list_new)
        cls_num_list = torch.Tensor(cls_num_list).to(self.device)
        
        self.criterion = CompensatingLogitAdjustedLoss(cls_num_list=cls_num_list)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "proto_manager" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.proto_manager, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.proto_manager, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("proto_manager", self.model.proto_manager, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None
        
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = self.criterion(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, output_new_vt, output_new_vv = self.model(image)
            loss1 = self.criterion(output, label)
            loss2 = self.criterion(output_new_vt, label)
            loss3 = self.criterion(output_new_vv, label)
            loss = loss1 + loss2 + loss3 
            self.model_backward_and_update(loss)
        
        n = self.num_classes_all
        m = self.num_classes_train 
        
        loss_summary = {
            "loss": loss.item(),
            "loss1": loss1.item(),
            "loss2": loss2.item(),
            "loss3": loss3.item(),
            "acc": compute_accuracy((output_new_vv + output_new_vt)[:-n, :m], label[:-n])[0].item(),
        }
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, input):
        output, output_new_vt, output_new_vv = self.model(input, phase="test")     
        m = self.num_classes_train
        
        # (batchsize, :num_classes_train)
        output_base, output_new_vt_base, output_new_vv_base = output[..., :m], output_new_vt[..., :m], output_new_vv[..., :m]
        # (batchsize, num_classes_train:)
        output_new, output_new_vt_new, output_new_vv_new = output[..., m:], output_new_vt[..., m:], output_new_vv[..., m:]
        
        if(self.task == "B2Ntrainbase" or self.task == "XDtrainall"):
            return output_new_vt_base + output_new_vv_base
        elif(self.task == "B2Ntestnew" or self.task == "XDtestall"):
            return output_new_vt_new  + output_new_vv_new 
        else:
            return output_new_vt + output_new_vv

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        
        # add labels of visual prototypes
        proto_label = torch.tensor([i for i in range(self.num_classes_all)])
        label = torch.concat((label, proto_label), dim=0)

        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

        
    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
            
        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            
            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} "
                  'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
