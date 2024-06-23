import os
import pickle
import math
import random
import numpy as np
from collections import defaultdict, Counter

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing


@DATASET_REGISTRY.register()
class OxfordPets(DatasetBase):

    dataset_dir = "oxford-iiit-pet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json") 
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            trainval = self.read_data(split_file="trainval.txt")
            test = self.read_data(split_file="test.txt")
            train, val = self.split_trainval(trainval)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
                          
        self.all_classnames = self.get_classnames(train)
        
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)
        
        imb_ratio = cfg.DATASET.IMBALANCE_RATIO
        train = self.gen_imbalanced_data(train, imb_ratio)
        
        super().__init__(train_x=train, val=val, test=test)
        
    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1  # convert to 0-based index
                item = Datum(impath=impath, label=label, classname=breed)
                items.append(item)

        return items

    @staticmethod
    def gen_imbalanced_data(train, imb_ratio, imb_type='exp'):
        if (imb_ratio == 1.0):
            return train
             
        labels = []
        for item in train:
            label = item.label
            labels.append(label)
        
        labels_np = np.array(labels, dtype=np.int64)
        classes = np.unique(labels)
        cls_num = len(classes)

        # Calculate a list of pairs (label, class_size), where class_size is the number of samples of that label
        # Sort the list according to class_size, descendingly, E.g. [(2, 7), (0, 5), (1, 3)]  
        
        label_count = Counter(labels).most_common(cls_num)        
        # print(label_count)

        _, img_max = label_count[0]

        # Use img_max to calculate img_num_per_cls, E.g. [7, 4, 1] 
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = max(img_max * (imb_ratio**(cls_idx / (cls_num - 1.0))), 1)
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_ratio))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
            
        # print(img_num_per_cls)
        # Initialize an empty list, and assign img_num to the corresponding label, E.g. [4, 1, 7]    
        img_num_per_cls_reassigned = [0 for i in range(cls_num)]
        
        for i in range(cls_num):
            label_idx, _ = label_count[i]
            img_num_per_cls_reassigned[label_idx] = img_num_per_cls[i]
        
        # print(img_num_per_cls_reassigned)
        # 5. Drafting designated number of samples from each classes of the original dataset 
        
        num_per_cls_dict = dict()  
        select_idx = []
        
        for the_class, the_img_num in zip(classes, img_num_per_cls_reassigned):
            num_per_cls_dict[the_class] = the_img_num
            idx = np.where(labels_np == the_class)[0]
            np.random.shuffle(idx)
            select_idx.extend(idx[:the_img_num])
            
        train_new = []
 
        for idx in range(len(train)):
            item = train[idx]
            if idx not in select_idx:
                continue
            train_new.append(item)
 
        return train_new
        
        
    
    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test
    
    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args
        
        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)
        
        return output

    @staticmethod
    def get_classnames(data_source):
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return classnames
    
    def get_cls_num_list(self):
        cls_num_list = [0] * self.num_classes
        for item in self.train_x:
            cls_num_list[item.label] += 1
        return cls_num_list
