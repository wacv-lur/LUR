import pandas as pd
from operator import itemgetter
from PIL import Image

from glob import glob
import random
from os.path import exists
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
from torchvision import transforms
from  torchvision.transforms.functional import pil_to_tensor


class NuscenesDataset(Dataset):
    def __init__(self,
                 base_dir:str="./",
                 label_file_path:str="./nuscenes_intention.xlsx",
                 exp_type:str = "regular", # ood_min, ood_max
                 split_type:str="train", # test
                 resize_to = (224,224),
                 mean = [0.485, 0.456, 0.406], 
                 std = [0.229, 0.224, 0.225], 
                 ood :bool =False,
                 ):
        self.split_type = split_type
        self.df = self._load_df(label_file_path, split_type, exp_type, ood)
        self.num_classes = self.df.intention.nunique()
        self.class_labels = sorted(self.df.intention.unique().tolist())
        self.label2id = {label: i for i, label in enumerate(self.class_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.exp_type = exp_type
        self.transform =  Compose([ApplyTransformToKey(
                                key="video",
                                transform=Compose([
                                        UniformTemporalSubsample(16),
                                        Lambda(lambda x: x / 255.0),
                                        Normalize(mean, std),
                                        Resize(resize_to), 
                                ]),),])
        self.flip_transform = RandomHorizontalFlip(p=1.0)
        
    def __len__(self):
        return len(self.df)
    
    def load_target(self, label):
        label =  self.label2id[label]
        one_hot_label = F.one_hot(torch.tensor(label), 
                                  self.num_classes).float()
        return one_hot_label
    
    def label_flip(self, original_label):
        flip_dict={'left lane change':'right lane change',
                   'right lane change':'left lane change',
                   'left turn': 'right turn',
                   'right turn': 'left turn',}
        
        if self.exp_type == "ood_min":
            del flip_dict["right lane change"], flip_dict["left lane change"]
        
        if original_label in list(flip_dict.keys()):
            flipped_label = flip_dict[original_label]
            new_label = self.label2id[flipped_label]
            return F.one_hot(torch.tensor(new_label), self.num_classes).float() 
        else:
            return self.load_target(original_label)
        
    def flip(self,label, video):
        if random.random() > 0.5:        
            video = torch.stack([self.flip_transform(frame) for frame in video]) 
        target = self.label_flip(label)
        return target, video
            
    def _load_df(self,
                 label_file_path:str, 
                 split_type:str='train', 
                 exp_type:str='regular',
                 ood:bool=False):
        df = pd.read_excel(label_file_path).drop(columns={"Unnamed: 0"})
        df = df[df["split"]==split_type]
        if (exp_type == "ood_min") and ood is True:
            df = df[df["intention"] == "right lane change"]
        if (exp_type == "ood_min") and ood is False:
            df = df[df["intention"] != "right lane change"] 
        if (exp_type == "ood_max") and ood is True:
            df = df[df["intention"] == "moving forward"]
        if (exp_type == "ood_max") and ood is False:
            df = df[df["intention"] != "moving forward"]
        df = df.reset_index().drop(columns="index")
        return df
    
    def __getitem__(self, index, model_frame_length=16):
        label = self.df.loc[index]["intention"]
        target = self.load_target(label)
        frame_paths = sorted(literal_eval(self.df.loc[index]["frames"]))
        num_frames = len(frame_paths)
        frames = torch.stack([pil_to_tensor(Image.open(f"./CAM_FRONT/{x}")) for x in frame_paths])
        if num_frames < model_frame_length:
            required_frames = model_frame_length - num_frames
            padding_frames = torch.stack([torch.zeros_like(frames[:1,:,...]) for _ in range(required_frames)], dim=1)
            if len(padding_frames.shape) ==5:
                padding_frames = padding_frames.squeeze()
            if len(padding_frames.shape) ==3:
                padding_frames = padding_frames.unsqueeze(dim=0)
            frames = torch.concat([frames, padding_frames])
        
        if (self.split_type=="train"):
            frames = self.transform({"video":frames.permute(1,0,2,3)})["video"]
        elif (self.split_type=="test"):
            frames = self.transform_test({"video":frames.permute(1,0,2,3)})["video"]
              
        if self.split_type == "train":
            target, frames = self.flip(label, frames)
        return {"label":target, "exterior_video":frames, "idx": index}
