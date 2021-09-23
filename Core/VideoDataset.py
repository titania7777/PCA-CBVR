import os
import csv
import json
import numpy as np
from PIL import Image
from glob import glob
from termcolor import cprint

import torch
from torch.utils.data import Dataset

from . import Transforms
from .Util import OptionError, PathError

class VideoDataset(Dataset):
    def __init__(self, frame_path:str, annotation_path:str=None, frame_size:int=112,sequence_length:int=16, max_interval:int=-1,
        random_interval:bool=False, random_start_position:bool=False, uniform_frame_sample:bool=True,
        random_pad_sample:bool=False, train:bool=False, channel_first:bool=True):

        self.frame_path = frame_path
        self.sequence_length = sequence_length
        self.max_interval = max_interval
        self.random_interval = random_interval
        self.random_start_position = random_start_position
        self.uniform_frame_sample = uniform_frame_sample
        self.random_pad_sample = random_pad_sample
        self.train = train
        self.channel_first = channel_first
        
        # annotation
        annotation_format = annotation_path.split(".")[-1]
        if annotation_format == "csv":
            cprint(f"custom annotation path: {annotation_path}", "cyan", end=", ")
            self.label_list, self.categorie_dict, self.frame_index_list = self._read_custom_annotation(annotation_path)
        elif annotation_format == "json":
            cprint(f"sampled annotation path: {annotation_path}", "cyan", end=", ")
            self.label_list, self.categorie_dict, self.frame_index_list = self._read_sampled_annotation(annotation_path)
        else:
            raise OptionError("should be specified at least one annotation path :(")
        
        # number of videos
        cprint(f"number of videos: {len(self.label_list)}", "cyan", end=", ")

        # number of classes
        self.num_classes = len(self.categorie_dict)
        cprint(f"number of classes: {self.num_classes}", "cyan")

        # transformer
        if self.train:
            self.transform = Transforms.Compose([
                Transforms.RandomResizedCrop((frame_size, frame_size)),
                Transforms.RandomHorizontalFlip(p=0.5),
                Transforms.ToTensor(),
                Transforms.Normalize(
                    mean = [0.4345, 0.4051, 0.3775],
                    std = [0.2768, 0.2713, 0.2737]
                ),
            ])
        else:
            self.transform = Transforms.Compose([
                Transforms.Resize(frame_size),
                Transforms.CenterCrop(frame_size),
                Transforms.ToTensor(),
                Transforms.Normalize(
                    mean = [0.4345, 0.4051, 0.3775],
                    std = [0.2768, 0.2713, 0.2737]
                ),
            ])

    def __len__(self):
        return len(self.label_list)

    def get_category(self, label:int):
        return self.categorie_dict[label]

    def get_few_shot_sampler(self, num_iters:int, way:int, shot:int, query:int):
        return CategoriesSampler(label_list=[label for _, label in self.label_list], num_iters=num_iters, way=way, shot=shot, query=query)

    def _read_custom_annotation(self, annotation_path:str):
        label_list = [] # [[sub directory path, label]]
        categorie_dict = {} # {label: category}
        with open(annotation_path, "r") as f:
            for rows in csv.reader(f):
                sub_directory_path = rows[0]
                label = int(rows[1])
                label_list.append([sub_directory_path, label])
                if label not in categorie_dict:
                    categorie_dict[label] = rows[2] # category

        return label_list, categorie_dict, None
    
    def _read_sampled_annotation(self, annotation_path:str):
        label_list = [] # [[sub directory path, label]]
        categorie_dict = {} # {label: category}
        frame_index_list = []
        with open(annotation_path, "r") as f:
            data = json.load(f)
            for sub_directory_path in data:
                sub_data = data[sub_directory_path]
                label = int(sub_data["label"])
                label_list.append([sub_directory_path, label])
                frame_index_list.append(sorted(sub_data["index"][:self.sequence_length]))
                if label not in categorie_dict:
                    categorie_dict[label] = sub_data["category"] # category
        return label_list, categorie_dict, frame_index_list
    
    def _frame_pad_sampler(self, num_frames:int):
        # length -> array
        original_sequence = np.arange(num_frames)
        required_frames = self.sequence_length - num_frames

        # add pads
        if self.random_pad_sample:
            pad_sequence = np.random.choice(original_sequence, required_frames)
        else:
            pad_sequence = np.repeat(original_sequence[0], required_frames)

        # sorting
        new_sequence = sorted(np.append(original_sequence, pad_sequence, axis=0))

        return new_sequence

    def _frame_index_sampler(self, num_frames:int):
        if self.uniform_frame_sample:
            # start position
            start_position = 0
            if self.random_start_position:
                start_position = np.random.randint(num_frames - self.sequence_length + 1)

            # interval: number of frames between prev and next frame
            # (interval + 1) x (sequence_length - 1) <= num_frames - start_position - 1
            interval = int((((num_frames - start_position - 1) / (self.sequence_length - 1)) - 1))
            
            # max interval
            if self.max_interval != -1 and interval > self.max_interval:
                interval = self.max_interval
            
            # random interval
            if self.random_interval:
                interval = np.random.randint(interval + 1)
            
            sampled_index = np.arange(start=start_position, stop=num_frames, step=interval + 1)[:self.sequence_length]
        else:
            sampled_index = sorted(np.random.permutation(num_frames)[:self.sequence_length])
            
        return sampled_index

    def __getitem__(self, index):
        sub_directory_path, label = self.label_list[index]

        # for hmdb51
        replaced_sub_directory_path = sub_directory_path.replace("]", "?")

        # sorting
        image_path_list = np.array(sorted(glob(os.path.join(self.frame_path, replaced_sub_directory_path, "*")), key=lambda x: int(x.split("/")[-1].split(".")[0])))

        # path check
        num_frames = len(image_path_list)
        if num_frames == 0:
            raise PathError(f"'{sub_directory_path}' not exists or empty :(")
        
        # sampling 
        if num_frames >= self.sequence_length:
            if self.frame_index_list:
                # sampled frame index
                frame_index = self.frame_index_list[index]
            else:
                frame_index = self._frame_index_sampler(num_frames)
        else:
            frame_index = self._frame_pad_sampler(num_frames)
            
        image_path_list = image_path_list[frame_index]

        # to tensor
        data = torch.stack([self.transform(Image.open(image_path)) for image_path in image_path_list], dim=1 if self.channel_first else 0)

        return data, label, sub_directory_path

class CategoriesSampler():
    def __init__(self, label_list:list, num_iters:int, way:int, shot:int, query:int):
        self.num_iters = num_iters
        self.way = way
        self.num_shots = shot + query
        label_list = np.array(label_list)
        
        self.label_index_dict = {}
        for label in np.unique(label_list):
            self.label_index_dict[label] = np.argwhere(label_list == label).reshape(-1)

    def __len__(self):
        return self.num_iters
    
    def __iter__(self):
        for _ in range(self.num_iters):
            batch_list = []
            for label in np.random.permutation(list(self.label_index_dict.keys()))[:self.way]:
                batch_list.append(np.random.permutation(self.label_index_dict[label])[:self.num_shots])
                
            yield np.stack(batch_list).T.reshape(-1)