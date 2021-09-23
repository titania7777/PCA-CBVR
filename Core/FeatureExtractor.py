import os
import sys
import json
import inspect
import numpy as np

import torch
from torch.utils.data import DataLoader

from .Util import PathManager, PrintManager, ModelManager
from .VideoDataset import VideoDataset

class FeatureExtractor(object):
    def __init__(self, frame_path:str, db_annotation_path:str, query_annotation_path:str, save_path:str, model_path:str=None,
        model_name:str="R3D18", shortcut:str="B", pretrained:bool=False, frame_size:int=112, sequence_length:int=16,
        max_interval:int=-1, random_interval:int=False, random_start_position:bool=False, uniform_frame_sample:bool=True,
        random_pad_sample:bool=False, batch_size:int=128, only_cpu:bool=False, gpu_number:int=0):
        
        _, _, _, self.args_dict = inspect.getargvalues(inspect.currentframe())
        self.frame_path = frame_path
        self.db_annotation_path = db_annotation_path
        self.query_annotation_path = query_annotation_path
        self.save_path = save_path
        self.model_path = model_path
        self.model_name = model_name
        self.shortcut = shortcut
        self.pretrained = pretrained
        self.frame_size = frame_size
        self.sequence_length = sequence_length
        self.max_interval = max_interval
        self.random_interval = random_interval
        self.random_start_position = random_start_position
        self.uniform_frame_sample = uniform_frame_sample
        self.random_pad_sample = random_pad_sample
        self.batch_size = batch_size
        self.only_cpu = only_cpu
        self.gpu_number = gpu_number
        self.meta_data = {} # {meta data key (label|category): {save_target: sub-directory path}}

        # path check
        PathManager(frame_path, db_annotation_path, query_annotation_path, model_path).exist(raise_error=True)

    def run(self):
        # path check
        save_path_manager = PathManager(self.save_path)
        if save_path_manager.exist(raise_error=False):
            if save_path_manager.remove(enforce=False):
                save_path_manager.create()
            else:
                return
        else:
            save_path_manager.create()

        # display arguments
        PrintManager(self.save_path).args(args_dict=self.args_dict)

        # initialize
        self._initialize()

        # extraction loop
        self._extraction_loop("db")
        self._extraction_loop("prototype")
        self._extraction_loop("query")
        
        # save meta data
        with open(os.path.join(self.save_path, "meta.json"), "w") as f:
            json.dump(self.meta_data, f, indent=4)
    
    def _initialize(self):
        # video dataset
        self.db = VideoDataset(
            frame_path=self.frame_path,
            annotation_path=self.db_annotation_path,
            frame_size=self.frame_size,
            sequence_length=self.sequence_length,
            max_interval=self.max_interval,
            random_interval=self.random_interval,
            random_start_position=self.random_start_position,
            uniform_frame_sample=self.uniform_frame_sample,
            random_pad_sample=self.random_pad_sample,
            train=False,
            channel_first=True
        )

        self.query = VideoDataset(
            frame_path=self.frame_path,
            annotation_path=self.query_annotation_path,
            frame_size=self.frame_size,
            sequence_length=self.sequence_length,
            max_interval=self.max_interval,
            random_interval=self.random_interval,
            random_start_position=self.random_start_position,
            uniform_frame_sample=self.uniform_frame_sample,
            random_pad_sample=self.random_pad_sample,
            train=False,
            channel_first=True
        )

        # loader
        self.db_loader = DataLoader(self.db, batch_size=self.batch_size, shuffle=False, num_workers=0 if os.name == 'nt' else 4)
        self.query_loader = DataLoader(self.query, batch_size=self.batch_size, shuffle=False, num_workers=0 if os.name == 'nt' else 4)

        # model
        model_manager = ModelManager(only_cpu=self.only_cpu, gpu_number=self.gpu_number, model_path=self.model_path)
        self.encoder = model_manager.get_encoder(model_name=self.model_name, shortcut=self.shortcut, pretrained=self.pretrained, progress=True)
        self.encoder.eval()
        self.device = model_manager.get_device()

    def _extraction_loop(self, save_target:str):
        target_save_path = os.path.join(self.save_path, save_target)
        if save_target == "db":
            dataset = self.db
            dataset_loader = self.db_loader
        elif save_target == "prototype":
            prototype_save_path = os.path.join(self.save_path, "prototype")
            os.makedirs(prototype_save_path) # create directory
            for i, meta_data_key in enumerate(self.meta_data.keys()):
                category = meta_data_key.split("|")[1]
                db_save_path = os.path.join(self.save_path, "db", category)
                prototype_feature_save_path = os.path.join(prototype_save_path, category)

                # meta data
                self.meta_data[meta_data_key]["prototype"] = category
                
                # save prototype
                np.save(prototype_feature_save_path, np.mean(
                        [np.load(os.path.join(db_save_path, sub_directory_path + ".npy")) for sub_directory_path in self.meta_data[meta_data_key]["db"]], axis=0
                    )
                )
                sys.stdout.write("\r['{}' '{}' Calculating... {:.2f}%] ".format(self.model_name, prototype_save_path, ((i+1) / len(self.meta_data.keys())) * 100))
            print(); return
        else:
            dataset = self.query
            dataset_loader = self.query_loader

        with torch.no_grad():
            for i, (data_batch, label_batch, sub_directory_path_batch) in enumerate(dataset_loader):
                feature_batch = self.encoder(data_batch.to(self.device)).detach().cpu().numpy()
                for feature, label, sub_directory_path in zip(feature_batch, label_batch, sub_directory_path_batch):
                    # get category
                    category = dataset.get_category(int(label))
                    
                    # feature path check
                    feature_path = os.path.join(target_save_path, category)
                    if not os.path.exists(feature_path):
                        os.makedirs(feature_path)
                    
                    # get feature path for saving feature
                    splitted_sub_directory_path = sub_directory_path.split("/")
                    if len(splitted_sub_directory_path) > 1:
                        sub_directory_path = "/".join(splitted_sub_directory_path[1:]) # this sub-directory path is not the same as the csv version
                    feature_save_path =  os.path.join(feature_path, sub_directory_path)

                    # save feature
                    np.save(feature_save_path, feature)

                    # meta data
                    meta_data_key = f"{label}|{category}"
                    if meta_data_key in self.meta_data and save_target in self.meta_data[meta_data_key]:
                        self.meta_data[meta_data_key][save_target].append(sub_directory_path)
                    elif meta_data_key in self.meta_data:
                        self.meta_data[meta_data_key][save_target] = [sub_directory_path]
                    else:
                        self.meta_data[meta_data_key] = {save_target: [sub_directory_path]}
                    
                    sys.stdout.write("\r['{}' '{}' Extracting... {:.2f}%] ".format(self.model_name, target_save_path, ((i+1) / len(dataset_loader)) * 100))
        print()