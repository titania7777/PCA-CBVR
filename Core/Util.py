import os
import sys
import time
import shutil
from typing import OrderedDict
from termcolor import cprint, colored

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.hub import load_state_dict_from_url

from .ResNet.R3D import r3d_18, r3d_34, r3d_50
from .ResNet.R2Plus1D import r2plus1d_18, r2plus1d_34, r2plus1d_50

class OptionError(Exception):
    def __init__(self, message:str):
        super().__init__(colored(message ,"red"))

class PathError(Exception):
    def __init__(self, message:str):
        super().__init__(colored(message ,"red"))

class StatusCalculator():
    def __init__(self, num_batchs:int, topk:int=1):
        self.num_batchs = num_batchs
        self.topk = topk
        self.reset()

        # for best loss score
        self.best_loss = 10000000000.0

        # for best accuracy score
        self.best_acc = 0.0

    def set_loss(self, loss:torch.Tensor):
        with torch.no_grad():
            # counting and memorize loss value
            self.counter_loss += 1
            self.current_loss = loss.item()
            self.total_loss += self.current_loss
            
            # mean loss
            self.mean_loss = self.total_loss / self.counter_loss
            
            # best loss (low value)
            if self.counter_loss == self.num_batchs and self.mean_loss < self.best_loss:
                self.best_loss = self.mean_loss

    def get_loss(self, best:bool=False, mean:bool=False):
        if best:
            return self.best_loss
        else:
            if mean:
                return self.mean_loss
            else:
                return self.current_loss

    def set_acc(self, pred_batch:torch.Tensor, label_batch:torch.Tensor):
        with torch.no_grad():
            batch_size = len(label_batch)
            self.counter_batch += 1
            self.counter_acc += batch_size

            # topk accuracy score
            _, indices = torch.topk(pred_batch, k=self.topk, dim=1, largest=True, sorted=False)
            self.current_corrects = sum(torch.any(indices == label_batch.unsqueeze(1).repeat(1, self.topk), dim=1)).item()
            self.total_corrects += self.current_corrects

            # score calculation
            self.current_acc = self.current_corrects / batch_size
            self.mean_acc = self.total_corrects / self.counter_acc

            # best accuracy (high value)
            if self.counter_batch == self.num_batchs and self.mean_acc > self.best_acc:
                self.best_acc = self.mean_acc

    def get_acc(self, best:bool=False, mean:bool=False):
        if best:
            return self.best_acc
        else:
            if mean:
                return self.mean_acc
            else:
                return self.current_acc

    def reset(self, best=False):
        # for loss
        self.total_loss = 0.0
        self.current_loss = 0.0
        self.mean_loss = 0.0
        self.counter_loss = 0

        # for accuracy
        self.total_corrects = 0
        self.current_corrects = 0
        self.mean_acc = 0.0
        self.current_acc = 0.0
        self.counter_batch = 0
        self.counter_acc = 0

        if best:
            self.best_loss = 10000000000.0
            self.best_acc = 0.0

class ModelSaver():
    def __init__(self, per:int, save_path:str):
        self.counter = 0
        self.per = per
        self.save_path = save_path
        self.best_acc = 0.0
        self.best_loss = 10000000000.0

    def auto_save(self, model:nn.Sequential, best_acc:float=None, best_loss:float=None):
        self.counter += 1
        if self.counter % self.per == 0:
            torch.save(model.state_dict(), os.path.join(self.save_path, f"{self.counter}ep.pth"))
        
        if best_acc:
            if best_acc > self.best_acc:
                self.best_acc = best_acc
                torch.save(model.state_dict(), os.path.join(self.save_path, f"best_acc.pth"))
        
        if best_loss:
            if best_loss < self.best_loss:
                self.best_loss = best_loss
                torch.save(model.state_dict(), os.path.join(self.save_path, f"best_loss.pth"))

class PathManager(object):
    def __init__(self, *path_list:list):
        self.path_list = path_list

    def exist(self, raise_error:bool=False):
        for path in self.path_list:
            exist = os.path.exists(path)
            if raise_error and not exist:
                raise PathError(f"'{path}' does not exist :(")
            else:
                return exist # true or false

    def remove(self, enforce:bool=False):
        for path in self.path_list:
            if self.exist(path):
                if enforce:
                    self._remove(path)
                else:
                    while True:
                        cprint(f"'{path}' already exists, do you want to continue after remove that ? [yes(y)/no(n)/skip(s)] ", "cyan", end="")
                        response = input()

                        # yes
                        if response == "y" or response == "Y" or response == "yes":
                            self._remove(path)
                            return True

                        # no
                        if response == "n" or response == "N" or response == "no":
                            cprint("script terminated :(", "red")
                            exit()
                        
                        # skip
                        if response == "s" or response == "S" or response == "skip":
                            cprint("skipping...", "cyan")
                            return False

    def _remove(self, path:str):
        cprint("removing...", "cyan", end=" ", flush=True)
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
        cprint("complete !!", "cyan")
    
    def create(self, raise_error:bool=False):
        for path in self.path_list:
            exist = os.path.exists(path)
            if raise_error and exist:
                raise PathError(f"'{path}' already exists :(")
            elif not exist:
                os.makedirs(path)

class PrintManager(object):
    def __init__(self, save_path:str=None):
        self.save_path = save_path

    def args(self, args_dict:dict):
        lines = []
        lines.append("=================================================")
        for i, key in enumerate(args_dict):
            if i == 0:
                continue
            lines.append("[{}] {}:{}".format(i, key, args_dict[key]))
        lines.append("=================================================")

        # print
        [print(line)for line in lines]
        
        # save
        if self.save_path:
            with open(os.path.join(self.save_path, "args.txt"), "w") as f:
                f.write("\n".join(lines))

    def layers(self, encoder_freeze:nn.Sequential, encoder_tune:nn.Sequential, classifier:nn.Sequential=None):
        lines = []
        if "children" in dir(encoder_freeze):
            lines.append("====================FREEZING LAYER====================")
            lines.append(str(encoder_freeze))
        if "children" in dir(encoder_tune):
            lines.append("=====================TUNING LAYER=====================")
            lines.append(str(encoder_tune))
        if "children" in dir(classifier):
            lines.append("===================CLASSIFIER LAYER===================")
            lines.append(str(classifier))
        lines.append("======================================================")
        
        # print
        [print(line)for line in lines]
        
        # save
        if self.save_path:
            with open(os.path.join(self.save_path, "models.txt"), "w") as f:
                f.write("\n".join(lines))

    def status(self, state:str, current_epoch:int, total_epoch:int, current_batch:int, total_batch:int, current_loss:float, loss_mean:float, current_acc:float, acc_mean:float):
        msg = "[{}]-[Epoch {}/{}] [Batch {}/{}] [Loss: {:.4f} (mean: {:.4f}), Acc: {:.2f}% (mean: {:.2f}%)]".format(
                state,
                current_epoch,
                total_epoch,
                current_batch,
                total_batch,
                current_loss,
                loss_mean,
                current_acc,
                acc_mean
            )
        
        sys.stdout.write("\r" + msg)

        if self.save_path:
            with open(os.path.join(self.save_path, "status.txt"), "a") as f:
                f.write(msg + "\n")


class ModelManager(object):
    def __init__(self, only_cpu:bool, gpu_number:int, cudnn_benchmark:bool=False, model_path:str=None):
        self.only_cpu = only_cpu
        self.gpu_number = gpu_number
        self.cudnn_benchmark = cudnn_benchmark
        self.model_path = model_path
        self.hidden_size = None

        # pretrained on kinetics 700
        # original weights: https://drive.google.com/drive/folders/1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4
        self.weights_urls = {
            "R3D18": "https://www.dropbox.com/s/afoqrewod4meewk/r3d18.pth?dl=1",
            "R3D34": "https://www.dropbox.com/s/0vn1vlts8rjo7cw/r3d34.pth?dl=1",
            "R3D50": "https://www.dropbox.com/s/21e539j3kw0dg1s/r3d50.pth?dl=1",
            "R2Plus1D18": "https://www.dropbox.com/s/jby9zmcdz28bwbo/r2plus1d18.pth?dl=1",
            "R2Plus1D34": "https://www.dropbox.com/s/ei1qi0vap7huk7c/r2plus1d34.pth?dl=1",
            "R2Plus1D50": "https://www.dropbox.com/s/8zoz9yeffckvi5y/r2plus1d50.pth?dl=1",
        }

    def get_device(self):
        if torch.cuda.is_available() and not self.only_cpu:
            if self.cudnn_benchmark:
                cudnn.benchmark = True
            return torch.device(f"cuda:{self.gpu_number}")
        else:
            return torch.device("cpu")

    def get_encoder(self, model_name:str, shortcut:str="B", pretrained:bool=False, progress:bool=True):
        if shortcut != "A" and shortcut != "B":
            raise OptionError(f"'{shortcut}' is not supported for model shortcut option :(")

        if model_name == "R3D18":
            model = r3d_18(shortcut)
            self.hidden_size = 512
        elif model_name == "R3D34":
            model = r3d_34(shortcut)
            self.hidden_size = 512
        elif model_name == "R3D50":
            model = r3d_50(shortcut)
            self.hidden_size = 2048
        elif model_name == "R2Plus1D18":
            model = r2plus1d_18(shortcut)
            self.hidden_size = 512
        elif model_name == "R2Plus1D34":
            model = r2plus1d_34(shortcut)
            self.hidden_size = 512
        elif model_name == "R2Plus1D50":
            model = r2plus1d_50(shortcut)
            self.hidden_size = 2048
        else:
            raise OptionError(f"'{model_name}' is not supported for model name option :(")

        device = self.get_device()
        model.to(device)
        cprint(f"{model_name} model load on {device} completed", "cyan")

        if pretrained:
            if self.model_path:
                model.load_state_dict(torch.load(self.model_path, map_location=device))
                cprint(f"{model_name} Weights load completed from '{self.model_path}'", "cyan")
            else:
                if shortcut == "A":
                    raise OptionError("Shortcut A is not supported for pre-trained option :(")
                model.load_state_dict(load_state_dict_from_url(self.weights_urls[model_name], progress=progress))
                cprint(f"{model_name} Weights load completed which pre-trained on kinetics-700", "cyan")
        else:
            model.apply(self.initialize)
            cprint(f"{model_name} Weights initialize completed", "cyan")

        return model

    def get_classifier(self, num_classes:int):
        classifier = nn.Linear(self.hidden_size, num_classes)
        device = self.get_device()
        classifier.to(device)
        cprint(f"Classifier load on {device} completed", "cyan")

        classifier.apply(self.initialize)
        cprint("Classifier Weights initialize completed", "cyan")

        return classifier
    
    def split_model(self, model:nn.Module, layer:int):
        # (layer0 / header)-(layer1)-(layer2)-(layer3)-(layer4)-(layer5 / avgpool)
        #      [freeze layer]                           [tune layer]
        # 1 => [(layer0)-(layer1)-(layer2)-(layer3)]    [(layer4)-(layer5)]
        # 2 => [(layer0)-(layer1)-(layer2)]             [(layer3)-(layer4)-(layer5)]
        # 3 => [(layer0)-(layer1)]                      [(layer2)-(layer3)-(layer4)-(layer5)]
        # 4 => [(layer0)]                               [(layer1)-(layer2)-(layer3)-(layer4)-(layer5)]
        layer_map = {1: -2, 2: -3, 3: -4, 4: -5}
        model = list(model.children())
        encoder_freeze = nn.Sequential(*model[:layer_map[layer]]).apply(self.freeze_all).eval()
        encoder_tune = nn.Sequential(*model[layer_map[layer]:])
        
        return encoder_freeze, encoder_tune

    def merge_model(self, model_freeze:nn.Sequential, model_tune:nn.Sequential):
        model = OrderedDict()
        layer_num = 0

        if "children" in dir(model_freeze):
            for module in list(model_freeze.children()):
                model[f"layer{layer_num}"] = module
                layer_num += 1
        if "children" in dir(model_tune):
            for module in list(model_tune.children()):
                model[f"layer{layer_num}"] = module
                layer_num += 1

        return nn.Sequential(model)
    
    def freeze_all(self, model:nn.Module):
        for param in model.parameters():
            param.requires_grad = False

    def initialize(self, model:nn.Module):
        for module in model.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.xavier_normal_(module.weight)
                if module.bias != None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias != None:
                    nn.init.constant_(module.bias, 0)
