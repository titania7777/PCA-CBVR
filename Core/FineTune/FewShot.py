import os
import inspect
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from termcolor import cprint

from ..Util import PathManager, PrintManager, ModelManager, StatusCalculator, ModelSaver
from ..VideoDataset import VideoDataset

class FewShot(object):
    def __init__(self, frame_path:str, train_annotation_path:str, val_annotation_path:str, save_path:str, model_path:str=None,
        model_name:str="R3D18", shortcut:str="B", pretrained:bool=False, frame_size:int=112, sequence_length:int=16,
        max_interval:int=-1, random_interval:int=False, random_start_position:bool=False, uniform_frame_sample:bool=True,
        random_pad_sample:bool=False, train_iter_size:int=100, val_iter_size:int=200, way:int=5, shot:int=1, query:int=10,
        only_cpu:bool=False, gpu_number:int=0, cudnn_benchmark:bool=True, num_workers:int=4, layer:int=1, num_epochs:int=30,
        learning_rate:float=1e-4, scheduler_step_size:float=10, scheduler_gamma:float=0.9, momentum:float=0.9, weight_decay=1e-3):
        
        _, _, _, self.args_dict = inspect.getargvalues(inspect.currentframe())
        self.frame_path = frame_path
        self.train_annotation_path = train_annotation_path
        self.val_annotation_path = val_annotation_path
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
        self.train_iter_size = train_iter_size
        self.val_iter_size = val_iter_size
        self.way = way
        self.shot = shot
        self.query = query
        self.only_cpu = only_cpu
        self.gpu_number = gpu_number
        self.cudnn_benchmark = cudnn_benchmark
        self.num_workers = num_workers
        self.layer = layer
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.momentum = momentum
        self.weight_decay = weight_decay

        # path check
        PathManager(frame_path, train_annotation_path, val_annotation_path, model_path).exist(raise_error=True)

    def run(self):
        # path check
        save_path_manager = PathManager(self.save_path)
        if save_path_manager.exist(raise_error=False):
            if save_path_manager.remove(enforce=False):
                save_path_manager.create()
            else:
                return # skip
        else:
            save_path_manager.create()
        
        # display arguments
        PrintManager(self.save_path).args(args_dict=self.args_dict)

        # initialize
        self._initialize()

        # ================================================
        # MAIN EPOCHS
        # ================================================
        cprint(f"[train] number of videos: {len(self.train)}, [val] number of videos: {len(self.val)}", "cyan")
        cprint(f"{self.way}-way {self.shot}-shot {self.query}-query", "cyan")
        cprint(f"train episodes: {self.num_epochs * self.train_iter_size}, val episodes: {self.num_epochs * self.val_iter_size}", "cyan")

        for current_epoch in range(1, self.num_epochs):
            # train
            self._train_iteration(current_epoch)

            # val
            self._val_iteration(current_epoch)

            # initialize calculator
            self.train_calculator.reset()
            self.val_calculator.reset()

            # lr scheduler
            self.lr_scheduler.step()
    
    def _initialize(self):
        # video dataset
        self.train = VideoDataset(
            frame_path=self.frame_path,
            annotation_path=self.train_annotation_path,
            frame_size=self.frame_size,
            sequence_length=self.sequence_length,
            max_interval=self.max_interval,
            random_interval=self.random_interval,
            random_start_position=self.random_start_position,
            uniform_frame_sample=self.uniform_frame_sample,
            random_pad_sample=self.random_pad_sample,
            train=True,
            channel_first=True
        )
        self.val = VideoDataset(
            frame_path=self.frame_path,
            annotation_path=self.val_annotation_path,
            frame_size=self.frame_size,
            sequence_length=self.sequence_length,
            max_interval=-1,
            random_interval=False,
            random_start_position=False,
            uniform_frame_sample=True,
            random_pad_sample=False,
            train=False,
            channel_first=True
        )

        # loader
        self.train_loader = DataLoader(
            self.train,
            batch_sampler=self.train.get_few_shot_sampler(self.train_iter_size, self.way, self.shot, self.query),
            pin_memory=True,
            num_workers=0 if os.name == 'nt' else self.num_workers
        )
        self.val_loader = DataLoader(
            self.val,
            batch_sampler=self.val.get_few_shot_sampler(self.val_iter_size, self.way, self.shot, self.query),
            pin_memory=True,
            num_workers=0 if os.name == 'nt' else self.num_workers
        )
        
        # model (encoder)
        self.model_manager = ModelManager(only_cpu=self.only_cpu, gpu_number=self.gpu_number, model_path=self.model_path, cudnn_benchmark=self.cudnn_benchmark)
        encoder = self.model_manager.get_encoder(model_name=self.model_name, shortcut=self.shortcut, pretrained=self.pretrained, progress=True)
        if self.layer == -1: # for full training
            self.encoder_freeze, self.encoder_tune = lambda x: x, encoder
        else:
            self.encoder_freeze, self.encoder_tune = self.model_manager.split_model(encoder, layer=self.layer)
        self.device = self.model_manager.get_device()

        # display layers
        PrintManager(self.save_path).layers(encoder_freeze=self.encoder_freeze, encoder_tune=self.encoder_tune)

        # optimizer and scheduler
        self.optimizer = torch.optim.SGD(self.encoder_tune.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)

        # initialize calculator
        self.train_calculator = StatusCalculator(num_batchs=len(self.train_loader), topk=1) # top 1 acc
        self.val_calculator = StatusCalculator(num_batchs=len(self.val_loader), topk=1) # top 1 acc
        self.saver = ModelSaver(per=10, save_path=self.save_path) # save per 10 iteration after
        self.printer = PrintManager(self.save_path)

    def inference(self, data_batch, encoder):
        data_batch = data_batch.to(self.device)
        pivot = self.way * self.shot
        
        # shot, query, label
        data_batch_shot, data_batch_query = data_batch[:pivot], data_batch[pivot:]
        label_batch = torch.arange(self.way).repeat(self.query).to(self.device)

        # prediction
        data_batch_shot = encoder(data_batch_shot)
        data_batch_shot = data_batch_shot.reshape(self.shot, self.way, -1).mean(dim=0) # prototype
        data_batch_shot = F.normalize(data_batch_shot, dim=-1)
        data_batch_query = F.normalize(encoder(data_batch_query), dim=-1)
        pred_batch = torch.mm(data_batch_query, data_batch_shot.t())

        return pred_batch, label_batch
    
    def _train_iteration(self, current_epoch:int):
        self.train.transform.initialize() # initialize random parameters
        self.encoder_tune.train()
        for i, (data_batch, _, _) in enumerate(self.train_loader):
            # inference
            pred_batch, label_batch = self.inference(data_batch, lambda x: self.encoder_tune(self.encoder_freeze(x)))

            # loss
            loss = F.cross_entropy(pred_batch, label_batch)

            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # set
            self.train_calculator.set_loss(loss)
            self.train_calculator.set_acc(pred_batch, label_batch)

            # status print
            self.printer.status(
                "train", current_epoch, self.num_epochs, i + 1, len(self.train_loader),
                self.train_calculator.get_loss(), self.train_calculator.get_loss(mean=True),
                self.train_calculator.get_acc() * 100, self.train_calculator.get_acc(mean=True) * 100
            )
        print()

    def _val_iteration(self, current_epoch:int):
        self.encoder_tune.eval()
        with torch.no_grad():
            for i, (data_batch, _, _) in enumerate(self.val_loader):
                # inference
                pred_batch, label_batch = self.inference(data_batch, lambda x: self.encoder_tune(self.encoder_freeze(x)))

                # loss
                loss = F.cross_entropy(pred_batch, label_batch)

                # set
                self.val_calculator.set_loss(loss)
                self.val_calculator.set_acc(pred_batch, label_batch)

                # print result
                self.printer.status(
                    "val", current_epoch, self.num_epochs, i + 1, len(self.val_loader),
                    self.val_calculator.get_loss(), self.val_calculator.get_loss(mean=True),
                    self.val_calculator.get_acc() * 100, self.val_calculator.get_acc(mean=True) * 100
                )
                
            self.saver.auto_save(self.model_manager.merge_model(self.encoder_freeze, self.encoder_tune), best_acc=self.val_calculator.get_acc(best=True), best_loss=self.val_calculator.get_loss(best=True))
            print(f" Best Accuracy: {self.val_calculator.get_acc(best=True) * 100:.2f}%")