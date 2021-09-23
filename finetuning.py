import os
import torch
import argparse

from Core.FineTune.Categorical import Categorical
from Core.FineTune.FewShot import FewShot

parser = argparse.ArgumentParser()
parser.add_argument("--db_root_path", type=str, default="./DB")
parser.add_argument("--model_root_path", type=str, default="./Models")
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--tune_target", type=str, default="UCF101", choices=["UCF101", "HMDB51", "ActivityNet"])
parser.add_argument("--model", type=str, default="R3D18", choices=["R3D18", "R3D34", "R3D50", "R2Plus1D18", "R2Plus1D34", "R2Plus1D50"])
parser.add_argument("--tune_layer", type=int, default=1, choices=[-1, 1, 2, 3, 4])
parser.add_argument("--tune_type", type=str, default="Categorical", choices=["Categorical", "FewShot"])
parser.add_argument("--train_iter_size", type=int, default=100)
parser.add_argument("--val_iter_size", type=int, default=200)
parser.add_argument("--tune_way", type=int, default=5)
parser.add_argument("--tune_shot", type=int, default=1)
parser.add_argument("--tune_query", type=int, default=10)
parser.add_argument("--shortcut", type=str, default="B", choices=["A", "B"])
parser.add_argument("--max_interval", type=int, default=-1)
parser.add_argument("--random_interval", action="store_true")
parser.add_argument("--random_start_position", action="store_true")
parser.add_argument("--uniform_frame_sample", action="store_true")
parser.add_argument("--random_pad_sample", action="store_true")
parser.add_argument("--fine_tuner_frame_size", type=int, default=112)
parser.add_argument("--sequence_length", type=int, default=16)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--scheduler_step_size", type=int, default=10)
parser.add_argument("--scheduler_gamma", type=float, default=0.9)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--gpu_number", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--cudnn_benchmark", action="store_true")
parser.add_argument("--only_cpu", action="store_true")
args = parser.parse_args()

database_path = os.path.join(args.db_root_path, args.tune_target)
save_path = os.path.join(args.model_root_path, args.tune_type)
frame_path = os.path.join(database_path, "frames")

if args.tune_type == "Categorical":
    save_path = os.path.join(save_path, args.model, f"{args.tune_target}_all" if args.tune_layer == -1 else f"{args.tune_target}_{args.tune_layer}")
    train_annotation_path = os.path.join(database_path, "labels/custom/categorical/train.csv")
    val_annotation_path = os.path.join(database_path, "labels/custom/categorical", "test.csv" if args.tune_target == "UCF101" else "val.csv")

    Categorical(
        frame_path=frame_path,
        train_annotation_path=train_annotation_path,
        val_annotation_path=val_annotation_path,
        save_path=save_path,
        model_path=None,
        model_name=args.model,
        shortcut=args.shortcut,
        pretrained=args.pretrained,
        frame_size=args.fine_tuner_frame_size,
        sequence_length=args.sequence_length,
        max_interval=args.max_interval,
        random_interval=args.random_interval,
        random_start_position=args.random_start_position,
        uniform_frame_sample=args.uniform_frame_sample,
        random_pad_sample=args.random_pad_sample,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        layer=args.tune_layer,
        learning_rate=args.learning_rate,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        only_cpu=(False or args.only_cpu) if torch.cuda.is_available() else True,
        gpu_number=args.gpu_number,
        cudnn_benchmark=args.cudnn_benchmark,
        num_workers=args.num_workers,
    ).run()

else:
    save_path = os.path.join(save_path, f"{args.tune_way}way", f"{args.tune_shot}shot", args.model, f"{args.tune_target}_all" if args.tune_layer == -1 else f"{args.tune_target}_{args.tune_layer}")
    train_annotation_path = os.path.join(database_path, "labels/custom/fewshot/train.csv")
    val_annotation_path = os.path.join(database_path, "labels/custom/fewshot/val.csv")

    FewShot(
        frame_path=frame_path,
        train_annotation_path=train_annotation_path,
        val_annotation_path=val_annotation_path,
        save_path=save_path,
        model_path=None,
        model_name=args.model,
        shortcut=args.shortcut,
        pretrained=args.pretrained,
        frame_size=args.fine_tuner_frame_size,
        sequence_length=args.sequence_length,
        max_interval=args.max_interval,
        random_interval=args.random_interval,
        random_start_position=args.random_start_position,
        uniform_frame_sample=args.uniform_frame_sample,
        random_pad_sample=args.random_pad_sample,
        train_iter_size=args.train_iter_size,
        val_iter_size=args.val_iter_size,
        way=args.tune_way,
        shot=args.tune_shot,
        query=args.tune_query,
        num_epochs=args.num_epochs,
        layer=args.tune_layer,
        learning_rate=args.learning_rate,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        only_cpu=(False or args.only_cpu) if torch.cuda.is_available() else True,
        gpu_number=args.gpu_number,
        cudnn_benchmark=args.cudnn_benchmark,
        num_workers=args.num_workers,
    ).run()