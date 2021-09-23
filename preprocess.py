import os
import argparse

import torch

from Core.FrameExtractor import FrameExtractor
from Core.FeatureExtractor import FeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument("--db_root_path", type=str, default="./DB")
parser.add_argument("--metadb_root_path", type=str, default="./MetaDB")
parser.add_argument("--model_root_path", type=str, default="./Models")
parser.add_argument("--annotation_type", type=str, default="Categorical", choices=["Categorical", "Sampled"])
parser.add_argument("--extract_target", type=str, default="UCF101", choices=["UCF101", "HMDB51", "ActivityNet"])
parser.add_argument("--model", type=str, default="R3D18", choices=["R3D18", "R3D34", "R3D50", "R2Plus1D18", "R2Plus1D34", "R2Plus1D50"])
parser.add_argument("--tune_type", type=str, default="WithoutTune", choices=["WithoutTune", "Categorical", "FewShot"])
parser.add_argument("--tune_layer", type=int, default=1, choices=[1, 2, 3, 4])
parser.add_argument("--tune_target", type=str, default="UCF101", choices=["UCF101", "HMDB51", "ActivityNet"])
parser.add_argument("--tune_way", type=int, default=5)
parser.add_argument("--tune_shot", type=int, default=1)
parser.add_argument("--frame_extractor_frame_size", type=int, default=240)
parser.add_argument("--feature_extractor_frame_size", type=int, default=112)
parser.add_argument("--sequence_length", type=int, default=16)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=-1)
parser.add_argument("--gpu_number", type=int, default=0)
parser.add_argument("--only_cpu", action="store_true")
args = parser.parse_args()

"""
without feature path = ./{metadb path}/{annotation type}/{extract target}/{tune type}/{model}
categorical feature path = ./{MetaDB path}/{annotation type}/{extract target}/{tune type}/{model}/{tune target}_{tune layer}
fewshot feature path = ./{MetaDB path}/{annotation type}/{extract target}/{tune type}/{tune way}way/{tune shot}shot/{model}/{tune target}_{tune layer}

"""

database_path = os.path.join(args.db_root_path, args.extract_target)
video_path = os.path.join(database_path, "videos")
frame_path = os.path.join(database_path, "frames")

feature_path = os.path.join(args.metadb_root_path, args.annotation_type, args.extract_target, args.tune_type)
if args.tune_type == "WithoutTune":
    feature_path = os.path.join(feature_path, args.model)
elif args.tune_type == "Categorical":
    feature_path = os.path.join(feature_path, args.model, f"{args.tune_target}_{args.tune_layer}")
else:
    feature_path = os.path.join(feature_path, f"{args.tune_way}way", f"{args.tune_shot}shot", args.model, f"{args.tune_target}_{args.tune_layer}")
    
# annotation: categorical, sampled
if args.annotation_type == "Categorical":
    db_annotation_path = os.path.join(database_path, "labels/custom/categorical/train.csv")
    query_annotation_path = os.path.join(database_path, "labels/custom/categorical", "val.csv" if args.extract_target == "ActivityNet" else "test.csv")
else:
    db_annotation_path = os.path.join(database_path, "labels/custom/sampled/train.json")
    query_annotation_path = os.path.join(database_path, "labels/custom/sampled", "val.json" if args.extract_target == "ActivityNet" else "test.json")

FrameExtractor(video_path, frame_path, args.frame_extractor_frame_size, args.num_workers).run()

# pretrained models: without tune, categorical, fewshot
if args.tune_type == "WithoutTune":
    model_path = None
elif args.tune_type == "Categorical":
    # ex) ./Models/Categorical/R3D18/UCF101_1/best_acc.pth
    model_path = os.path.join(args.model_root_path, args.tune_type, args.model, f"{args.tune_target}_{args.tune_layer}/best_acc.pth")
else:
    # ex) ./Models/FewShot/5way/1shot/R3D18/UCF101_1/best_acc.pth
    model_path = os.path.join(args.model_root_path, args.tune_type, f"{args.tune_way}way", f"{args.tune_shot}shot", args.model, f"{args.tune_target}_{args.tune_layer}/best_acc.pth")

FeatureExtractor(
    frame_path=frame_path,
    db_annotation_path=db_annotation_path,
    query_annotation_path=query_annotation_path,
    save_path=feature_path,
    model_name=args.model,
    model_path=model_path,
    shortcut="B",
    pretrained=True,
    frame_size=args.feature_extractor_frame_size,
    sequence_length=args.sequence_length,
    max_interval=-1,
    random_interval=False,
    random_start_position=False,
    uniform_frame_sample=True,
    random_pad_sample=False,
    batch_size=args.batch_size,
    only_cpu=(False or args.only_cpu) if torch.cuda.is_available() else True,
    gpu_number=args.gpu_number
).run()