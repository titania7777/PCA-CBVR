import os
import numpy as np
from glob import glob

# one thread
# import mkl
# mkl.set_num_threads(1)

def cosine_similarity(a: np.array, b: np.array, eps: float = 1e-12):
    a_temp = a / np.expand_dims(np.fmax(np.linalg.norm(a, axis=1), eps), axis=1)
    b_temp = b / np.expand_dims(np.fmax(np.linalg.norm(b, axis=1), eps), axis=1)
    return np.matmul(a_temp, b_temp.T)

def euclidean_distance(a: np.array, b: np.array):
    a_temp = np.repeat(np.expand_dims(a, axis=1), b.shape[0], axis=1)
    b_temp = np.repeat(np.expand_dims(b, axis=0), a.shape[0], axis=0)
    return np.sqrt(np.sum((a_temp - b_temp)**2, axis=-1))

def accuracy(scores:np.ndarray, test_label:np.ndarray, topk:int=1):
    indices = np.argpartition(scores, -topk, axis=1)[:,-topk:] # topk indices
    corrects = sum(np.any(indices == np.expand_dims(test_label, axis=1).repeat(topk, axis=1), axis=1))
    return corrects / len(test_label)
    
def prototype_matching(db_feature_path:str, query_feature_path:str):
    prototype_feature_list = []
    label_dict = {}
    for i, sub_db_featur_path in enumerate(glob(os.path.join(db_feature_path, "*"))):
        feature_path_list = glob(os.path.join(sub_db_featur_path, "*"))
        
        # label
        label_dict[sub_db_featur_path.split("/")[-1]] = i
        
        # feature
        prototype_feature_list.append(np.mean([np.load(feature_path) for feature_path in feature_path_list], axis=0))

    prototype_feature_list = np.stack(prototype_feature_list, axis=0)

    query_feature_list = []; query_label_list = []
    for sub_query_feature_path in glob(os.path.join(query_feature_path, "*")):
        feature_path_list = glob(os.path.join(sub_query_feature_path, "*"))

        # label
        query_label_list = query_label_list + [label_dict[sub_query_feature_path.split("/")[-1]]] * len(feature_path_list)

        # feature
        [query_feature_list.append(np.load(feature_path)) for feature_path in feature_path_list]
    
    query_feature_list = np.stack(query_feature_list, axis=0)
    query_label_list = np.stack(query_label_list, axis=0)

    similarity_score_list = cosine_similarity(query_feature_list, prototype_feature_list) # similarity scores
    similarity_top_1_acc = accuracy(similarity_score_list, query_label_list, topk=1)
    similarity_top_5_acc = accuracy(similarity_score_list, query_label_list, topk=5)
    print(f"similarity top1 acc: {similarity_top_1_acc:.3}")
    print(f"similarity top5 acc: {similarity_top_5_acc:.3}")

    distance_score_list = -euclidean_distance(query_feature_list, prototype_feature_list) # distance scores
    distance_top_1_acc = accuracy(distance_score_list, query_label_list, topk=1)
    distance_top_5_acc = accuracy(distance_score_list, query_label_list, topk=5)
    print(f"distance top1 acc: {distance_top_1_acc:.3}")
    print(f"distance top5 acc: {distance_top_5_acc:.3}")

print("==========UCF101==========")
prototype_matching(db_feature_path = "./MetaDB/Categorical/UCF101/WithoutTune/R3D18/db/", query_feature_path = "./MetaDB/Categorical/UCF101/WithoutTune/R3D18/query/")