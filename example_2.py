import os
import numpy as np
from glob import glob

def cosine_similarity(a: np.array, b: np.array, eps: float = 1e-12):
    a_temp = a / np.expand_dims(np.fmax(np.linalg.norm(a, axis=1), eps), axis=1)
    b_temp = b / np.expand_dims(np.fmax(np.linalg.norm(b, axis=1), eps), axis=1)
    return np.matmul(a_temp, b_temp.T)

def retrieval(db_feature_path:str, query_feature_path:str, topk=1):
    prototype_feature_list = []
    label_path_list = []
    label_feature_dict = {}
    for _, sub_db_featur_path in enumerate(glob(os.path.join(db_feature_path, "*"))):
        # label
        label_path_list.append(sub_db_featur_path)

        # feature
        prototype_feature = np.mean([np.load(feature_path) for feature_path in glob(os.path.join(sub_db_featur_path, "*"))], axis=0)
        prototype_feature_list.append(prototype_feature)
        label_feature_dict[sub_db_featur_path] = prototype_feature

    prototype_feature_list = np.stack(prototype_feature_list, axis=0)
    label_path_list = np.array(label_path_list)

    query_feature_list = []
    [query_feature_list.append(np.load(feature_path)) for feature_path in glob(os.path.join(query_feature_path, "*"))]
    query_feature_list = np.stack(query_feature_list, axis=0)

    # category searching
    estimated_label_path_list = label_path_list[np.argmax(cosine_similarity(query_feature_list, prototype_feature_list), 1)]
    
    # fine seraching
    for estimated_label_path in estimated_label_path_list:
        fine_feature_list = []
        fine_label_path_list = []
        for estimated_feature_path in glob(os.path.join(estimated_label_path, "*")):
            fine_label_path_list.append(estimated_feature_path)
            fine_feature_list.append(np.load(estimated_feature_path))

        fine_label_path_list = np.array(fine_label_path_list)
        fine_feature_list = np.array(fine_feature_list)

        print(f"query: {estimated_label_path.split('/')[-1]}")
        print(fine_label_path_list[np.argsort(np.squeeze(cosine_similarity(fine_feature_list, np.expand_dims(label_feature_dict[estimated_label_path], 0))))[:topk]])

retrieval(db_feature_path = "./MetaDB/Categorical/UCF101/WithoutTune/R3D18/db/", query_feature_path = "./Test/", topk=5)