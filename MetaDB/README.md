# MetaDB
## This directory will be used to save the extracted deep features.
```
MetaDB
+-- Categorical (annotation type)
|   +-- UCF101 (target dataset for feature extraction)
|       +-- WithoutTune (fine-tune type)
|           +-- R3D18 (model type)
|               +-- db (where stored numpy array)
|               +-- prototype
|               +-- query
|               +-- args.txt (arguments of feature extractor)
|               +-- meta.json (it needs when conduct matching test)
|                             ({label|category: {db: sub-directory path list for db, prototype: ... for prototype, query: ... for query}})
|           +-- R3D34
|               +-- same above R3D18
|           +-- R3D50
|               +-- same above R3D18
|           +-- R2Plus1D18
|               +-- same above R3D18
|           +-- R2Plus1D34
|               +-- same above R3D18
|           +-- R2Plus1D50
|               +-- same above R3D18
|       +-- Categorical
|           +-- same above WithoutTune
|       +-- FewShot
|           +-- same above WithoutTune
|   +-- HMDB51
|       +-- same above UCF101
|   +-- ActivityNet
|       +-- same above UCF101
+-- Sampled
    +-- same above Categorical
```