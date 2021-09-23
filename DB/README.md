# DB
## This directory will be used to save the datasets.
```
DB
+-- UCF101
|   +-- videos (avi)
|   +-- frames (jpeg)
|   +-- labels
|       +-- custom
|           +-- categorical (csv)
|               +-- train.csv (default split number is 1 for UCF101 and HMDB51, version 3 for ActivityNet)
|               +-- test.csv
|           +-- fewshot (csv)
|               +-- train.csv (default split rate 71:30[train:test] for UCF101)
|                             (default split rate 36:15[train:test] for HMDB51)
|                             (default split rate 140:60[train:test] for ActivityNet)
|               +-- test.csv
|           +-- sampled (json)
|               +-- train.json (default split number is 1 for UCF101 and HMDB51, version 3 for ActivityNet)
|                              ({sub-directory path: {label: label number, category: category name, index: sampeld frame index list}})
|               +-- test.json
|       +-- official
+-- HMDB51
|   +-- same above UCF101
+-- ActivityNet
    +-- same above UCF101
```