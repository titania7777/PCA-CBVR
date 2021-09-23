# ActivityNet
## Data preparation
1. [download](https://docs.google.com/forms/d/e/1FAIpQLSeKaFq9ZfcmZ7W0B0PbEhfbTHY41GeEgwsa7WobJgGUhn4DTQ/viewform)
```
$ wget https://www.dropbox.com/s/hgnruj6vvol2pk1/activitynet_labels.zip
```
2. decompress
```
$ mkdir videos && tar -xzvf v1-2_train.tar.gz && rm -rf v1-2_train.tar.gz && mv v1-2/train/* videos
$ tar -xzvf v1-2_val.tar.gz && rm -rf v1-2_val.tar.gz && mv v1-2/val/* videos
$ tar -xzvf v1-3_train_val.tar.gz && rm -rf v1-3_train_val.tar.gz && mv v1-3/train_val/* videos
$ rm -rf v1-2 && rm -rf v1-3
```
☆ we changed the name like below.  
Copy of v1-2_train.tar.gz => v1-2_train.tar.gz  
Copy of v1-2_val.tar.gz => v1-2_val.tzr.gz  
Copy of v1-3_train_val.tar.gz => v1-3_train_val.tar.gz

☆ the labels for version 1.3