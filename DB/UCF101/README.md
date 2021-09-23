# UCF101
## Data preparation
1. download
```
$ wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
$ wget https://www.dropbox.com/s/j4brxpdcqsp848n/ucf101_labels.zip
```
2. decompress
```
$ unrar x UCF101.rar && mv UCF-101 videos && rm -rf UCF101.rar
$ unzip ucf101_labels.zip && rm -rf ucf101_labels.zip
```

☆ the labels for split id 1