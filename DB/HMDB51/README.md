# HMDB51
## Data preparation
1. download
```
$ wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
$ wget https://www.dropbox.com/s/7r9ouxz9c4ai5ex/hmdb51_labels.zip
```
2. decompress
```
$ mkdir videos && unrar x hmdb51_org.rar videos && rm -rf hmdb51_org.rar
$ for i in videos/*.rar; do unrar x $i videos && rm -rf $i; done
$ unzip ucf101_labels.zip && rm -rf ucf101_labels.zip
```

☆ the labels for split id 1