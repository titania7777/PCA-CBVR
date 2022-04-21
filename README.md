# The original repository of PCA-CBVR
We will update the example code soon.

The PCA-CBVR (Prototypical Category Approximation Content-Based Video Retrieval) is a proposed method among CBVR applications.

The PCA-CBVR process consists of two-step. The first step is query video classification, in this step we utilize [prototype technique](https://arxiv.org/abs/1703.05175). The second step is a fine-search on the database videos which corresponds to the estimated query video category.

The character of PCA-CBVR shows good performance on untrimmed videos such as [ActivityNet](http://activity-net.org/).

Please, for more detailed things are referred to the paper.

## Example

## Preprocessing Arguments
```
python preprocess.py --args(Default) [choices]
```
1.  --db_root_path(./DB): the location of the [DB](DB/README.md)
2.  --metadb_root_path(./MetaDB): the location of the [MetaDB](MetaDB/README.md)
3.  --model_root_path(./Models): the location of the [Models](Models/README.md)
4.  --annotation_type(Categorical) [Categorical, Sampled]: the annotation type to build [MetaDB](MetaDB/README.md)
5.  --extract_target(UCF101) [UCF101, HMDB51, ActivityNet]: the target dataset to build [MetaDB](MetaDB/README.md)
6.  --model(R3D18) [R3D18, R3D34, R3D50, R2Plus1D18, R2Plus1D34, R2Plus1D50]: the model to use for feature extraction
7.  --tune_type(WithoutTune) [WithoutTune, Categorical, FewShot]: the weight as by fine-tuned types to use for feature extraction
8.  --tune_layer(1) [1, 2, 3, 4]: the number of residual blocks to use for feature extraction
9.  --tune_target(UCF101) [UCF101, HMDB51, ActivityNet]: the used target dataset when conducted fine-tuning
10. --tune_way(5): the few-shot parameter
11. --tune_shot(1): the few-shot parameter
12. --frame_extractor_frame_size(240): adjust frame size when performing frame extraction
13. --feature_extractor_frame_size(112): adjust frame size when performing feature extraction
14. --sequence_length(16): the number of frames
15. --batch_size(128): the batch size for feature extraction
16. --num_workers(-1): the number of cores to use for feature extraction
17. --gpu_number(0): the index number of the gpu to use for feature extraction
18. --only_cpu(store_true): performing all processes on the cpu

☆ (tune_layer, tune_target, tune_way, tune_shot) options are depends on (tune_type) option

## Fine-tuning Arguments
```
python finetuning.py --args(Default) [choices]
```
1.  --db_root_path(./DB): the location of the [DB](DB/README.md)
2.  --model_root_path(./Models): the location of the [Models](Models/README.md)
3.  --pretrained(store_true): using a model which pretrained on kinetics-700
4.  --tune_target(UCF101) [UCF101, HMDB51, ActivityNet]: the target dataset to train a model
5.  --model(R3D18) [R3D18, R3D34, R3D50, R2Plus1D18, R2Plus1D34, R2Plus1D50]: the model to use for training
6.  --tune_layer(1) [-1, 1, 2, 3, 4]: the number of residual blocks to use for training, -1 means using all
7.  --tune_type(Categorical) [Categorical, FewShot]: the type of learning strategy
8.  --train_iter_size(100): the number of train iteration for few-shot learning
9.  --val_iter_size(200): the number of val iteration for few-shot learning
10. --tune_way(5): the few-shot parameter
11. --tune_shot(1): the few-shot parameter
12. --tune_query(10): the few-shot parameter
13. --shortcut(B) [A, B]: resnet shortcut type
14. --max_interval(-1): fix the frame sampling interval, -1 means does not fix
15. --random_interval(store_true): use the random interval with range 0 to max_interval
16. --random_start_position(store_true): use the random start position for frame sampling
17. --uniform_frame_sample(store_true): use the uniform frame sampling strategy, will ignore the random_interval option if this option is activated
18. --random_pad_sample(store_true): use the random pad sampling strategy; the default is to repeat the first
19. --frame_size(112): adjust frame size when performing a training
20. --sequence_length(16): the number of frames
21. --batch_size(64): the batch size for training
22. --num_epochs(100): the number of epochs for training
23. --learning_rate(1e-3): the parameter for [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html?highlight=sgd#torch.optim.SGD)
24. --scheduler_step_size(10): the parameter for [StepRL](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html)
25. --scheduler_gamma(0.9): the parameter for [StepRL](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html)
26. --momentum(0.9): the parameter for [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html?highlight=sgd#torch.optim.SGD)
27. --weight_decay(1e-3): the parameter for [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html?highlight=sgd#torch.optim.SGD)
28. --gpu_number(0): the index number of the gpu to use for training
29. --num_workers(4): the number of cores to use for training
30. --cudnn_benchmark(store_true): activate cuDNN benchmark option, see [BACKENDS](https://pytorch.org/docs/stable/backends.html)
31. --only_cpu(store_true): performing all process on the cpu

☆ (tune_way, tune_shot) options are depends on (tune_type) option

## Citation
If you use this code in your work, please cite our work
```bibtex
@ARTICLE{9737137,
  author={Yoon, Hyeok and Han, Ji-Hyeong},
  journal={IEEE Access}, 
  title={Content-Based Video Retrieval With Prototypes of Deep Features}, 
  year={2022},
  volume={10},
  number={},
  pages={30730-30742},
  doi={10.1109/ACCESS.2022.3160214}}
```
