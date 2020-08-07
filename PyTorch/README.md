# How to run the experiments
* build the docker image and exec the container
```
cd /path_to_whole_project/remote_torch
docker build -t dlf_benchmark_torch .
docker-compose up -d
docker-compose exec dlf_benchmark_torch_container bash 
```

* run the experiments in the container
```
cd /workspace
cd /an_project_like_imageclassification
python test_*.py
```

# About dataset
* dataset horse2zebra for CycleGAN

you can download dataset horse2zebra from [here](https://www.kaggle.com/arnaud58/horse2zebra) 
and then put it under directoy like `/path_to_whole_project/PyTorch/data/horse2zebra/trainA`
