# How to run the experiments
* build the docker image and exec the container
```
cd /path_to_whole_project/remote_paddle
docker build -t dlf_benchmark_paddle .
docker-compose up -d
docker-compose exec dlf_benchmark_paddle_container bash 
```

* run the experiments in the container
```
cd /workspace
cd /an_project_like_imageclassification
python test_*.py
```

# About tasks
* task1 imagecalssification and task2 textclassification_bilstm

For the implementaion of these two tasks are by notebook, please start the jupyter notebook in the conainter and then run the task in the notebook via a browser:
```
cd /workspace/Paddle
jupyter notebook --ip=0.0.0.0 --port=8080 --allow-root
```

* task3 textclassification_bert

Please run `python download_pretrained_model.py` to download bert pretrained model at first
