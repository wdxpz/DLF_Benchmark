# How to run the experiments
* build the docker image and exec the container
```
cd /path_to_whole_project/remote_tf
docker build -t dlf_benchmark_tf .
docker-compose up -d
docker-compose exec dlf_benchmark_tf_container bash 
```

* run the experiments
```
cd /workspace
cd /an_project_like_imageclassification
python test_*.py
```

