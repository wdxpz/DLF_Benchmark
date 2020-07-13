#Connect remote docker host for dev in vscode

## 1. direct attach remote running containers
* modify user setting.json by menu->preference->settings, or command+shift+p to open commend box and input settings
```
"docker.host":"ssh://your-remote-user@your-remote-machine-fqdn-or-ip-here"
```

* at reomote host, start container with host network setting and directory mapping, and gpu enalbed
```
docker run -it -v ~/DLF_Benchmark:/project --network host --name dlf_benchmark --gpus 1 pytorch/pytorch
```

* close all vscode window, restart without any connection to existed local container

* click `Remote Explorer' at the right side bar of vscode to attach wanted container, 
** if the container is running as wanted, there will be a white dot under its icon **

* install needed extension: docker, python in remote container (if python extension can not install, uninstall local python extension, and then retart the vscode by attaching the container)
)  

## 2. open direcotry form container (recommended)
* 2.1 clone project dirctory to remote host at `\remote_path_to_dev_dir`
* 2.2 in the dev dir of local host `\local_path_to_dev_dir`, create subdir `remote`
    ```
    mkdir remote
    cd remote
    ```
* 2.3 in `\local_path_to_dev_dir\remote`, create docker-compose.yml with required settings, like network, gpu, and dir mount,
    **noted: the src in `volumes` should be the `\remote_path_to_dev_dir` **
    `docker-compose.yml`
    ```
    version: '2.3'
    services:
      dlf_benchmark_container:
      image: "pytorch/pytorch"
    volumes:
      - remote_path_to_dev_dir:/workspace:cached
    network_mode: "host"
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ```
    
    To enable para `runtime: nvidia`, need to do the following works: (refer: https://github.com/docker/compose/issues/6691)
    ```
    at remote host:
    
    * install nvidia-docker-runtime:
    https://github.com/NVIDIA/nvidia-container-runtime#docker-engine-setup

    * add to /etc/docker/daemon.json
    {
    "runtimes": {
    "nvidia": {
    "path": "/usr/bin/nvidia-container-runtime",
    "runtimeArgs": []
    }
    }
    }

    * after modify /etc/docker/daemon.json, restart docker service
    systemctl restart docker

    * use Compose format 2.3 and add runtime: nvidia to your GPU service. Docker Compose must be version 1.19.0 or higher, docker-compose file:
        
        version: '2.3'

        services:
        nvsmi:
        image: ubuntu:16.04
        runtime: nvidia
        environment:
        - NVIDIA_VISIBLE_DEVICES=all
    ```

* 2.4 start vscode, and open local dir of `\local_path_to_dev_dir\remote`
* 2.5 from command panel, choose `Remote-Containers: open folder in container`, and then `choose docker compose file`
* 2.6 after the container build, from command panel, choose `open locally`
* 2.7 in `\local_path_to_dev_dir\remote\.devcontainer`, and modify `docker-compose.yml`
    ```
     volumes:
      # Update this to wherever you want VS Code to mount the folder of your project
      - \remote_path_to_dev_dir:/workspace:cached
    ```
* 2.7 from command panel, choose `reopen in container`, and choose `rebuild container` from the prommpt or from the command panel
* 2.8 click any .py file to install Python extension, and then install pylint, reload docker extension and python extension

* Done!
