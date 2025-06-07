#!/bin/bash

docker_cmd="docker run --gpus all -it -p 7860:7860 -v ${PWD}:/home/user/slam3r/ --rm --entrypoint /bin/bash"

# 检查是否包含 --quick_start 选项
for arg in "$@"; do
    if [[ "$arg" == "--quick_start" ]]; then
        docker_cmd="docker run --gpus all -t -p 7860:7860 --rm"
        break
    fi
done

docker_cmd+=" slam3r_ntu:latest"

# 执行命令
eval $docker_cmd