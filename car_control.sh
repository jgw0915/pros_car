#!/bin/bash

IMAGE_NAME="ghcr.io/screamlab/pros_car_docker_image:latest"
NETWORK_NAME="compose_my_bridge_network"

# 1. 統一管理 -v 參數
VOLUME_ARGS="-v $(pwd)/src:/workspaces/src -v $(pwd)/launch:/workspaces/launch"

# Port mapping check
PORT_MAPPING=""
if [ "$1" = "--port" ] && [ -n "$2" ] && [ -n "$3" ]; then
    PORT_MAPPING="-p $2:$3"
    shift 3  # Remove the first three arguments
fi

# 檢查系統架構與作業系統
ARCH=$(uname -m)
OS=$(uname -s)

# 初始化 GPU 相關變數
GPU_FLAGS=""
USE_GPU=false

# 檢查是否為 Linux 並且支援 NVIDIA GPU
if [ "$OS" = "Linux" ]; then
    GPU_CANDIDATES=()

    if [ -f "/etc/nv_tegra_release" ]; then
        GPU_CANDIDATES+=("--runtime=nvidia")
    else
        DOCKER_RUNTIMES=$(docker info --format '{{json .Runtimes}}' 2>&1)
        DOCKER_INFO_STATUS=$?

        if [ $DOCKER_INFO_STATUS -ne 0 ]; then
            echo "Could not query Docker runtimes: $DOCKER_RUNTIMES"
        elif echo "$DOCKER_RUNTIMES" | grep -q "nvidia"; then
            GPU_CANDIDATES+=("--gpus all")
        fi

        if command -v nvidia-smi > /dev/null 2>&1; then
            if nvidia-smi > /dev/null 2>&1; then
                GPU_CANDIDATES+=("--gpus all")
            else
                echo "Host NVIDIA driver exists, but nvidia-smi failed. Docker GPU may not work."
            fi
        fi
    fi

    for CANDIDATE in "${GPU_CANDIDATES[@]}"; do
        [ -z "$CANDIDATE" ] && continue

        echo "Testing Docker run with GPU flags: $CANDIDATE"
        GPU_TEST_OUTPUT=$(docker run --rm $CANDIDATE "$IMAGE_NAME" /bin/bash -c "echo GPU test" 2>&1)
        if [ $? -eq 0 ]; then
            GPU_FLAGS="$CANDIDATE"
            USE_GPU=true
            break
        fi

        echo "GPU test failed with '$CANDIDATE':"
        echo "$GPU_TEST_OUTPUT"
    done
fi

if [ "$USE_GPU" != true ]; then
    echo "Docker GPU support was not detected; continuing without GPU flags."
fi

echo "Detected OS: $OS, Architecture: $ARCH"
echo "GPU Flags: $GPU_FLAGS"

if ! docker info > /dev/null 2>&1; then
    echo "Docker is not available for this user."
    echo "Try: sudo usermod -aG docker $USER"
    echo "Then log out and log back in before running this script again."
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "Missing .env in $(pwd). Run this script from the pros_car folder."
    exit 1
fi

if ! docker network inspect "$NETWORK_NAME" > /dev/null 2>&1; then
    echo "Docker network '$NETWORK_NAME' does not exist; creating it..."
    if ! docker network create --driver bridge "$NETWORK_NAME" > /dev/null; then
        echo "Failed to create Docker network '$NETWORK_NAME'."
        exit 1
    fi
fi

# 設定適當的 Docker 參數
device_options=""

# 檢查設備並加入 --device 參數
if [ -e /dev/usb_front_wheel ]; then
    device_options+=" --device=/dev/usb_front_wheel"
fi
if [ -e /dev/usb_rear_wheel ]; then
    device_options+=" --device=/dev/usb_rear_wheel"
fi
if [ -e /dev/usb_robot_arm ]; then
    device_options+=" --device=/dev/usb_robot_arm"
fi

# 根據不同架構選擇適當的 Docker 圖像
if [ "$ARCH" = "aarch64" ]; then
    echo "Detected architecture: arm64"
    docker run -it --rm \
        --network "$NETWORK_NAME" \
        $PORT_MAPPING \
        $device_options \
        --runtime=nvidia \
        --env-file .env \
        -v "$(pwd)/src:/workspaces/src" \
        "$IMAGE_NAME" \
        /bin/bash

elif [ "$ARCH" = "x86_64" ] || ([ "$ARCH" = "arm64" ] && [ "$OS" = "Darwin" ]); then
    echo "Detected architecture: amd64 or macOS arm64"

    if [ "$OS" = "Darwin" ]; then
        echo "Running Docker on macOS (without GPU support)..."
        docker run -it --rm \
            --network "$NETWORK_NAME" \
            $PORT_MAPPING \
            $device_options \
            --env-file .env \
            $VOLUME_ARGS \
            "$IMAGE_NAME" \
            /bin/bash
    else
        if [ "$USE_GPU" = true ]; then
            echo "Trying to run with GPU support..."
        else
            echo "Running without GPU support..."
        fi
        docker run -it --rm \
            --network "$NETWORK_NAME" \
            $PORT_MAPPING \
            $GPU_FLAGS \
            $device_options \
            --env-file .env \
            $VOLUME_ARGS \
            "$IMAGE_NAME" \
            /bin/bash

        RUN_STATUS=$?

        if [ $RUN_STATUS -ne 0 ]; then
            if [ "$USE_GPU" = true ]; then
                echo "Docker run failed with GPU flags, falling back to CPU mode..."
                docker run -it --rm \
                    --network "$NETWORK_NAME" \
                    $PORT_MAPPING \
                    --env-file .env \
                    $device_options \
                    $VOLUME_ARGS \
                    "$IMAGE_NAME" \
                    /bin/bash
            else
                echo "Docker run failed without GPU flags. This is probably not a GPU detection problem."
                echo "Check Docker permissions, .env, and image availability."
                exit $RUN_STATUS
            fi
        fi
    fi
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi
