This repository contains the implementation and experimental environment used for the EE6483 course classification task.  
It includes a complete **Docker-based GPU environment**, **model training pipeline**, and **batch prediction utilities** to ensure full reproducibility
# Environment Setup
## Base Docker Image
```bash
docker pull nvidia/cuda:12.1.0-devel-ubuntu22.04
```
## Run the Docker Container
```bash
docker run -d \
  -e DISPLAY=unix${DISPLAY} \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --privileged \
  --ipc=host \
  --shm-size=8g \
  -v /lib/modules:/lib/modules \
  -v /home/ryan/NTU_LEARN/EE6483/project/workspace:/workspace \
  -it \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --name HomeworkEE6483Ver1.0 \
  ee6483project:1.0 \
  /bin/bash 
```
## Install PyTorch
``` bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
## Install YOLO (Ultralytics)
```bash
cd ultralytics
pip install -e .
```
## Install GUI Libraries in Docker
```bash
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgtk-3-0 \
  && rm -rf /var/lib/apt/lists/*
```
## Fix Import Errors(Optional)
```bash
conda install -y -c conda-forge ittapi tbb=2021.* intel-openmp
```
# Usage Guide
## Training
### Modify Dataset Path
In `train.py`:
```bash
train_data_path = "/workspace/code/datasets"
```
### Run the Training Script
```
python train.py
```
After execution, a project folder named **`classification_task`** will be created automatically, containing all training logs, weights, and results.
# Prediction
## Configure Paths
- in `test.py`
``` bash
# Model weights
weights = "/workspace/classification_projects/classification_task11/weights/best.pt"

# Save visualized prediction results
tester.predict_folder_save_images(
    folder="/workspace/code/test", 
    save_dir="/workspace/code/image_results", 
    prefix="test_predict_"
)

# Test image directory
test_folder = "/workspace/code/test"

# Output CSV file
output_csv = "/workspace/code/test_results.csv"

```
## Run the Test Script
```bash
python test.py
```
The script will generate:
- A CSV file containing prediction results
- Visualized prediction images saved to your chosen directory

