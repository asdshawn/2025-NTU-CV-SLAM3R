<!-- # SLAM3R    

Paper: [arXiv](http://arxiv.org/abs/2412.09401)

TL;DR: A real-time RGB SLAM system that performs dense 3D reconstruction via points regression with feed-forward neural networks. -->


<p align="center">
  <h2 align="center">[2025-NTU-Computer-Vision] Final Project: SLAM3R</h2>
<!-- <div style="line-height: 1;" align=center>
  <a href="https://arxiv.org/abs/2412.09401" target="_blank" style="margin: 2px;">
    <img alt="Arxiv" src="https://img.shields.io/badge/Arxiv-SLAM3R-red" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div> -->

  <div align="center"></div>
</p>

<div align="center">
  <img src="./media/replica.gif" width="49%" /> 
  <img src="./media/wild.gif" width="49%" />
</div>

<p align="center">
<strong>SLAM3R</strong> is a real-time dense scene reconstruction system that regresses 3D points from video frames using feed-forward neural networks, without explicitly estimating camera parameters. 
</p>
<be>

## Prerequisites

- Ubuntu20.04+

- You installed the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit).
  
  1. Configure the production repository:
     ```bash
     curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
     && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
     ```
     Optionally, configure the repository to use experimental packages:
     ```bash
     sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
     ```
  2. Update the packages list from the repository:
     ```bash
     sudo apt-get update
     ```
  3. Install the NVIDIA Container Toolkit packages:
     ```bash
     sudo apt-get install -y nvidia-container-toolkit
     ```
  4. Configure the container runtime by using the nvidia-ctk command:
     ```bash
     sudo nvidia-ctk runtime configure --runtime=docker
     ```
  6. Restart the Docker daemon:
     ```bash
     sudo systemctl restart docker
     ```

- You installed [Docker](https://www.docker.com/).
  
  To verify installation, run:
  ```bash
  docker run hello-world
  ```
  or
  ```bash
  sudo docker run hello-world
  ```
  
  It should display:
  ```
  Hello from Docker!
  This message shows that your installation appears to be working correctly.
  
  To generate this message, Docker took the following steps:
   1. The Docker client contacted the Docker daemon.
   2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
      (amd64)
   3. The Docker daemon created a new container from that image which runs the
      executable that produces the output you are currently reading.
   4. The Docker daemon streamed that output to the Docker client, which sent it
      to your terminal.
  
  To try something more ambitious, you can run an Ubuntu container with:
   $ docker run -it ubuntu bash
  
  Share images, automate workflows, and more with a free Docker ID:
   https://hub.docker.com/
  
  For more examples and ideas, visit:
   https://docs.docker.com/get-started/
  ```

## Installation

1. Clone SLAM3R
```bash
git clone https://github.com/asdshawn/2025-NTU-CV-SLAM3R.git && cd 2025-NTU-CV-SLAM3R/
```

2. Prepare environment
```bash
docker build -t slam3r_ntu .
```

3. Run slam3r_ntu
```bash
# Gradio interface
docker run --gpus all -t -p 7860:7860 slam3r_ntu
# or CLI mode
# docker run --gpus all -it -p 7860:7860 --entrypoint /bin/bash slam3r_ntu
```

## Demo
### Replica dataset
To run our demo on Replica dataset, download the sample scene [here](https://drive.google.com/file/d/1NmBtJ2A30qEzdwM0kluXJOp2d1Y4cRcO/view?usp=drive_link) and unzip it to `./data/Replica_demo/`. Then run the following command to reconstruct the scene from the video images 

 ```bash
 bash scripts/demo_replica.sh
 ```

The results will be stored at `./results/` by default.

### Self-captured outdoor data
We also provide a set of images extracted from an in-the-wild captured video. Download it [here](https://drive.google.com/file/d/1FVLFXgepsqZGkIwg4RdeR5ko_xorKyGt/view?usp=drive_link) and unzip it to `./data/wild/`.  

Set the required parameter in this [script](./scripts/demo_wild.sh), and then run SLAM3R by using the following command
 
 ```bash
 bash scripts/demo_wild.sh
 ```

When `--save_preds` is set in the script, the per-frame prediction for reconstruction will be saved at `./results/TEST_NAME/preds/`. Then you can visualize the incremental reconstruction process with the following command

 ```bash
 bash scripts/demo_vis_wild.sh
 ```

A Open3D window will appear after running the script. Please click `space key` to record the adjusted rendering view and close the window. The code will then do the rendering of the incremental reconstruction.

You can run SLAM3R on your self-captured video with the steps above. Here are [some tips](./docs/recon_tips.md) for it


## Gradio interface
We also provide a Gradio interface, where you can upload a directory, a video or specific images to perform the reconstruction. After setting the reconstruction parameters, you can click the 'Run' button to start the process. Modifying the visualization parameters at the bottom allows you to directly display different visualization results without rerunning the inference.

The interface can be launched with the following command:

 ```bash
 python app.py
 ```

Here is a demo GIF for the Gradio interface (accelerated).

<img src="media/gradio_office.gif" style="zoom: 66%;" />

## Evaluation on the 7-Scenes dataset

1. Download the 7-Scenes dataset generated by TA of Computer Vision course in NTU:
```bash
cd data
gdown 1r172cIGZKBc3b7_b1-cscPnVFj8bl8HF # or 1IIDaxvauiNtZX49lSckrok82XJchOtD3
unzip 7SCENES.zip
rm -rf 7SCENES.zip
```

2. Obtain the GT pointmaps by running the following command:
```bash
bash ./scripts/gen_gt.sh
```
The processed GT will be saved at `./results/gt_points/`.

3. Evaluate the reconstruction on the Replica dataset with the following command:

```bash
python evaluation/eval.py -rec path/to/results/ -gt path/to/ground_truth # Add -b option for bonus sequence
```

## Evaluation on the Replica dataset

1. Download the Replica dataset generated by the authors of iMAP:
```bash
cd data
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
rm -rf Replica.zip
```

2. Obtain the GT pointmaps and valid masks for each frame by running the following command:
```bash
python evaluation/process_gt.py
```
The processed GT will be saved at `./results/gt/replica`.

3. Evaluate the reconstruction on the Replica dataset with the following command:

```bash
bash ./scripts/eval_replica.sh
```

Both the numerical results and the error heatmaps will be saved in the directory `./results/TEST_NAME/eval/`.

> [!NOTE]
> Different versions of CUDA, PyTorch, and xformers can lead to slight variations in the predicted point cloud. These differences may be amplified during the alignment process in evaluation. Consequently, the numerical results you obtain might differ from those reported in the paper. However, the average values should remain approximately the same.

## Training

### Datasets

We use ScanNet++, Aria Synthetic Environments and Co3Dv2 to train our models. For data downloading and pre-processing, please refer to [here](./docs/data_preprocess.md). 

### Pretrained weights from DUSt3R

```bash
# download the pretrained weights from DUSt3R
mkdir checkpoints 
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth -P checkpoints/
```

### Pretrained weights from SLAM3R

```bash
# download the pretrained weights from SLAM3R
mkdir checkpoints 
wget https://huggingface.co/siyan824/slam3r_i2p/resolve/main/slam3r_i2p.pth -P checkpoints/
wget https://huggingface.co/siyan824/slam3r_l2w/resolve/main/slam3r_l2w.pth -P checkpoints/
```

### Hyperparameters for 7-Scenes dataset

```bash
TRAIN_DATASET="1000 @ SevenScenes_Seq(scene_id='chess',seq_id=1,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='chess',seq_id=2,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='chess',seq_id=4,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='chess',seq_id=6,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='fire',seq_id=1,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='fire',seq_id=2,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='heads',seq_id=2,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='office',seq_id=1,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='office',seq_id=3,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='office',seq_id=4,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='office',seq_id=5,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='office',seq_id=8,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='office',seq_id=10,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='pumpkin',seq_id=2,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='pumpkin',seq_id=3,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='pumpkin',seq_id=6,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='pumpkin',seq_id=8,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='redkitchen',seq_id=1,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='redkitchen',seq_id=2,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='redkitchen',seq_id=5,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='redkitchen',seq_id=7,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='redkitchen',seq_id=8,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='redkitchen',seq_id=11,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='redkitchen',seq_id=13,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='stairs',seq_id=2,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='stairs',seq_id=3,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='stairs',seq_id=5,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='stairs',seq_id=6,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20)"

TEST_DATASET="1000 @ SevenScenes_Seq(scene_id='chess',seq_id=3,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='chess',seq_id=5,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='fire',seq_id=3,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='fire',seq_id=4,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='heads',seq_id=1,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='office',seq_id=2,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='office',seq_id=6,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='office',seq_id=7,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='office',seq_id=9,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='pumpkin',seq_id=1,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='pumpkin',seq_id=7,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='redkitchen',seq_id=3,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='redkitchen',seq_id=4,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='redkitchen',seq_id=6,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='redkitchen',seq_id=12,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='redkitchen',seq_id=14,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='stairs',seq_id=1,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20) + \
1000 @ SevenScenes_Seq(scene_id='stairs',seq_id=4,resolution=(224,224), num_views=5, start_freq=1, sample_freq=20)"

# Stage 1: Train the i2p model for pointmap prediction
PRETRAINED="checkpoints/slam3r_i2p/slam3r_i2p.pth"
TRAIN_OUT_DIR="checkpoints/i2p/slam3r_i2p_stage1"
```

### Start training

```bash
# train the Image-to-Points model and the retrieval module
bash ./scripts/train_i2p.sh
# train the Local-to-Wrold model
bash ./scripts/train_l2w.sh
```
> [!NOTE]
> They are not strictly equivalent to what was used to train SLAM3R, but they should be close enough.


## Citation

If you find our work helpful in your research, please consider citing: 
```
@article{slam3r,
  title={SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos},
  author={Liu, Yuzheng and Dong, Siyan and Wang, Shuzhe and Yin, Yingda and Yang, Yanchao and Fan, Qingnan and Chen, Baoquan},
  journal={arXiv preprint arXiv:2412.09401},
  year={2024}
}
```

## Acknowledgments

Our implementation is based on several awesome repositories:

- [Croco](https://github.com/naver/croco)
- [DUSt3R](https://github.com/naver/dust3r)
- [NICER-SLAM](https://github.com/cvg/nicer-slam)
- [Spann3R](https://github.com/HengyiWang/spann3r)
- [SLAM3R](https://github.com/PKU-VCL-3DV/SLAM3R)

We thank the respective authors for open-sourcing their code.
