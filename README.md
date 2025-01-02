
## Installation

Download the pretrained `SAM 2` checkpoints:

```bash
cd checkpoints
bash download_ckpts.sh
```

Download the pretrained `Grounding DINO` checkpoints:

```bash
cd gdino_checkpoints
bash download_ckpts.sh
```

### Installation without docker

Install PyTorch environment first. We use `python=3.10`, as well as `torch >= 2.3.1`, `torchvision>=0.18.1` and `cuda-12.1` in our environment to run this demo. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended. You can easily install the latest version of PyTorch as follows:

```bash
pip3 install torch torchvision torchaudio
```

Since we need the CUDA compilation environment to compile the `Deformable Attention` operator used in Grounding DINO, we need to check whether the CUDA environment variables have been set correctly (which you can refer to [Grounding DINO Installation](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install) for more details). You can set the environment variable manually as follows if you want to build a local GPU environment for Grounding DINO to run Grounded SAM 2:

```bash
export CUDA_HOME=/path/to/cuda-12.1/
```

Install `Segment Anything 2`:

```bash
pip install -e .
```

Install `Grounding DINO`:

```bash
pip install --no-build-isolation -e grounding_dino
```

### Installation with docker
Build the Docker image and Run the Docker container:

```
cd Grounded-SAM-2
make build-image
make run
```
After executing these commands, you will be inside the Docker environment. The working directory within the container is set to: `/home/appuser/Grounded-SAM-2`

Once inside the Docker environment, you can start the demo by running:
```
python grounded_sam2_tracking_demo.py
```

## Grounded SAM 2 Demos
### Grounded SAM 2 Image Demo (with Grounding DINO)
Note that `Grounding DINO` has already been supported in [Huggingface](https://huggingface.co/IDEA-Research/grounding-dino-tiny), so we provide two choices for running `Grounded SAM 2` model:
- Use huggingface API to inference Grounding DINO (which is simple and clear)

```bash
python grounded_sam2_hf_model_demo.py
```

> [!NOTE]
> 🚨 If you encounter network issues while using the `HuggingFace` model, you can resolve them by setting the appropriate mirror source as `export HF_ENDPOINT=https://hf-mirror.com`

- Load local pretrained Grounding DINO checkpoint and inference with Grounding DINO original API (make sure you've already downloaded the pretrained checkpoint)

```bash
python grounded_sam2_local_demo.py
```


**Preparing Hand Segmentation Data**
```bash
python masking_code.py  \ 
   --input_dir "./path_to_your_input_videos" \
   --output_dir "./path_to_your_output_data" \
   --fps 15 --text_input "hand"

```
