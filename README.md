# NeRP: Neutral-Reference Prompting for Vision–Language Models (ICML 2026)

This repository provides the official PyTorch implementation for our ICML 2026 paper:  
**Neutral-Reference Prompting for Vision–Language Models**  

📄 [NeRP Paper Link](#) 




## 📢📢 News

- 🗓️ 2025/05/07: NeRP code released!✅
- 🗓️ 2025/05/01: NeRP is accepted at ICML 2026, and all reviewers give positive feedback! 🎉  
- 🗓️ 2025/01/27: NeRP is submitted to ICML 2026.💪
- 🗓️ 2026/01/24: NeRP is withdrawn from CVPR 2026.😊
- 🗓️ 2026/01/23: NeRP receives negative & non-constructive reviewer comments.😮‍💨
- 🗓️ 2025/11/13: NeRP is submitted to CVPR 2026.💪



## 🔧🔧 Installation  

To set up the runtime environment, you can use the step-by-step instructions below (recommended) to create and configure your environment.

* Setup miniconda environment.
```bash
# Create a conda environment
conda create -y -n nerp python=3.10

# Activate the environment
conda activate nerp

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

* Clone NeRP code repository
```bash
git clone https://github.com/Sheldon04/NeRP.git
cd NeRP
```

* Install [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation
# Some cfgs & codes are modified to avoid potential bugs
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library
pip install -e .
cd ..
```


## 📦📦 Data Preparation

* Prepare the datasets. 

Please follow the instructions detailed in [DATASETS.md](docs/DATASETS.md). For download convenience, They maintain a repository at huggingface, which contains all the datasets to be used (except imagenet because it is too large).   [[HuggingFace_Download_Links](https://huggingface.co/zhengli97/prompt_learning_dataset)]

* Download CLIP weights.

Download the original ViT-B/16 model weights from the official OpenAI website. Then place these models in the `~/.cache/clip` folder.  
[[ViT-B/16 CLIP](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)]

* Download demo MMRL weights.

For a quick reporoduction, we provide a pretrained demo MMRL weights. Please download the "MultiModalRepresentationLearner" folder from [[Google Drive](#)] and put it into the "demo_output_mmrl" folder.


## 🚀🚀 Running the Code  

NeRP is built upon foundational prompt learning methods. Therefore, before reproducing the results of NeRP in the paper, it is necessary to first reproduce these methods as baselines ([CoCoOp](https://github.com/KaiyangZhou/CoOp), [PromptSRC](https://github.com/muzairkhattak/PromptSRC), [MMA](https://github.com/ZjjConan/Multi-Modal-Adapter),  [MMRL](https://github.com/yunncheng/MMRL) ).

### Quick Demo

Since training the baseline models takes some time, we provide a model fine-tuned with the MMRL method on the EuroSAT dataset for the base-to-novel task, enabling users to quickly get started with NeRP.

Before running the following demo command, make sure you have downloaded demo MMRL weights in **Data preparation**.

```bash
# You can specify the GPU index to use via the CUDA_VISIBLE_DEVICES. For example, when 4 GPUs are available, you can specify 0, 1, 2, or 3. 
CUDA_VISIBLE_DEVICES=0 bash test_nerp_demo.sh
```

### Reproduction

We provide various scripts for different experimental settings (including baseline methodes training & NeRP testing). The main scripts are:

Baselines:
- `train_baseline_b2n.sh` (Base-to-Novel Generalization)
- `train_baseline_cd.sh` (Cross-Dataset Evaluation and Domain Generalization)
- Detailed bash scripts in `scripts/cocoop`, `scripts/promptsrc`, `scripts/mma`, `scripts/mmrl`

NeRP:
- `test_nerp_b2n.sh` (Base-to-Novel Generalization)
- `test_nerp_cd.sh` (Cross-Dataset Evaluation and Domain Generalization)
- Detailed bash scripts in `scripts/cocoop_nerp`, `scripts/promptsrc_nerp`, `scripts/mma_nerp`, `scripts/mmrl_nerp`

To run the experiments, navigate to the root directory and execute the corresponding script. Make sure to replace `DATA=PATH_TO_YOUR_DATASETS` with the path to your dataset in `scripts`.  

**Base-to-Novel Generalization**  

Run the following command:  

```bash
# The script uses MMRL as the default baseline. 
# If you need to reproduce other models, please uncomment the corresponding lines.
CUDA_VISIBLE_DEVICES=0 bash train_baseline_b2n.sh
CUDA_VISIBLE_DEVICES=0 bash test_nerp_b2n.sh
```

**Cross-Dataset Evaluation and Domain Generalization**  

Run the following command:  

```bash
# The script uses MMRL as the default baseline. 
# If you need to reproduce other models, please uncomment the corresponding lines.
CUDA_VISIBLE_DEVICES=0 bash train_baseline_cd.sh
CUDA_VISIBLE_DEVICES=0 bash test_nerp_cd.sh
```


## ✨✨ Acknowledgment  
NeRP is built upon [PromptKD](https://github.com/KaiyangZhou/CoOp) and [MMRL](https://github.com/muzairkhattak/multimodal-prompt-learning).  We sincerely appreciate their contributions!



## 📌📌 Citation  

If you find this repository useful for your research, please consider citing:  

```bibtex
TODO
```