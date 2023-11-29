# Bi-Temporal Adapter Network for Remote Sensing Image Change Detection



Our code is developed on top of [Open-CD v1.1.0](https://github.com/likyoo/open-cd/tree/main).



## Usage

### Install

- Create a conda virtual environment and activate it:

```bash
conda create -n BAN python=3.8 -y
conda activate BAN
```

- Install `PyTorch` and `torchvision` :

For examples, to install torch==2.0.0 with CUDA==11.8:

```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

- Install `OpenMMLab` Toolkits as Python packages:

```bash
pip install -U openmim
mim install mmengine==0.10.1
mim install mmcv==2.1.0
mim install mmpretrain==1.1.1
pip install mmsegmentation==1.2.1
pip install mmdet==3.2.0
```

- Install Open-CD

```bash
git clone https://github.com/likyoo/open-cd.git
cd open-cd
pip install -v -e .
cd ..
```

- Install other requirements:

```bash
pip install ftfy regex
```

- Clone this repo:

```bash
git clone https://github.com/likyoo/BAN.git
cd BAN
```

### Data Preparation

Download datasets ([LEVIR-CD](https://justchenhao.github.io/LEVIR/), [S2Looking](https://github.com/S2Looking/Dataset), [BANDON](https://github.com/fitzpchao/BANDON), [WHU-CD](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html))  and move (or link) them to `BAN/data`.


### Evaluation

To evaluate our `BAN` on LEVIR-CD test, run:

```bash
python test.py <config-file> <checkpoint>
```

You can download checkpoint files from [here](https://huggingface.co/likyoo/BAN/tree/main/checkpoint).

For example, to evaluate the `BAN-vit-l14-clip_mit-b0` with a single GPU:

```bash
python test.py configs/ban/ban_vit-l14-clip_mit-b0_512x512_40k_levircd.py checkpoint/ban_vit-l14-clip_mit-b0_512x512_40k_levircd.pth
```

### Training

To train the `BAN`, run:

```bash
python train.py <config-file>
```

For example, to train the `BAN-vit-l14-clip_mit-b0` with a single GPU on LEVIR-CD, run:

```bash
python train.py configs/ban/ban_vit-l14-clip_mit-b0_512x512_40k_levircd.py
```

**Note**: You can download pretrained files from [here](https://huggingface.co/likyoo/BAN/tree/main/pretrain).
