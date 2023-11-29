# BAN

[Bi-Temporal Adapter Network for Remote Sensing Image Change Detection]()

## Introduction

[Code Snippet](https://github.com/likyoo/BAN/blob/main/opencd_custom/models/decode_heads/ban.py)

## Abstract
Change detection (CD) is a critical task in observing and analyzing the dynamic processes of land cover. Although numerous deep learning-based CD models have obtained excellent performance, their further performance improvements are constrained by the limited knowledge extracted from the given labelled data. On the other hand, the foundation models emerged recently contain a huge amount of knowledge by scaling up across data modalities and proxy tasks. In this paper, we propose a Bi-Temporal Adapter Network (BAN), which aims to fully utilize the knowledge of foundation models for CD. BAN is a universal foundation model-based CD adaptation framework which consists of three parts, i.e. frozen foundation model (e.g., CLIP), Bi-Temporal Adapter Branch (Bi-TAB), and bridging modules between them. Specifically, Bi-TAB can be either an existing arbitrary CD model or some hand-crafted stacked blocks. The bridging modules focus on aligning the general features with the task/domain-specific features and injecting the selected general knowledge into the Bi-TAB. To our knowledge, this is the first universal framework to extensively adapt the foundation model to CD. Extensive experiments demonstrate the effectiveness of our proposed BAN on improving the performance of existing CD models, e.g., up to 4.08\% IoU improvement, with only a few additional learnable parameters. More importantly, these successful practices show us the potential of foundation models in remote sensing CD. The code is available at https://github.com/likyoo/BAN and will be supported in [Open-CD](https://github.com/likyoo/open-cd).

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/likyoo/BAN/blob/main/resources/BAN.png" width="100%"/>
</div>


```bibtex

```

## Results and models

### LEVIR-CD

| Method |       Pretrain       |     Bi-TAB      | Crop Size | Lr schd | Precision | Recall | F1-Score |  IoU  |   config   |
| :----: | :------------------: | :-------------: | :-------: | :-----: | :-------: | :----: | :------: | :---: | :--------: |
|  BAN   |    ViT-L/14, CLIP    |       BiT       |  512x512  |  40000  |   92.83   | 90.89  |  91.85   | 84.93 | [config]() |
|  BAN   |    ViT-B/16, CLIP    | ChangeFormer-b0 |  512x512  |  40000  |   93.25   | 90.21  |  91.71   | 84.68 | [config]() |
|  BAN   |    ViT-L/14, CLIP    | ChangeFormer-b0 |  512x512  |  40000  |   93.47   | 90.30  |  91.86   | 84.94 | [config]() |
|  BAN   |    ViT-L/14, CLIP    | ChangeFormer-b1 |  512x512  |  40000  |   93.48   | 90.76  |  92.10   | 85.36 | [config]() |
|  BAN   |    ViT-L/14, CLIP    | ChangeFormer-b2 |  512x512  |  40000  |   93.61   | 91.02  |  92.30   | 85.69 | [config]() |
|  BAN   | ViT-B/32, RemoteCLIP | ChangeFormer-b0 |  512x512  |  40000  |   93.28   | 90.26  |  91.75   | 84.75 | [config]() |
|  BAN   | ViT-L/14, RemoteCLIP | ChangeFormer-b0 |  512x512  |  40000  |   93.44   | 90.46  |  91.92   | 85.05 | [config]() |
|  BAN   |   ViT-B/16, IN-21K   | ChangeFormer-b0 |  512x512  |  40000  |   93.59   | 89.80  |  91.66   | 84.60 | [config]() |
|  BAN   |   ViT-L/16, IN-21K   | ChangeFormer-b0 |  512x512  |  40000  |   93.27   | 90.11  |  91.67   | 84.61 | [config]() |

### S2Looking

| Method |    Pretrain    |     Bi-TAB      | Crop Size | Lr schd | Precision | Recall | F1-Score |  IoU  |   config   |
| :----: | :------------: | :-------------: | :-------: | :-----: | :-------: | :----: | :------: | :---: | :--------: |
|  BAN   | ViT-L/14, CLIP |       BiT       |  512x512  |  80000  |   75.06   | 58.00  |  65.44   | 48.63 | [config]() |
|  BAN   | ViT-L/14, CLIP | ChangeFormer-b0 |  512x512  |  80000  |   74.63   | 60.30  |  66.70   | 50.04 | [config]() |

### BANDON-SCD (In-domain Test)

| Method |    Pretrain    |     Bi-TAB      | Crop Size | Lr schd | Precision | Recall | F1-Score |  IoU  |   config   |
| :----: | :------------: | :-------------: | :-------: | :-----: | :-------: | :----: | :------: | :---: | :--------: |
|  BAN   | ViT-L/14, CLIP | ChangeFormer-b0 |  512x512  |  40000  |   78.19   | 67.71  |  72.57   | 56.95 | [config]() |
|  BAN   | ViT-L/14, CLIP | ChangeFormer-b2 |  512x512  |  40000  |   79.66   | 70.44  |  74.77   | 59.70 | [config]() |


- All metrics are based on the category "change".
- All scores are computed on the test set.
