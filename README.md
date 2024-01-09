<div align="center">

# [ICCV 2023 Oral🔥] G2L: Semantically Aligned and Uniform Video Grounding via Geodesic and Game Theory

[![Paper](http://img.shields.io/badge/Paper-arxiv.2303.14369-FF6B6B.svg)](https://arxiv.org/abs/2307.14277)
</div>

## 📌 Citation
If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:
```
@inproceedings{li2023g2l,
  title={G2l: Semantically aligned and uniform video grounding via geodesic and game theory},
  author={Li, Hongxiang and Cao, Meng and Cheng, Xuxin and Li, Yaowei and Zhu, Zhihong and Zou, Yuexian},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12032--12042},
  year={2023}
}
```

## 🚀 Quick Start
### Setup

#### Setup code environment
Our code is developed on the [third-party implementation of 2D-TAN](https://github.com/ChenJoya/2dtan), so we have similar dependencies with it, such as:

```
yacs h5py terminaltables tqdm pytorch transformers torch-geometric
```

### Datasets

* Download the [video feature](https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav)  provided by [2D-TAN](https://github.com/microsoft/2D-TAN).
* Extract and put the feature in the corresponding dataset in the  `dataset` folder. For configurations of feature/groundtruth's paths, please refer to `./g2l/config/paths_catalog.py`. (ann_file is the annotation, feat_file is the video feature)


### Train
```
bash scripts\anet_train.sh anet
```

## 🎗️ Acknowledgement

We appreciate [MMN](https://github.com/MCG-NJU/MMN), [2D-TAN](https://github.com/microsoft/2D-TAN) for video feature and configurations, and the [third-party implementation of 2D-TAN](https://github.com/ChenJoya/2dtan) for its implementation with `DistributedDataParallel`. Disclaimer: the performance gain of this [third-party implementation](https://github.com/ChenJoya/2dtan) is due to a tiny mistake of adding val set into training, yet our reproduced result is similar to the reported result in 2D-TAN paper.
