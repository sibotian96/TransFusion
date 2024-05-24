# [IEEE RA-L] TransFusion
#### TransFusion: A Practical and Effective Transformer-based Diffusion Model for 3D Human Motion Prediction

[Sibo Tian](https://scholar.google.com/citations?hl=en&user=fv-tcZIAAAAJ)<sup>1</sup>, [Minghui Zheng](https://engineering.tamu.edu/mechanical/profiles/zheng-minghui.html)<sup>1,\*</sup>, [Xiao Liang](https://engineering.tamu.edu/civil/profiles/liang-xiao.html)<sup>2,\*</sup>

<sup>1</sup>J. Mike Walker ’66 Department of Mechanical Engineering, Texas A&M University, <sup>2</sup>Zachry Department of Civil and Environmental Engineering, Texas A&M University, <sup>\*</sup>Corresponding Authors

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfusion-a-practical-and-effective/human-pose-forecasting-on-amass)](https://paperswithcode.com/sota/human-pose-forecasting-on-amass?p=transfusion-a-practical-and-effective)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfusion-a-practical-and-effective/human-pose-forecasting-on-human36m)](https://paperswithcode.com/sota/human-pose-forecasting-on-human36m?p=transfusion-a-practical-and-effective)

[[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10530938)] | [[Code](https://github.com/sibotian96/TransFusion)]

> Predicting human motion plays a crucial role in ensuring a safe and effective human-robot close collaboration in intelligent remanufacturing systems of the future. Existing works can be categorized into two groups: those focusing on accuracy, predicting a single future motion, and those generating diverse predictions based on observations. The former group fails to address the uncertainty and multi-modal nature of human motion, while the latter group often produces motion sequences that deviate too far from the ground truth or become unrealistic within historical contexts. To tackle these issues, we propose TransFusion, an innovative and practical diffusion-based model for 3D human motion prediction which can generate samples that are more likely to happen while maintaining a certain level of diversity. Our model leverages Transformer as the backbone with long skip connections between shallow and deep layers. Additionally, we employ the discrete cosine transform to model motion sequences in the frequency space, thereby improving performance. In contrast to prior diffusion-based models that utilize extra modules like cross-attention and adaptive layer normalization to condition the prediction on past observed motion, we treat all inputs, including conditions, as tokens to create a more practical and effective model compared to existing approaches. Extensive experimental studies are conducted on benchmark datasets to validate the effectiveness of our human motion prediction model.

## 📢 News

**[2024/05/23]: Code released!**

**[2024/04/28]: Our work is accepted by IEEE Robotics and Automation Letters (RA-L)!**

**[2024/03/25]: TransFusion prediction demos released!**

## 🛠 Setup

### 1. Python/Conda Environment

```
sh install.sh
```

### 2. Datasets

**Datasets for [Human3.6M](http://vision.imar.ro/human3.6m/description.php), [HumanEva-I](http://humaneva.is.tue.mpg.de/) and [AMASS](https://amass.is.tue.mpg.de/)**:

For Human3.6M and HumanEva-I, we adopt the data preprocessing from [GSPS](https://github.com/wei-mao-2019/gsps). For AMASS, we carefully adopt the data preprocessing from [BeLFusion](https://github.com/BarqueroGerman/BeLFusion). We provide all the processed data [here](https://drive.google.com/drive/folders/1J_8XyZC_sgRYZg6TQm09ZhlcsjjYO9Y8?usp=sharing) for convenience. Download all files into the `./data` directory and the final `./data` directory structure is shown below:

```
data
├── data_3d_amass.npz
├── data_3d_amass_test.npz
├── data_3d_h36m.npz
├── data_3d_h36m_test.npz
├── data_3d_humaneva15.npz
├── data_3d_humaneva15_test.npz
├── data_multi_modal
│   ├── data_candi_t_his25_t_pred100_skiprate20.npz
│   └── t_his25_1_thre0.500_t_pred100_thre0.100_filtered_dlow.npz
└── humaneva_multi_modal
    ├── data_candi_t_his15_t_pred60_skiprate15.npz
    └── t_his15_1_thre0.500_t_pred60_thre0.010_index_filterd.npz
```

### 3. Pretrained Models

We provide the pretrained models for all three datasets [here](https://drive.google.com/drive/folders/16iPASM7pnYEixBXaVFnp2pGbjgg-Ppxq?usp=sharing). Download all files into the `./checkpoints` directory and the final `./checkpoints` directory structure is shown below:

```
checkpoints
├── humaneva_ckpt.pt
├── h36m_ckpt.pt
└── amass_ckpt.pt
```

## 🔎 Evaluation
Evaluate on Human3.6M:

```
python main.py --cfg h36m --mode eval --ckpt ./checkpoints/h36m_ckpt.pt
```

Evaluate on HumanEva-I:

```
python main.py --cfg humaneva --mode eval --ckpt ./checkpoints/humaneva_ckpt.pt
```

Evaluate on AMASS:

```
python main.py --cfg amass --mode eval --ckpt ./checkpoints/amass_ckpt.pt --seed 6
```

**Note**: We change the random seed to 6 instead of 0 for AMASS dataset to fairly compared with [BeLFusion](https://github.com/BarqueroGerman/BeLFusion). GPU is required for evaluation.

## ⏳ Training
For training TransFusion from scratch for all three datasets, run the following scripts:
```
python main.py --cfg h36m --mode train
```
```
python main.py --cfg humaneva --mode train
```
```
python main.py --cfg amass --mode train --multimodal_threshold 0.4 --seed 6 --milestone [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800]
```

## 🎥 Visualization
Run the following scripts for visualization purpose:
```
python main.py --cfg h36m --mode pred --vis_row 3 --vis_col 10 --ckpt ./checkpoints/h36m_ckpt.pt
```
```
python main.py --cfg humaneva --mode pred --vis_row 3 --vis_col 10 --ckpt ./checkpoints/humaneva_ckpt.pt
```
```
python main.py --cfg amass --mode pred --vis_row 3 --vis_col 10 --ckpt ./checkpoints/amass_ckpt.pt
```

## 🎞 Demos of Human Motion Prediction

More prediction demos can be found in `./assets`.

#### Human3.6M -- Walking
![](assets/H36M_Walking.gif)

#### Human3.6M -- Walk Together
![](assets/H36M_WalkTogether.gif)

#### Human3.6M -- Photo
![](assets/H36M_Photo.gif)

#### Human3.6M -- Purchases
![](assets/H36M_Purchases.gif)

#### HumanEva-I -- Jog
![](assets/HumanEva_Jog.gif)

#### HumanEva-I -- ThrowCatch
![](assets/HumanEva_ThrowCatch.gif)

#### HumanEva-I -- Walking
![](assets/HumanEva_Walking.gif)

#### HumanEva-I -- Gestures
![](assets/HumanEva_Gestures.gif)

#### AMASS -- DanceDB
![](assets/AMASS_DanceDB.gif)

#### AMASS -- DFaust
![](assets/AMASS_DFaust.gif)

#### AMASS -- SSM
![](assets/AMASS_SSM.gif)

#### AMASS -- Transitions
![](assets/AMASS_Transitions.gif)


## 🌹 Acknowledgment
Project structure is borrowed from [HumanMAC](https://github.com/LinghaoChan/HumanMAC). We would like to thank the authors for making their code publicly available.

## 📝 Citation
If you find our work useful in your research, please consider citing our paper:
```
@article{tian2024transfusion,
  title={TransFusion: A practical and effective transformer-based diffusion model for 3d human motion prediction},
  author={Tian, Sibo and Zheng, Minghui and Liang, Xiao},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```

## 📚 License
The software in this repository is freely available for free non-commercial use (see [license](https://github.com/sibotian96/TransFusion/blob/main/LICENSE) for further details).
