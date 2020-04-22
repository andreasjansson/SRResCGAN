# Super-Resolution Residual Convolutional Generative Adversarial Network (SRResCGAN)
A PyTorch implementation of the [SRResCGAN](https://github.com/RaoUmer/SRResCGAN) model as described in the paper [Deep Generative Adversarial Residual Convolutional Networks for Real-World Super-Resolution](https://arxiv.org/). This work is participated in the [NTIRE 2020](https://data.vision.ee.ethz.ch/cvl/ntire20/) RWSR challenges on the [Real-World Super-Resolution](https://arxiv.org/).

#### Abstract
Most current deep learning based single image super-resolution (SISR) methods focus on  designing deeper / wider models to learn the non-linear mapping between low-resolution (LR) inputs and the high-resolution (HR) outputs from a large number of paired (LR/HR) training data. They usually take as assumption that the LR image is a bicubic down-sampled version of the HR image. However, such degradation process is not available in real-world settings i.e. inherent sensor noise, stochastic noise, compression artifacts, possible mismatch between image degradation process and camera device. It reduces significantly the performance of current SISR methods due to real-world image corruptions. To address these problems, we propose a deep Super-Resolution Residual Convolutional Generative Adversarial Network (SRResCGAN) to follow the real-world degradation settings by adversarial training the model with pixel-wise supervision in the HR domain from its generated LR counterpart. The proposed network exploits the residual learning by minimizing the energy-based objective function with powerful image regularization and convex optimization techniques. We demonstrate our proposed approach in quantitative and qualitative experiments that generalize robustly to real input and it is easy to deploy for other down-scaling operators and mobile/embedded devices.

#### Pre-trained Models
| |[DSGAN](https://github.com/ManuelFritsche/real-world-sr/tree/master/dsgan)|[SRResCGAN](https://github.com/RaoUmer/SRResCGAN)|
|---|:---:|:---:|
|NTIRE2020 RWSR|[Source-Domain-Learning](https://github.com/RaoUmer/SRResCGAN)|[SR-learning](https://github.com/RaoUmer/SRResCGAN)|

#### BibTeX
    @inproceedings{UmerCVPRW2020,
        title={Deep Generative Adversarial Residual Convolutional Networks for Real-World Super-Resolution},
        author={Rao Muhammad Umer and Gian Luca Foresti and Christian Micheloni},
        booktitle={CVPR Workshops},
        year={2020},
        }

## Quick Test
#### Dependencies
- [Python 3.7](https://www.anaconda.com/distribution/) (version >= 3.0)
- [PyTorch >= 1.0](https://pytorch.org/) (CUDA version >= 8.0 if installing with CUDA.)
- Python packages:  `pip install numpy opencv-python`

#### Test models
describe here later.

## SRResCGAN Architecture
#### Overall Representative diagram
<p align="center">
  <img height="120" src="figs/srrescgan.png">
</p>

#### SR Generator Network
<p align="center">
  <img height="180" src="figs/generator.png">
</p>

## Quantitative Results
describe here later.
| <sub>Dataset (HR/LR pairs)</sub> | <sub>SR methods</sub> | <sub>#Params</sub> | <sub>PSNR&#x2191;</sub> | <sub>SSIM&#x2191;</sub> | <sub>LPIPS&#x2193;</sub> | <sub>Artifacts</sub> |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| <sub>Bicubic</sub>| <sub>EDSR</sub>| <sub>43M</sub> |<sub>24.48</sub>|<sub>0.53</sub>|<sub>0.6800</sub>|<sub>Sensor noise (&#x03C3; = 8)</sub>|
| <sub>Bicubic</sub>| <sub>EDSR</sub>| <sub>43M</sub> |<sub>23.75</sub>|<sub>0.62</sub>|<sub>0.5400</sub>|<sub>JPEG compression (quality=30)</sub>|
| <sub>Bicubic</sub>| <sub>ESRGAN</sub>| <sub>16.7M</sub> |<sub>17.39</sub>|<sub>0.19</sub>|<sub>0.9400</sub>|<sub>Sensor noise (&#x03C3; = 8)</sub>|
| <sub>Bicubic</sub>| <sub>ESRGAN</sub>| <sub>16.7M</sub> |<sub>22.43</sub>|<sub>0.58</sub>|<sub>0.5300</sub>|<sub>JPEG compression (quality=30)</sub>|
| <sub>CycleGAN</sub>| <sub>ESRGAN-FT</sub>| <sub>16.7M</sub> |<sub>22.42</sub>|<sub>0.55</sub>|<sub>0.3645</sub>|<sub>Sensor noise (&#x03C3; = 8)</sub>|
| <sub>CycleGAN</sub>| <sub>ESRGAN-FT</sub>| <sub>16.7M</sub> |<sub>22.80</sub>|<sub>0.57</sub>|<sub>0.3729</sub>|<sub>JPEG compression (quality=30)</sub>|
| <sub>DSGAN</sub>| <sub>ESRGAN-FS</sub>| <sub>16.7M</sub> |<sub>22.52</sub>|<sub>0.52</sub>|<sub>0.3300</sub>|<sub>Sensor noise (&#x03C3; = 8)</sub>|
| <sub>DSGAN</sub>| <sub>ESRGAN-FS</sub>| <sub>16.7M</sub> |<sub>20.39</sub>|<sub>0.50</sub>|<sub>0.4200</sub>|<sub>JPEG compression (quality=30)</sub>|
| <sub>DSGAN</sub>| <sub>SRResCGAN (ours)</sub>| <sub>380K</sub> |<sub>25.46</sub>|<sub>0.67</sub>|<sub>0.3604</sub>|<sub>Sensor noise (&#x03C3; = 8)</sub>|
| <sub>DSGAN</sub>| <sub>SRResCGAN (ours)</sub>| <sub>380K</sub> |<sub>23.34</sub>|<sub>0.59</sub>|<sub>0.4431</sub>|<sub>JPEG compression (quality=30)</sub>|
| <sub>DSGAN</sub>| <sub>SRResCGAN+ (ours)</sub>| <sub>380K</sub> |<sub>26.01</sub>|<sub>0.71</sub>|<sub>0.3871</sub>|<sub>Sensor noise (&#x03C3; = 8)</sub>|
| <sub>DSGAN</sub>| <sub>SRResCGAN+ (ours)</sub>| <sub>380K</sub> |<sub>23.69</sub>|<sub>0.62</sub>|<sub>0.4663</sub>|<sub>JPEG compression (quality=30)</sub>|
| <sub>DSGAN</sub>| <sub>SRResCGAN (ours)</sub>| <sub>380K</sub> |<sub>25.05</sub>|<sub>0.67</sub>|<sub>0.3357</sub>|<sub>unknown (validset)</sub>|
| <sub>DSGAN</sub>| <sub>SRResCGAN+ (ours)</sub>| <sub>380K</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|<sub>unknown (validset)</sub>|
| <sub>DSGAN</sub>| <sub>ESRGAN-FS</sub>| <sub>16.7M</sub> |<sub>25.96</sub>|<sub>0.71</sub>|<sub>0.3401</sub>|<sub>unknown (testset)</sub>|
| <sub>DSGAN</sub>| <sub>SRResCGAN (ours)</sub>| <sub>380K</sub> |<sub>24.87</sub>|<sub>0.68</sub>|<sub>0.3250</sub>|<sub>unknown (testset)</sub>|

#### The NTIRE2020 RWSR Challenge Results (Track-1)
| <sub>Team</sub> | <sub>PSNR&#x2191;</sub> | <sub>SSIM&#x2191;</sub> | <sub>LPIPS&#x2193;</sub> | <sub>MOS&#x2193;</sub> |
|:---:|:---:|:---:|:---:|:---:|
| <sub>Impressionism</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>Samsung-SLSI-MSL</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>BOE-IOT-AIBD</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>MSMers</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>KU-ISPL</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>InnoPeak-SR</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>ITS425</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>MLP-SR</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>Webbzhou</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>SR-DL</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>TeamAY</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>BIGFEATURE-CAMERA</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>BMIPL-UNIST-YH-1</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>SVNIT1-A</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>KU-ISPL2</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>SuperT</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>GDUT-wp</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>SVNIT1-B</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>SVNIT2</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>AITA-Noah-A</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>AITA-Noah-B</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>Bicubic</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|
| <sub>ESRGAN Supervised</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|

## Visual Results
#### Validation-set (Track-1)
<p align="center">
  <img height="200" src="figs/res_valset_1.png">
</p>
<p align="center">
  <img height="200" src="figs/res_valset_2.png">
</p>

#### Test-set (Track-1)
<p align="center">
  <img height="200" src="figs/res_testset_1.png">
</p>
<p align="center">
  <img height="200" src="figs/res_testset_2.png">
</p>

#### Real-World Smartphone images (Track-2)
<p align="center">
  <img height="200" src="figs/res_mobile_1.png">
</p>
<p align="center">
  <img height="200" src="figs/res_mobile_2.png">
</p>

## Code Acknowledgement
describe here later.
