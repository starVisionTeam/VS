# VS: Reconstructing Clothed 3D Human from Single Image via Vertex Shift
#### <p align="center">Leyuan Liu, Yuhan Li, Yunqi Gao, Changxin Gao, Yuanyuan Liu, Jingying Chen</p>

***
Various applications require high-fidelity and artifact-free 3D human reconstructions. However, current implicit function-based methods inevitably produce artifacts while existing deformation methods are difficult to reconstruct high-fidelity humans wearing loose clothing.
In this paper, we propose a two-stage deformation method named Vertex Shift (VS) for reconstructing clothed 3D humans from single images. 
Specifically, VS first stretches the estimated SMPL-X mesh into a coarse 3D human model using shift fields inferred from normal maps, then refines the coarse 3D human model into a detailed  3D human model via a graph convolutional network embedded with implicit-function-learned features. 
This ``stretch-refine'' strategy addresses large deformations required for reconstructing loose clothing and delicate deformations for recovering intricate and detailed surfaces, achieving high-fidelity reconstructions that faithfully convey the pose, clothing, and surface details from the input images. 
The graph convolutional network's ability to exploit neighborhood vertices coupled with the advantages inherited from the deformation methods ensure VS rarely produces artifacts like distortions and non-human shapes and never produces artifacts like holes, broken parts, and dismembered limbs. 
As a result, VS can reconstruct high-fidelity and artifact-less clothed 3D humans from single images, even under scenarios of challenging poses and loose clothing.
Experimental results on three benchmarks and two in-the-wild datasets demonstrate that VS significantly outperforms current state-of-the-art methods. 

## Qualitative Results
![](https://github.com/naivate/VS/blob/master/V2-ezgif.com-video-to-gif-converter%20(1).gif)


## Citation
Please consider citing the paper if you find the code useful in your research.
```
@InProceedings{VS_CVPR2024,
  author = {Liu, Leyuan and Li, Yuhan and Gao, Yunqi and Gao, Changxin and Liu, Yuanyuan and Chen, Jingying},
  booktitle = IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 
  title = {VS: Reconstructing Clothed 3D Human from Single Image via Vertex Shift}, 
  year = {2024},
  pages = {10498-10507}
}
```

## Installation

### Environment
- Ubuntu 20 / 18
- **CUDA=11.6, GPU Memory > 12GB**
- Python = 3.8
- PyTorch >= 1.13.0 (official [Get Started](https://pytorch.org/get-started/locally/))
- Cupy >= 11.3.0 (offcial [Installation](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi))
- PyTorch3D = 0.7.2 (official [INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md), recommend [install-from-local-clone](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#2-install-from-a-local-clone))
- 
### Install "Manifold" 
This code relies on the [Robust Watertight Manifold Software](https://github.com/hjwdzh/Manifold). 
First ```cd``` into the location you wish to install the software. For example, we used ```cd ~/code```.
Then follow the installation instructions in the Watertight README.
If you installed Manifold in a different path than ```~/code/Manifold/build```,  accordingly (see [this line](https://github.com/starVisionTeam/VS/blob/b36e4c7bfa3a2b7b6a4a6463ad96c14e56fe0f83/Mr/util/util.py#L9))

```bash
cd VS
conda env create -f environment.yaml
conda activate VS
pip install -r requirements.txt
```
### Download Pre-trained model and Related SMPL-X data 
Link：https://pan.baidu.com/s/1GDk1d6p5FEzd4Y1mSY9UTg

Access Code：vsvs.

 The `latest_net.pth` is saved under `./VS/Mr/checkpoints/debug/`,`pifuhd.pt` is saved under `./VS/pifuhd_ori/`,`data` is saved under `./VS/`.
## Quick Start

```bash
python -m apps.infer -in_dir ./examples -out_dir ./results
```


## Acknowledgements
Note that the *** code of this repo is based on ***. We thank the authors for their great job!

## Contact
We are still updating the code. If you have any trouble using this repo, please do not hesitate to E-mail Leyuan Liu (lyliu@mail.ccnu.edu.cn) or Yuhan Li (609806700@qq.com).
