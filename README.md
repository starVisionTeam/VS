# VS: Reconstructing Clothed 3D Human from Single Image via Vertex Shift

The code and models are coming soon!
## Installation
### Install "Manifold" 
This code relies on the [Robust Watertight Manifold Software](https://github.com/hjwdzh/Manifold). 
First ```cd``` into the location you wish to install the software. For example, we used ```cd ~/code```.
Then follow the installation instructions in the Watertight README.
If you installed Manifold in a different path than ```~/code/Manifold/build```,  accordingly (see [this line](https://github.com/starVisionTeam/VS/blob/b36e4c7bfa3a2b7b6a4a6463ad96c14e56fe0f83/Mr/util/util.py#L9))
### Environment
- Ubuntu 20 / 18
- **CUDA=11.6, GPU Memory > 12GB**
- Python = 3.8
- PyTorch >= 1.13.0 (official [Get Started](https://pytorch.org/get-started/locally/))
- Cupy >= 11.3.0 (offcial [Installation](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi))
- PyTorch3D = 0.7.2 (official [INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md), recommend [install-from-local-clone](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#2-install-from-a-local-clone))

```bash
cd VS
conda env create -f environment.yaml
conda activate VS
pip install -r requirements.txt
```
## Download Pre-trained model and Related SMPL-X data 
Link：https://pan.baidu.com/s/1GDk1d6p5FEzd4Y1mSY9UTg，Access Code：vsvs.

 The `latest_net.pth` is saved under `./VS/Mr/checkpoints/debug/`,`pifuhd.pt` is saved under `./VS/pifuhd_ori/`,`data` is saved under `./VS/`.
## Quick Start

```bash
python -m apps.infer -in_dir ./examples -out_dir ./results
```
## Citation
Please consider citing the paper if you find the code useful in your research.
```
@InProceedings{VS_CVPR2024,
  author = {Liu, Leyuan and Li, Yuhan and Gao, Yunqi and Gao, Changxin and Liu, Yuanyuan and Chen, Jingying},
  booktitle = IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 
  title = {VS: Reconstructing Clothed 3D Human from Single Image via Vertex Shift}, 
  year = {2024},
  pages = {1-10}
}
```
## VcgPoeple Dataset

## Acknowledgements
Note that the *** code of this repo is based on ***. We thank the authors for their great job!

## Contact
We are still updating the code. If you have any trouble using this repo, please do not hesitate to E-mail Leyuan Liu (lyliu@mail.ccnu.edu.cn) or Yuhan Li (609806700@qq.com).
