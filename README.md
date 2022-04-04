# iNeRF

### [Project Page](https://yenchenlin.me/inerf/) | [Video](https://www.youtube.com/watch?v=eQuCZaQN0tI&feature=emb_logo) | [Paper](https://arxiv.org/pdf/2012.05877.pdf)

<img src="https://user-images.githubusercontent.com/7057863/161620132-2ce16dca-53f6-413d-97ab-fe6086f1661c.gif" height=200>

PyTorch implementation of iNeRF, an RGB-only method that inverts neural radiance fields (NeRFs) for 6DoF pose estimation.

[iNeRF Inverting Neural Radiance Fields for Pose Estimation](https://yenchenlin.me/inerf/)  
 [Lin Yen-Chen](https://yenchenlin.me/)<sup>1</sup>,
 [Pete Florence](http://www.peteflorence.com/)<sup>2</sup>,
 [Jonathan T. Barron](http://jonbarron.info/)<sup>2</sup>,
 [Alberto Rodriguez](https://meche.mit.edu/people/faculty/ALBERTOR@MIT.EDU)<sup>1</sup>,
 [Phillip Isola](http://web.mit.edu/phillipi/)<sup>1</sup>,
  [Tsung-Yi Lin](https://scholar.google.com/citations?user=_BPdgV0AAAAJ&hl=en)<sup>2</sup><br>
 <sup>1</sup>MIT, <sup>2</sup>Google
 <br>
 [IROS 2021](https://www.iros2021.org/)

## Overview

This preliminary codebase currently only shows how to apply iNeRF with pixelNeRF. However, iNeRF can work with the original NeRF as well.

## Environment setup

To start, create the environment using conda:
```sh
cd pixel-nerf
conda env create -f environment.yml
conda activate pixelnerf
pip install mediapy
pip install jupyter
```

Please make sure you have up-to-date NVIDIA drivers supporting CUDA 10.2 at least.

## Quick start

1. Download all pixelNeRF's pretrained weight files from [here](https://drive.google.com/file/d/1UO_rL201guN6euoWkCOn-XpqR2e8o6ju/view?usp=sharing).
Extract this to `./pixel-nerf/checkpoints/`, so that `./pixel-nerf/checkpoints/srn_car/pixel_nerf_latest` exists.

2. Launch the Jupyter notebook.
```sh
cd pixel-nerf
jupyter notebook
```

3. Open `pose_estimation.ipynb` and run through it. You can preview the results [here](https://github.com/yenchenlin/iNeRF-public/blob/master/pixel-nerf/pose_estimation.ipynb). In the following, we show the overlay of images rendered with our predicted poses and the target image.

<img src="https://user-images.githubusercontent.com/7057863/161636178-c4f36310-eb62-44fc-abad-7d90b0637de6.gif" width=128>


# BibTeX

```
@inproceedings{yen2020inerf,
  title={{iNeRF}: Inverting Neural Radiance Fields for Pose Estimation},
  author={Lin Yen-Chen and Pete Florence and Jonathan T. Barron and Alberto Rodriguez and Phillip Isola and Tsung-Yi Lin},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems ({IROS})},
  year={2021}
}
```

# Acknowledgements

This implementation is based on Alex Yu's [pixel-nerf](https://github.com/sxyu/pixel-nerf).
