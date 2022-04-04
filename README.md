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

Although this codebase only shows how to apply iNeRF with pixelNeRF, iNeRF can work with the original NeRF as well.

## Environment setup

To start, create the environment using conda:
```sh
cd pixel-nerf
conda env create -f environment.yml
conda activate pixelnerf
```

Alternatively use `pip -r requirements.txt`.

Please make sure you have up-to-date NVIDIA drivers supporting CUDA 10.2 at least.

## Quick start

1. Download all pixelNeRF's pretrained weight files from [here](https://drive.google.com/file/d/1UO_rL201guN6euoWkCOn-XpqR2e8o6ju/view?usp=sharing).
Extract this to `./pixel-nerf/checkpoints/`, so that `./pixel-nerf/checkpoints/srn_car/pixel_nerf_latest` exists.




# BibTeX

```
@misc{yu2020pixelnerf,
      title={pixelNeRF: Neural Radiance Fields from One or Few Images},
      author={Alex Yu and Vickie Ye and Matthew Tancik and Angjoo Kanazawa},
      year={2020},
      eprint={2012.02190},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Acknowledgements

Parts of the code were based on from kwea123's NeRF implementation: https://github.com/kwea123/nerf_pl.
Some functions are borrowed from DVR https://github.com/autonomousvision/differentiable_volumetric_rendering
and PIFu https://github.com/shunsukesaito/PIFu
