# iNeRF

### [Project Page](https://yenchenlin.me/inerf/) | [Video](https://www.youtube.com/watch?v=eQuCZaQN0tI&feature=emb_logo) | [Paper](https://arxiv.org/pdf/2012.05877.pdf)

<img src="https://user-images.githubusercontent.com/7057863/161619992-fa198e28-2990-4ff7-87a9-9ba320021dff.png" height=200>

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

## Environment setup

To start, we prefer creating the environment using conda:
```sh
conda env create -f environment.yml
conda activate pixelnerf
```
Please make sure you have up-to-date NVIDIA drivers supporting CUDA 10.2 at least.

Alternatively use `pip -r requirements.txt`.

## Quick start

The main implementation is in the `src/` directory,
while evalutation scripts are in `eval/`.

First, download all pretrained weight files from
<https://drive.google.com/file/d/1UO_rL201guN6euoWkCOn-XpqR2e8o6ju/view?usp=sharing>.
Extract this to `<project dir>/checkpoints/`, so that `<project dir>/checkpoints/dtu/pixel_nerf_latest` exists.


## ShapeNet Multiple Categories (NMR)

1. Download NMR ShapeNet renderings (see Datasets section, 1st link)
2. Run using
    - `python eval/gen_video.py  -n sn64 --gpu_id <GPU(s)> --split test -P '2'  -D <data_root>/NMR_Dataset -S 0`
    - For unseen category generalization:
      `python eval/gen_video.py  -n sn64_unseen --gpu_id=<GPU(s)> --split test -P '2'  -D <data_root>/NMR_Dataset -S 0`

Replace `<GPU(s)>` with desired GPU id(s), space separated for multiple.  Replace `-S 0` with `-S <object_id>` to run on a different ShapeNet object id.
Replace `-P '2'` with `-P '<number>'` to use a different input view.
Replace `--split test` with `--split train | val` to use different data split.
Append `-R=20000` if running out of memory.

**Result will be at** `visuals/sn64/videot<object_id>.mp4` or `visuals/sn64_unseen/videot<object_id>.mp4`.
The script will also print the path.


Pre-generated results for all ShapeNet objects with comparison may be found at <https://www.ocf.berkeley.edu/~sxyu/ZG9yaWF0aA/pixelnerf/cross_v2/>

## ShapeNet Single-Category (SRN)

1. Download SRN car (or chair) dataset from Google drive folder in Datasets section.
Extract to `<srn data dir>/cars_<train | test | val>`
2. `python eval/gen_video.py -n srn_car --gpu_id=<GPU (s)> --split test -P '64 104' -D <srn data dir>/cars -S 1`

Use `-P 64` for 1-view (view numbers are from SRN).
The chair set case is analogous (replace car with chair).
Our models are trained with random 1/2 views per batch during training.
This seems to degrade performance especially for 1-view. It may be preferrable to use 
a fixed number of views instead.

## DTU

Make sure you have downloaded the pretrained weights above.

1. Download DTU dataset from Google drive folder in Datasets section. Extract to some directory, to get: `<data_root>/rs_dtu_4`
2. Run using `python eval/gen_video.py  -n dtu --gpu_id=<GPU(s)> --split val -P '22 25 28'  -D <data_root>/rs_dtu_4 -S 3 --scale 0.25`

Replace `<GPU(s)>` with desired GPU id(s). Replace `-S 3` with `-S <scene_id>` to run on a different scene. This is not DTU scene number but 0-14 in the val set.
Remove `--scale 0.25` to render at full resolution (quite slow).

**Result will be at** visuals/dtu/videov<scene_id>.mp4.
The script will also print the path.

Note that for DTU, I only use train/val sets, where val is used for test. This is due to the very small size of the dataset.
The model overfits to the train set significantly during training.

## Real Car Images

**Note: requires PointRend from detectron2.**
Install detectron2 by following https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md.

Make sure you have downloaded the pretrained weights above.

1. Download any car image.
Place it in `<project dir>/input`. Some example images are shipped with the repo.
The car should be fully visible.
2. Run the preprocessor script: `python scripts/preproc.py`. This saves `input/*_normalize.png`.
If the result is not reasonable, PointRend didn't work; please try another imge.
3. Run `python eval/eval_real.py`. Outputs will be in `<project dir>/output`

The Stanford Car dataset contains many example car images:
<https://ai.stanford.edu/~jkrause/cars/car_dataset.html>.
Note the normalization heuristic has been slightly modified compared to the paper. There may be some minor differences.
You can pass `-e -20` to `eval_real.py` to set the elevation higher in the generated video.


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
