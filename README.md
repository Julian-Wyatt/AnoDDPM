# AnoDDPM: Anomaly Detection with Denoising Diffusion Probabilistic Models using Simplex Noise

This is the github repository for an anomaly detection approach utilising DDPMs with simplex noise implemented in
pytorch.

The code was written by [Julian Wyatt](https://github.com/Julian-Wyatt) and is based off
the [Guided Diffusion Repo](https://github.com/openai/guided-diffusion) and a fork of
a [python simplex noise library](https://github.com/lmas/opensimplex).

The project was accepted at the CVPR Workshop: NTIRE 2022: [Project](https://julianwyatt.co.uk/anoddpm)
| [Paper](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.html)

## Simplex noise examples

<p align="center">
  <img alt="gif 1" src="https://github.com/Julian-Wyatt/JulianWyatt.github.io/blob/db50a67bec8aece87e185260572ece35d74b74df/assets/img/portfolio/anoddpm2-compressed.gif" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="gif 2" src="https://github.com/Julian-Wyatt/JulianWyatt.github.io/blob/db50a67bec8aece87e185260572ece35d74b74df/assets/img/portfolio/anoddpm3-compressed.gif" width="45%">
</p>

## Gaussian noise example

<p align="center">
  <img src='https://github.com/Julian-Wyatt/JulianWyatt.github.io/blob/db50a67bec8aece87e185260572ece35d74b74df/assets/img/portfolio/anoddpmGauss.gif' width=45%>
</p>

## File structure:

- dataset.py - custom dataset loader
- detection.py - code for generating measures & initial testing and experimentation.
- diffusion_training.py - training procedure
- evaluation.py - functions for measures and metrics
- GaussianDiffusion.py - Gaussian architecture with custom detection, forked from https://github.
  com/openai/guided-diffusion
- generate_images.py - generates images for Figs in paper
- graphs.py - reduce graph quality, load and visualise graphs
- helpers.py - helper functions for use in several places ie checkpoint loading
- perlin.py - Generating Fig 2 and testing octaves
- simplex.py - Simplex class - forked from https://github.com/lmas/opensimplex with added multi-scale code.
- UNet.py - UNet architecture, forked from https://github.com/openai/guided-diffusion
- test_args/args{i}.json - primary example seen below
- model/diff-params-ARGS={i}/params-final.pt - checkpoint for i'th arg
- Examples/ - demonstration of early testing
- diffusion-videos/ARGS={i}/ - video outputs of varying args across training, testing and detection
- diffusion-training-images/ARGS={i}/ - detection images
- metrics/ - storage of varying metrics
- final-outputs/ - outputs from generate_images.py

For access to checkpoints, please get in touch. For access to datasets, please refer to the paper's citations.

## How To...

### Train

To train a model, run `python3 diffusion_training.py ARG_NUM` where `ARG_NUM` is the number relating to the json arg
file. These arguments are stored in ./test_args/ and are called args1.json for example.

### Evaluate

To evaluate a model, run `python3 detection.py ARG_NUM`, and ensure the script runs the correct sub function.

### Datasets

To perform the same tests, store the anomalous dataset
in `./DATASETS/CancerousDataset/EdinburghDataset/Anomalous-T1/raw` and the training dataset in
`./DATASETS/Train/`. The training dataset contained 100 folders where each contained the raw file and the numpy
extracted file. To add a new dataset, edit the `dataset.py` file and ensure the new dataset is loaded via the script
you're running.

## Example args:

{
"img_size": [256,256],
"Batch_Size": 1,
"EPOCHS": 3000,
"T": 1000,
"base_channels": 128,
"beta_schedule": "linear",
"channel_mults": "",
"loss-type": "l2",
"loss_weight": "none",
"train_start": true,
"lr": 1e-4,
"random_slice": true,
"sample_distance": 800,
"weight_decay": 0.0,
"save_imgs":false,
"save_vids":true,
"dropout":0,
"attention_resolutions":"16,8",
"num_heads":2,
"num_head_channels":-1,
"noise_fn":"simplex",
"dataset": "mri"
}

## Citation:

If you use this code for your research, please cite:<br>
AnoDDPM: Anomaly Detection With Denoising Diffusion Probabilistic Models Using Simplex Noise<br>
[Julian Wyatt](https://github.com/Julian-Wyatt), [Adam Leach](https://github.com/qazwsxal)
, [Sebastian M. Schmon](https://scholar.google.com/citations?user=hs2WrYYAAAAJ&hl=en&oi=ao)
, [Chris G. Willcocks](https://github.com/cwkx); Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR) Workshops, 2022

```
@InProceedings{Wyatt_2022_CVPR,
    author    = {Wyatt, Julian and Leach, Adam and Schmon, Sebastian M. and Willcocks, Chris G.},
    title     = {AnoDDPM: Anomaly Detection With Denoising Diffusion Probabilistic Models Using Simplex Noise},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {650-656}
}
```

