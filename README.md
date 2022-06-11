# AnoDDPM: Anomaly Detection with Denoising Diffusion Probabilistic Models using Simplex Noise

This is the github repository for an anomaly detection approach utilising DDPMs with simplex noise implemented in pytorch.

The code was written by [Julian Wyatt](https://github.com/Julian-Wyatt) and is based off the [Guided Diffusion Repo](https://github.com/openai/guided-diffusion) and a fork of a [python simplex noise library](https://github.com/lmas/opensimplex).

The project was accepted at the CVPR Workshop: NTIRE 2022: [Project](https://julianwyatt.co.uk/anoddpm) | [Paper](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.html)


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

## Citation:

If you use this code for your research, please cite:<br>
AnoDDPM: Anomaly Detection With Denoising Diffusion Probabilistic Models Using Simplex Noise<br>
[Julian Wyatt](https://github.com/Julian-Wyatt), [Adam Leach](https://github.com/qazwsxal), [Sebastian M. Schmon](https://scholar.google.com/citations?user=hs2WrYYAAAAJ&hl=en&oi=ao), [Chris G. Willcocks](https://github.com/cwkx); Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2022

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

