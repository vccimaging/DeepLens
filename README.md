# DeepLens

**DeepLens** is a differentiable optical lens simulator that supports multiple optical models (eg., geometric, diffractive, hybrid, neural, and interpolation). It can be used for (1) image simulation, (2) optical design, and (3) end-to-end optics-algorithm co-design ([End2end-Imaging](https://github.com/vccimaging/End2endImaging)). DeepLens enables researchers to rapidly prototype and optimize custom optical systems through differentiable simulation.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/singer-yang/DeepLens)

<div style="text-align:center;">
    <img src="assets/logo.png"/>
</div>

<p align="center">
    <a href="https://vccimaging.org/DeepLens/">Docs</a> •
    <a href="https://github.com/singer-yang/DeepLens-tutorials">Tutorials</a> •
    <a href="#community">Community</a> •
    <a href="https://pypi.org/project/deeplens-core/">PyPI</a> •
    <a href="https://deepwiki.com/singer-yang/DeepLens">DeepWiki</a>
</p>


## Features

1. **Differentiable Simulation.** DeepLens builds on differentiable physical simulation and enables accurate, efficient gradient calculation for lens optimization.
2. **Automated Design.** DeepLens demonstrates outstanding optimization power compared with classical optimization, especially for complex optical systems (e.g., mobile lenses, metasurfaces, and AR/VR displays). Automated lens design is demonstrated with curriculum learning and optical regularization losses.
3. **Multiple Optical Models.** DeepLens supports not only geometric ray tracing, but also various other optical models, including hybrid ray-wave models, neural lens representations, and reference-data interpolation.
4. **Image Simulation.** DeepLens delivers photorealistic image simulations with spatially varying and depth-dependent aberration simulation, bridging sim-to-real gaps when combined with [End2end-Imaging](https://github.com/vccimaging/End2endImaging).

Additional features (available via collaboration):

1. **Kernel Acceleration.** Achieves >10x speedup and >90% GPU memory reduction with custom GPU kernels across NVIDIA and AMD platforms.
2. **Polarization Ray Tracing.** Simulates polarization ray tracing and differentiable optimization of coating films.
3. **Non-Sequential Ray Tracing.** Simulates differentiable non-sequential ray tracing model for stray light analysis and optimization.
4. **Distributed Optimization.** Supports distributed simulation and optimization for billion-level of ray tracing and high-resolution (>100k x 100k) diffractive propagation.

## Applications

#### 1. Lens Analysis and Image Simulation

DeepLens supports comprehensive lens analysis (spot diagram, PSF, MTF, distortion, etc.) and photorealistic image simulation with spatially-varying, depth-dependent aberrations.

<div align="center">
    <img src="assets/feature.png" alt="Lens Analysis and Image Simulation"/>
</div>

#### 2. Automated geometric lens design

Fully automated lens design from scratch with differentiable optimization. Try it with [AutoLens](https://github.com/vccimaging/AutoLens)!

[![paper](https://img.shields.io/badge/NatComm-2024-orange)](https://www.nature.com/articles/s41467-024-50835-7) [![quickstart](https://img.shields.io/badge/Project-green)](https://github.com/vccimaging/AutoLens)

<div align="center">
    <img src="assets/autolens1.gif" alt="AutoLens" height="270px"/>
    <img src="assets/autolens2.gif" alt="AutoLens" height="270px"/>
</div>

#### 3. Neural Lens PSF Representation

A surrogate network for efficient lens PSF representation ang image simulation (spatially-varying aberration + defocus).

[![paper](https://img.shields.io/badge/TPAMI-2023-orange)](https://ieeexplore.ieee.org/document/10209238) [![link](https://img.shields.io/badge/Project-green)](https://github.com/vccimaging/Aberration-Aware-Depth-from-Focus)

<div align="center">
    <img src="assets/implicit_net.png" alt="Implicit" height="150px"/>
</div>

#### 4. Hybrid Ray-Wave Optical Model

Design hybrid refractive-diffractive lenses with differentiable ray-wave model.

[![report](https://img.shields.io/badge/SiggraphAsia-2024-orange)](https://arxiv.org/abs/2406.00834)

<div align="center">
    <img src="assets/hybridlens.png" alt="Implicit" height="200px"/>
</div>

#### 5. End-to-End Computational Imaging

DeepLens serves as the differentiable optics engine in [**End2endImaging**](https://github.com/vccimaging/End2endImaging), an end-to-end differentiable computational imaging framework. End2endImaging integrates optics (DeepLens), sensor/ISP simulation, and neural reconstruction networks into a single PyTorch computation graph, enabling joint optimization of the entire camera pipeline.

<div align="center">
    <img src="assets/end2end.png" alt="End2endImaging" height="200px"/>
</div>

## Installation

Clone this repo:

```
git clone https://github.com/singer-yang/DeepLens
cd DeepLens
```

Create a conda environment:
```
conda create -n deeplens_env python=3.12
conda activate deeplens_env

# Linux and Mac
pip install torch torchvision
# Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```
or
```
conda env create -f environment.yml -n deeplens_env
```

Run the demo code:
```
python 0_hello_geolens.py
```

DeepLens repo structure:

```
DeepLens/
│
├── deeplens/
│   ├── geolens.py          (multi-element refractive lens)
│   ├── hybridlens.py       (refractive + diffractive hybrid lens)
│   ├── diffraclens.py      (pure diffractive lens)
│   ├── defocuslens.py     (defocus / circle-of-confusion model)
│   ├── psfnetlens.py       (neural surrogate lens)
│   ├── geometric_surface/  (spheric, aspheric, aperture, etc.)
│   ├── diffractive_surface/(DOE surfaces)
│   ├── phase_surface/      (phase-only surfaces)
│   ├── light/              (Ray, ComplexWave)
│   ├── material/           (glass catalogs)
│   ├── imgsim/             (PSF convolution, monte carlo)
│   ├── geolens_pkg/        (eval, optim, vis, io mixins)
│   └── surrogate/          (MLP, Siren neural surrogates)
│
├── 0_hello_geolens.py     (code tutorials)
├── ...
└── write_your_own_code.py
```

## Community

Join our [Slack](https://join.slack.com/t/deeplens/shared_invite/zt-2wz3x2n3b-plRqN26eDhO2IY4r_gmjOw) workspace and WeChat Group (singeryang1999) to connect with our core contributors, receive the latest industry updates, and be part of our community. For any inquiries, contact Xinge Yang (xinge.yang@kaust.edu.sa).

## Contribution

We welcome all contributions. To get started, please read our [Contributing Guide](./CONTRIBUTING.md) or check out [open questions](https://github.com/users/singer-yang/projects/2). All project participants are expected to adhere to our [Code of Conduct](./CODE_OF_CONDUCT.md). A list of contributors can be viewed in [Contributors](./CONTRIBUTORS.md) and below:

<a href="https://github.com/singer-yang/DeepLens/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=singer-yang/DeepLens" />
</a>

## Citation

If you use DeepLens in your research, please cite the paper. See more in [History of DeepLens](./CITATION.md).

```bibtex
@article{yang2024curriculum,
  title={Curriculum learning for ab initio deep learned refractive optics},
  author={Yang, Xinge and Fu, Qiang and Heidrich, Wolfgang},
  journal={Nature communications},
  volume={15},
  number={1},
  pages={6572},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
