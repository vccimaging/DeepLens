# Network API Reference

The `deeplens.network` module provides neural network components for two roles:

1. **Surrogate modelling** — MLP/SIREN networks that predict spatially-varying
   PSFs from `(fov, depth, focus_distance)` without running ray tracing.
2. **Image reconstruction** — restoration networks (NAFNet, UNet, Restormer)
   for end-to-end computational imaging pipelines.

---

## Surrogate Networks

Surrogate networks approximate the PSF of a `GeoLens` at orders-of-magnitude
faster speed. The recommended entry point is `PSFNetLens` in
`deeplens.optics`, which wraps a `GeoLens` together with the surrogate and
handles training automatically. The low-level building blocks are documented
here.

### MLP

Simple multi-layer perceptron with normalised outputs.

```python
from deeplens.network import MLP

model = MLP(in_features=3, out_features=64, hidden_features=64, hidden_layers=3)
out = model(x)   # (B, out_features)
```

::: deeplens.network.MLP
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### MLPConv

MLP encoder followed by a convolutional decoder; predicts spatial kernel images.

```python
from deeplens.network import MLPConv

model = MLPConv(in_features=3, ks=64, channels=3, activation="relu")
kernel = model(condition)   # (B, 3, 64, 64)
```

::: deeplens.network.MLPConv
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### Siren

Sinusoidal Representation Network layer for implicit optical field modelling.

```python
from deeplens.network import Siren

layer = Siren(dim_in=2, dim_out=256, w0=30.0, is_first=True)
```

::: deeplens.network.Siren
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### ModulateSiren

Full modulated SIREN network with FiLM conditioning on a latent vector.

```python
from deeplens.network import ModulateSiren

model = ModulateSiren(
    dim_in=2, dim_hidden=256, dim_out=1,
    dim_latent=64, num_layers=5,
    image_width=64, image_height=64, w0_initial=30.0,
)
```

::: deeplens.network.ModulateSiren
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

---

## Reconstruction Networks

Standard image restoration networks used as the decoder in end-to-end
optics–network co-design.

### NAFNet

Nonlinear Activation Free Network. State-of-the-art restoration quality with
no nonlinear activations — fast and memory efficient.

```python
from deeplens.network import NAFNet

model = NAFNet(
    img_channel=3, width=32,
    middle_blk_num=1,
    enc_blk_nums=[1, 1, 1, 28],
    dec_blk_nums=[1, 1, 1, 1],
)
```

::: deeplens.network.NAFNet
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### UNet

Standard encoder-decoder UNet for image restoration.

```python
from deeplens.network import UNet

model = UNet(in_channels=3, out_channels=3, base_channels=64, num_scales=4)
restored = model(degraded_image)
```

::: deeplens.network.UNet
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### Restormer

Transformer-based restoration with multi-scale attention. Best for large
spatially-varying degradations.

```python
from deeplens.network import Restormer

model = Restormer(
    inp_channels=3, out_channels=3, dim=48,
    num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8],
    ffn_expansion_factor=2.66, bias=False,
)
```

::: deeplens.network.Restormer
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

---

## Loss Functions

### PerceptualLoss

VGG-based perceptual loss for better visual quality than pixel-wise metrics.

```python
from deeplens.network import PerceptualLoss

loss_fn = PerceptualLoss(
    model='vgg19',
    layers=['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4'],
    weights=[1.0, 1.0, 1.0, 1.0],
    device='cuda'
)
loss = loss_fn(pred, target)
```

::: deeplens.network.PerceptualLoss
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### PSNRLoss

::: deeplens.network.PSNRLoss
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true

### SSIMLoss

```python
from deeplens.network import SSIMLoss
loss_fn = SSIMLoss(window_size=11, size_average=True)
```

::: deeplens.network.SSIMLoss
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true

**Combined loss example:**

```python
def combined_loss(pred, target):
    mse  = torch.nn.functional.mse_loss(pred, target)
    ssim = SSIMLoss()(pred, target)
    perc = PerceptualLoss()(pred, target)
    return mse + 0.5 * (1 - ssim) + 0.1 * perc
```

---

## Datasets

### ImageDataset

::: deeplens.network.ImageDataset
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### PhotographicDataset

::: deeplens.network.PhotographicDataset
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

---

## End-to-End Training

### Joint Lens + Network Optimization

```python
import torch
from deeplens import GeoLens
from deeplens.network import UNet, SSIMLoss

lens    = GeoLens(filename='initial_design.json', device='cuda')
network = UNet(in_channels=3, out_channels=3).cuda()

lens_params  = lens.get_optimizer_params(lrs=[1e-4, 1e-4, 1e-2, 1e-4])
opt_lens     = torch.optim.Adam(lens_params)
opt_net      = torch.optim.Adam(network.parameters(), lr=1e-4)
ssim_loss    = SSIMLoss()

for img_clean in dataloader:
    img_degraded = lens.render(img_clean, depth=-10000.0, method='ray_tracing', spp=32)
    img_restored = network(img_degraded)

    loss  = torch.nn.functional.mse_loss(img_restored, img_clean)
    loss += 0.5 * (1 - ssim_loss(img_restored, img_clean))
    loss_reg, _ = lens.loss_reg()
    loss += 0.05 * loss_reg

    opt_lens.zero_grad(); opt_net.zero_grad()
    loss.backward()
    opt_lens.step(); opt_net.step()
```

### Task-Specific Optimization

Optimize the lens directly for a downstream vision metric (e.g. classification):

```python
import torchvision.models as models

classifier = models.resnet18(weights='IMAGENET1K_V1').cuda().eval()
opt = torch.optim.Adam(lens.get_optimizer_params(lrs=[1e-4, 1e-4, 0, 0]))

for img, label in dataloader:
    img_rendered = lens.render(img.cuda(), depth=-10000.0, spp=32)
    loss = torch.nn.functional.cross_entropy(classifier(img_rendered), label.cuda())
    loss_reg, _ = lens.loss_reg()
    (loss + 0.01 * loss_reg).backward()
    opt.step(); opt.zero_grad()
```

### Training Utilities

```python
# Learning rate scheduling
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    loss = criterion(model(data), target)
scaler.scale(loss).backward()
scaler.step(optimizer); scaler.update()

# Checkpointing
torch.save({'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict()}, 'ckpt.pth')
```

---

## Best Practices

- **Start simple**: begin with a smaller model (NAFNet width=16) and scale up
- **Learning rates**: lens LR (1e-3 – 1e-4) should be lower than network LR (1e-4 – 1e-5)
- **SPP**: use 32 (`SPP_RENDER`) during training, 64+ for evaluation
- **Mixed precision**: AMP gives ~2× speedup with negligible quality loss
- **Alternating optimisation**: update network 5× per lens update for stability
- **Physical constraints**: always include `lens.loss_reg()` to prevent unphysical designs
