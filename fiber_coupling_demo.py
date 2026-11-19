import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Circle

from deeplens import GeoLens

lens = GeoLens(filename="./datasets/lenses/cellphone/cellphone80deg.json")
lens.astype(torch.float64)

result = lens.fiber_coupling(
    points=[0.0, 0.0, -10000.0],
    mode_field_diameter=0.0104,  # 10.4 µm, in mm
    wvln=0.587,  # design wavelength; this lens glass stops at 1.014 µm
    spp=1_000_000,
    ks=128,
)

field = result["field"].detach().cpu()
mode = result["mode"].detach().cpu()
center = slice(field.shape[0] // 2 - 16, field.shape[0] // 2 + 16)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
panels = [
    (field.abs(), "Incident amplitude"),
    (field.angle(), "Incident phase"),
    (mode[center, center].abs(), "Fiber mode, zoomed"),
    (mode[center, center].angle(), "Fiber phase, zoomed"),
]
for ax, (image, title) in zip(axes.ravel(), panels):
    im = ax.imshow(image.numpy(), cmap="viridis")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)

efficiency = result["efficiency"].item()
loss_db = result["loss_db"].item()
summary = f"{efficiency:.2%} coupled, {loss_db:.2f} dB"
fig.suptitle(summary)
fig.tight_layout()
fig.savefig("fiber_coupling.png", dpi=150)
print(f"Coupling efficiency: {efficiency:.2%}")
print(f"Insertion loss: {loss_db:.2f} dB")
print("wrote fiber_coupling.png")

# Geometric multimode coupling: circular 50 µm core with NA = 0.22.
multimode = lens.multimode_fiber_coupling(
    points=[0.0, 0.0, -10000.0],
    core_diameter=0.050,
    numerical_aperture=0.22,
    wvln=0.587,
    num_rays=100_000,
)

positions = multimode["positions"].detach().cpu()
valid = multimode["valid"].detach().cpu()
coupled = multimode["coupled"].detach().cpu()
fiber_center = multimode["fiber_center"].detach().cpu()
relative_positions_um = (positions[:, :2] - fiber_center[:2]) * 1_000
core_radius_um = 25.0

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(
    relative_positions_um[valid, 0],
    relative_positions_um[valid, 1],
    s=0.2,
    c="lightgray",
    label="Valid rays",
)
ax.scatter(
    relative_positions_um[coupled, 0],
    relative_positions_um[coupled, 1],
    s=0.4,
    c="tab:blue",
    label="Coupled rays",
)
ax.add_patch(Circle((0, 0), core_radius_um, fill=False, color="tab:red", lw=2))
ax.set_aspect("equal")
ax.set_xlabel("Facet x relative to core centre [µm]")
ax.set_ylabel("Facet y relative to core centre [µm]")
ax.set_title("Geometric multimode-fiber capture")
ax.legend(markerscale=10)
fig.tight_layout()
fig.savefig("multimode_fiber_capture.png", dpi=150)

print(f"Multimode geometric coupling: {multimode['efficiency'].item():.2%}")
print(f"Multimode insertion loss: {multimode['loss_db'].item():.2f} dB")
print(f"Lens throughput: {multimode['lens_throughput'].item():.2%}")
print(f"Core throughput: {multimode['core_throughput'].item():.2%}")
print(f"NA throughput: {multimode['na_throughput'].item():.2%}")
print("wrote multimode_fiber_capture.png")
