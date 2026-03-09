
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import sys
sys.path.append("src")

from gan_control.inference.controller import Controller
from gan_control.projection.lpips.lpips import PerceptualLoss

print("HELLOOOO")
# === IMPORT FUNCTIONS YOU HAVE IN YOUR CODE =========================
from gan_control.projection.projection import (
    load_source_images,
    get_avg_latent,
    noise_regularize,
    noise_normalize_,
    get_lr,
    latent_noise,
)

# -------------------------
# CONFIGURATION
# -------------------------
IMAGE_PATH = "../outputs/selected/00002.png"  # Your input image
CONTROLLER_PATH = "resources/gan_models/controller_age015id025exp02hai04ori02gam15"

OUTPUT_LATENT = "projection_w_gc.npy"
OUTPUT_IMAGE = "projection_result.png"

DEVICE = "cuda"
STEPS = 500
INITIAL_LR = 0.1
RES = 256  # GAN-Control resolution


# =========================================================
# Load Controller + Models
# =========================================================
controller = Controller(CONTROLLER_PATH)
g_model = controller.model

target = load_source_images([IMAGE_PATH], res=RES).to(DEVICE)

# =========================================================
# Initialize latent in W-space   (1,18,512)
# =========================================================
with torch.no_grad():
    w_avg = g_model.module.mean_latent(4096)

# latent = w_avg.unsqueeze(0).clone().detach().to(DEVICE)
latent = w_avg.unsqueeze(0).unsqueeze(1).repeat(1, 14, 1).to(DEVICE)
print(latent.shape)
latent.requires_grad = True   # Optimize this


optimizer = torch.optim.Adam([latent], lr=INITIAL_LR)

# Loss networks
loss_lpips = PerceptualLoss().to(DEVICE)
loss_mse = torch.nn.MSELoss()


# =========================================================
# Projection Loop
# =========================================================
print("\n[INFO] Starting Projection...\n")

for step in range(STEPS):

    # Learning rate schedule
    t = step / STEPS
    lr = get_lr(t, INITIAL_LR)
    optimizer.param_groups[0]["lr"] = lr

    # Add noise to latent
    latent_n = latent_noise(latent, 0.05 * (1 - t))

    # Generate image
    gen_img, _, _ = controller.gen_batch(
        latent=latent_n,
        input_is_latent=True,
        normalize=False,
        static_noise=False
    )

    # Losses
    lp = loss_lpips(gen_img, target)
    lm = loss_mse(gen_img, target)
    loss = lp + 0.1 * lm
    
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



    if step % 50 == 0 or step == STEPS - 1:
        print(f"[{step:04d}/{STEPS}]  LPIPS={lp.item():.4f} | MSE={lm.item():.4f}")

# =========================================================
# Save Latent
# =========================================================
final_latent = latent.detach().cpu().numpy()
np.save(OUTPUT_LATENT, final_latent)
print(f"\n[INFO] Saved latent → {OUTPUT_LATENT}")


# =========================================================
# Save final reconstructed image
# =========================================================
with torch.no_grad():
    final_img, _, _ = controller.gen_batch(
        latent=latent,
        input_is_latent=True,
        normalize=False,
        static_noise=False
    )

save_image(final_img.clamp(-1, 1)*0.5 + 0.5, OUTPUT_IMAGE)

print(f"[INFO] Saved reconstructed image → {OUTPUT_IMAGE}\n")



