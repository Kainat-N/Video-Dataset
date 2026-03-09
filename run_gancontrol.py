
# import os
# import torch
# from PIL import Image
# import numpy as np
# import sys

# sys.path.append("src")
# from src.gan_control.inference.controller import Controller
# from src.gan_control.utils.spherical_harmonics_utils import sh_eval_basis_1

# # ---------------- Setup ----------------
# controller_path = 'resources/gan_models/controller_age015id025exp02hai04ori02gam15'
# controller = Controller(controller_path)

# batch_size = 1
# num_images = 50
# truncation = 0.7
# resize = 480
# output_root = "../outputs/gancontroloutputs"

# os.makedirs(output_root, exist_ok=True)

# # ---------------- Loop to generate images ----------------
# for i in range(1, num_images + 1):
#     folder_name = f"gancontrol_{i:03d}"
#     folder_path = os.path.join(output_root, folder_name)
#     os.makedirs(folder_path, exist_ok=True)

#     # -------- Generate original image --------
#     initial_image_tensors, initial_latent_z, initial_latent_w = controller.gen_batch(
#         batch_size=batch_size, truncation=truncation
#     )
#     img = controller.make_resized_grid_image(initial_image_tensors, resize=resize, nrow=1)
#     img.save(os.path.join(folder_path, f"{folder_name}_original.png"))

#     # -------- Right Pose --------
#     right_pose_control = torch.tensor([[-30., -10., 0.]])
#     right_image_tensors, _, _ = controller.gen_batch_by_controls(
#         latent=initial_latent_w, input_is_latent=True, orientation=right_pose_control
#     )
#     right_pose_img = controller.make_resized_grid_image(right_image_tensors, resize=resize, nrow=1)
#     right_pose_img.save(os.path.join(folder_path, f"{folder_name}_rightpose.png"))

#     # -------- Left Pose --------
#     left_pose_control = torch.tensor([[30., -10., 0.]])
#     left_image_tensors, _, _ = controller.gen_batch_by_controls(
#         latent=initial_latent_w, input_is_latent=True, orientation=left_pose_control
#     )
#     left_pose_img = controller.make_resized_grid_image(left_image_tensors, resize=resize, nrow=1)
#     left_pose_img.save(os.path.join(folder_path, f"{folder_name}_leftpose.png"))

#     # -------- Right Illumination --------
#     strength = 1.
#     right_illum_control = torch.tensor([sh_eval_basis_1(1, 0, 0)]) * strength
#     right_illum_tensors, _, _ = controller.gen_batch_by_controls(
#         latent=initial_latent_w, input_is_latent=True, gamma=right_illum_control
#     )
#     right_illum_img = controller.make_resized_grid_image(right_illum_tensors, resize=resize, nrow=1)
#     right_illum_img.save(os.path.join(folder_path, f"{folder_name}_rightillum.png"))

#     # -------- Left Illumination --------
#     left_illum_control = torch.tensor([sh_eval_basis_1(-1, 0, 0)]) * strength
#     left_illum_tensors, _, _ = controller.gen_batch_by_controls(
#         latent=initial_latent_w, input_is_latent=True, gamma=left_illum_control
#     )
#     left_illum_img = controller.make_resized_grid_image(left_illum_tensors, resize=resize, nrow=1)
#     left_illum_img.save(os.path.join(folder_path, f"{folder_name}_leftillum.png"))

#     print(f"Saved images in {folder_path}")


















import os
import torch
from PIL import Image
import numpy as np
import sys
import shutil
import clip
from tqdm import tqdm
import time

sys.path.append("src")
from src.gan_control.inference.controller import Controller
from src.gan_control.utils.spherical_harmonics_utils import sh_eval_basis_1

# ---------------- Setup ----------------
controller_path = 'resources/gan_models/controller_age015id025exp02hai04ori02gam15'
controller = Controller(controller_path)

batch_size = 1
num_images = 2000
truncation = 0.7
resize = 480
output_root = "../outputs/Final_dataset"
temp_folder = "../outputs/temp"

os.makedirs(output_root, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)

# ---------------- CLIP Setup ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
prompt = "smiling man"
threshold = 0.24

def check_clip_match(img_path, prompt, threshold=0.22):
    """Return True if image matches prompt according to CLIP."""
    image = clip_preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    text_tokens = clip.tokenize([prompt]).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).item()
    
    return similarity >= threshold, similarity

# ---------------- Loop to generate images ----------------
start_time = time.time()
for i in range(1, num_images + 1):
    # -------- Generate original image --------
    initial_image_tensors, initial_latent_z, initial_latent_w = controller.gen_batch(
        batch_size=batch_size, truncation=truncation
    )
    original_img = controller.make_resized_grid_image(initial_image_tensors, resize=resize, nrow=1)
    
    temp_img_path = os.path.join(temp_folder, f"temp_{i:03d}.png")
    original_img.save(temp_img_path)

    # -------- CLIP check --------
    match, score = check_clip_match(temp_img_path, prompt, threshold)
    if not match:
        print(f"[INFO] Image {i:03d} skipped by CLIP (score={score:.3f})")
        continue  # skip this image if it doesn't match prompt

    # Only now create final folder
    folder_name = f"gancontrol_{i:03d}"
    folder_path = os.path.join(output_root, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Save original image in final folder
    original_img.save(os.path.join(folder_path, f"{folder_name}_original.png"))

    # -------- Right Pose --------
    right_pose_control = torch.tensor([[-30., -10., 0.]])
    right_image_tensors, _, _ = controller.gen_batch_by_controls(
        latent=initial_latent_w, input_is_latent=True, orientation=right_pose_control
    )
    right_pose_img = controller.make_resized_grid_image(right_image_tensors, resize=resize, nrow=1)
    right_pose_img.save(os.path.join(folder_path, f"{folder_name}_rightpose.png"))

    # -------- Left Pose --------
    left_pose_control = torch.tensor([[30., -10., 0.]])
    left_image_tensors, _, _ = controller.gen_batch_by_controls(
        latent=initial_latent_w, input_is_latent=True, orientation=left_pose_control
    )
    left_pose_img = controller.make_resized_grid_image(left_image_tensors, resize=resize, nrow=1)
    left_pose_img.save(os.path.join(folder_path, f"{folder_name}_leftpose.png"))

    # -------- Right Illumination --------
    strength = 1.
    right_illum_control = torch.tensor([sh_eval_basis_1(1, 0, 0)]) * strength
    right_illum_tensors, _, _ = controller.gen_batch_by_controls(
        latent=initial_latent_w, input_is_latent=True, gamma=right_illum_control
    )
    right_illum_img = controller.make_resized_grid_image(right_illum_tensors, resize=resize, nrow=1)
    right_illum_img.save(os.path.join(folder_path, f"{folder_name}_rightillum.png"))

    # -------- Left Illumination --------
    left_illum_control = torch.tensor([sh_eval_basis_1(-1, 0, 0)]) * strength
    left_illum_tensors, _, _ = controller.gen_batch_by_controls(
        latent=initial_latent_w, input_is_latent=True, gamma=left_illum_control
    )
    left_illum_img = controller.make_resized_grid_image(left_illum_tensors, resize=resize, nrow=1)
    left_illum_img.save(os.path.join(folder_path, f"{folder_name}_leftillum.png"))

    print(f"[INFO] Saved images in {folder_path} (CLIP score={score:.3f})")

end_time = time.time()

print(f"Total time taken by GANControl on GPU for {num_images} : ", end_time - start_time)
