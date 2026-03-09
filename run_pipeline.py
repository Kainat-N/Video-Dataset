# # run_pipeline.py with styleclip eidts although not working i delete styleclip images here 
# import os
# import sys
# import shutil
# from modules.generator import StyleGANGenerator
# from modules.clip_cluster_selector import CLIPSelector
# from modules.styleclip_optimizer import StyleCLIPOptimizer

# # ---------------- CONFIG ----------------
# NUM_IMAGES = 5
# POOL_DIR = "outputs/pool"
# SELECTED_DIR = "outputs/selected"
# CLIP_PROMPT = "smiling woman"
# STYLECLIP_PROMPT = "a person smiling, realistic, high quality"
# THRESHOLD = 0.22
# TRUNCATION_PSI = 0.7
# IMAGE_SIZE = 512
# SEED = 42
# DATASET_PROMPT = "young smiling woman"

# # Optimization hyperparams
# OPT_STEPS = 300
# OPT_LR = 0.05
# OPT_L2 = 0.01
# OPT_ID_LAMBDA = 0.1
# WORK_IN_STYLESPACE = False
# SAVE_INTERMEDIATE_EVERY = 0
# # ----------------------------------------

# def main():
#     # ---------------- LOGGING ----------------
#     log_file = "terminal_output.txt"
#     sys.stdout = open(log_file, "w")
#     sys.stderr = sys.stdout
#     print(f"[INFO] Logging terminal output to {log_file}\n")

#     # ---------------- STEP 1: GENERATE IMAGES ----------------
#     print("\n[STEP 1] Generating images with StyleGAN...")
#     generator = StyleGANGenerator("models/stylegan_ffhq.pkl")
#     generator.generate_images(
#         num_images=NUM_IMAGES,
#         outdir=POOL_DIR,
#         size=IMAGE_SIZE,
#         truncation_psi=TRUNCATION_PSI,
#         seed=SEED,
#         save_latents=True
#     )

#     # ---------------- STEP 2: CLIP SELECTION ----------------
#     print("\n[STEP 2] Selecting images with CLIP...")
#     selector = CLIPSelector()
#     selected = selector.select_top(
#         image_folder=POOL_DIR,
#         prompt=CLIP_PROMPT,
#         threshold=THRESHOLD,
#         outdir=SELECTED_DIR       # Already stores selected PNGs
#     )
#     print(f"[INFO] {len(selected)} images selected.")

#     # ---------------- REMOVE EVERYTHING EXCEPT PNGs ----------------
#     print("\n[INFO] Cleaning selected folder (keeping only PNG files)...")

#     for f in os.listdir(SELECTED_DIR):
#         if not f.lower().endswith(".png"):
#             os.remove(os.path.join(SELECTED_DIR, f))

#     print("[INFO] Selected folder cleaned.")

#     # ---------------- STEP 3: STYLECLIP EDITS ----------------
#     print("\n[STEP 3] Applying StyleCLIP edits...")

#     optimizer = StyleCLIPOptimizer(
#         styleclip_repo_path="StyleCLIP",
#         stylegan_ckpt_path=os.path.join("StyleCLIP", "pretrained_models", "stylegan2-ffhq-config-f.pt"),
#         ir_se50_path=os.path.join("StyleCLIP", "pretrained_models", "model_ir_se50.pth"),
#         stylegan_size=1024
#     )

#     optimizer.edit_folder(
#         selected_root=SELECTED_DIR,
#         prompt=STYLECLIP_PROMPT,
#         steps=OPT_STEPS,
#         lr=OPT_LR,
#         l2_lambda=OPT_L2,
#         id_lambda=OPT_ID_LAMBDA,
#         truncation=TRUNCATION_PSI,
#         work_in_stylespace=WORK_IN_STYLESPACE,
#         save_intermediate_every=SAVE_INTERMEDIATE_EVERY
#     )

#     #-------------Stable Diffusion Generation------------------------
#     import torch, os, random
#     from diffusers import StableDiffusionXLPipeline

#     # Config-----------------

#     base_model = "stabilityai/stable-diffusion-xl-base-1.0"
#     lora_path = "real_humans_sd/real-humans-PublicPrompts.safetensors"  # local path


#     prompt = f"portrait photo of a {DATASET_PROMPT}"
#     file_save = "Smiling_Woman"
#     prompt_2 = "Ultra realistic, 8K, detailed skin texture, grainy, low resolution, full face"
#     negative_prompt = "cartoon, unrealistic, cgi, painting, blurry, doll, makeup, plastic skin, black and white"

#     latest_select_image_id = 0
#     output_dir = f"outputs/selected"
#     os.makedirs(output_dir, exist_ok=True)
#     num_images = 8  # increase to test uniqueness
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("device =", device)
#     # ------------------------------
#     # LOAD PIPELINE
#     # ------------------------------
#     # pipe = StableDiffusionXLPipeline.from_pretrained(
#     #     base_model,
#     #     dtype= torch.float32 if device == "cuda" else torch.float32,
#     #     safety_checker=None,
#     # ).to(device)
#     pipe = StableDiffusionXLPipeline.from_pretrained(base_model).to(device)

#     # Load LoRA
#     pipe.load_lora_weights(
#         ".",  # folder containing the .safetensors
#         weight_name="real-humans-PublicPrompts.safetensors"
#     )

#     # Fuse LoRA for faster inference
#     pipe.fuse_lora(lora_scale=1.3)

#     # ------------------------------
#     # GENERATION
#     # ------------------------------
#     for i in range(num_images):
#         seed = random.randint(0, 999999)      # ← random seed ensures unique identities
#         generator = torch.Generator(device).manual_seed(seed)

#         result = pipe(
#             prompt=prompt,
#             prompt_2=prompt_2,
#             negative_prompt=negative_prompt,
#             negative_prompt_2=negative_prompt,
#             generator=generator,
#             num_inference_steps=30,
#             guidance_scale=4.5,
#         )

#         image = result.images[0]
#         filename = f"{output_dir}/{file_save}_{seed}.png"
#         selected_image_id += 1 
#         image.save(filename)

#         print(f"Saved: {filename}")


#     print("\n[INFO] Pipeline completed successfully!")
#     sys.stdout.close()
#     print(f"[INFO] Terminal output saved to {log_file}")


# if __name__ == "__main__":
#     main()
































# run_pipeline.py
import os
import sys
import shutil
import warnings
import contextlib
import io

# ---------------- HELPER: SILENCE STDOUT ----------------
@contextlib.contextmanager
def suppress_stdout():
    """Temporarily suppress all stdout prints."""
    saved_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = saved_stdout


# ---------------- CONFIG ----------------
NUM_IMAGES = 100
POOL_DIR = "outputs3/pool"
SELECTED_DIR = "outputs3/selected"
CLIP_PROMPT = "smiling man"
THRESHOLD = 0.23
TRUNCATION_PSI = 0.7
IMAGE_SIZE = 512
#SEED = 41
DATASET_PROMPT = "smiling man"
import random
SEED = random.randint(0, 2**32 - 1)


# Variation generation config
NETWORK_PKL = "E:/FYP/ffhq.pkl"
PROJECTOR_SCRIPT = "E:/FYP/stylegan2-ada-pytorch/projector.py"
VARIATION_COUNT = 20
OUTPUT_VARIATIONS_ROOT = "E:/FYP/outputs3/Final_dataset"
TARGET_RESOLUTION = (256, 256)


def main():

    # ---------------- LOGGING ----------------
    log_file = "terminal_output.txt"
    sys.stdout = open(log_file, "w")
    sys.stderr = sys.stdout

    print(f"[INFO] Logging terminal output to {log_file}\n")
    warnings.filterwarnings("ignore")

    from modules.generator import StyleGANGenerator
    from modules.clip_cluster_selector import CLIPSelector
    from modules.styleclip_optimizer import StyleCLIPOptimizer

    # -------------------------------------------------------------------------
    # STEP 1 — STYLEGAN IMAGE GENERATION
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Generating images with StyleGAN...")

    with suppress_stdout():
        generator = StyleGANGenerator("models/stylegan_ffhq.pkl")
        generator.generate_images(
            num_images=NUM_IMAGES,
            outdir=POOL_DIR,
            size=IMAGE_SIZE,
            truncation_psi=TRUNCATION_PSI,
            seed=SEED,
            save_latents=True
        )

    print("[INFO] Finished generating images.")

    # -------------------------------------------------------------------------
    # STEP 2 — CLIP SELECTION
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Selecting images with CLIP...")

    selector = CLIPSelector()
    selected = selector.select_top(
        image_folder=POOL_DIR,
        prompt=CLIP_PROMPT,
        threshold=THRESHOLD,
        outdir=SELECTED_DIR
    )

    print(f"[INFO] {len(selected)} images selected.")

    # Clean non-png files
    for f in os.listdir(SELECTED_DIR):
        if not f.lower().endswith(".png"):
            os.remove(os.path.join(SELECTED_DIR, f))

    # # -------------------------------------------------------------------------
    # # STEP 3 — SDXL IMAGE GENERATION
    # # -------------------------------------------------------------------------
    # print("\n[STEP 3] Generating additional images with Stable Diffusion XL...\n")

    # import torch, random
    # from diffusers import StableDiffusionXLPipeline

    # base_model = "stabilityai/stable-diffusion-xl-base-1.0"

    # prompt = f"portrait photo of a {DATASET_PROMPT}"
    # prompt_2 = "Ultra realistic, 8K, detailed skin texture, grainy, low resolution, full face"
    # negative_prompt = "cartoon, unrealistic, cgi, painting, blurry, doll, makeup, plastic skin, black and white"

    # num_images = 1
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # os.makedirs(SELECTED_DIR, exist_ok=True)

    # # Determine next filename
    # existing = sorted([f for f in os.listdir(SELECTED_DIR) if f.endswith(".png")])
    # max_id = 0
    # for f in existing:
    #     try:
    #         idx = int(os.path.splitext(f)[0])
    #         max_id = max(max_id, idx)
    #     except:
    #         pass

    # next_id = max_id + 1
    # print(f"[INFO] Starting Stable Diffusion images from ID {next_id:05d}")

    # pipe = StableDiffusionXLPipeline.from_pretrained(base_model).to(device)
    # pipe.load_lora_weights("real_human_sd", weight_name="real-humans-PublicPrompts.safetensors")
    # pipe.fuse_lora(lora_scale=1.3)

    # for i in range(num_images):
    #     seed = random.randint(0, 999999)
    #     g = torch.Generator(device).manual_seed(seed)

    #     result = pipe(
    #         prompt=prompt,
    #         prompt_2=prompt_2,
    #         negative_prompt=negative_prompt,
    #         negative_prompt_2=negative_prompt,
    #         generator=g,
    #         num_inference_steps=30,
    #         guidance_scale=4.5,
    #     )

    #     img = result.images[0]
    #     save_path = os.path.join(SELECTED_DIR, f"{next_id:05d}.png")
    #     img.save(save_path)
    #     print(f"Saved: {save_path}")
    #     next_id += 1

    # -------------------------------------------------------------------------
    # STEP 4 — VARIATION GENERATION (YOUR ADDED CODE)
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Generating variations using StyleGAN2 projection...\n")
    import torch, random

    import subprocess
    import numpy as np
    from PIL import Image
    import legacy

    device = torch.device("cuda")

    os.makedirs(OUTPUT_VARIATIONS_ROOT, exist_ok=True)

    print("Loading StyleGAN2 generator for variations...")
    with open(NETWORK_PKL, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)

    # Get all selected images
    selected_imgs = sorted([f for f in os.listdir(SELECTED_DIR) if f.endswith(".png")])

    for img_file in selected_imgs:
        identity = os.path.splitext(img_file)[0]   # "00001"
        img_path = os.path.join(SELECTED_DIR, img_file)

        print(f"\nProcessing identity {identity}...")

        # Output folder: testimgresult/00001/
        id_out = os.path.join(OUTPUT_VARIATIONS_ROOT, identity)
        os.makedirs(id_out, exist_ok=True)

        # Temporary projection path
        proj_dir = os.path.join("projected_tmp", identity)
        os.makedirs(proj_dir, exist_ok=True)

        # Run projector.py
        subprocess.run([
            sys.executable,
            PROJECTOR_SCRIPT,
            "--outdir", proj_dir,
            "--target", img_path,
            "--network", NETWORK_PKL,
            "--save-video", "false",
            "--num-steps", "100"
        ], check=True)

        # Load projected W+
        w_path = os.path.join(proj_dir, "projected_w.npz")
        if not os.path.exists(w_path):
            print(f"[ERR] Projection failed for {identity}")
            continue

        w = np.load(w_path)["w"]
        w = torch.tensor(w).to(device)

        # Generate variations
        for i in range(VARIATION_COUNT):
            noise = torch.randn_like(w) * 0.5
            w_mod = w + noise

            img_tensor = G.synthesis(w_mod, noise_mode="const")[0]
            img_tensor = (img_tensor + 1) * 127.5
            img_tensor = img_tensor.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()

            img_pil = Image.fromarray(img_tensor)
            img_pil = img_pil.resize(TARGET_RESOLUTION, Image.LANCZOS)

            save_path = os.path.join(id_out, f"{identity}_var{i}.png")
            img_pil.save(save_path)

        print(f"[OK] Finished identity {identity}. Variations saved in {id_out}")

    print("\n[INFO] Pipeline completed successfully!")
    sys.stdout.close()


if __name__ == "__main__":
    main()
