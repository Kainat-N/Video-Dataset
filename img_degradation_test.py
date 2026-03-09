import cv2
import numpy as np
import random

# Load image
img = cv2.imread("00068.png")

if img is None:
    raise ValueError("Image not found")

# 1. Add Gaussian noise
def gaussian_noise(image):
    noise = np.random.normal(0, 20, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# 2. Motion blur
def motion_blur(image):
    size = random.randint(5, 15)
    kernel = np.zeros((size, size))
    kernel[size//2, :] = 1
    kernel = kernel / size
    return cv2.filter2D(image, -1, kernel)

# 3. JPEG compression
def jpeg_artifacts(image):
    quality = random.randint(25, 50)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, 1)

# 4. Resolution degradation
def downscale(image):
    h, w = image.shape[:2]
    small = cv2.resize(image, (w//3, h//3))
    return cv2.resize(small, (w, h))

# 5. Random occlusion
def occlusion(image):
    h, w = image.shape[:2]
    x = random.randint(0, w-60)
    y = random.randint(0, h-60)
    image[y:y+60, x:x+60] = np.random.randint(0,255,(60,60,3))
    return image

# Apply degradations
img = gaussian_noise(img)
img = motion_blur(img)      #blurs the image a little
img = jpeg_artifacts(img)
img = downscale(img)
# img = occlusion(img)

# Save result
cv2.imwrite("degraded_face_00068.png", img)

print("Degraded image saved as degraded_face.png")