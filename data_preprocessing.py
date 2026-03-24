import numpy as np
from PIL import Image
import cv2

def preprocessing_image_and_mask(image_path, mask_path):
    #original image
    raw_image = np.array(Image.open(image_path).convert("RGB"))
    raw_mask = np.array(Image.open(mask_path))

    #Resize
    img_res = cv2.resize(raw_image, (256, 256))
    msk_res = cv2.resize(raw_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

    #Remapping
    msk_binary = (msk_res == 1).astype(np.uint8)

    return img_res, msk_binary