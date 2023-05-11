import cv2
import os
import sys
# sys.path.append("..")
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

img= cv2.imread("/sam/sam/input/sincerely-media-VDPauwJ_sHo-unsplash.jpg")
model_type = "vit_h"


sam = sam_model_registry[model_type](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(img)

print(masks)