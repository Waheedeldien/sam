import cv2
import os
import sys
sys.path.append("..")
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor







sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(input/sincerely-media-VDPauwJ_sHo-unsplash.jpg)