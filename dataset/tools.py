import cv2
import torch
import numpy as np

def get_the_cv2_image(tensor):
    img_raw = torch.squeeze(tensor).data.cpu().numpy().transpose(1, 2, 0)
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
    return img_raw

def get_the_mask_contours(mask_img):
    mask_i = torch.squeeze(mask_img).data.cpu().numpy()
    mask_i[mask_i == 1] = 5
    mask_i[mask_i == 0] = 255
    ret, mask_b = cv2.threshold(mask_i, 127, 255, 0)
    img_temp, contours, hierarchy = cv2.findContours(mask_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_the_segment_contours(segment):
    segment_i = torch.squeeze(segment).data.cpu().numpy().astype(np.uint8)
    segment_i[segment_i == 1] = 255
    ret, segment_b = cv2.threshold(segment_i, 127, 255, 0)
    img_temp, contours, hierarchy = cv2.findContours(segment_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours