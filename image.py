import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Optional

"""
TODO:

start from larger image, take smaller and smaller slices (x + i*offset:x + width - i*offset)

"""

def cs2xywh(center_x: int, center_y: int, size: int) -> Tuple[int, int, int, int]:
  """
  converts center + size (x, y, size) to (left, top, width, height)
  """
  half = size // 2
  return (center_x - half, center_y - half, size, size)

def get_img_slice_padded(img: np.ndarray, x: int, y: int, slice_width: int, slice_height: int) -> np.ndarray:
  """
  returns a padded slice of an image
  """
  height, width, channels = img.shape
  x1 = max(x, 0)
  y1 = max(y, 0)
  x2 = min(x + slice_width, width)
  y2 = min(y + slice_height, height)

  target_x = x1 - x
  target_y = y1 - y
  slice_img = np.zeros((slice_height, slice_width, channels))
  slice_img[target_y:target_y + slice_height, target_x:target_x + slice_width] = img[y1:y2, x1:x2]

  return slice_img

def get_heatmap(img: np.ndarray, kernel_size: int = 5, guassian_strength: float = 0):
  if guassian_strength > 0:
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), guassian_strength)

  heatmap = cv2.resize(img, (kernel_size, kernel_size))
  return heatmap

def get_lod_heatmaps(img: np.ndarray, x: int, y: int, kernel_size: int = 5, lod: int = 4, scale_factor: float = 0, guassian_strength: float = 0):
  """
  this version grows based on param
  """
  if scale_factor == 0:
    scale_factor = kernel_size

  heatmaps = []
  slice_images = []

  for i in range(1, lod + 1):
    chunk_size = int(kernel_size * (scale_factor ** i))
    # if chunk_size % 2 == 0:
    #   chunk_size += 1

    img_slice = get_img_slice_padded(img, *cs2xywh(x, y, chunk_size))
    slice_images.append(img_slice)

    heatmap = get_heatmap(img_slice, kernel_size, guassian_strength)
    heatmaps.append(heatmap)

  return heatmaps, slice_images

def get_log_image(img: np.ndarray, x: int, y: int, kernel_size: int = 5, lod: int = 4, scale_factor: float = 2):
  """
  this version grows based on param and superimposes the resized images
  """
  resized_images = []

  for i in range(1, lod + 1):
    chunk_size = int(kernel_size * (scale_factor ** i))
    if chunk_size % 2 == 0:
      chunk_size += 1

    resized_img, sliced_img = get_lod_heatmaps(img, x, y, chunk_size, guassian_strength)
    resized_images.append(resized_img)

    kernel_size += 4

  total = resized_images[-1]
  for i in range(lod - 2, 0, -1):
    offset = 2 * (lod - i - 1)
    total[offset:-offset, offset:-offset] = resized_images[i][:, :]

  return total

def process_and_show_image(image_path):
  img = plt.imread(image_path)
  img = img.astype(np.float32)

  lod = 4
  kernel_size = 9
  scale_factor = 3
  guassian_strength = 1
  # x, y = img.shape[1] // 2, img.shape[0] // 2
  x, y = 250, 500

  # kernel_size = 128
  # padded_slice = get_img_slice_padded(img, *cs2xywh(x, y, kernel_size))
  # heatmap = get_heatmap(padded_slice, kernel_size = 11, guassian_strength = 15)
  # # resized_img = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

  # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
  # axs[0].imshow(padded_slice)  # show original
  # axs[0].set_title("Original Image")
  # axs[0].axis('off')
  # axs[1].imshow(heatmap)
  # axs[1].set_title("LOD Composite")
  # axs[1].axis('off')
  # plt.show()

  # return

  resized_images, slice_images = get_lod_heatmaps(img, x, y, kernel_size, lod, scale_factor, guassian_strength)

  # Create a grid to show all LOD images
  fig, axes = plt.subplots(2, len(resized_images), figsize=(15, 3))
  if len(resized_images) == 1:
    axes = [axes]
  
  for i, lod_img in enumerate(resized_images):
    axes[0, i].imshow(slice_images[i])
    axes[1, i].imshow(lod_img)
    axes[0, i].axis('off')
    axes[1, i].axis('off')
    axes[0, i].set_title(f'Slice {i+1}')
    axes[1, i].set_title(f'Resized {i+1}')
  
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  process_and_show_image("img2.png")
