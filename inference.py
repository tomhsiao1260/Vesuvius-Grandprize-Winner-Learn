import cv2
import torch
import numpy as np
import pytorch_lightning as pl
from timesformer_pytorch import TimeSformer
from torch.utils.data import Dataset

segment_path = 'train_scrolls'
fragment_id = '20230509182749'
checkpoint_path = 'checkpoints/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt'

in_chans = 26
tile_size = 64
stride = tile_size // 3

def read_image_mask(fragment_id, start_idx=18, end_idx=38, rotation=0):
  image_stack = []
  mid = 65 // 2
  start = mid - in_chans // 2
  end = mid + in_chans // 2

  for i in range(start_idx, end_idx):
    image = cv2.imread(f"{segment_path}/{fragment_id}/layers/{i:02}.tif", 0)
    pad0 = (256 - image.shape[0] % 256)
    pad1 = (256 - image.shape[1] % 256)
    image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
    image = np.clip(image, 0, 200)
    image_stack.append(image)

  image_stack = np.stack(image_stack, axis=2)
  fragment_mask = cv2.imread(f"{segment_path}/{fragment_id}/{fragment_id}_mask.png", 0)
  fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)

  return image_stack, fragment_mask

def get_img_splits(fragment_id, start_idx, end_idx, rotation=0):
  image_stack, fragment_mask = read_image_mask(fragment_id, start_idx, end_idx)

  images = []
  coords = []
  x_list = list(range(0, image_stack.shape[1]-tile_size+1, stride))
  y_list = list(range(0, image_stack.shape[0]-tile_size+1, stride))

  for ymin in y_list:
    for xmin in x_list:
      ymax = ymin + tile_size
      xmax = xmin + tile_size
      if not np.any(fragment_mask[ymin:ymax, xmin:xmax]==0):
        images.append(image_stack[ymin:ymax, xmin:xmax])
        coords.append([xmin, ymin, xmax, ymax])

  test_dataset = CustomDatasetTest(images, np.stack(coords))

  print('Dataset length:', len(test_dataset))
  print('1st data coord:', test_dataset[0][1])
  print('1st data shape:', test_dataset[0][0].shape)

class CustomDatasetTest(Dataset):
    def __init__(self, images, coords):
        self.images = images
        self.coords = coords

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        coord = self.coords[idx]
        return image, coord

class RegressionPLModel(pl.LightningModule):
  def __init__(self, pred_shape, size=64, enc='', with_norm=False):
    super(RegressionPLModel, self).__init__()
    self.save_hyperparameters()
    self.backbone = TimeSformer(
      dim = 512,
      image_size = 64,
      patch_size = 16,
      num_frames = 30,
      num_classes = 16,
      channels=1,
      depth = 8,
      heads = 6,
      dim_head =  64,
      attn_dropout = 0.1,
      ff_dropout = 0.1
    )

if __name__ == "__main__":
  model = RegressionPLModel.load_from_checkpoint(checkpoint_path, map_location=torch.device('cpu'), strict=False)
  model.eval()

  start_idx = 0
  end_idx = start_idx + in_chans

  get_img_splits(fragment_id, start_idx, end_idx)


