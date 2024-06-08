import cv2
import torch
import numpy as np
import pytorch_lightning as pl
from timesformer_pytorch import TimeSformer

in_chans = 26
segment_path = 'train_scrolls'
fragment_id = '20230509182749'
checkpoint_path = 'checkpoints/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt'

def read_image_mask(fragment_id, start_idx=18, end_idx=38, rotation=0):
  images = []
  mid = 65 // 2
  start = mid - in_chans // 2
  end = mid + in_chans // 2

  for i in range(start_idx, end_idx):
    image = cv2.imread(f"{segment_path}/{fragment_id}/layers/{i:02}.tif", 0)
    pad0 = (256 - image.shape[0] % 256)
    pad1 = (256 - image.shape[1] % 256)
    image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
    image = np.clip(image, 0, 200)
    images.append(image)

  images = np.stack(images, axis=2)
  fragment_mask = cv2.imread(f"{segment_path}/{fragment_id}/{fragment_id}_mask.png", 0)
  fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)

  return images, fragment_mask

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

  start_f = 0
  end_f = start_f + in_chans

  images, fragment_mask = read_image_mask(fragment_id, start_f, end_f)
  print(images.shape, fragment_mask.shape, np.max(images[:, :, 0]))


