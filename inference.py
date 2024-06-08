import cv2
import torch
import numpy as np
import scipy.stats as st
from tqdm.auto import tqdm
import pytorch_lightning as pl
from timesformer_pytorch import TimeSformer
from torch.utils.data import Dataset, DataLoader

segment_path = 'train_scrolls'
fragment_id = '20230509182749'
checkpoint_path = 'checkpoints/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt'

in_chans = 26
tile_size = 64
stride = tile_size // 3
batch_size = 256
num_workers = 4
size = 64

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()

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

  coords = np.stack(coords)
  # test_dataset = CustomDatasetTest(images, coords)
  test_dataset = CustomDatasetTest(images[:1000], coords[:1000])
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
  image_shape = (image_stack.shape[0], image_stack.shape[1])

  return test_loader, coords, image_shape, fragment_mask

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

  test_loader, coords, image_shape, fragment_mask = get_img_splits(fragment_id, start_idx, end_idx)

  kernel = gkern(size, 1)
  kernel = kernel / kernel.max()

  for step, (image, coord) in tqdm(enumerate(test_loader), total=len(test_loader)):
    print('Batch image shape: ', image.shape)
    print('Batch coord shape: ', coord.shape)


