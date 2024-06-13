import cv2
import torch
import numpy as np
import scipy.stats as st
from tqdm.auto import tqdm
import pytorch_lightning as pl
from timesformer_pytorch import TimeSformer
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

segment_path = 'train_scrolls'
# fragment_id = '20230509182749'
fragment_id = 'pi_small'
checkpoint_path = 'checkpoints/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt'

in_chans = 26
tile_size = 64
stride = tile_size // 3
batch_size = 8
# batch_size = 256
num_workers = 4
size = 64

device_type = 'cpu'
device = torch.device(device_type)

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

  transform = A.Compose([
    A.Resize(size, size),
    A.Normalize(mean= [0] * in_chans, std= [1] * in_chans),
    ToTensorV2(transpose_mask=True),
  ])

  coords = np.stack(coords)
  dataset = CustomDatasetTest(images, coords, transform=transform)
  # dataset = CustomDatasetTest(images[:1000], coords[:1000], transform=transform)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
  image_shape = (image_stack.shape[0], image_stack.shape[1])

  return loader, coords, image_shape, fragment_mask

class CustomDatasetTest(Dataset):
    def __init__(self, images, coords, transform=None):
        self.images = images
        self.coords = coords
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        coord = self.coords[idx]

        if self.transform:
          data = self.transform(image=image)
          image = data['image'].unsqueeze(0) 

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

  def forward(self, x):
    print('Batch image shape: ', x.shape)
    if x.ndim==4: x=x[:,None]
    print('TimeSformer input shape: ', torch.permute(x, (0, 2, 1, 3, 4)).shape)
    x = self.backbone(torch.permute(x, (0, 2, 1, 3, 4)))
    print('TimeSformer output shape: ', x.shape)
    x = x.view(-1, 1, 4, 4)   
    print('Reshape shape: ', x.shape)     
    return x

if __name__ == "__main__":
  model = RegressionPLModel.load_from_checkpoint(checkpoint_path, map_location=device, strict=False)
  model.eval()

  if (device_type == 'cuda'): model.cuda()

  start_idx = 17
  end_idx = start_idx + in_chans

  loader, coords, image_shape, fragment_mask = get_img_splits(fragment_id, start_idx, end_idx)

  kernel = gkern(size, 1)
  kernel = kernel / kernel.max()

  for step, (images, coords) in tqdm(enumerate(loader), total=len(loader)):
    images = images.to(device)
    batch_size = images.size(0)

    with torch.no_grad():
      with torch.autocast(device_type=device_type):
        y_preds = model(images)
    y_preds = torch.sigmoid(y_preds).to('cpu')
