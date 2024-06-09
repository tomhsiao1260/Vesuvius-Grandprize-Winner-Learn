import os
import torch.nn as nn
import torch.nn.functional as F
from timesformer_pytorch import TimeSformer
import torch
import random
import gc
import pytorch_lightning as pl
import scipy.stats as st
from torch.utils.data import DataLoader
import numpy as np
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import PIL.Image
from tap import Tap
import glob
import gc

PIL.Image.MAX_IMAGE_PIXELS = 933120000
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cpu'
device = torch.device(device_type)

class InferenceArgumentParser(Tap):
    segment_id: list[str] =['20230509182749']
    segment_path:str='./train_scrolls'
    model_path:str= './checkpoints/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt'
    out_path:str=""
    stride: int = 2
    start_idx:int=15
    workers: int = 4
    batch_size: int = 512
    size:int=64
    reverse:int=0
    device:str='cuda'
    format='tif'
args = InferenceArgumentParser().parse_args()
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    # comp_dir_path = './'
    comp_dir_path = './'
    comp_folder_name = './'
    comp_dataset_path = f'./'
    
    exp_name = 'pretraining_all'
    # ============== model cfg =============
    in_chans = 26 # 65
    encoder_depth=5
    # ============== training cfg =============
    size = 64
    tile_size = 64
    stride = tile_size // 3

    train_batch_size = 256 # 32
    valid_batch_size = 1
    # valid_batch_size = 256
    use_amp = True

    epochs = 50 # 30

    # adamW warmupあり
    warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    lr = 1e-4 / warmup_factor
    min_lr = 1e-6
    num_workers = 4
    # num_workers = 16
    seed = 42
    # ============== augmentation =============
    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]

def read_image_mask(fragment_id, start_idx=18, end_idx=38, rotation=0):
    images = []
    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    for i in range(start_idx, end_idx):
        image = cv2.imread(f"{args.segment_path}/{fragment_id}/layers/{i:02}.{args.format}", 0)
        pad0 = (256 - image.shape[0] % 256)
        pad1 = (256 - image.shape[1] % 256)
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        image = np.clip(image, 0, 200)
        images.append(image)
    images = np.stack(images, axis=2)

    fragment_mask = cv2.imread(CFG.comp_dataset_path + f"{args.segment_path}/{fragment_id}/{fragment_id}_mask.png", 0)
    fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)

    return images, fragment_mask

def get_img_splits(fragment_id, s, e, rotation=0):
    images = []
    xyxys = []
    image, fragment_mask = read_image_mask(fragment_id, s, e, rotation)
    x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
    y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size
            if not np.any(fragment_mask[y1:y2, x1:x2]==0):
                images.append(image[y1:y2, x1:x2])
                xyxys.append([x1, y1, x2, y2])

    test_dataset = CustomDatasetTest(images[:1],np.stack(xyxys)[:1], CFG,transform=A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(
            mean= [0] * CFG.in_chans,
            std= [1] * CFG.in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]))

    test_loader = DataLoader(test_dataset,
                              batch_size=CFG.valid_batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False,
                              )
    return test_loader, np.stack(xyxys), (image.shape[0], image.shape[1]), fragment_mask

def get_transforms(data, cfg):
    if data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug

class CustomDatasetTest(Dataset):
    def __init__(self, images,xyxys, cfg, transform=None):
        self.images = images
        self.xyxys=xyxys
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        xy=self.xyxys[idx]
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)
        return image,xy
    
class RegressionPLModel(pl.LightningModule):
    def __init__(self,pred_shape,size=64,enc='',with_norm=False):
        super(RegressionPLModel, self).__init__()
        self.save_hyperparameters()
        self.backbone=TimeSformer(
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
        if x.ndim==4: x=x[:, None]
        x = self.backbone(torch.permute(x, (0, 2, 1, 3, 4)))
        x=x.view(-1,1,4,4)        
        return x

def predict_fn(test_loader, model, device, test_xyxys,pred_shape):
    mask_pred = np.zeros(pred_shape)
    mask_count = np.zeros(pred_shape)
    kernel = gkern(CFG.size,1)
    kernel = kernel / kernel.max()
    model.eval()

    for step, (images,xys) in tqdm(enumerate(test_loader),total=len(test_loader)):
        images = images.to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            with torch.autocast(device_type=device_type):
                y_preds = model(images)
        y_preds = torch.sigmoid(y_preds).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xys):
            mask_pred[y1:y2, x1:x2] += np.multiply(F.interpolate(y_preds[i].unsqueeze(0).float(), scale_factor=16,mode='bilinear').squeeze(0).squeeze(0).numpy(), kernel)
            mask_count[y1:y2, x1:x2] += np.ones((CFG.size, CFG.size))

    mask_pred /= mask_count
    return mask_pred

if __name__ == "__main__":
    model=RegressionPLModel.load_from_checkpoint(args.model_path, map_location=device, strict=False)
    # model.cuda()
    model.eval()

    preds=[]
    rotation = 0
    fragment_id = '20230509182749'
    start_f = 0
    end_f = start_f + 26
    test_loader,test_xyxz,test_shape,fragment_mask = get_img_splits(fragment_id,start_f,end_f)

    mask_pred = predict_fn(test_loader, model, device, test_xyxz, test_shape)
    mask_pred = np.clip(np.nan_to_num(mask_pred),a_min=0,a_max=1)
    mask_pred /= mask_pred.max()

    preds.append(mask_pred)

    gc.collect()

    # CV2 image
    image_cv = (mask_pred * 255).astype(np.uint8)
    cv2.imwrite(f"{fragment_id}_prediction.png", image_cv)

    del mask_pred, test_loader, model
    torch.cuda.empty_cache()
    gc.collect()
