import torch
import pytorch_lightning as pl
import numpy as np
from timesformer_pytorch import TimeSformer

checkpoint_path = 'checkpoints/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt'

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

if __name__ == "__main__":
  model=RegressionPLModel.load_from_checkpoint(checkpoint_path, map_location=torch.device('cpu'), strict=False)
