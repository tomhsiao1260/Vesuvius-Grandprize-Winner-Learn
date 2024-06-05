import torch
import pytorch_lightning as pl
import numpy as np

class RegressionPLModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        print("model init")

if __name__ == "__main__":
    model=RegressionPLModel()
