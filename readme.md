# Introduction

Try to learn [Vesuvius Grandprize Winner](https://github.com/younader/Vesuvius-Grandprize-Winner) step by step.

先載套件

```
pip install -r requirements.txt
```

載 segments 資料，需輸入使用者名稱和密碼，記得先下載 rclone

```
./download.sh
```

在 [這裡](https://drive.google.com/drive/folders/1rn3GMOvtJRMBHOxVhWFVSY6IVI6xUnYp) 下載模型，放到 checkpoints 資料夾

```
timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt
```

開始預測

```
python inference_timesformer.py
```

# Details

## read_image_mask

回傳 fragment_mask 和 images，其中 fragment_mask 大小為 (h, w) 是 mask 資料，images 大小為 (h, w, l) 是 layers 資料夾下產生的 stack，數值被裁切到介於 0~200 之間，l 是以中央為基準取出的層數。此外，這兩種資料的 w, h 都被 padding 到 256 的整數倍以方便批量載入數據



