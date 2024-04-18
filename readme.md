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

在[這裡](https://drive.google.com/drive/folders/1rn3GMOvtJRMBHOxVhWFVSY6IVI6xUnYp)下載模型，放到 checkpoints 資料夾

```
timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt
```

開始預測

```
python inference_timesformer.py
```