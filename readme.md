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

## 參數

in_chans：使用的層數，最大為 64
tile_size：批量時每筆資料的窗戶大小
stride：批量時的跨步大小
batch_size：一次要批進多少筆資料
num_workers：想使用的 worker 數量
size：為 kernal 大小

## read_image_mask

回傳 fragment_mask 和 image_stack，其中 fragment_mask 大小為 (h, w) 是 mask 資料，image_stack 大小為 (h, w, layer) 是 layers 資料夾下產生的 stack，數值被裁切到介於 0~200 之間，layer 是以中央為基準取出的層數。此外，這兩種資料的 w, h 都被 padding 到 256 的整數倍以方便批量載入數據

## CustomDatasetTest

建立資料集，基於 torch 的 Dataset，其中 __len__ 可以獲得批量資料的數量 (e.g. len(dataset))，__getitem__ 能獲得某筆指定的資料 (e.g. dataset[idx])，這個應用回傳的是單筆的 image_stack 和對應的 coord。最後定義好一個 dataset 後就能放進 DataLoader 裡，決定要怎麼批量訪問這些資料

比較特別的是，資料本身有透過 albumentations 套件做ㄧ些轉換，像是 resize, normalize, to tensor， unsqueeze，為的是滿足 model 預測所需的輸入資料。細節上，會讓原本資料從 (size, size, layer) 變為 (1, layer, size, size)

當你使用下面方式遍歷 DataLoader 資料時，會以 tensor 形式回傳，images 大小是 (batch_size, 1, layer, size, size)，coords 大小是 (batch_size, 4)

```python
for step, (images, coords) in tqdm(enumerate(loader), total=len(loader)):
```

## get_img_splits

把資料裁切後建立 Dataset, DataLoader，並回傳 loader, coords, image_shape, fragment_mask

images 是個列表，囤放了所有裁切後的 stack，每個小方塊大小為 (tile_size, tile_size, layer)
coords 是個列表，囤放了 images 資料所對應的資料座標，[xmin, ymin, xmax, ymax]，最後以 numpy 回傳
image_shape 是 padding 後圖片的長寬大小 (h, w)
fragment_mask 就是 read_image_mask 回傳的東西
loader 是最後建立好的 DataLoader

## gkern

產生一個二維高斯函數，作為 kernal

## predict_fn

首先，`torch.no_grad` 會先把梯度下降關掉，`torch.autocast` 則是自動選擇必要的浮點數，然後就能夠把資料批量的丟給 model 預測。

也就是呼叫 `model(images)` 方法，這會執行 model 內部的 forward 方法，這個方法先把原本的資料轉換為 (batch_size, layer, 1, size, size)，作為 `TimeSformer` 的輸入，輸出會是 (batch_size, patch_size)，最後再轉為 (..., 1, 4, 4) 的大小，然後經過一個 `tourch.sigmoid` 函數作為輸出結果






