# How to run

Install packages

```bash
pip install -r requirements.txt
```

Download the data

```bash
./download.sh
```

Run the script

```python
python inference_timesformer.py --segment_id 20230509182749 --segment_path $(pwd)/train_scrolls --model_path checkpoints/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt
```