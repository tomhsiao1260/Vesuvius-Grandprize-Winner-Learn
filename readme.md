# How to run

Install packages

```bash
pip install -r requirements.txt
```

Download the data

```bash
./download.sh
```

Put the model into `checkpoints` folder. You can find the weights of the canonical timesformer uploaded [here](https://drive.google.com/drive/folders/1rn3GMOvtJRMBHOxVhWFVSY6IVI6xUnYp).

Run the script and check out the `predict` folder to see the result.

```python
python inference.py
```