# A region-based convolutional fusion network for typhoon intensityestimation in satellite images

## create environment
```bash
conda env create -f read-data.yaml
conda activate read-data
```

## train-single-gpu
```bash
git checkout single-gpu
python trainMS.py --gpu=X --model=XXX

```

## train-multi-gpu
```bash
git checkout master
python trainMS.py
```
