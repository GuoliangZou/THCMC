# Stop Pushing Friends Away: Trustworthy Hybrid Contrastive Multi-modal Clustering

Official implementation/cleaned training repository for **Trustworthy Hybrid Contrastive Multi-Modal Clustering (THCMC)**.

> **Paper:** Guoliang Zou, Shizhe Hu, Tongji Chen, Yangdong Ye, *Stop Pushing Friends Away: Trustworthy Hybrid Contrastive Multi-modal Clustering*, **Information Fusion**, Vol. 136, Article 104565, 2026.  
> **DOI:** https://doi.org/10.1016/j.inffus.2026.104565

## Overview

Contrastive multi-modal clustering can mistakenly repel semantically similar samples when they are treated as negatives. THCMC addresses this problem by jointly learning discriminative information within and across modalities and improving the reliability of the contrastive objective. The implementation in this repository contains the training path, model definitions, losses, graph utilities, evaluation code, experiment configurations, and the processed Caltech datasets required by the included configurations.

The repository has been cleaned to remove files unrelated to the normal `train.py` training path, such as IDE settings, Python caches, a local virtual environment, visualization scripts, convergence-analysis scripts, result text files, plotting assets, ablation shell scripts, standalone test code, and data-inspection helpers. The retained training source files are not modified by the cleanup.



## Environment

Python 3.8+ is recommended. Create a clean environment instead of using the removed project-local `venv` folder.

```bash
conda create -n thcmc python=3.8 -y
conda activate thcmc
```

Install PyTorch for your CUDA/CPU environment first, then install the remaining dependencies:

```bash
pip install torch
pip install -r requirements.txt
```

The configuration classes use the Pydantic 1.x API, so `requirements.txt` restricts Pydantic to `<2` for compatibility.

## Dataset Format

Datasets are loaded from `data/processed/<dataset-name>.npz` by `src/data/load.py`.

The expected keys are:

```text
labels
n_views
view_0
view_1
...
```

Included data:

| Dataset | Samples | Classes | Included view dimensions |
|---|---:|---:|---|
| Caltech-2V | 1400 | 7 | 40, 254 |



## Training

Run training from the `src` directory so the package imports resolve correctly:

```bash
cd src
python models/train.py -c caltech2v
```

The corresponding experiment is defined in:

```text
src/config/experiments/Caltech2V_mlp.py
```

The THCMC-style contrastive configuration uses the following loss terms in the supplied code:

```text
ddc_1 | ddc_2 | ddc_3 | contrast | GraphCL | Contrastive_label_level
```

Training checkpoints and configuration snapshots are saved under the repository-level `models/` directory by `ModelSaver`.

### Optional configuration overrides

Configuration values can be overridden from the command line using the nested `__` naming convention implemented in `src/config/__init__.py`. For example:

```bash
python models/train.py \
  -c caltech2v \
  --batch_size 100 \
  --n_epochs 100
```

## Main Training Flow



Files imported by this chain are retained even when some imported components are not activated by the default experiment, which avoids introducing behavior changes solely for repository cleanup.

## Reproducibility

`train.py` sets NumPy and PyTorch random seeds and configures deterministic cuDNN behavior:

```python
np.random.seed(cfg.seed)
th.manual_seed(cfg.seed)
th.cuda.manual_seed_all(cfg.seed)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False
```

Exact numerical results can still depend on the PyTorch/CUDA version, hardware, and low-level operators.

## Citation

If this code is useful in your research, please cite:

```bibtex
@article{zou2026thcmc,
  title   = {Stop Pushing Friends Away: Trustworthy Hybrid Contrastive Multi-modal Clustering},
  author  = {Zou, Guoliang and Hu, Shizhe and Chen, Tongji and Ye, Yangdong},
  journal = {Information Fusion},
  volume  = {136},
  pages   = {104565},
  year    = {2026},
  doi     = {10.1016/j.inffus.2026.104565}
}
```

## Publication

- Journal: **Information Fusion**
- Volume: **136**
- Article: **104565**
- Publication date: **December 2026**
- DOI: **10.1016/j.inffus.2026.104565**
- Publisher page: https://www.sciencedirect.com/science/article/pii/S1566253526004434
