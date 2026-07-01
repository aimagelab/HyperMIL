# HyperMIL

HyperMIL is a modular MIL pipeline for multivariate time-series classification that combines:

1. **Intra-sample channel reasoning** with a channel-level hypergraph encoder.
2. **Temporal aggregation** with a transformer-like MIL head (Nyström attention).

> Source code for the paper:  
> **HyperMIL: Hypergraph-based channel reasoning for Multiple Instance Learning on Multivariate Time Series**  
> Del Gaudio et al., ICPR 2026.

## Installation

```bash
pip install -r requirements.txt
```

PyTorch should be installed separately to match your CUDA version — see the [official instructions](https://pytorch.org/get-started/locally/). The code requires `torch>=1.13`.

## Project structure

```text
HyperMIL/
├── main.py                  # CLI entrypoint for training on aeon datasets
├── train_model.py           # training/evaluation loops and metrics
├── lookhead.py              # Lookahead optimizer wrapper
├── utils.py                 # IO/logging/reproducibility helpers
├── requirements.txt
└── models/
    ├── hypermil.py
    ├── channels_hypergraph.py
    ├── timemil.py
    ├── nystrom_attention.py
    └── common.py
```

## Quick start

Data is loaded directly from the [aeon](https://www.aeon-toolkit.org/) time-series library — no manual download required.

```bash
python main.py --dataset PenDigits --num_epochs 50 --batchsize 64 --encoding wavelet --pooling cls
```

## Main CLI arguments

- `--dataset`: aeon dataset name (train/test split fetched automatically)
- `--num_epochs`, `--batchsize`, `--lr`, `--optimizer`, `--scheduler`
- `--encoding`: `wavelet | sinusoidal | none`
- `--pooling`: `cls | mean | max | attention | conjunct`
- `--k_prototypes`, `--num_convs`, `--tau`, `--intra_embed`, `--embed`
- `--seed`: random seed for reproducibility (default: `0`)
- `--save_dir`: checkpoint output directory (default: `./savemodel/`)

## Design notes

- `main.py` is intentionally thin and orchestrates only parsing, data loading, and handoff to `train_loop`.
- Model components are separated by responsibility:
  - `channels_hypergraph.py`: channel-graph encoding
  - `timemil.py`: temporal MIL head and pooling
  - `hypermil.py`: composition layer

## Current scope

This release version targets aeon-based equal-length classification workflows while preserving support paths for variable-length collation in the training utilities.

## Acknowledgements

This work builds on [TimeMIL](https://arxiv.org/abs/2405.03140) by Chen et al. — we thank the authors for releasing their code, which we used as the foundation for the temporal MIL head in this repository.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{del2026hypermil,
  title={HyperMIL: Hypergraph-based channel reasoning for Multiple Instance Learning on Multivariate Time Series},
  author={Del Gaudio, Livia and Cuculo, Vittorio and Cucchiara, Rita},
  booktitle={Proceedings of the 28th International Conference on Pattern Recognition},
  year={2026}
}
```

