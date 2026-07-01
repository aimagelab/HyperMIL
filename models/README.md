# `models/` folder

This folder contains the model building blocks for HyperMIL.

## Files and responsibilities

- `hypermil.py`
  - Composition module that connects intra-hypergraph encoding to temporal MIL.
  - Handles per-sample sequence length masking and batch padding.

- `channels_hypergraph.py`
  - Channel-level hypergraph feature extraction.
  - Learns prototype-based motif hyperedges and performs hypergraph convolution.

- `timemil.py`
  - Temporal MIL classifier head.
  - Provides positional encodings (`wavelet`, `sinusoidal`, `none`) and pooling (`cls`, `mean`, `max`, `attention`, `conjunct`).

- `nystrom_attention.py`
  - Nyström self-attention implementation used by temporal blocks.

- `common.py`
  - Shared low-level utility layers used by model variants.

## Architectural pattern

The design follows a **composition-over-monolith** approach:

1. Encode local/channel structure (`channels_hypergraph.py`)
2. Aggregate temporally (`timemil.py`)
3. Wire modules in a thin orchestrator (`hypermil.py`)

This separation keeps experimentation and testing localized per module.
