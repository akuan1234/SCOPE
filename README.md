# From Decoupled Heads to Self-Correcting Inference: SCOPE for Training-Free Open-Vocabulary Segmentation

<p align="center">
  <img width="1612" height="964" alt="image" src="https://github.com/user-attachments/assets/267c259b-5d64-47a9-a030-8b586123a8c9" />
</p>


## Overview

**SCOPE** is a fully training-free framework for open-vocabulary semantic segmentation built on top of **SAM3**.  
Instead of treating SAM3's semantic map, instance proposals, and presence score as passive endpoints, SCOPE turns them into internal signals for **diagnosis**, **correction**, and **stabilization** during inference.

SCOPE is organized around three tightly coupled modules:

- **Semantic-Adaptive Branch (SAB):** identifies unreliable queries through foreground-focused uncertainty and conditionally triggers adaptive re-inference.
- **Collaborative Dual-Head Decoding (CDHD):** uses dense semantic support to refine instance-derived responses through asymmetric cross-head interaction.
- **Multi-View Consensus (MVC):** fuses aligned predictions from the original and horizontally mirrored views to improve stability.

The full pipeline runs **without parameter updates, auxiliary training, or external refinement backbones**.

---

## Highlights

- Fully **training-free** open-vocabulary segmentation.
- Built directly on **SAM3's native decoupled outputs**.
- No additional training, no fine-tuning, and no auxiliary model composition in the core pipeline.
- Strong results on **eight standard OVSS benchmarks**.
- Clean inference design with three complementary stages: **SAB**, **CDHD**, and **MVC**.

---

## Main Results

SCOPE improves the frozen SAM3 baseline from **57.4** to **63.3** average mIoU on eight standard natural-image OVSS benchmarks.

| Method | Backbone | VOC21 | VOC20 | PC60 | PC59 | Object | Stuff | ADE | City | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM3 baseline | SAM3 (PE-L+/14) | 79.1 | 96.2 | 42.9 | 50.7 | 66.1 | 32.2 | 28.6 | 63.9 | 57.4 |
| **SCOPE** | **SAM3 (PE-L+/14)** | **80.3** | **97.1** | **51.0** | **59.0** | **68.4** | **42.9** | **37.7** | **70.1** | **63.3** |

Benchmarks: **VOC21, VOC20, PC60, PC59, Object, Stuff, ADE, City**.

---

## Method Overview

<p align="center">
  <img width="2222" height="1135" alt="image" src="https://github.com/user-attachments/assets/637cc47f-6888-455d-a0b7-18aff0d4b6e4" />
</p>


For each text query, SCOPE first performs standard SAM3 prompt inference to obtain:

- a semantic response map,
- a set of instance masks with objectness scores,
- and a presence-aware confidence score.

These decoupled outputs are then processed as follows:

1. **SAB** estimates query ambiguity from semantic uncertainty and presence confidence, and conditionally performs a second inference pass under mildly reweighted multi-scale evidence.
2. **CDHD** combines semantic and instance predictions into an initial query response and then locally refines top-ranked instance proposals using semantic support.
3. **MVC** averages aligned predictions from the original view and its horizontally flipped counterpart.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/akuan1234/SCOPE.git
cd SCOPE
```

### 2. Create an environment

```bash
conda create -n scope python=3.10 -y
conda activate scope
```

### 3. Install dependencies

Install the required packages from your environment file:

```bash
pip install -r requirements.txt
```

At minimum, this project depends on the following components:

- PyTorch
- MMSegmentation
- MMEngine
- MMCV
- Pillow
- SAM3 and its tokenizer / processor dependencies

> If your released repository uses a different dependency file name, replace the command above accordingly.

---

## Checkpoint Preparation

The current implementation expects the SAM3 checkpoint and tokenizer assets in the following locations:

```text
weights/sam3/sam3.pt
sam3/assets/bpe_simple_vocab_16e6.txt.gz
```

Please download or place the corresponding files there before evaluation.

---

## Dataset Preparation

Please organize each dataset according to the paths expected by its corresponding config file in `configs/`.

The paper reports results on the following natural-image benchmarks:

- Pascal VOC21
- Pascal VOC20
- Pascal Context60 (PC60)
- Pascal Context59 (PC59)
- COCO-Object
- COCO-Stuff
- ADE20K
- Cityscapes

You should also ensure that the class-name files used by each config are correctly specified, since the model loads text queries from the provided `classname_path`.

> Add dataset tree examples here if you want the README to be directly reproducible for new users.

---

## Usage

### Single-GPU

```bash
python eval.py --config configs/cfg_DATASET.py
```

### Multi-GPU

```bash
bash dist_test.sh configs/cfg_DATASET.py NUM_GPU
```

### Example

```bash
python eval.py --config configs/cfg_city_scapes.py
```

---

## Qualitative Results

<p align="center">
  <img width="1674" height="1063" alt="image" src="https://github.com/user-attachments/assets/32b090a1-249b-49a9-8825-fb2066874f48" />
</p>


Compared with prior training-free baselines, SCOPE typically produces:

- cleaner object/background separation,
- stronger recovery of fragmented structures,
- more complete fine-grained regions,
- and better robustness in semantically crowded scenes.

---

## Repository Structure

A minimal project layout is expected to look like this:

```text
SCOPE/
├── configs/
├── weights/
│   └── sam3/
│       └── sam3.pt
├── sam3/
│   └── assets/
│       └── bpe_simple_vocab_16e6.txt.gz
├── eval.py
├── dist_test.sh
├── requirements.txt
└── README.md
```

If your repository contains additional scripts for visualization, ablations, or dataset conversion, you can extend this section accordingly.

---

## Notes

- This repository is focused on **inference-only**, **training-free** evaluation.
- SCOPE is designed to preserve the native SAM3 pipeline while reorganizing its inference structure.
- If you are adapting the code for new datasets, the main points to check are the config file, dataset path, and class-name file.

