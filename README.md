# Deep ST-PLIF

**A fully event-driven Spiking Transformer with Parametric LIF neurons for decoding fine finger movements from EEG**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-snnTorch-red.svg)](#)
[![Status](https://img.shields.io/badge/Status-Research-orange.svg)](#)

Deep ST-PLIF is a low-power, event-driven Spiking Transformer architecture for decoding individual finger motor execution from non-invasive scalp EEG. It targets a core bottleneck of wearable brain-computer interfaces (BCIs): continuous deep learning models rely on dense Multiply-Accumulate (MAC) operations that are too energy-hungry for battery-powered, untethered neuroprosthetics.

The architecture combines Parametric Leaky Integrate-and-Fire (PLIF) neurons with a Softmax-free spiking attention mechanism, and introduces a biologically inspired **Split-Polarity (ON/OFF) spatial encoding** scheme that resolves the "Dead Neuron" phase-cancellation problem standard LIF neurons hit on fast-oscillating, zero-mean EEG signals. In addition to the architecture, this repository documents a **sliding-window data leakage audit** contrasting naive window-stratified splits against strict, zero-overlap cross-trial evaluation.

<p align="center">
  <img src="assets/architecture-diagram.jpeg" alt="Deep ST-PLIF architecture" width="850">
  <br>
  <em>Figure 1 — Deep ST-PLIF pipeline: preprocessing → Split-Polarity encoding → stacked Spiking Transformer blocks → classification head</em>
</p>

---

## Table of Contents

- [Highlights](#highlights)
- [Problem](#problem)
- [Method](#method)
  - [1. EEG Preprocessing](#1-eeg-preprocessing)
  - [2. Split-Polarity Encoding](#2-split-polarity-encoding)
  - [3. Deep ST-PLIF Architecture](#3-deep-st-plif-architecture)
  - [4. Training Setup](#4-training-setup)
- [Dataset](#dataset)
- [Evaluation Regimes & Results](#evaluation-regimes--results)
- [Sliding-Window Leakage Audit](#sliding-window-leakage-audit)
- [Energy Efficiency](#energy-efficiency)
- [License](#license)

---

## Highlights

- **Strict leak-free cross-trial decoding:** Under zero-overlap trial-grouped validation (`StratifiedGroupKFold` with an 8-window embargo buffer), Deep ST-PLIF consistently outperforms chance across unseen 5 s trials on Subject 1 — **59.74%** on 2-class Thumb vs. Pinky (+9.74% over chance), **44.35%** on 3-class (+11.02% over chance), and **35.06%** on 4-class (+10.06% over chance).
- **Biologically consistent separability:** Leak-free evaluation preserves expected motor-cortex somatotopy, with spatially distant digits (**Thumb vs. Pinky: 59.74%**) separating more cleanly than anatomically adjacent digits (**Index vs. Middle: 53.25%**).
- **Ultra-low-power spiking inference:** Achieves **84.91×–102.64×** theoretical energy efficiency over an iso-architecture ANN (with **27.00%–31.45%** spike firing rates) and **13.87×** theoretical efficiency over a 2-layer LSTM baseline on 45 nm CMOS hardware.
- **Pre-synaptic Split-Polarity (ON/OFF) encoder:** Eliminates phase-cancellation in unipolar spiking neurons without discarding raw temporal phase information via frequency-domain transforms.
- **Softmax-free spiking attention:** Scaled Query–Key dot products drive PLIF neurons directly, generating sparse binary attention maps instead of dense floating-point probability distributions.
- **Transparent leakage audit:** Directly quantifies how 87.4% sliding-window overlap inflates apparent EEG accuracy from **35.06%–59.74%** (unseen trials) to **78.24%–93.97%** (within-trial window splits).

## Problem

Wearable BCIs increasingly rely on deep learning (CNNs, LSTMs, Transformers) to decode motor intent from EEG. However, continuous architectures depend on dense, synchronous MAC operations at every timestep, creating high energy consumption, thermal overhead, and short battery life on edge neuroprosthetic hardware.

Spiking Neural Networks (SNNs) offer an event-driven, low-power alternative — but feeding raw, fast-oscillating, zero-mean EEG directly into standard unipolar Leaky Integrate-and-Fire (LIF) neurons triggers the **"Dead Neuron" problem**: alternating positive and negative wave phases cancel out inside the neuron's membrane potential before reaching firing threshold, silencing spikes and halting surrogate-gradient updates.

Deep ST-PLIF addresses this gap by combining Split-Polarity pre-synaptic encoding with PLIF-driven spiking self-attention, evaluated rigorously under leak-free trial-independent splits.

## Method

Figure 1 shows the three-stage pipeline: **Data Preprocessing → Spiking Transformer Design → Classification Head**.

### 1. EEG Preprocessing

Continuous EEG (128-channel BioSemi system, downselected to 32 sensorimotor channels C1–C32) is cleaned and standardized before encoding:

| Step | Detail |
|---|---|
| Line-noise removal | FIR notch filter at 60 Hz |
| Band-pass filtering | Zero-phase FIR (`firwin`), 4.0–40.0 Hz (isolates Mu/Alpha 8–13 Hz and Beta 13–30 Hz rhythms) |
| Downsampling | 1024 Hz → 100 Hz (zero-padded prior to resampling to limit aliasing) |
| Re-referencing | Common Average Reference (CAR) across the 32 retained sensorimotor channels |
| Artifact removal | ICA (10–30 components fitted on first 900 s); ocular/myogenic components rejected via spatial topography and spectrum inspection |
| Epoching & windowing | 5 s post-stimulus trial window; sliding window augmentation (1.0 s window, 0.126 s stride → 87.4% overlap) |
| Cross-trial split hygiene | `StratifiedGroupKFold` grouped strictly by parent trial ID with an 8-window embargo buffer (zero trial overlap across folds) |
| Epoch-level rejection | Channel std-dev thresholded in the 20–200 µV range, retaining ≥10% of cleanest epochs |
| Class balancing | Random downsampling of the majority (Thumb) class to match minority class counts |

The CAR transform for channel *i* is:

<p align="center">
  <strong>V<sub>i</sub><sup>CAR</sup>(t) = V<sub>i</sub>(t) − (1/N) ∑<sub>j=1</sub><sup>N</sup> V<sub>j</sub>(t), &nbsp; N = 32</strong>
</p>

### 2. Split-Polarity Encoding

Rather than delta-modulation (which loses amplitude and waveform information when converting continuous EEG to spikes), Deep ST-PLIF uses a biologically inspired **ON/OFF pathway split**, analogous to retinal encoding:

- A ReLU-based split separates each of the 32 sensorimotor channels into a **positive (ON)** and **negative (OFF)** channel → 64 channels total.
- Both positive peaks and negative troughs contribute additively to membrane potential integration instead of cancelling out.
- This preserves fine-grained temporal phase dynamics that frequency-domain or spatial-filtering front ends in prior spiking BCI models (e.g., SpiTranNet-LIF, S²M-Former) discard.

### 3. Deep ST-PLIF Architecture

Implemented in PyTorch with `snnTorch`:

| Stage | Detail |
|---|---|
| Input representation | Split-Polarity encoding: 32 → 64 channels |
| Spatial encoding | Pointwise 1D convolution, 64 → 256 channels; batch norm; PLIF neuron layer with **learnable** decay factor β and firing threshold V<sub>th</sub> |
| Spiking attention | Softmax replaced entirely by PLIF neurons — scaled Q · K acts as input current, producing sparse binary attention maps; **4 heads**, 256-dim embedding (64-dim per head) |
| Feed-forward | Spiking FFN: Linear 256→512 → PLIF → Linear 512→256 |
| Residuals | Skip connections around both the spiking attention block and the spiking FFN, stabilizing training over 100-timestep sequences |
| Stacking | 2 Spiking Transformer blocks |
| Output | Rate encoding (mean over time) → fully connected 256 → *C* classes (2, 3, or 4); trial-level voting aggregates predictions across each unseen 5 s trial |

### 4. Training Setup

| Hyperparameter | Value |
|---|---|
| Gradient method | Surrogate gradient descent (fast sigmoid, slope = 5) |
| Batch size | 64 |
| Max epochs | 500 |
| Optimizer | Adam, initial LR = 2 × 10⁻³ |
| LR schedule | Cosine annealing with warm restarts (T<sub>0</sub> = 50 epochs, T<sub>mult</sub> = 2) |

## Dataset

This work uses the **CMU Motor Execution dataset** (Ding et al., *Nature Communications*, 2025) — 21 subjects, 128-channel BioSemi EEG recorded during individual-finger motor execution for robotic hand control. Due to high non-stationarity and inter-subject variability in EEG, models are trained and evaluated **strictly per-subject**, not pooled across subjects.

- **Classes:** Thumb, Index, Middle, Pinky
- **Subject 1:** 960 Thumb trials, 192 trials each for Index / Middle / Pinky
- **Subjects 2–5:** 800 Thumb trials, 160 trials each for Index / Middle / Pinky
- Majority class (Thumb) is randomly downsampled to balance all classes prior to training.

## Evaluation Regimes & Results

Because sliding-window augmentation (1.0 s window, 0.126 s stride) introduces **87.4% temporal overlap** between adjacent windows within each 5 s trial, Deep ST-PLIF is reported under two explicitly separated evaluation regimes:

1. **Within-Trial Continuous Decoding (Window-Stratified Split):** Overlapping windows from the same movement trials are shared across train/test splits (reflecting within-session state tracking, but susceptible to temporal leakage).
2. **Cross-Trial Generalization (Strict Trial-Grouped Split):** Enforces zero trial overlap via `StratifiedGroupKFold` grouped by parent trial ID with an 8-window embargo buffer, aggregating spike-rate predictions across each unseen 5 s trial.

| Task Grouping (Subject 1) | Chance Baseline | Within-Trial (1 s Window Split) | Cross-Trial (Unseen 5 s Trial Voting) | Cross-Trial Macro F1 | Firing Rate (Sparsity) | Energy Efficiency vs. ANN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2-Class: Thumb vs. Pinky** | 50.00% | 83.71% (S1–S5 avg) | **59.74%** | **59.74%** | 27.01% | **96.23×** |
| **2-Class: Index vs. Middle** | 50.00% | 91.41% (S1–S5 avg) | **53.25%** | **52.86%** | 31.45% | **84.91×** |
| **3-Class: Thumb / Index / Pinky** | 33.33% | — | **44.35%** | **44.42%** | 27.00% | **102.64×** |
| **4-Class: All 4 Fingers** | 25.00% | 78.24% | **35.06%** | **33.95%** | 28.23% | **102.10×** |

**Key findings (Cross-Trial Generalization):**

- **Consistent above-chance decoding:** Deep ST-PLIF outperforms chance across every unseen-trial grouping — **+9.74%** on Thumb vs. Pinky, **+11.02%** on 3-class, and **+10.06%** on 4-class — without class collapse (Macro F1 tracks accuracy closely across all tasks).
- **Anatomical ordering holds:** Unlike the leaked window-split regime (where Index–Middle artificially outscored Thumb–Pinky), strict cross-trial validation restores expected sensorimotor somatotopy: spatially distant **Thumb vs. Pinky (59.74%) > anatomically adjacent Index vs. Middle (53.25%)**.
- **Physical ceiling of scalp EEG:** Individual finger representations sit millimeters apart in primary motor cortex and are heavily blurred by skull volume conduction; the ~35% 4-class and ~60% binary unseen-trial accuracies reflect the realistic signal-to-noise ceiling of non-invasive single-finger decoding without trial leakage.

## Sliding-Window Leakage Audit

Comparing the two columns above exposes a widespread methodological pitfall in sliding-window EEG deep learning:

- When overlapping windows (87.4% shared samples) are randomly split across train and test sets, high-capacity temporal models exploit **trial-specific baseline drift and shared signal segments** rather than generalizable motor-execution features, inflating apparent accuracy to **78.24% (4-class)** and **91.41%–93.97% (binary)**.
- Enforcing parent-trial grouping (`StratifiedGroupKFold`) plus an **8-window embargo buffer** completely removes this leakage and establishes an honest, reproducible benchmark for spiking EEG transformers on the CMU Motor Execution dataset.

## Energy Efficiency

Theoretical inference energy was estimated on **45 nm CMOS hardware** by comparing dense continuous Multiply-Accumulate (MAC) operations against sparse, event-driven Accumulate (AC) operations governed by empirical spike firing rates (**27.00%–31.45%**):

- **Vs. iso-architecture ANN:** **84.91×–102.64×** theoretical energy reduction across the 2-, 3-, and 4-class tasks (see Results table above).
- **Vs. 2-layer LSTM baseline** (hidden size 128, 32-channel input, 100 timesteps/inference, 3.2 pJ/MAC):

| Model | Operations / inference | Energy / inference (45 nm CMOS) |
|---|---|---|
| 2-layer LSTM baseline | 21.4 × 10⁶ MACs | 68.48 × 10⁶ pJ |
| **Deep ST-PLIF** | Sparse AC ops (27%–31% firing rate) | **4.93 × 10⁶ pJ** |

**→ ≈13.87× theoretical energy efficiency improvement** over the 2-layer LSTM baseline, and **~85×–102×** over an equivalent continuous Transformer/ANN.

## License

MIT
