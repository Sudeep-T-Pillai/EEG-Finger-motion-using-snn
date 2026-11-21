# Archive — Deprecated Within-Trial Confusion Matrices

> [!WARNING]
> **Stale / Inflated Results (Data Leakage):** The confusion matrices in this directory (`S01`–`S05`) were generated under an earlier **window-stratified split** where overlapping sliding windows (1.0 s window, 0.126 s stride → **87.4% temporal overlap**) from the same 5 s trials were shared across training and validation sets. Do not cite these plots as generalizable cross-trial decoding performance.

## Why These Results Are Archived

When overlapping EEG windows are split randomly across folds, sequence models can exploit trial-specific baseline drift and shared temporal segments, inflating apparent accuracy to **78.24%** (4-class) and up to **93.97%** (binary pairs).

These artifacts are preserved solely as a transparent record for the **Sliding-Window Leakage Audit**.

## Current Leak-Free Benchmarks

For the corrected, zero-overlap **cross-trial generalization** results (`StratifiedGroupKFold` grouped by parent trial ID with an 8-window embargo buffer), refer to the [main README](../README.md):

- **2-Class (Thumb vs. Pinky):** 59.74% (chance: 50.00%)
- **2-Class (Index vs. Middle):** 53.25% (chance: 50.00%)
- **3-Class (Thumb / Index / Pinky):** 44.35% (chance: 33.33%)
- **4-Class (All 4 Fingers):** 35.06% (chance: 25.00%)
