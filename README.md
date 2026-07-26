# Cross-Dataset Comparison of CNN-LSTM Fall Detection Models

A controlled, matched-configuration comparison of a CNN-LSTM fall detection model trained and evaluated identically on two public wearable-IMU benchmarks — **SisFall** and **FallAllD** — using the same architecture, sensor modality, sampling rate, window length, and Leave-One-Subject-Out (LOSO) cross-validation protocol.



---

## Overview

Reported fall-detection performance across benchmark datasets is rarely compared under matched experimental conditions, making cross-dataset claims hard to interpret. This project holds architecture, preprocessing, and validation protocol fixed while varying only the underlying dataset, isolating dataset properties as the explanatory variable for the performance gap.

**Key result:** under an identical configuration (CNN-LSTM, waist-only, 6 features, 50 Hz, 5 s window, LOSO), the model achieves **98.81% AUC-ROC on SisFall** vs. **86.67% on FallAllD**.

## Repository Structure

```
.
├── notebooks/
│   ├── sisfall-cnn-lstm-fall-detection-50hz-waist.ipynb
│   ├── fallalld-cnn-lstm-fall-detection-50hz-waist.ipynb
│  
├── results/
│   └── (LOSO fold metrics, feature-ablation CSVs)
├── figures/
│   ├── fig_signals.png        # raw fall vs ADL signal visualization
│   └── fig_featimp.png        # channel-ablation feature importance

└── README.md
```

## Method Summary

- **Datasets:** [SisFall](http://sistemic.udea.edu.co/en/research/projects/english-falls/) (38 subjects) and [FallAllD](https://dx.doi.org/10.21227/bnya-mn34) (15 subjects), waist device only, 6-channel input (3-axis accelerometer + 3-axis gyroscope)
- **Preprocessing:** resampled to 50 Hz, 5 s sliding windows (80% overlap), StandardScaler
- **Model:** CNN-LSTM — 3× (Conv1D + BatchNorm + MaxPool + Dropout) → 2× LSTM → Dense → sigmoid (~106.6K parameters)
- **Training:** Adam (lr=1e-3), class-weighted BCE loss, EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
- **Validation:** Leave-One-Subject-Out (LOSO) cross-validation, threshold tuned via grid search
- **Cross-dataset transfer:** zero-shot evaluation + lightweight fine-tuning (frozen CNN, retrained LSTM+Dense on 20% of target data)

## Results

| Metric    | SisFall | FallAllD |
|-----------|---------|----------|
| AUC-ROC   | 98.81%  | 86.67%   |
| Recall    | 94.46%  | 82.81%   |
| Precision | 92.24%  | 62.50%   |
| F1 Score  | 93.25%  | 71.24%   |
| Accuracy  | 94.35%  | 79.48%   |

See the [full report](report/Fall_Detection_IEEE_Report.pdf) for the complete methodology, cross-dataset fine-tuning results, feature-importance analysis, and discussion.

## Requirements

```
tensorflow>=2.15
numpy
pandas
scikit-learn
matplotlib
```

## Author

**Mandavi Singh**
B.Sc. (Hons.) Data Science and Artificial Intelligence, Indian Institute of Technology Guwahati
Internship Report — Trimester 9

## Datasets Citation

- A. Sucerquia, J. D. López, and J. F. Vargas-Bonilla, "SisFall: A Fall and Movement Dataset," *Sensors*, 17(1):198, 2017.
- M. Saleh and R. Le Bouquin Jeannes, "FallAllD: An Open Dataset of Human Falls and Activities of Daily Living for Classical and Deep Learning Applications," *IEEE Sensors Journal*, 21(2):1849–1858, 2021.

## License

This repository is shared for academic and educational purposes as part of an internship report submission.
