# Multimodal Artificial Intelligence for Gas Detection: Robust Sensor Fusion via Gated Multimodal Unit (GMU)

**Author:** Muhammet ÖZCAN (Student ID: 25434977001)  
**Course:** YZM5110 - Multimodal Learning  
**Institution:** Bursa Technical University, Graduate School  

---

## 1. Overview & Motivation
This repository contains the official implementation, reproduction baseline, and proposed fault-tolerant architecture for the **YZM5110 Research Project**. The study builds upon the foundational work by Narkhede et al. (2021), *"Gas Detection and Identification Using Multimodal Artificial Intelligence Based Sensor Fusion"*, which combines 7 semiconductor metal-oxide (MQ) gas sensors with thermal camera imagery to classify four gas states (`NoGas`, `Perfume`, `Smoke`, `Mixture`).

### Research Question
> *"In the CNN and LSTM-based early fusion architecture proposed by Narkhede et al. (2021)—once the inter-modal dimensionality imbalance (i.e., 64-dimensional image features versus 5-dimensional sensor features) is resolved and a learnable Gated Multimodal Unit (GMU) mechanism is incorporated—to what extent can the model's robustness be enhanced in scenarios involving sensor failure, calibration drift, or thermal camera degradation, while maintaining performance on clean data?"*

### Key Contributions
1. **Data Alignment & Scaling Fix (Reproduction):** Identified and resolved a critical image-sensor alphabetical sorting mismatch (`0, 1000, 1001...`) in standard directory loaders by deterministically mapping CSV rows via the `Corresponding Image Name` column (6,400/6,400 exact match). Combined with Z-score standardization (`StandardScaler`), the unimodal LSTM accuracy increased from the paper's reported **82.00%** to **94.30%**.
2. **Dimensionality Balancing & Symmetric Projection Ablation:** Projecting both the 64-D CNN features and 5-D LSTM features into a shared 32-D latent space improved clean test accuracy from **95.55%** (Baseline Early Fusion) to **98.05%** (Symmetric No-Gate Ablation).
3. **Fault Tolerance via Gated Multimodal Unit (GMU):** Introduced a learnable sigmoid gating mechanism trained with Modality Corruption Augmentation. Under severe thermal camera degradation, while Baseline Early Fusion collapses to **45.70%**, the proposed GMU model dynamically shifts reliance to the gas sensor array and maintains **90.94%** accuracy (**+45.23%** gain).

---

## 2. Dataset & Preprocessing Protocol
* **Source Dataset:** Multimodal Gas Detection Dataset ([Mendeley Data / Narkhede et al., 2021](https://doi.org/10.3390/asi4010003))
* **Total Samples:** 6,400 paired samples across 4 balanced classes (1,600 samples per class): `0: NoGas`, `1: Perfume`, `2: Smoke`, `3: Mixture`.
* **Input Modalities:**
  * **Gas Sensor Array (1D Sequence):** 7 channels (`MQ2`, `MQ3`, `MQ5`, `MQ6`, `MQ7`, `MQ8`, `MQ135`), standardized via `StandardScaler` (fitted strictly on the training split) and reshaped to `(1, 7)`.
  * **Thermal Imagery (2D RGB):** Captured via Seek Thermal camera, resized to `128 × 128 × 3`, and normalized to `[0, 1]`.
* **Splitting Protocol (`SEED = 42`, Stratified):** 4,096 Training (64%), 1,024 Validation (16%), 1,280 Test (20% — 320 samples per class).

---

## 3. Experimental Results

### 3.1. Baseline Reproduction vs. Original Paper (Clean Test Set)
| Model / Method | Paper Reported Acc (%) | Reproduced Test Acc (%) | Bootstrap 95% CI | Macro-F1 | Total Parameters | Inference Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM Only (Gas Sensor)** | 82.00 | 94.30 | 94.30% ± 0.66% [92.97, 95.70] | 0.94 | 284 | 0.202 |
| **CNN Only (Thermal Image)** | 93.00 | 94.06 | 94.09% ± 0.66% [92.81, 95.39] | 0.94 | 1,699,204 | 10.147 |
| **Early Fusion (Baseline)** | 96.00 | 95.55 | 95.56% ± 0.57% [94.38, 96.56] | 0.96 | 1,701,860 (Head: 2,372) | 10.477 (Head: 0.128) |
| **Late Fusion (Max / Avg)** | 96.00 | 97.19 | — | 0.97 | 1,699,488 | 10.349 |
| **Ablation (Symmetric No-Gate)** | — | 98.05 | 98.07% ± 0.38% [97.34, 98.75] | 0.98 | 1,702,916 (Head: 3,428) | 10.550 (Head: 0.201) |
| **Proposed Gated Fusion (GMU)** | — | **96.48** | **96.50% ± 0.50% [95.47, 97.34]** | **0.96** | **1,705,284 (Head: 5,796)** | **10.645 (Head: 0.296)** |

### 3.2. Stress Test: Clean vs. Corrupted Modality Conditions
| Test Scenario | Baseline Early Fusion (%) | Ablation (No-Gate) (%) | Proposed Gated Fusion (GMU) (%) | Diff (Proposed - Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Clean Input** | 95.55 | **98.05** | 96.48 | **+0.94%** |
| **2. Noisy Gas Sensor ($\sigma=2.0$)** | 91.33 | 43.28 | **93.67** | **+2.34%** |
| **3. Missing Gas Sensor ($\mathbf{f}_{\text{lstm}}=\mathbf{0}$)** | **95.08** | 91.95 | 94.61 | -0.47% |
| **4. Corrupted Thermal Camera (90% loss + $\sigma=2.0$)** | 45.70 | 80.70 | **90.94** | **+45.23%** |

---

## 4. Repository Structure

```text
Multimodal_gas_detection/
│
├── Multimodal_Gas_Detection.ipynb       # Complete Google Colab notebook (Data loading, baselines, GMU, stress tests)
├── Ozcan_YZM5110_Research.pdf           # Final research report
└── README.md                            # Project documentation and reproduction instructions
```

---

## 5. How to Run & Reproduce

### Requirements
The experiments were executed on **Google Colab (CPU Runtime)** using `Python >= 3.10`, `tensorflow >= 2.15.0`, `scikit-learn >= 1.2.2`, `pandas`, `numpy`, `matplotlib`, `seaborn`, and `pillow`.

To install dependencies locally:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn pillow
```

### Step-by-Step Execution in Google Colab
1. Download the dataset archives (`NoGas.zip`, `Perfume.zip`, `Smoke.zip`, `Mixture.zip`) and `Gas_Sensors_Measurements.csv` from the original dataset source ([Mendeley Data](https://doi.org/10.3390/asi4010003)), and place them in your Google Drive directory: `/content/drive/MyDrive/MultimodalGasData/`.
2. Open `Multimodal_Gas_Detection.ipynb` in Google Colab and run all cells sequentially:
   * **Cells 1–2:** Mounts Google Drive, extracts the ZIP archives to `/content/extracted_images`, and deterministically maps image filenames to CSV sensor rows via the `Corresponding Image Name` column (6,400/6,400 exact match).
   * **Cell 3:** Loads and rescales thermal images to `(128, 128, 3)`, performs a stratified split into Training (4,096), Validation (1,024), and Test (1,280) sets (`SEED = 42`), and applies `StandardScaler` to the 7 MQ sensor channels.
   * **Cells 4–5:** Trains and evaluates the unimodal (`LSTM Only`, `CNN Only`) and baseline fusion (`Early Fusion`, `Max Fusion`, `Average Fusion`) models, generating the 5-panel confusion matrix.
   * **Cells 6–8:** Trains the `Proposed Gated Fusion (GMU)` and `Ablation (Symmetric No-Gate)` models, runs the 4-scenario stress test, computes 1,000-iteration Bootstrap 95% confidence intervals, performs sample-level failure analysis, and exports all result tables and model weights.

---

## 6. References
1. **Narkhede, P., Walambe, R., Mandaokar, S., Chandel, P., Kotecha, K., & Ghinea, G. (2021).** Gas detection and identification using multimodal artificial intelligence based sensor fusion. *Applied System Innovation*, 4(1), 3. [https://doi.org/10.3390/asi4010003](https://doi.org/10.3390/asi4010003)
2. **Sharma, A., Khullar, V., Kansal, I., Chhabra, G., Arora, P., Popli, R., & Kumar, R. (2024).** Gas detection and classification using multimodal data based on federated learning. *Sensors*, 24(18), 5904.
3. **Jadhav, P., Sairam, V. A., Bhojane, N., Singh, A., Gite, S., Pradhan, B., ... & Alamri, A. M. (2025).** Multimodal Gas Detection Using E-Nose and Thermal Images: An Approach Utilizing SRGAN and Sparse Autoencoder. *Comput. Mater. Continua*, 83(2), 3493-3517.
4. **Zhang, E., & Zhang, E. (2025).** Gas pipeline leakage detection based on multiple multimodal deep feature selections and optimized deep forest classifier. *Frontiers in Environmental Science*, 13, 1569621.
5. **Attallah, O. (2023).** Multitask Deep Learning-Based Pipeline for Gas Leakage Detection via E-Nose and Thermal Imaging Multimodal Fusion. *Chemosensors*, 11(7), 364.
6. **Rahate, A., Mandaokar, S., Chandel, P., Walambe, R., Ramanna, S., & Kotecha, K. (2023).** Employing multimodal co-learning to evaluate the robustness of sensor fusion for industry 5.0 tasks. *Soft Computing*, 27(7), 4139-4155.

To install dependencies locally:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn pillow
