# RC-BCJ-Dataset — Reinforced Concrete Beam–Column Joint Failure Image Dataset

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-blue)](https://doi.org/10.5281/zenodo.link)

Labeled image dataset of reinforced concrete (RC) beam–column joint failures with
expert-verified annotations for structural damage recognition, multimodal learning,
and generative modeling.

---

## Dataset Overview

| Property | Details |
|---|---|
| Images | ~400 annotated beam–column joint photos |
| Format | PNG (images) + CSV (annotations) |
| Annotation | Expert-verified, multi-attribute labels |
| Source | Laboratory specimens + field images |

### Annotation Fields

| Field | Categories |
|---|---|
| `joint_type` | Interior (+), Interior (T), Exterior (inverted-T), Exterior (L) |
| `failure_mechanism` | Beam (B), Joint (J), Beam–Joint (BJ) |
| `damage_type` | Flexural cracking, Diagonal shear, Spalling, Crushing, Rebar exposure |
| `damage_severity` | DS0 (none) → DS4 (near collapse) |
| `description` | Free-text structural diagnostic narrative |

---

## Repository Structure
```
├── images/                  # Beam–column joint images (.png)
├── metadata/
│   └── metadata.csv         # Annotations and labels
└── experiments/
    ├── classification/      # Failure-mode classification baselines
    ├── image_to_text/       # Image captioning baselines
    └── text_to_image/       # Conditional image generation baselines
```

---

## Baseline Results

### Failure-Mode Classification (10-fold CV)
| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| Swin-T | **81.69%** | 81.69% | 85.63% |
| ViT-B/16 | 80.28% | **82.05%** | **88.97%** |
| ResNet-50 | 77.46% | 80.95% | 85.24% |

### Image-to-Text (BLEU-4 / METEOR / ROUGE-L)
| Model | BLEU-4 | METEOR | ROUGE-L |
|---|---|---|---|
| Gemma-3-4B | **0.449** | 0.684 | 0.604 |
| Qwen3-VL-4B | 0.431 | **0.688** | 0.596 |
| R2Gen | 0.355 | 0.476 | 0.565 |

---

## Getting Started
```bash
git clone https://github.com/nubcico/rc-bcj-dataset
cd rc-bcj-dataset
pip install -r requirements.txt
```
```python
import pandas as pd
meta = pd.read_csv("metadata/metadata.csv")
print(meta.head())
```

---

## Citation

If you use this dataset, please cite:

---

## License

This dataset is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Free for academic and research use. Commercial use is prohibited.
