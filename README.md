# MADX

# Setup
```bash
conda create -n madx python=3.11
conda activate madx
pip install -r requirements.txt
pip install -e .
```
if there is a problem with pytorch, try installing it manually from [here](https://pytorch.org/)

# Usage
- For training please running notebook located at `notebooks/train.ipynb`
- For playing with the model, please run python script located at `madx/play.py`


# Evaluation

## 1-Frame Evaluation

### Random Action Dataset
| Model | Dataset | PSNR | LPIPS | Weights |
|-------|---------|------|-------|---------|
| Naive | Random action space | 26.7938 | 0.0216 | [drive](https://drive.google.com/file/d/1xB9N3vSRu8LWIn4G-gC41z61FNQN3Qrp/view?usp=drive_link) |
| Concat | Random action space | 26.7645 | 0.021 | [drive](https://drive.google.com/file/d/1zaeAqz3hRL7Cl0dWT8JSEo5A5aHZxEgz/view?usp=drive_link) |
| Attention | Random action space | 26.3499 | 0.0226 | [drive](https://drive.google.com/file/d/1DIel1G043TJUk4QxofhRoZ-eQea5yV8a/view?usp=drive_link) |

### Multi-RL Agent Dataset

| Model | Dataset | PSNR | LPIPS | Weights |
|-------|---------|------|-------|---------|
| Naive | Multi-RL agent (17k) | 26.6786 | 0.023 | [drive](https://drive.google.com/file/d/1xB9N3vSRu8LWIn4G-gC41z61FNQN3Qrp/view?usp=drive_link) |
| Concat | Multi-RL agent (17k) | 26.6448 | 0.0225 | [drive](https://drive.google.com/file/d/1zaeAqz3hRL7Cl0dWT8JSEo5A5aHZxEgz/view?usp=drive_link) |
| Attention | Multi-RL agent (17k) | 26.6783 | 0.0228 | [drive](https://drive.google.com/file/d/1DIel1G043TJUk4QxofhRoZ-eQea5yV8a/view?usp=drive_link) |

### Human Play Dataset
| Model | Dataset | PSNR | LPIPS | Weights |
|-------|---------|------|-------|---------|
| Naive | Human Play (17k) | 27.6249 | 0.0197 | [drive](https://drive.google.com/file/d/1xB9N3vSRu8LWIn4G-gC41z61FNQN3Qrp/view?usp=drive_link) |
| Concat | Human Play (17k) | 28.0194 | 0.0187 | [drive](https://drive.google.com/file/d/1zaeAqz3hRL7Cl0dWT8JSEo5A5aHZxEgz/view?usp=drive_link) |
| Attention | Human Play (17k) | 28.8662 | 0.0158 | [drive](https://drive.google.com/file/d/1DIel1G043TJUk4QxofhRoZ-eQea5yV8a/view?usp=drive_link) |

## 5-Frame Evaluation
