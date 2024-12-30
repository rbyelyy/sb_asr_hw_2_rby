This repository provides code for solving the **Automatic Speech Recognition (ASR)** task using **PyTorch**.

---

## ⚙️ Installation

### **Prerequisites:**
- 🐍 Python 3.x (Check with `python3 --version`)
- 📦 pip (Check with `pip --version`)

### **Steps:**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rbyelyy/sb_asr_hw_2_rby.git
   ```

2. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up pre-commit:**
   ```bash
   pre-commit install
   ```

## 🚀 Training

To train the model, run the following command:

```bash
python3 train.py
```

### Default Training Configuration:
- **Config file**: `baseline.yaml`
- **Batch size**: 200 samples
- **Training epochs**: 50
📄 *More details are available in the configuration file.*

---

### Training Results:
- 📊 **WandB Logs**: [View Run Details](https://wandb.ai/hse_rbyelyy/pytorch_template_asr_example/runs/42uh8hwr?nw=nwuserrbyelyy)
- 📁 **Saved Model**: `saved/testing/model_best.pth`
  - **Test CER (Character Error Rate)**: 0.2045
  - **Test WER (Word Error Rate)**: 0.5851

---

### 🔍 Inference
Run inference using the pre-trained model:

```bash
python3 inference.py
```
💡 Optionally, you can pass custom HYDRA_CONFIG_ARGUMENTS to modify configurations.
