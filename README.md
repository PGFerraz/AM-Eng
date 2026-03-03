# AM-Eng
A command-line chat interface for a llm, desinigned to ask questions about computer science and engineering. It was created and trained with a standard GPU-equipped (6GB VRAM or more) home computer in mind. In the future it will be trained with aditional databases,for a better comprehension of the subjects. I will also comment every step of the code, so it can be easily understood and modified by anyone. The system uses 4-bit quantization to reduce memory usage while maintaining good inference quality.

Future updates will include expanded training databases and fully commented source code for educational purposes.

---

# Features

- Local LLM inference (no API required)
- 4-bit quantization (BitsAndBytes)
- Persistent memory system (`fact_unit.json`)
- Context-aware conversation
- CLI-based interface
- Lightweight design for 6GB VRAM GPUs

---

# Model

Default model:

Qwen/Qwen2.5-3B-Instruct

Loaded with:
- 4-bit quantization (nf4)
- float16 compute
- automatic device mapping (CPU/GPU)

---

# Requirements

- Linux (recommended)
- Python 3.10 or 3.11 (3.14 not recommended)
- NVIDIA GPU with 6GB+ VRAM
- CUDA properly installed
- 15GB+ free disk space (first install downloads large CUDA wheels)

---

# Installation

## 1 - Clone the repository

```bash
git clone https://github.com/PGFerraz/AM_Eng.git
cd AM_Eng
```

## 2 - Create a Virtual Environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```
## 3 - Install PyTorch

GPU Version (CUDA 12.1 example)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
CPU Version (if no GPU)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 4 - Install Required Libraries
Install the remaining dependencies manually:
```bash
pip install transformers accelerate bitsandbytes
```

# Troubleshooting

## CUDA Out of Memory
If you see:
```bash
torch.OutOfMemoryError
```
Try:
- Closing other GPU applications
- Reducing max_new_tokens in the code
- Restarting your system

## No Space Left on Device
Clear pip cache:
```bash
pip cache purge
rm -rf ~/.cache/pip
```
