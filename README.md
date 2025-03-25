# Rotation-Invariant Completion Network (RICNet)
Our paper has been accepted by *PRCV 2023* 🚀🚀🚀

![Teaser](https://via.placeholder.com/800x300.png?text=RICNet+Architecture+Teaser)

A novel rotation-invariant framework for robust 3D point cloud completion under arbitrary poses.

## Key Features
✨ ​**Rotation-Invariant Learning**  
Dual Pipeline Completion Network (DPCNet) with RIConv++ operator enables pose-agnostic feature extraction

🧠 ​**Dual-Path Architecture**  
- Reconstruction path with VAE framework
- Completion path for partial inputs
- Shared encoder-decoder with distribution alignment

🔍 ​**Detail Enhancement**  
RENet-based enhancer recovers fine-grained structures using self-attention and multi-scale fusion

## Getting Started
### Installation
```bash
# Clone repo
git clone https://github.com/yourusername/RICNet.git
cd RICNet

# Install dependencies
pip install -r requirements.txt
