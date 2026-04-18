#!/bin/bash
#SBATCH --job-name=char_classifier
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH -A r02072
#SBATCH --output=/N/scratch/akspate/classifier_%j.out
#SBATCH --error=/N/scratch/akspate/classifier_%j.err

echo "Running on: $(hostname)"
echo "Starting at: $(date)"

set -e

PROJECT_DIR=/geode2/home/u010/akspate/BigRed200/hackathon/Hackathon
cd $PROJECT_DIR

mkdir -p models

module load python/gpu/3.11.5

echo "Python location:"
which python
python -V

echo "GPU status:"
nvidia-smi

echo "Checking PyTorch CUDA:"
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

echo "Starting classifier training (wider VGG, 32x32, 60 epochs, batch=256)..."
python -u training/train_classifier.py \
    --epochs 60 \
    --batch-size 256 \
    --lr 1e-3

echo "Finished at: $(date)"
