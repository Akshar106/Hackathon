#!/bin/bash
#SBATCH --job-name=neural_compression
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=akspate@iu.edu
#SBATCH --account=r02072

# ── Project path (absolute — matches your BigRed200 directory) ────────────────
PROJECT_DIR=/N/u/akspate/BigRed200/hackathon/Hackathon
cd $PROJECT_DIR

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p logs models

# ── Modules ───────────────────────────────────────────────────────────────────
module load python/gpu/3.11.5

# ── Virtual environment ───────────────────────────────────────────────────────
# First run: create and install (uncomment the next 3 lines, run once, then re-comment)
# python -m venv $HOME/envs/hackathon
# source $HOME/envs/hackathon/bin/activate
# pip install -r $PROJECT_DIR/requirements.txt

# Every run after that: just activate
source $HOME/envs/hackathon/bin/activate

# ── Print job info ─────────────────────────────────────────────────────────────
echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Partition    : $SLURM_JOB_PARTITION"
echo "GPU          : $CUDA_VISIBLE_DEVICES"
echo "Project dir  : $PROJECT_DIR"
echo "Start time   : $(date)"
echo "============================================"

python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# ── Stage 1: Train Denoising U-Net ────────────────────────────────────────────
echo ""
echo ">>> [1/3] Training DenoisingUNet on SimulatedNoisyOffice..."
echo "Start: $(date)"

python training/train_denoiser.py \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-4 \
    --patch-size 256

echo "Denoiser done: $(date)"

# ── Stage 2: Train Character Classifier ───────────────────────────────────────
echo ""
echo ">>> [2/3] Training CharClassifier on EMNIST..."
echo "Start: $(date)"

python training/train_classifier.py \
    --epochs 50 \
    --batch-size 128 \
    --lr 1e-3

echo "Classifier done: $(date)"

# ── Stage 3: Evaluate both models ─────────────────────────────────────────────
echo ""
echo ">>> [3/3] Running evaluation..."

python training/evaluate.py --n-samples 5000

echo ""
echo "============================================"
echo "All done: $(date)"
echo "Weights : $PROJECT_DIR/models/"
echo "Logs    : $PROJECT_DIR/logs/"
echo "============================================"
