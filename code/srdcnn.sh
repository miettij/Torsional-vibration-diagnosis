#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=70G
#SBATCH --constraint=volta
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6

cd $WRKDIR/gear/gear/code2/

module load anaconda3/latest
python3 main.py --dataset traditional --arch SRDCNN --lr 0.0001 --batch-size 64 --epochs 20 --tw-stride 128 --tw-len 2048
