#!/bin/bash
#SBATCH --account=rrg-csubakan
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --exclude=fc10713
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --array=0-2
#SBATCH --job-name=infer
#SBATCH --output=slurm-infer-%A_%a.out
python /home/jingyi49/scratch/trm-codec-rates-20261022/infer_wavlm.py --input /home/jingyi49/dns_real.list --checkpoint /home/jingyi49/scratch/trm-codec-rates-20261022/experiments/50hz/checkpoints/best.pt --output-dir /home/jingyi49/scratch/trm-codec-rates-20261022/dns_real/50hz
python /home/jingyi49/scratch/trm-codec-rates-20261022/infer_wavlm.py --input /home/jingyi49/dns_real.list --checkpoint /home/jingyi49/scratch/trm-codec-rates-20261022/experiments/25hz/checkpoints/best.pt --output-dir /home/jingyi49/scratch/trm-codec-rates-20261022/dns_real/25hz
python /home/jingyi49/scratch/trm-codec-rates-20261022/infer_wavlm.py --input /home/jingyi49/dns_real.list --checkpoint /home/jingyi49/scratch/trm-codec-rates-20261022/experiments/12_5hz/checkpoints/best.pt --output-dir /home/jingyi49/scratch/trm-codec-rates-20261022/dns_real/12_5hz