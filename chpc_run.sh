#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=3
#SBATCH --gres=gpu:2080ti:3
#SBATCH --time=8:00:00
#SBATCH --mem=2GB
#SBATCH --mail-user=u1380656@umail.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o cv_large-%j

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ct-tf

OUT_DIR=/uufs/chpc.utah.edu/common/home/u1380656/Normative-Modeling-Using-Deep-Generative-Models/results
FILE_PATH=/uufs/chpc.utah.edu/common/home/u1380656/Normative-Modeling-Using-Deep-Generative-Models/data/cleaned_data/megasample_cleaned.csv

mkdir -p ${OUT_DIR}

python cv_large.py --output_dir ${OUT_DIR} --file_path ${FILE_PATH}