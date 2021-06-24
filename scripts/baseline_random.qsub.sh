#$ -S /bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=40:00:00

#$ -l gpu=true
#$ -N baseline

#$ -cwd

#module purge
#module load default/python/3.8.5
source /share/apps/source_files/python/python-3.8.5.source
source rectenv/bin/activate

python ./RectAngle/train.py --train ./miccai_us_data/train.h5 \
--val ./miccai_us_data/val.h5 \
--ensemble 5 \
--lr_schedule exponential \
--label random \
--odir ./baseline_data/random \
--epochs 50 \
--seed 0
