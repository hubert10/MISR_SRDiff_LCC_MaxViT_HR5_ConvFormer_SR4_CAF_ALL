#!/bin/bash 
#SBATCH --job-name=exp_misr_joint_srdiff_lcc_train_backbone_maxvit_hr5_maxvit_sr4_caf_focal_all
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100m40:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH --mail-user=kanyamahanga@ipi.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output logs/exp_misr_srdiff_lcc_train_backbone_maxvit_hr5_convformer_sr4_caf_all_%j.out
#SBATCH --error logs/exp_misr_srdiff_lcc_train_backbone_maxvit_hr5_convformer_sr4_caf_all_%j.err
source load_modules.sh
export CONDA_ENVS_PATH=$HOME/.conda/envs
export DATA_DIR=$BIGWORK
conda activate flair_venv
which python
cd $HOME/MISR_SRDiff_LCC_MaxViT_HR5_ConvFormer_SR4_CAF_ALL
# srun python trainer.py --config configs/diffsr_highresnet_ltae.yaml --config_file flair-config-server.yml --exp_name misr/srdiff_highresnet_ltae_ckpt --hparams="cond_net_ckpt=/bigwork/nhgnkany/Results/MISR_SRDiff_LCC_MaxViT_HR5_ConvFormer_SR4_CAF_ALL/results/checkpoints/misr/highresnet_ltae_ckpt" --reset
srun python trainer.py  --config_file==./configs/train_main/ --exp_name misr/srdiff_highresnet_ltae_ckpt --hparams="cond_net_ckpt=/bigwork/nhgnkany/pretrain_weights/MISR_JOINT_SRDiff_HIGHRESNET_PRETRAINED/results/checkpoints/misr/highresnet_ltae_ckpt" --reset
