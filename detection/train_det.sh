CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py \
--config-file configs/pascal_voc_R_50_C4_24k_moco.yaml \
--num-gpus 8 MODEL.WEIGHTS ./output199.pkl