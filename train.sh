export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_moco.py \
 -a resnet50 \
 --lr 0.03 \
 --batch-size 256 --moco-k 65536\
 --mlp --moco-t 0.2 --aug-plus --cos \
 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
 /data2/AllData/ILSVRC2012

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_simsiam_lincls.py \
  -a resnet50 \
  --batch-size 4096 \
  --pretrained checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --lars \
  /data2/AllData/ILSVRC2012

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_lincls.py \
  -a resnet50 \
  --lr 10.0 \
  --batch-size 256 \
  --pretrained checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /data2/AllData/ILSVRC2012


