CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval_svm_voc.py --pretrained ../checkpoint_0199.pth.tar \
  -a resnet50 \
  --low-shot \
  /data2/AllData
