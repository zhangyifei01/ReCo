## Beyond Instance Discrimination: Relation-aware Contrastive Self-supervised Learning

<p align="center">
  <img src=figs/reco.png width="500">
</p>



### Unsupervised Training

This implementation references the code of MoCo.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_reco.py \
 -a resnet50 \
 --lr 0.03 \
 --batch-size 256 --moco-k 65536\
 --mlp --moco-t 0.2 --aug-plus --cos \
 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
 [your imagenet-folder with train and val folders]
```


### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_lincls.py \
  -a resnet50 \
  --lr 10.0 \
  --batch-size 256 \
  --pretrained checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```
or
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_simsiam_lincls.py \
  -a resnet50 \
  --batch-size 4096 \
  --pretrained checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --lars \
  [your imagenet-folder with train and val folders]
```

Linear classification results on ImageNet using this repo with 8 GPUs :
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">backbone</th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">MoCo v2<br/>top-1 acc.</th>
<th valign="bottom">ReCo<br/>top-1 acc.</th>
<!-- TABLE BODY -->
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">67.5</td>
<td align="center">71.3</td>
</tr>
</tbody></table>


### Transferring to Object Detection

See [./detection](detection).
