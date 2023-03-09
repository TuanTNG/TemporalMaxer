export CUDA_VISIBLE_DEVICES=0
python ./eval.py configs/temporalmaxer_multithumos_i3d.yaml ckpt/multithumos/bestmodel.pth.tar
