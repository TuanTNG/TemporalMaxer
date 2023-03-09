export CUDA_VISIBLE_DEVICES=0
python ./train.py configs/temporalmaxer_multithumos_i3d.yaml --save_ckpt_dir "./ckpt/multithumos"
