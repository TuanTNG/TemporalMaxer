export CUDA_VISIBLE_DEVICES=0
python ./train.py configs/temporalmaxer_epic_slowfast_noun.yaml --save_ckpt_dir "./ckpt/noun_epic"
