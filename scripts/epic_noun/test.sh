export CUDA_VISIBLE_DEVICES=0
python ./eval.py configs/temporalmaxer_epic_slowfast_noun.yaml ./ckpt/noun_epic/bestmodel.pth.tar
