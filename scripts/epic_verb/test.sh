export CUDA_VISIBLE_DEVICES=2
python ./eval.py configs/temporalmaxer_epic_slowfast_verb.yaml ./ckpt/verb_epic/bestmodel.pth.tar
