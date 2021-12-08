# txt2img-score-sde

CS236 Final Project (Autumn 2021)

Contributors: Rui Yan, Wenna Qin, Yixing Wang

This repository contains the code for text-to-image generation via score-based model.

First set up the enviroment ```conda env create -f score_env.yml```

Then download the pretrained checkpoints for unconditional score-based model and image captioning model following the instructions in the below repositories:
- pretrained score-based model on CIFAR-10: https://github.com/yang-song/score_sde
- pretrained image captioning model on MSCOCO: https://github.com/ruotianluo/ImageCaptioning.pytorch

Finally run the conditional sampling ```python main.py coco_workdir coco_ve sample --sample_mode cond```.

This repository is built based on https://github.com/yang-song/score_sde and https://github.com/tim-kuechler/SemanticSynthesisForScoreBasedModels.
