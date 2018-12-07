# Formosa-Grand-Challenge
Code of [Formosa Speech Grand Challenge - Talk to AI Warm-Up Match](https://fgc.stpi.narl.org.tw/index) / 2017

Collaboration with [Daikon-Sun](https://github.com/Daikon-Sun) and [Ray-Wu](https://github.com/raywu0123)

Three methods are ensembled to perform Response Selection (in Chinese):

* [Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-based Chatbots](https://arxiv.org/abs/1612.01627), implemented in Tensorflow (refer to `SMN/`).

* RNN encoder model (not present in this repo)

* simple averaging of word vectors (refer to `avg_wordvec.py`)
