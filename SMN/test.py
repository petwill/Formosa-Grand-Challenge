#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import copy
from model import SMN_model
import pickle
import config
args = config.parse_arguments()
args.fac = int(args.use_bidirection) + 1

dct = open(args.vocab_file, 'r').read().splitlines()
dct = dict([[word, i] for i, word in enumerate(dct)])
args.vocab_size = len(dct)
wordvec = np.load(args.wordvec_file)
assert args.vocab_size == wordvec.shape[0]
args.embed_dim = wordvec.shape[1]

args.pred_file += \
  '_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.use_bidirection,
                                                         args.rnn_type,
                                                         args.sent_max_len,
                                                         args.matching_dim,
                                                         args.hidden_size,
                                                         args.batch_size,
                                                         args.rnn_layer_num,
                                                         args.cnn_layer_num,
                                                         args.max_grad_norm,
                                                         args.keep_prob,
                                                         args.init_scale,
                                                         args.filter_num,
                                                         args.kernel_size,
                                                         args.pool_size,
                                                         args.learning_rate)

with tf.Graph().as_default():
  initializer = tf.random_uniform_initializer(-args.init_scale, args.init_scale)
  with tf.name_scope('test'):
    test_args = copy.deepcopy(args)
    test_args.mode = 2
    test_args.batch_size = args.option_num
    with tf.variable_scope('model', reuse=None):
      test_model = SMN_model(args=test_args)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.graph_options.optimizer_options.global_jit_level =\
    tf.OptimizerOptions.ON_1
  sv = tf.train.Supervisor(logdir=args.log_dir)


  with sv.managed_session(config=config) as sess:
    sess.run(test_model.embed_init, feed_dict={test_model.embed : wordvec})
    global_step = sess.run(test_model.step)
    print('global step = {}'.format(global_step))
    print('model_mode= ',sess.run(test_model.mode))
    args.pred_file += '_{}.txt'.format(global_step)
    score_array = np.zeros([args.test_num, args.option_num])
    with open(args.pred_file, 'w') as f:
      f.write('id,ans\n')
      for i in range(1, args.test_num+1):
        prediction = sess.run(test_model.pred)
        f.write('{},{}\n'.format(i, np.argmax(prediction)))
        if (args.get_weights):
          score_array[i-1] = prediction


    if args.get_weights:
      with open(args.weights_file,'wb+') as weights_file:
        pickle.dump(score_array,weights_file)

