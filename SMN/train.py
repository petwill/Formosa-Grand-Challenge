#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import copy
from model import SMN_model
import config

def run_epoch(sess, model):
  '''Runs the model for one epoch'''
  fetches = {}
  fetches['loss'] = model.loss
  if model.is_train():
    fetches['eval'] = model.eval
  vals = sess.run(fetches)
  return vals['loss']

if __name__ == '__main__':
  args = config.parse_arguments()
  args.fac = int(args.use_bidirection) + 1

  dct = open(args.vocab_file, 'r').read().splitlines()
  dct = dict([[word, i] for i, word in enumerate(dct)])
  args.vocab_size = len(dct)
  wordvec = np.load(args.wordvec_file)
  assert args.vocab_size == wordvec.shape[0]
  args.embed_dim = wordvec.shape[1]

  with tf.Graph().as_default() as graph:
    initializer = tf.random_uniform_initializer(-args.init_scale, args.init_scale)
    with tf.name_scope('train'):
      train_args = copy.deepcopy(args)
      valid_args = copy.deepcopy(args)
      train_args.mode = 0
      valid_args.mode = 1
      with tf.variable_scope('model', reuse=None, initializer=initializer) as scope:
        train_model = SMN_model(args=train_args)
        scope.reuse_variables()
        val_model = SMN_model(args=valid_args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.optimizer_options.global_jit_level =\
      tf.OptimizerOptions.ON_1
    sv = tf.train.Supervisor(logdir=args.log_dir,
                             save_model_secs=args.save_model_secs)

    with sv.managed_session(config=config) as sess:
      sess.run(train_model.embed_init, feed_dict={train_model.embed : wordvec})
      global_step = sess.run(train_model.step)
      for i in range(global_step+1, args.max_epoch+1):
        train_loss = run_epoch(sess, train_model)
        if i % args.info_epoch == 0:
          acc=sess.run(val_model.pred)
          print('Epoch: %d Training Loss: %.5f, Validation_Accuracy: %.5f'%(i, train_loss,acc))
