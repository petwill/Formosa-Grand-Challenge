#!/usr/bin/python3

import os
import re
import argparse
import tensorflow as tf
from collections import Counter, defaultdict
from tqdm import tqdm
from gensim.models import word2vec
import numpy as np
import random

dft_input_dir = 'data/cleaned_data'
dft_output_file = 'data/train.tfr'
dft_all_sents_file = 'data/all_sents.tfr'
dft_jieba_dict = 'data/dict.txt.big'
dft_vocab_file = 'data/vocab.txt'
dft_wordvec_file = 'data/wv.npy'
dft_dial_min_len = 2
dft_dial_len = 4
dft_sent_max_len = 29
dft_vocabulary_size = 16000
dft_embed_dim = 300
dft_valid_file = 'data/valid.tfr'
dft_valid_ratio = 0.1

parser = argparse.ArgumentParser(description='parse cleaned data into the '
                                             'format of TFRecorder file.')
parser.add_argument('-nj', '--no_jieba', action='store_true',
                    help='don\'t use jieba (every chinese character is a word) '
                    '(default:False)')
parser.add_argument('-id', '--input_dir', type=str, default=dft_input_dir,
                    help='the root directory storing the all training data '
                    '(default: %(default)s)')
parser.add_argument('-of', '--output_file', type=str, default=dft_output_file,
                    help='the output TFRecord file (default: %(default)s)')
parser.add_argument('-asf', '--all_sents_file', type=str,
                    default=dft_all_sents_file, help='TFR file storing all '
                    'sentences (default: %(default)s)')
parser.add_argument('-vf', '--vocab_file', type=str, default=dft_vocab_file,
                    help='output filename of all selected vocabularies '
                    '(default: %(default)s)')
parser.add_argument('-wf', '--wordvec_file', type=str, default=dft_wordvec_file,
                    help='output filename of word vector in numpy format '
                    '(default: %(default)s)')
parser.add_argument('-jd', '--jieba_dict', type=str, default=dft_jieba_dict,
                    help='the dictionary for jieba (default: %(default)s)')
parser.add_argument('-dml', '--dial_min_len', type=int,
                    default=dft_dial_min_len, help='dialogue that are too '
                    'short is discarded (default: %(default)s)')
parser.add_argument('-dl', '--dial_len', type=int,
                    default=dft_dial_len, help='output dialogue '
                    ' length for training (default: %(default)s)')
parser.add_argument('-sml', '--sent_max_len', type=int,
                    default=dft_sent_max_len, help='sentences that are too '
                    'long are considered a break point between two dialogue'
                    ' (default: %(default)s)')
parser.add_argument('-vs', '--vocabulary_size', type=int,
                    default=dft_vocabulary_size, help='vocabulary size, '
                    'unknown words are replaced by UNK (default: %(default)s)')
parser.add_argument('-ed', '--embed_dim', type=int,
                    default=dft_embed_dim, help='embedding dimension of '
                    'word vector (default: %(default)s)')
parser.add_argument('--valid_file', type=str,
                    default=dft_valid_file, help='TFR file storing validation data (default: %(default)s)')
parser.add_argument('--valid_ratio', type=float,
                    default=dft_valid_ratio, help='Ratio of validation data (default: %(default)s)')
args = parser.parse_args()

if not args.no_jieba:
  import jieba
  jieba.set_dictionary(args.jieba_dict)

def cut(s):
  if not args.no_jieba:
    return list(jieba.cut(s))
  else:
    return list(s)

print('reading cleaned data...')

all_dials, all_sents, all_words = [], [], []
nulls = [[] * (args.dial_len-2)]
all_files = os.listdir(args.input_dir)
remove_patt = '[^\n\u4e00-\u9fff]+'

for filename in tqdm(all_files):
  dials = open(os.path.join(args.input_dir, filename), 'r').read()
  #dials = re.sub(remove_patt, ' ', dials)
  # remove trailing whitespaces
  #dials = re.sub('[ ]+\n', '\n', dials).split('\n\n')
  dials = dials.split('\n\n')

  dials = [dial.split('\n') for dial in dials]
  dials = [[cut(sent) for sent in dial] for dial in dials]

  for sents in dials:
    for sent in sents:
      all_sents.append(['BOS'] + sent + ['EOS'])
      all_words.extend(sent)

  for sents in dials:
    if len(sents) > args.dial_min_len:
      l = 0
      for i, sent in enumerate(sents):
        if len(sent) > args.sent_max_len-2:
          if l >= 1:
            all_dials.append(nulls + sents[i-l : i])
          l = 0
        else:
          l += 1
      if l >= 1:
        all_dials.append(nulls + sents[-l : ])
print('len = {}'.format(len(all_sents)))

print('training word2vec...')
model = word2vec.Word2Vec(all_sents, size=args.embed_dim)
wv = []

print('building dictionary...')
vocabs = Counter(all_words).most_common(args.vocabulary_size)
PAD, BOS, EOS, UNK = 0, 1, 2, 3
vocabs = ['PAD', 'BOS', 'EOS', 'UNK'] + [vocab[0] for vocab in vocabs]
dct = defaultdict(lambda: UNK)
with open(args.vocab_file, 'w') as f:
  for i, vocab in enumerate(vocabs):
    f.write(vocab + '\n')
    dct[vocab] = i
    if vocab == 'PAD' or vocab == 'UNK':
      wv.append(np.zeros((args.embed_dim)))
    else:
      wv.append(model.wv[vocab])
wv = np.array(wv)
print('size of wv = {}'.format(wv.shape))
np.save(args.wordvec_file, wv)

print('writing training tfrdata...')
def IntList(val):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=val))
num_val, num_train, num_ex, unk_cnt, tot_cnt = 0, 0, 0, 0, 0
writer = tf.python_io.TFRecordWriter(args.output_file)
writer_val = tf.python_io.TFRecordWriter(args.valid_file)
for sents in tqdm(all_dials):
  for j in range(0, len(sents)-args.dial_len+1):
    feat = {}
    for k in range(j, j+args.dial_len):
      word_ids = [BOS] + [dct[word] for word in sents[k]] + [EOS]

      for word_id in word_ids:
        if word_id == UNK:
          unk_cnt += 1
        else:
          tot_cnt += 1

      word_ids += [PAD] * (args.sent_max_len - len(word_ids))
      assert len(word_ids) == args.sent_max_len
      feat['len'+str(k-j)] = IntList([len(sents[k])+2])
      feat['sent'+str(k-j)] = IntList(word_ids)
    example = tf.train.Example(features=tf.train.Features(feature=feat))
    serialized = example.SerializeToString()
    if random.random() > args.valid_ratio :
      writer.write(serialized)
      num_train+=1
    else:
      num_val+=1
      writer_val.write(serialized)

    num_ex += 1
writer.close()

print('number of training examples = {}'.format(num_train))
print('number of validation examples = {}'.format(num_val))
print('number of total examples = {}'.format(num_ex))
print('unk / tot = {}'.format(unk_cnt / tot_cnt))

print('writing all-sentence tfrdata...')
writer = tf.python_io.TFRecordWriter(args.all_sents_file)
for sent in tqdm(all_sents):
  word_ids = [dct[word] for word in sent]
  if len(word_ids) > args.sent_max_len:
    continue
  feat = {}
  feat['len'] = IntList([len(word_ids)])
  word_ids += [PAD] * (args.sent_max_len - len(word_ids))
  assert len(word_ids) == args.sent_max_len
  feat['sent'] = IntList(word_ids)
  example = tf.train.Example(features=tf.train.Features(feature=feat))
  serialized = example.SerializeToString()
  writer.write(serialized)
writer.close()
