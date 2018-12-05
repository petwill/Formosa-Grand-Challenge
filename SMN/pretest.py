#!/usr/bin/python3

import os
import csv
import argparse
import tensorflow as tf
from collections import defaultdict

dft_input_file = 'data/test.txt'
dft_output_file = 'data/test.tfr'
dft_vocab_file = 'data/vocab.txt'
dft_jieba_dict = 'data/dict.txt.big'
dft_dial_len = 4
dft_option_num = 6
dft_sent_max_len = 29

parser = argparse.ArgumentParser(description='parse cleaned data into the '
                                             'format of TFRecorder file.')
parser.add_argument('-nj', '--no_jieba', action='store_true',
                    help='don\'t use jieba (every chinese character is a word) '
                    '(default:False)')
parser.add_argument('-on', '--option_num', type=int,
                    default=dft_option_num, help='number of options per '
                    'question (default:%(default)s)')
parser.add_argument('-jd', '--jieba_dict', type=str, default=dft_jieba_dict,
                    help='the dictionary for jieba (default: %(default)s)')
parser.add_argument('-if', '--input_file', type=str, default=dft_input_file,
                    help='input testing file (default: %(default)s)')
parser.add_argument('-of', '--output_file', type=str, default=dft_output_file,
                    help='the output TFRecord file (default: %(default)s)')
parser.add_argument('-vf', '--vocab_file', type=str, default=dft_vocab_file,
                    help='input vocabulary file (default: %(default)s)')
parser.add_argument('-dl', '--dial_len', type=int,
                    default=dft_dial_len, help='output dialogue '
                    ' length for testing (default: %(default)s)')
parser.add_argument('-sml', '--sent_max_len', type=int,
                    default=dft_sent_max_len, help='sents that are too '
                    'long are truncated (default: %(default)s)')
args = parser.parse_args()

if not args.no_jieba:
  import jieba
  jieba.set_dictionary(args.jieba_dict)

def cut(s):
  if not args.no_jieba:
    return list(jieba.cut(s))
  else:
    return list(s)

print('building dictionary...')
PAD, BOS, EOS, UNK = 0, 1, 2, 3
vocabs = open(args.vocab_file, 'r').read().splitlines()
dct = defaultdict(lambda: UNK)
for i, vocab in enumerate(vocabs):
  dct[vocab] = i

print('parsing testing file...')
def process(s):
  rtns = s.split('\t')
  rtns = [cut(rtn[rtn.find(':')+1 : ]) for rtn in rtns]
  return rtns
def IntList(val):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=val))
num_ex, unk_cnt, tot_cnt = 0, 0, 0
writer = tf.python_io.TFRecordWriter(args.output_file)
with open(args.input_file, 'r') as f:
  csvf = csv.reader(f)
  next(csvf)
  for row in csvf:
    Us = process(row[1])
    Rs = process(row[2])
    Us = [[]] * (args.dial_len - 1 - len(Us)) + Us
    assert len(Us)+1 == args.dial_len
    for i in range(args.option_num):
      URs = Us + [Rs[i]]
      assert len(URs) == args.dial_len
      feat = {}
      for j, U in enumerate(URs):
        word_ids = [BOS] + [dct[word] for word in U] + [EOS]

        assert len(word_ids) <= args.sent_max_len

        for word_id in word_ids:
          if word_id == UNK:
            unk_cnt += 1
          else:
            tot_cnt += 1

        feat['len'+str(j)] = IntList([len(word_ids)])
        word_ids += [PAD] * (args.sent_max_len - len(word_ids))
        assert len(word_ids) == args.sent_max_len
        feat['sent'+str(j)] = IntList(word_ids)
      example = tf.train.Example(features=tf.train.Features(feature=feat))
      num_ex += 1
      serialized = example.SerializeToString()
      writer.write(serialized)
writer.close()
print('number of examples = {}'.format(num_ex))
print('unk / tot = {}'.format(unk_cnt / tot_cnt))
