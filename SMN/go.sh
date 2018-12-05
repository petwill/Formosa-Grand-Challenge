#!/bin/bash -ex

echo "Usage : ./go.sh [pretrain] [pretest] [train] [test]"

#sorted alphabetically
all_sents_file=data/all_sents.tfr
batch_size=750
cnn_layer_num=1
dial_len=4
dial_min_len=2
dialogue_min_len=2
embed_dim=400
filter_num=12
hidden_size=500
info_epoch=1
init_scale=0.01
jieba_dict=data/dict.txt.big
keep_prob=1
kernel_size=2
learning_rate=0.003
decay_steps=20
decay_rate=0.96
log_dir=logs
matching_dim=50
max_epoch=200000
max_grad_norm=1
option_num=6
pool_size=2
pred_file=submission/pred
rnn_layer_num=1
rnn_type=1 # 0: LSTM, 1: GRU
save_model_secs=100
sent_max_len=29
test_file=data/test.tfr
train_file=data/train.tfr
use_bidirection=''
vocab_file=data/vocab.txt
vocabulary_size=20000
wordvec_file=data/wv.npy
valid_file=data/valid.tfr
valid_ratio=0.1
get_weights=True
weights_file='smn_weights.pickle'
test_num=$1 # number of test data



for var in "$@"
do
  if [ "$var" == "pretest" ]
  then
    pretest_input_file=data/test.txt
    pretest_output_file=data/test.tfr
    ./pretest.py \
      --input_file   $pretest_input_file \
      --output_file  $pretest_output_file \
      --vocab_file   $vocab_file \
      --jieba_dict   $jieba_dict \
      --dial_len     $dial_len \
      --option_num   $option_num \
      --sent_max_len $sent_max_len
  elif [ "$var" == "pretrain" ]
  then
    pretrain_input_dir=data/cleaned_data
    pretrain_output_file=data/train.tfr
    ./pretrain.py \
      --input_dir       $pretrain_input_dir \
      --output_file     $pretrain_output_file \
      --all_sents_file  $all_sents_file \
      --jieba_dict      $jieba_dict \
      --vocab_file      $vocab_file \
      --wordvec_file    $wordvec_file \
      --dial_min_len    $dial_min_len \
      --dial_len        $dial_len \
      --sent_max_len    $sent_max_len \
      --vocabulary_size $vocabulary_size \
      --embed_dim       $embed_dim\
      --valid_file      $valid_file\
      --valid_ratio      $valid_ratio
  elif [ "$var" == "train" ]
  then
    ./train.py \
      --rnn_type        $rnn_type \
      --sent_max_len    $sent_max_len \
      --dial_len        $dial_len \
      --matching_dim    $matching_dim \
      --hidden_size     $hidden_size \
      --batch_size      $batch_size \
      --rnn_layer_num   $rnn_layer_num \
      --cnn_layer_num   $cnn_layer_num \
      --option_num      $option_num \
      --max_grad_norm   $max_grad_norm \
      --keep_prob       $keep_prob \
      --init_scale      $init_scale \
      --max_epoch       $max_epoch \
      --info_epoch      $info_epoch \
      --filter_num      $filter_num \
      --kernel_size     $kernel_size \
      --pool_size       $pool_size \
      --test_num        $test_num \
      --save_model_secs $save_model_secs \
      --learning_rate   $learning_rate \
      --decay_rate      $decay_rate \
      --decay_steps     $decay_steps \
      --vocab_file      $vocab_file \
      --wordvec_file    $wordvec_file \
      --train_file      $train_file \
      --all_sents_file  $all_sents_file \
      --test_file       $test_file \
      --pred_file       $pred_file \
      --log_dir         $log_dir \
      --valid_file      $valid_file\
      --valid_ratio      $valid_ratio\
      $use_bidirection
  elif [ "$var" == "test" ]
  then
    ./test.py \
      --rnn_type        $rnn_type \
      --sent_max_len    $sent_max_len \
      --dial_len        $dial_len \
      --matching_dim    $matching_dim \
      --hidden_size     $hidden_size \
      --batch_size      $batch_size \
      --rnn_layer_num   $rnn_layer_num \
      --cnn_layer_num   $cnn_layer_num \
      --option_num      $option_num \
      --max_grad_norm   $max_grad_norm \
      --keep_prob       $keep_prob \
      --init_scale      $init_scale \
      --max_epoch       $max_epoch \
      --info_epoch      $info_epoch \
      --filter_num      $filter_num \
      --kernel_size     $kernel_size \
      --pool_size       $pool_size \
      --test_num        $test_num \
      --save_model_secs $save_model_secs \
      --learning_rate   $learning_rate \
      --decay_rate      $decay_rate \
      --decay_steps     $decay_steps \
      --vocab_file      $vocab_file \
      --wordvec_file    $wordvec_file \
      --train_file      $train_file \
      --all_sents_file  $all_sents_file \
      --test_file       $test_file \
      --pred_file       $pred_file \
      --log_dir         $log_dir \
      --get_weights     $get_weights\
      $use_bidirection
  fi
done