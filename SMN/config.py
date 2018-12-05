import argparse

dft_rnn_type = 1 # 0: LSTM, 1: GRU
dft_sent_max_len = 29
dft_dial_len = 4
dft_matching_dim = 50
dft_hidden_size = 350
dft_batch_size = 700
dft_rnn_layer_num = 2
dft_cnn_layer_num = 1
dft_option_num = 6
dft_max_grad_norm = 2
dft_keep_prob = 1
dft_init_scale = 0.001
dft_max_epoch = 30000
dft_info_epoch = 1
dft_filter_num = 12
dft_kernel_size = 2
dft_pool_size = 2
dft_test_num = 500 # number of test data
dft_save_model_secs = 1200
dft_learning_rate = 0.0001
dft_decay_steps=10000
dft_decay_rate=0.96
dft_vocab_file = 'data/vocab.txt'
dft_wordvec_file = 'data/wv.npy'
dft_train_file = 'data/train.tfr'
dft_all_sents_file = 'data/all_sents.tfr'
dft_test_file = 'data/test.tfr'
dft_pred_file = 'submission/pred'
dft_log_dir = 'logs'
dft_valid_file = 'data/valid.tfr'
dft_valid_ratio = 0.1
dft_get_weights = True
dft_weights_file = 'SMN_weights.pickle'

def parse_arguments():

  argparser = argparse.ArgumentParser(description='Sequential Matching Network')

  argparser.add_argument('-rct', '--rnn_type', type=int, default=dft_rnn_type,
    										 help='rnn cell type: 0->LSTM, 1->GRU '
                         '(default:%(default)s)')
  argparser.add_argument('-sml', '--sent_max_len', type=int,
                         default=dft_sent_max_len, help='input sentence maximum'
                         ' length, (default:%(default)s)')
  argparser.add_argument('-dl', '--dial_len', type=int,
                         default=dft_dial_len, help='input dialogue length, '
                         '(default:%(default)s)')
  argparser.add_argument('-md', '--matching_dim', type=int,
                         default=dft_matching_dim, help='dimension '
                         'of matching vector (default:%(default)s)')
  argparser.add_argument('-hu', '--hidden_size', type=int,
                         default=dft_hidden_size, help='hidden units of rnn '
                         'cell (default:%(default)s)')
  argparser.add_argument('-bs', '--batch_size', type=int,
                         default=dft_batch_size,
                         help='batch size (default:%(default)s)')
  argparser.add_argument('-on', '--option_num', type=int,
                         default=dft_option_num,
                         help='number of options per question '
                         '(default:%(default)s)')
  argparser.add_argument('-cln', '--cnn_layer_num', type=int,
                         default=dft_cnn_layer_num,
                         help='number of cnn layers (default:%(default)s)')
  argparser.add_argument('-rln', '--rnn_layer_num', type=int,
                         default=dft_rnn_layer_num,
                         help='number of rnn layers (default:%(default)s)')
  argparser.add_argument('-fn', '--filter_num', type=int,
                         default=dft_filter_num,
                         help='number of filters (default:%(default)s)')
  argparser.add_argument('-ks', '--kernel_size', type=int,
                         default=dft_kernel_size,
                         help='size of kernel for 2d cnn (default:%(default)s)')
  argparser.add_argument('-ps', '--pool_size', type=int,
                         default=dft_pool_size,
                         help='size of max pooling (default:%(default)s)')
  argparser.add_argument('-mgn', '--max_grad_norm', type=int,
                         default=dft_max_grad_norm,
                         help='maximum gradient norm (default:%(default)s)')
  argparser.add_argument('-kp', '--keep_prob', type=float,
                         default=dft_keep_prob, help='keep probability '
                         'of dropout layer (default:%(default)s)')
  argparser.add_argument('-lr', '--learning_rate', type=float,
                         default=dft_learning_rate, help='learning rate '
                         '(default:%(default)s)')
  argparser.add_argument('-ds', '--decay_steps', type=float,
                        default=dft_decay_steps, help='decay steps '
                        '(default:%(default)s)')
  argparser.add_argument('-dr', '--decay_rate', type=float, 
                        default=dft_decay_rate, help='decay rate '
                        '(default:%(default)s)')
  argparser.add_argument('-is', '--init_scale', type=float,
                         default=dft_init_scale, help='initialization scale for'
                         ' tensorflow initializer (default:%(default)s)')
  argparser.add_argument('-me', '--max_epoch', type=int, default=dft_max_epoch,
                         help='maximum training epoch '
                         '(default:%(default)s)')
  argparser.add_argument('-ie', '--info_epoch', type=int,
                         default=dft_info_epoch, help='show training '
                         'information for each (default:%(default)s) epochs')
  argparser.add_argument('-tn', '--test_num', type=int,
                         default=dft_test_num, help='number of test '
                         'data (default:%(default)s)')
  argparser.add_argument('-sms', '--save_model_secs', type=int,
                         default=dft_save_model_secs, help='save model for '
                         'every SAVE_MODEL_SECS seconds (default:%(default)s)')
  argparser.add_argument('-vf', '--vocab_file', type=str,
                         default=dft_vocab_file, help='vocabulary filename '
                         '(default:%(default)s)')
  argparser.add_argument('-wf', '--wordvec_file', type=str,
                         default=dft_wordvec_file, help='filename for trained '
                         'word vector in numpy format (default:%(default)s)')
  argparser.add_argument('-trf', '--train_file', type=str,
                         default=dft_train_file, help='input training filename '
                         '(default:%(default)s)')
  argparser.add_argument('-asf', '--all_sents_file', type=str,
                         default=dft_all_sents_file, help='file containing all '
                         'sentences (default:%(default)s)')
  argparser.add_argument('-tef', '--test_file', type=str,
                         default=dft_test_file, help='test filename '
                         '(default:%(default)s)')
  argparser.add_argument('-pf', '--pred_file', type=str,
                         default=dft_pred_file, help='filename for submission '
                         '(default:%(default)s)')
  argparser.add_argument('-ld', '--log_dir', type=str,
                         default=dft_log_dir, help='log directory '
                         '(default:%(default)s)')
  argparser.add_argument('-ub', '--use_bidirection', action='store_true',
                          help='use bidirectional rnn (default:False)')
  argparser.add_argument('--valid_file', type=str,
                          default=dft_valid_file, help='TFR file storing validation data (default: %(default)s)')
  argparser.add_argument('--valid_ratio', type=float,
                          default=dft_valid_ratio, help='Ratio of validation data (default: %(default)s)')
  argparser.add_argument('--get_weights', type=bool,
                          default=dft_get_weights, help = 'Program will output weights of all options if True.')
  argparser.add_argument('--weights_file', type=str,
                         default=dft_weights_file, help='default name of weights file.')

  return argparser.parse_args()