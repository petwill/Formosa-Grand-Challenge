import tensorflow as tf

class SMN_model():
  '''
  Sequential Matching Network
  '''
  def __init__(self, args):
    self.__dict__ = args.__dict__.copy()
    rnn_cells, W_E = self.initialize()
    self._step = tf.contrib.framework.get_or_create_global_step()
    if self.is_train():
      slens, sents, bslens, bsents = self.get_input()
      with tf.variable_scope('build'):
        self._loss =\
          self.build_model(rnn_cells, W_E, sents[:-1], slens[:-1], sents[-1],
                           slens[-1], 1)
        tf.get_variable_scope().reuse_variables()
        self._loss +=\
          self.build_model(rnn_cells, W_E, sents[:-1], slens[:-1], bsents,
                           bslens, 0)
        self._loss /= (2 * self.batch_size)
      self._eval = self.optimize(self._loss)
      _vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      for _var in _vars:
        name = '  '.join('{}'.format(v) for v in _var.name.split('/'))
        print('{:85} {}'.format(name, _var.get_shape()))
    elif self.is_test():
      slens, sents = self.get_input()
      with tf.variable_scope('build'):
        self._pred = self.build_model(rnn_cells, W_E, sents[:-1], slens[:-1], sents[-1],
                         slens[-1], 1)
    else: # validation
      slens, sents, bslens, bsents = self.get_input()
      with tf.variable_scope('build'):
        pos_pred = self.build_model(rnn_cells, W_E, sents[:-1], slens[:-1], sents[-1],
                         slens[-1], 1)
        tf.get_variable_scope().reuse_variables()
        neg_pred = self.build_model(rnn_cells, W_E, sents[:-1], slens[:-1], bsents,
                           bslens, 0)

        full_pred = tf.argmax(tf.concat([neg_pred, pos_pred], axis=1), axis=1)  # shape:[batch_size, 1]
        self._pred = tf.reduce_mean(tf.cast(full_pred,tf.float32))

  def build_model(self, rnn_cells, W_E, Us, U_lens, R, R_len, tar):
    Uhs, Vs = self.first_part(rnn_cells, W_E, Us, U_lens, R, R_len)

    Hs = self.second_part(rnn_cells, Uhs, Vs)
    G = self.third_part(Uhs, Hs)

    if  self.is_train():
      return self.calc_loss(G, tar)
    elif self.is_test():
      return G[:, 1]
    else:
      return tf.expand_dims(G[:, 1],axis=1)

  def initialize(self):
    if self.rnn_type == 0:#LSTM
      def unit_cell(fac):
        return tf.contrib.rnn.LSTMCell(fac, use_peepholes=True)
    elif self.rnn_type == 1:#GRU
      def unit_cell(fac):
        return tf.contrib.rnn.GRUCell(fac)
    rnn_cell = unit_cell
    #dropout layer
    if not self.is_test() and self.keep_prob < 1:
      def rnn_cell(fac):
        return tf.contrib.rnn.DropoutWrapper(unit_cell(fac),
                                             output_keep_prob=self.keep_prob)
    def rnn_cells(fac):
      return tf.contrib.rnn.MultiRNNCell([rnn_cell(fac)
                                          for _ in range(self.rnn_layer_num)])
    W_E = tf.get_variable('W_E', [self.vocab_size, self.embed_dim],
                          dtype=tf.float32)
    self.embed = tf.placeholder(tf.float32, [self.vocab_size, self.embed_dim])
    self.embed_init = W_E.assign(self.embed)
    return rnn_cells, W_E

  def get_input(self):
    #feed in data in batches
    if self.is_train():
      slen, sent, bslen, bsent = self.get_single_example()
      slens, sents, bslens, bsents =\
        tf.train.shuffle_batch([slen, sent, bslen, bsent], self.batch_size,
                               capacity=5000, min_after_dequeue=1000)
      slens = tf.transpose(slens)
      sents = tf.transpose(sents, [1, 0, 2])
      return slens, sents, bslens, bsents
    elif self.is_test():
      slen, sent = self.get_single_example()
      slens, sents = tf.train.batch([slen, sent], self.batch_size)
      slens = tf.transpose(slens)
      sents = tf.transpose(sents, [1, 0, 2])
      return slens, sents
    else:
      slen, sent, bslen, bsent = self.get_single_example()
      slens, sents, bslens, bsents =\
        tf.train.shuffle_batch([slen, sent, bslen, bsent], self.batch_size,
                               capacity=5000, min_after_dequeue=1000)
      slens = tf.transpose(slens)
      sents = tf.transpose(sents, [1, 0, 2])
      return slens, sents, bslens, bsents

  def is_train(self): return self.mode == 0
  def is_valid(self): return self.mode == 1
  def is_test(self): return self.mode == 2

  def rnn_output(self, inp, seq_len, f_cells, b_cells, scope):
    if self.use_bidirection:
      oup, st = tf.nn.bidirectional_dynamic_rnn(f_cells, b_cells, inp, seq_len,
                                               dtype=tf.float32, scope=scope)
      oup = tf.concat(oup, -1)
      st = tf.concat(st, -1)
    else:
      oup, st =\
        tf.nn.dynamic_rnn(f_cells, inp, seq_len, dtype=tf.float32, scope=scope)
    oup =\
      tf.reshape(oup, [self.batch_size, -1, self.fac * f_cells.state_size[0]])
    return oup, st[-1]

  def first_part(self, rnn_cells, W_E, Us, U_lens, R, R_len):
    '''
    Utterance-Response Matching
    '''
    f1_cells = rnn_cells(self.hidden_size)
    if self.use_bidirection:
      b1_cells = rnn_cells(self.hidden_size)
    else:
      b1_cells = None

    A = tf.get_variable('A', [1, self.fac * self.hidden_size,
                              self.fac * self.hidden_size], dtype=tf.float32)
    A = tf.tile(A, [self.batch_size, 1, 1])
    R1 = R2 = tf.nn.embedding_lookup(W_E, R)
    R1 = tf.transpose(R1, [0, 2, 1])
    R2, Rh = self.rnn_output(R2, R_len, f1_cells, b1_cells, 'first')
    R2 = tf.transpose(R2, [0, 2, 1])

    def conv2d(inp):
      return tf.layers.conv2d(inp, self.filter_num, [self.kernel_size] * 2)
    def maxpool2d(inp):
      return tf.layers.max_pooling2d(inp, [self.pool_size] * 2, self.pool_size)
    Uhs, Vs = [], []
    for i in range(self.dial_len-1):
      Ui1 = Ui2 = tf.nn.embedding_lookup(W_E, Us[i])
      Ui2, Uh = self.rnn_output(Ui2, U_lens[i], f1_cells, b1_cells, 'first')
      Uhs.append(Uh)
      #if self.is_train() and self.keep_prob < 1:
      #  Ui = tf.nn.dropout(Ui, self.keep_prob)
      M1 = Ui1 @ R1
      M2 = Ui2 @ A @ R2
      M = tf.stack([M1, M2], 3)
      with tf.variable_scope('cnn', reuse=(i>0)):
        for j in range(self.cnn_layer_num):
          M = maxpool2d(conv2d(M))
        M = tf.reshape(M, [self.batch_size, -1])
        V = tf.layers.dense(M, self.fac * self.matching_dim)
      Vs.append(V)
    Uhs = tf.stack(Uhs, 1)
    Vs = tf.stack(Vs, 1)
    return Uhs, Vs

  def second_part(self, rnn_cells, Uhs, Vs):
    '''
    Matching Accumulation
    '''
    f2_cells = rnn_cells(self.matching_dim)
    if self.use_bidirection:
      b2_cells = rnn_cells(self.matching_dim)
    else:
      b2_cells = None
    Hs, _ = self.rnn_output(Vs, [self.dial_len-1] * self.batch_size,
                                 f2_cells, b2_cells, 'second')
    return Hs

  def third_part(self, Uhs, Hs):
    '''
    Matching Prediction
    '''
    W11 =\
      tf.get_variable('W11', [1, self.fac * self.hidden_size,
                              self.fac * self.matching_dim], dtype=tf.float32)
    W11 = tf.tile(W11, [self.batch_size, 1, 1])
    W12 =\
      tf.get_variable('W12', [1, self.fac * self.matching_dim,
                              self.fac * self.matching_dim], dtype=tf.float32)
    W12 = tf.tile(W12, [self.batch_size, 1, 1])
    W2 = tf.get_variable('W2', [self.fac * self.matching_dim, 2],
                         dtype=tf.float32)
    B1 = tf.get_variable('B1', [self.fac * self.matching_dim],
                         dtype=tf.float32)
    B2 = tf.get_variable('B2', [2], dtype=tf.float32)
    Ts = tf.get_variable('Ts', [1, self.fac * self.matching_dim, 1])
    Ts = tf.tile(Ts, [self.batch_size, 1, 1])

    Ti = tf.tanh(Uhs @ W11 + Hs @ W12 + B1)
    Ai = tf.nn.softmax(Ti @ Ts, 1)
    L = tf.reduce_sum(Hs * Ai, 1)
    G = L @ W2 + B2
    return G

  def calc_loss(self, G, tar):
    labels = tf.one_hot([tar] * self.batch_size, 2)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=G)
    return tf.reduce_sum(loss)

  def optimize(self, loss):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
               self.max_grad_norm)
    self.learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                    self.step,
                                                    self.decay_steps,
                                                    self.decay_rate)
    opt = tf.train.AdamOptimizer(self.learning_rate)
    return opt.apply_gradients(zip(grads, tvars), global_step=self._step)

  def get_single_example(self):
    '''get one example from TFRecorder file using tf default queue runner'''
    if self.is_test():
      f_queue0 = tf.train.string_input_producer([self.test_file])
    else:
      if self.is_train():
        f_queue0 = tf.train.string_input_producer([self.train_file])
      else:
        f_queue0 = tf.train.string_input_producer([self.valid_file])
      f_queue1 = tf.train.string_input_producer([self.all_sents_file])
      reader1 = tf.TFRecordReader()
      _, serialized_example1 = reader1.read(f_queue1)

    reader0 = tf.TFRecordReader()
    _, serialized_example0 = reader0.read(f_queue0)

    features0 = {}
    for i in range(self.dial_len):
      features0['len'+str(i)] = tf.FixedLenFeature([1], tf.int64)
      features0['sent'+str(i)] = tf.FixedLenFeature([self.sent_max_len],
                                                    tf.int64)
    feat = tf.parse_single_example(serialized_example0, features=features0)
    sent, slen = [], []
    for i in range(self.dial_len):
      sent.append(feat['sent'+str(i)])
      slen.append(feat['len'+str(i)][0])

    if self.is_test():
      return slen, sent
    elif self.is_train():
      features1 = {'len' : tf.FixedLenFeature([1], tf.int64),
                   'sent' : tf.FixedLenFeature([self.sent_max_len], tf.int64)}
      bsent = tf.parse_single_example(serialized_example1, features=features1)
      return slen, sent, bsent['len'][0], bsent['sent']
    else:
      features1 = {'len' : tf.FixedLenFeature([1], tf.int64),
                   'sent' : tf.FixedLenFeature([self.sent_max_len], tf.int64)}
      bsent = tf.parse_single_example(serialized_example1, features=features1)
      return slen, sent, bsent['len'][0], bsent['sent']

  @property
  def pred(self): return self._pred
  @property
  def loss(self): return self._loss
  @property
  def eval(self): return self._eval
  @property
  def step(self): return self._step
