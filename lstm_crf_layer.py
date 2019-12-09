# encoding=utf-8

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf
class BLSTM_CRF(object):
    def __init__(self, embedded_chars, hidden_unit, cell_type,num_layers, droupout_rate,
                 initializers,num_labels, seq_length, labels, lengths, is_training):
        """
        BLSTM-CRF 网络
        :param embedded_chars: Fine-tuning embedding input
        :param hidden_unit: LSTM的隐含单元个数
        :param cell_type: RNN类型（LSTM OR GRU DICNN will be add in feature）
        :param num_layers: RNN的层数
        :param droupout_rate: droupout rate
        :param initializers: variable init class
        :param num_labels: 标签数量
        :param seq_length: 序列最大长度
        :param labels: 真实标签，这里的真实标签应该是可以不需要的啊
        :param lengths: [batch_size] 每个batch下序列的真实长度
        :param is_training: 是否是训练过程
        """
        self.hidden_unit = hidden_unit
        self.droupout_rate = droupout_rate
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedded_chars = embedded_chars
        self.initializers = initializers
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.lengths = lengths
        self.is_training = is_training

    def add_blstm_crf_layer(self):
        """
        blstm-crf网络
        :return: 
        """
        if self.is_training:
            # lstm input dropout rate set 0.5 will get best score
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.droupout_rate)
        #blstm
        lstm_output = self.blstm_layer(self.embedded_chars)
        #project
        logits = self.project_bilstm_layer(lstm_output)
        #crf，其实在这里，和多分类任务的区别就只有在以下的部分了，一个使用的是CRF，一个使用的是softmax损失函数来进行的
        #在多分类中这个里面的东西就是需要重新来进行处理的对象
        loss, trans = self.crf_layer(logits)
        # CRF decode, pred_ids 是一条最大概率的标注路径
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
        #那么在test中我们需要的是什么呢？
        return ((loss, logits, trans, pred_ids))   #在这里的loss存在一个问题啊，loss的作用是啥呢？   在这里trans传出去的话在那里会进行修改呢？会修改吗

    def add_blstm_crf_layer_print(self):
        """
        blstm-crf网络
        :return:
        """
        if self.is_training:
            # lstm input dropout rate set 0.5 will get best score
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.droupout_rate)
        #blstm
        lstm_output = self.blstm_layer(self.embedded_chars)
        #project
        logits = self.project_bilstm_layer(lstm_output)
        #crf，其实在这里，和多分类任务的区别就只有在以下的部分了，一个使用的是CRF，一个使用的是softmax损失函数来进行的
        #在多分类中这个里面的东西就是需要重新来进行处理的对象
        #zaiprint中我们需要的是什么呢？
        loss, trans = self.crf_layer(logits)    #zheli不需要loss
        # CRF decode, pred_ids 是一条最大概率的标注路径
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
        #那么在test中我们需要的是什么呢？
        return ((loss, logits, trans, pred_ids))

    def _witch_cell(self):
        """
        RNN 类型
        :return: 
        """
        cell_tmp = None
        if self.cell_type == 'lstm':
            cell_tmp = rnn.BasicLSTMCell(self.hidden_unit)
        elif self.cell_type == 'gru':
            cell_tmp = rnn.GRUCell(self.hidden_unit)
        # 是否需要进行dropout
        if self.droupout_rate is not None:
            cell_tmp = rnn.DropoutWrapper(cell_tmp, output_keep_prob=self.droupout_rate)
        return cell_tmp

    def _bi_dir_rnn(self):
        """
        双向RNN
        :return:
        """
        cell_fw = self._witch_cell()
        cell_bw = self._witch_cell()
        if self.droupout_rate is not None:
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.droupout_rate)
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.droupout_rate)
        return cell_fw, cell_bw
    def blstm_layer(self, embedding_chars):
        """
                
        :return: 
        """
        with tf.variable_scope('rnn_layer'):
            cell_fw, cell_bw = self._bi_dir_rnn()
            if self.num_layers > 1:
                cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
                cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding_chars,
                                                         dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)
        return outputs

    def project_bilstm_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.hidden_unit * 2, self.hidden_unit],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.hidden_unit], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.hidden_unit * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.hidden_unit, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss               #我们这么来考虑这个问题，这里计算了loss说明有目标函数，在这里的话如果是单纯的test的话，那么是不需要计算loss的啊
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(             #这里是不是就是相当于转移矩阵呢，一般的get_variable就是需要训练的数据
                "transitions",
                shape=[self.num_labels, self.num_labels],     #这里的维度是一个矩阵的形式
                initializer=self.initializers.xavier_initializer())
            log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(      #用到最大似然估计的优化方法
                inputs=logits,               #使用BILSTM处理之后输出转换为他要求的形状作为CRF层的输入
                tag_indices=self.labels,     #一个形状为[batch_size, max_seq_len] 的矩阵,其实就是真实标签
                transition_params=trans,      #形状为[num_tags, num_tags] 的转移矩阵
                sequence_lengths=self.lengths)   #一个形状为 [batch_size] 的向量,表示每个序列的长度
            return tf.reduce_mean(-log_likelihood), trans