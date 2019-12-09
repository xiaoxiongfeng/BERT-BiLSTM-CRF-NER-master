#! usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import request
import json

from flask import Flask, render_template, request

import logging

app = Flask(__name__, template_folder= "")
app.config['JSON_AS_ASCII'] = False

import tensorflow as tf
import codecs

from bert import modeling
from bert import optimization
from bert import tokenization
from lstm_crf_layer import BLSTM_CRF
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import estimator

import tf_metrics
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.flags

FLAGS = flags.FLAGS



bert_path = 'checkpoint/chinese_L-12_H-768_A-12'
root_path = ''

flags.DEFINE_string(
    "data_dir", os.path.join(root_path, 'NERdata'),
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", os.path.join(bert_path, 'bert_config.json'),
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", 'ner', "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", os.path.join(root_path, 'output_my'),
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", os.path.join(bert_path, 'bert_model.ckpt'),
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_boolean('clean', True, 'remove the files which created by last training')

#flags.DEFINE_bool("do_train", True, "Whether to run training.")
flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool("do_print", True, "Whether to run the model in inference mode on your data and return its answer.")

flags.DEFINE_integer("train_batch_size", 64, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 5.0, "Total number of training epochs to perform.")
flags.DEFINE_float('droupout_rate', 0.5, 'Dropout rate')
flags.DEFINE_float('clip', 5, 'Gradient clip')
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", os.path.join(bert_path, 'vocab.txt'),
                    "The vocabulary file that the BERT model was trained on.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")
flags.DEFINE_string('data_config_path', os.path.join(root_path, 'data.conf'),
                    'data config file, which save train and dev config')
# lstm parame
flags.DEFINE_integer('lstm_size', 128, 'size of lstm units')
flags.DEFINE_integer('num_layers', 1, 'number of rnn layers, default is 1')
flags.DEFINE_string('cell', 'lstm', 'which rnn cell used')




class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class NerProcessor(DataProcessor):      #在这里需要更改训练、测试等工作！
    def get_print_examples(self, data_print):
        examples = []
        guid = "%s-%s" % ("print", 0)
        text = tokenization.convert_to_unicode(data_print)
        print("myline0", text)
        vir = "I_videoName " * len(data_print.split(" "))
        label = tokenization.convert_to_unicode(vir)
        print("mylabel", label)
        examples.append(InputExample(guid=guid, text=text, label=label))
        return examples
    def get_labels(self):
        return ["O", "I_channelName", "B_channelName", "I_name", "B_name", "B_videoName",  "I_videoName", 'B_channelNo', 'B_endTime','I_endTime', 'I_channelNo', "B_startDate", "I_startDate", "B_startTime", "I_startTime", "B_persons", "I_persons", "B_area", "I_area", "B_category", "I_category", "B_modifier", "I_modifier", "B_season", "I_season", "B_episode", "I_episode", "B_startyear", "I_startyear", "B_endyear", "I_endyear", "X", "[CLS]", "[SEP]"]

def write_tokens(tokens, mode):
    """
    将序列解析结果写入到文件中
    只在mode=test的时候启用
    :param tokens:
    :param mode:
    :return:
    """
    if mode == "test" or mode == "print":
        path = os.path.join(FLAGS.output_dir, "token_" + mode + ".txt")    #zai这里有token_test.txt的信息
        wf = codecs.open(path, 'a', encoding='utf-8')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'wb') as w:
        pickle.dump(label_map, w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        # 分词，如果是中文，就是分字， 在这里会出现错误， 也就是可能数据的长度和标签的长度不一致吧
        #在这里还是要区分长度的不一致的情况的啊，在test的时候是不需要这些信息的啊
        try:
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:  # 一般不会出现else
                    labels.append("X")
        except:
            print("This is list!", textlist, label_list)
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    try:
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
    except:
        print("duild label err")
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 打印部分样本数据信息
    if ex_index < 1:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    if ex_index < 1:
        print(feature)
    # mode='test'的时候才有效
    write_tokens(ntokens, mode)     #这里就是单纯地输出token的结果而已
    return feature

def filed_based_convert_examples_to_features(      #也就是说在这里会生成数据，并写入TFRecord
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None
):
    """
    将数据转化为TF_Record 结构，作为模型数据输入
    :param examples:  样本
    :param label_list:标签list
    :param max_seq_length: 预先设定的最大序列长度
    :param tokenizer: tokenizer 对象
    :param output_file: tf.record 输出路径
    :param mode:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    # 遍历训练数据,
    #在这里将训练数据与测试数据进行分离吧，这样的话可以在测试中就不用加入label了，直接输出想要的答案就行了啊
    if mode !=  "print":
        for (ex_index, example) in enumerate(examples):
            if ex_index % 5000 == 0:
                tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
            # 对于每一个训练样本,
            feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)    #这里不能直接使用feature吗？为什么还需要使用create_int_feature
            features["input_mask"] = create_int_feature(feature.input_mask)   #也就是说这一套就是要写入TFRecord文件
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature(feature.label_ids)
            if ex_index < 1:
                print("1111", features, "22222")
            # features["label_mask"] = create_int_feature(feature.label_mask)
            # tf.train.Example/Feature 是一种协议，方便序列化？？？
            # 将一个样例转换为Example Protocol Buffer, 并将所有的信息写入这个数据结构
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))     #这里的tf.train.Features其实就是和tf.train.Example配合
            writer.write(tf_example.SerializeToString())    # 将一个Example写入TFRecord文件,那么在实际的测试的方法中，是不是可以直接使用，就不用先写入文件了呢？
    else:
        for (ex_index, example) in enumerate(examples):
            feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode)
            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)    #这里不能直接使用feature吗？为什么还需要使用create_int_feature
            features["input_mask"] = create_int_feature(feature.input_mask)   #也就是说这一套就是要写入TFRecord文件
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature(feature.label_ids)
            if ex_index < 1:
                print("1111", features, "22222")
            # features["label_mask"] = create_int_feature(feature.label_mask)
            # tf.train.Example/Feature 是一种协议，方便序列化？？？
            # 将一个样例转换为Example Protocol Buffer, 并将所有的信息写入这个数据结构
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))     #这里的tf.train.Features其实就是和tf.train.Example配合
            writer.write(tf_example.SerializeToString())    # 将一个Example写入TFRecord文件,那么在实际的测试的方法中，是不是可以直接使用，就不用先写入文件了呢？


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder, model = None):     #2.读取record 数据，组成batch
    if model != "print":
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
            # "label_ids":tf.VarLenFeature(tf.int64),
            # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
        }
    else:
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
            # "label_ids":tf.VarLenFeature(tf.int64),
            # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
        }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    """
    创建X模型
    :param bert_config: bert 配置
    :param is_training:
    :param input_ids: 数据的idx 表示
    :param input_mask:
    :param segment_ids:
    :param labels: 标签的idx 表示
    :param num_labels: 类别数量
    :param use_one_hot_embeddings:
    :return:
    """
    # 使用数据加载BertModel,获取对应的字embedding
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )
    # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
    embedding = model.get_sequence_output()
    print(embedding.shape, embedding.shape[1], embedding.shape[1].value)
    #(64, 128, 768) 128 128
    max_seq_length = embedding.shape[1].value

    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度

    #以下就是根据BERT的输入构建BLSTM的模型的过程了，这一部分具体在lstm_crf_layer中进行了定义
    blstm_crf = BLSTM_CRF(embedded_chars=embedding, hidden_unit=FLAGS.lstm_size, cell_type=FLAGS.cell, num_layers=FLAGS.num_layers,
                          droupout_rate=FLAGS.droupout_rate, initializers=initializers, num_labels=num_labels,
                          seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)
    rst = blstm_crf.add_blstm_crf_layer()
    return rst

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    #这个model_fn_builder是为了构造代码中默认调用的model_fn函数服务的，为了使用其他的参数
    #也就是通过model_fn_builder传入参数，但是返回的是一个构造的函数，将函数作为返回值来进行处理的
    """
    构建模型
    :param bert_config:
    :param num_labels:
    :param init_checkpoint:
    :param learning_rate:
    :param num_train_steps:
    :param num_warmup_steps:
    :param use_tpu:
    :param use_one_hot_embeddings:
    :return:
    """

    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        print('shape of input_ids', input_ids.shape)
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)    #注意，在这里已经定义了is_training，所以应该是在这里进行的规范
        #如果不是is_training那么应该就不需要使用label了吧
        # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
        #在这里把test区分开吧
        if is_training:
            (total_loss, logits, trans, pred_ids) = create_model(      #这里输出的是pre_ids
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                num_labels, use_one_hot_embeddings)
        else:
            (total_loss, logits, trans, pred_ids) = create_model(  # 这里输出的是pre_ids
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                num_labels, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        # 加载BERT模型
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")

        # 打印加载模型的参数
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(    #这里是构建，#Estimator要求返回一个EstimatorSpec对象
                #创建优化器并使用tf.contrib.tpu.TPUEstimatorSpec封装优化器和loss
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)  # 钩子，这里用来将BERT中的参数作为我们模型的初始值
        elif mode == tf.estimator.ModeKeys.EVAL:
            # 针对NER ,进行了修改
            def metric_fn(label_ids, logits, trans):
                # 首先对结果进行维特比解码
                # crf 解码

                weight = tf.sequence_mask(FLAGS.max_seq_length)
                precision = tf_metrics.precision(label_ids, pred_ids, num_labels, [2, 3, 4, 5, 6, 7], weight)
                recall = tf_metrics.recall(label_ids, pred_ids, num_labels, [2, 3, 4, 5, 6, 7], weight)
                f = tf_metrics.f1(label_ids, pred_ids, num_labels, [2, 3, 4, 5, 6, 7], weight)

                return {
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f": f,
                    # "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [label_ids, logits, trans])
            # eval_metrics = (metric_fn, [label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                #loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)  #
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=pred_ids,
                scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn   #返回回调函数

def myServer(myInput):
    myInput = " ".join(list(myInput))
    ans = {}
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "ner": NerProcessor
    }
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    task_name = FLAGS.task_name.lower()
    processor = processors[task_name]()
    label_list = processor.get_labels()     #在这里得到labels

    tokenizer = tokenization.FullTokenizer(          #所以这里的tokenizer就是提供的一个分词的API了
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    num_train_steps = None
    num_warmup_steps = None

    if os.path.exists(FLAGS.data_config_path):
        with codecs.open(FLAGS.data_config_path) as fd:
            data_config = json.load(fd)
    else:
        data_config = {}

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    #定义estimator封装器，这里的estimator封装器究竟是什么样的东西呢？
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    # 保存数据的配置文件，避免在以后的训练过程中多次读取训练以及测试数据集，消耗时间
    if not os.path.exists(FLAGS.data_config_path):
        with codecs.open(FLAGS.data_config_path, 'a', encoding='utf-8') as fd:
            json.dump(data_config, fd)

    if FLAGS.do_print:
        token_path = os.path.join(FLAGS.output_dir, "token_print.txt")
        if os.path.exists(token_path):
            os.remove(token_path)

        with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        predict_examples = processor.get_print_examples(myInput)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, label_list,
                                                 FLAGS.max_seq_length, tokenizer,
                                                 predict_file, mode="print")

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder,
            model = "print")

        result = estimator.predict(input_fn=predict_input_fn)      #同样都是predict为什么这里输出的是它的类别，然而multi中是概率呢
        print("fjfk", predict_examples, result)
        for predict_line, prediction in zip(predict_examples, result):
            #ans = {}
            idx = 0
            line = ''
            line_token = str(predict_line.text).split(' ')     #这里的predict为什么有text和label类型呢？在呢里定义的呢？
            label_token = str(predict_line.label).split(' ')
                #print("hello", line_token, label_token, prediction)
                #if len(line_token) != len(label_token):
                    #tf.logging.info(predict_line.text)
                    #tf.logging.info(predict_line.label)
            my_class_data = ""
            class_ans = ""
            for id in prediction:     #这里的id是数字
                if id == 0:
                    continue
                curr_labels = id2label[id]
                if curr_labels in ['[CLS]']:
                    continue
                if curr_labels in ['[SEP]']:
                    if class_ans != "":
                        if class_ans not in ans:
                            ans[class_ans] = my_class_data
                        else:
                            ans[class_ans] += " " + my_class_data
                        #print("xiaoxiong", class_ans, my_class_data)
                    continue
                    # 不知道为什么，这里会出现idx out of range 的错误。。。do not know why here cache list out of range exception!
                try:
                    line += line_token[idx] + ' ' + label_token[idx] + ' ' + curr_labels + '\n'
                except Exception as e:
                    tf.logging.info(e)
                    tf.logging.info(predict_line.text)
                    tf.logging.info(predict_line.label)
                    line = ''
                    break
                if curr_labels[:2] == "B_":
                    if class_ans != "":
                        if class_ans not in ans:
                            ans[class_ans] = my_class_data
                        else:
                            ans[class_ans] += " " + my_class_data
                        #print("xiaoxiong", class_ans, my_class_data)
                    class_ans = curr_labels[2:]
                    my_class_data = str(line_token[idx])  # 只在这里赋初值
                        # print("class_ans", class_ans, my_class_data)
                elif curr_labels[:2] == "I_" and my_class_data != "":
                    my_class_data += str(line_token[idx])  # 在为I的时候只需要添加就可以了
                else:
                    if my_class_data != "":
                        if class_ans not in ans:
                            ans[class_ans] = my_class_data
                        else:
                            ans[class_ans] += " " + my_class_data
                        #print("xiaoxiong", class_ans, my_class_data)
                    my_class_data = ""
                    class_ans = ""
                idx += 1
            return ans

@app.route('/index', methods=['GET'])
def index():
    return render_template('templates/index.html')

@app.route('/process', methods=['POST']) ###
def get_input():
    myinput = request.form.get("param")
    myret = myServer(myinput)
    return render_template('templates/index.html', res = myret, init=myinput)

if __name__ == "__main__":
    app.run(host = '0.0.0.0',port=5001, debug=True)


