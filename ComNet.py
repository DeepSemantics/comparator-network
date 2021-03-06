import utils
import reader

import argparse
import logging
import json
import sys
import os

import tensorflow as tf
from tensorflow.python.framework import graph_util

from utils import datafeeds
from utils import controler
from utils import utility
from utils import converter

_WORK_DIR = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(_WORK_DIR, '../../../common'))
#import log 


def load_config(config_file):
    """
    load config
    """
    with open(config_file, "r") as f:
        try:
            conf = json.load(f)
        except Exception:
            logging.error("load json file %s error" % config_file)
    conf_dict = {}
    unused = [conf_dict.update(conf[k]) for k in conf]
    logging.debug("\n".join(
        ["%s=%s" % (u, conf_dict[u]) for u in conf_dict]))
    return conf_dict


def train(conf_dict):
    """
    train
    """
    training_mode = conf_dict["training_mode"]
    net = utility.import_object(
        conf_dict["net_py"], conf_dict["net_class"])(conf_dict)
    if training_mode == "pointwise":
        datafeed = datafeeds.TFPointwisePaddingData(conf_dict)
        input_l, input_r, label_y = datafeed.ops()
        pred = net.predict(input_l, input_r)
        output_prob = tf.nn.softmax(pred, -1, name="output_prob")
        loss_layer = utility.import_object(
            conf_dict["loss_py"], conf_dict["loss_class"])()
        loss = loss_layer.ops(pred, label_y)
    elif training_mode == "pairwise":
        datafeed = datafeeds.TFPairwisePaddingData(conf_dict)
        input_l, input_r, neg_input = datafeed.ops()
        pos_score = net.predict(input_l, input_r)
        output_prob = tf.identity(pos_score, name="output_preb")
        neg_score = net.predict(input_l, neg_input)
        loss_layer = utility.import_object(
            conf_dict["loss_py"], conf_dict["loss_class"])(conf_dict)
        loss = loss_layer.ops(pos_score, neg_score)
    else:
        print(sys.stderr, "training mode not supported")
        sys.exit(1)
    # define optimizer
    lr = float(conf_dict["learning_rate"])
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # run_trainer
    controler.run_trainer(loss, optimizer, conf_dict)


def predict(conf_dict):
    """
    predict
    """
    net = utility.import_object(
        conf_dict["net_py"], conf_dict["net_class"])(conf_dict)
    conf_dict.update({"num_epochs": "1", "batch_size": "1",
                      "shuffle": "0", "train_file": conf_dict["test_file"]})
    test_datafeed = datafeeds.TFPointwisePaddingData(conf_dict)
    test_l, test_r, test_y = test_datafeed.ops()
    # test network
    pred = net.predict(test_l, test_r)
    controler.run_predict(pred, test_y, conf_dict)


def freeze(conf_dict):
    """
    freeze net for c api predict
    """
    model_path = conf_dict["save_path"]
    freeze_path = conf_dict["freeze_path"]
    saver = tf.train.import_meta_graph(model_path + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        var_graph_def = tf.get_default_graph().as_graph_def()
        const_graph_def = graph_util.convert_variables_to_constants(sess, var_graph_def, ["output_prob"])
        with tf.gfile.GFile(freeze_path, "wb") as f:
            f.write(const_graph_def.SerializeToString())


def sim_func(query_pair):
    '''
    输入:
        query_pair:文本对，制表符隔开
    返回:
        simlarity:文本对语义相似度
    '''
    simnet_process.input_pair = query_pair
    preds_list = []
    for iter, data in enumerate(batch_data()):
        output = executor.run(program, feed=infer_feeder.feed(data), fetch_list=fetch_targets)        
        if args.task_mode == "pairwise":
            preds_list += list(map(lambda item: str(item[0]), output[1]))
        else:
            preds_list += map(lambda item: str(np.argmax(item)), output[1])

    return float(preds_list[0])
            
            
            
def convert(conf_dict):
    """
    convert
    """
    converter.run_convert(conf_dict)


if __name__ == "__main__": 
#    log.init_log("./log/tensorflow")
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='train',
                        help='task: train/predict/freeze/convert, the default value is train.')
    parser.add_argument('--task_conf', default='./examples/cnn-pointwise.json',
                        help='task_conf: config file for this task')
    args = parser.parse_args()
    task_conf = args.task_conf
    config = load_config(task_conf)
    task = args.task
    if args.task == 'train':
        train(config)
    elif args.task == 'predict':
        predict(config)
    elif args.task == 'freeze':
        freeze(config)
    elif args.task == 'convert':
        convert(config)
    else:
        print(sys.stderr, 'task type error.')


