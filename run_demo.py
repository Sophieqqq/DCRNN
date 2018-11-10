import argparse
import numpy as np
import os
import sys
import tensorflow as tf
import yaml

from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor


def run_dcrnn(args):
    graph_pkl_filename = 'data/sensor_graph/adj_mx.pkl'
    with open(args.config_filename) as f:
        config = yaml.load(f)
    tf_config = tf.ConfigProto()
    if args.use_cpu_only:
        tf_config = tf.ConfigProto(device_count={'GPU': 0})
    tf_config.gpu_options.allow_growth = True
    _, _, adj_mx = load_graph_data(graph_pkl_filename)
    with tf.Session(config=tf_config) as sess:
        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **config)
        supervisor.load(sess, config['train']['model_filename'])

        # supervisor._test_model
        # Save the variables to disk.
        # save_path = supervisor._test_model.save(sess, "/tmp/test_model.ckpt")
        save_path = 'data/model/pretrained/'
        supervisor._saver.save(sess, save_path+"model.ckpt") #tf.train.Saver()
        print("Test_Model saved in path: %s" % save_path)

        ## Restore the Model
        saver = supervisor._saver#tf.train.import_meta_graph(save_path+'model.ckpt.meta', clear_devices=True)
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        # sess = tf.Session()
        saver.restore(sess, "./model.ckpt")
        print "GLOBAL Variables:"
        print tf.global_variables()

        #predict:
        # outputs = supervisor.evaluate(sess) # return prediction and groundtruth
        # np.savez_compressed(args.output_filename, **outputs)
        # print('Predictions saved as {}.'.format(args.output_filename))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    args = parser.parse_args()
    run_dcrnn(args)
