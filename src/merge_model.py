# -*- coding: utf-8 -*-
# email: zhoubingcheng@datagrand.com
# create  : 2020/7/6
import os
import sys

BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(BASEDIR)
from scripts.inference import Inference, LstmInference
import torch
import tensorflow as tf
from scripts.utils import load_charsets, decode_time_pos
from tensorflow.keras.models import Model
import time


def init_models():
    cnn_model_path = "document_onnx_cnn_graph.pb"
    cnn_model = Inference(tf_model_path=cnn_model_path, name="cnn")
    lstm1_model_path = "document_onnx_lstm1_graph.pb"
    lstm1_model = LstmInference(tf_model_path=lstm1_model_path, name="lstm1")
    embedding1_model_path = "document_onnx_embedding1_graph.pb"
    embedding1_model = Inference(tf_model_path=embedding1_model_path, name="embedding1")
    lstm2_model_path = "document_onnx_lstm2_graph.pb"
    lstm2_model = LstmInference(tf_model_path=lstm2_model_path, name="lstm2")
    embedding2_model_path = "document_onnx_embedding2_graph.pb"
    embedding2_model = Inference(tf_model_path=embedding2_model_path, name="embedding2")
    return {"cnn": cnn_model, "lstm1": lstm1_model, "lstm2": lstm2_model,
            "embedding1": embedding1_model, "embedding2": embedding2_model}


def main():
    model_path_dict = {"cnn": "document_onnx_cnn_graph.pb",
                       "lstm1": "document_onnx_lstm1_graph.pb",
                       "embedding1": "document_onnx_embedding1_graph.pb",
                       "lstm2": "document_onnx_lstm2_graph.pb",
                       "embedding2": "document_onnx_embedding2_graph.pb"}
    charsets_path = os.path.join(BASEDIR, "recognition/data/recognition_model/document/doc_charset.txt")
    alphabet = load_charsets(charsets_path) + '卐'
    graph = tf.Graph()
    with graph.as_default():
        cnn_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 1, 48, None], name="cnn_input")
        cnn_graph_def = tf.compat.v1.GraphDef()
        with open(model_path_dict["cnn"], "rb") as f:
            cnn_graph_def.ParseFromString(f.read())
        cnn_output = tf.import_graph_def(cnn_graph_def, name="cnn", input_map={"input:0": cnn_input},
                                         return_elements=["output:0"])
        lstm_input = tf.transpose(tf.squeeze(cnn_output), perm=[2, 0, 1], name='lstm1_input')
        input_shape = tf.unstack(tf.shape(lstm_input), name="rnn_input_shape")
        batch_size = graph.get_tensor_by_name("rnn_input_shape:1")
        time_pos = graph.get_tensor_by_name("rnn_input_shape:0")
        hidden_state = graph.get_tensor_by_name("rnn_input_shape:2")
        h0 = tf.zeros([2, batch_size, 256], dtype=tf.float32, name="lstm_input/h0")
        c0 = tf.zeros([2, batch_size, 256], dtype=tf.float32, name="lstm_input/c0")
        lstm1_graph_def = tf.compat.v1.GraphDef()
        with open(model_path_dict["lstm1"], "rb") as f:
            lstm1_graph_def.ParseFromString(f.read())
        lstm1_output, = tf.import_graph_def(lstm1_graph_def, name="lstm1",
                                            input_map={"input:0": lstm_input, "h0:0": h0, "c0:0": c0},
                                            return_elements=["output:0"])
        embedding1_input = tf.reshape(lstm1_output, shape=[-1, hidden_state])
        embedding1_graph_def = tf.compat.v1.GraphDef()
        with open(model_path_dict["embedding1"], "rb") as f:
            embedding1_graph_def.ParseFromString(f.read())
        embedding1_output, = tf.import_graph_def(embedding1_graph_def, name="embedding1",
                                                 input_map={"input:0": embedding1_input},
                                                 return_elements=["output:0"])
        lstm2_input = tf.reshape(embedding1_output, [time_pos, batch_size, 256])
        lstm2_graph_def = tf.compat.v1.GraphDef()
        with open(model_path_dict["lstm2"], "rb") as f:
            lstm2_graph_def.ParseFromString(f.read())
        lstm2_output, = tf.import_graph_def(lstm2_graph_def, name="lstm2",
                                            input_map={"input:0": lstm2_input, "h0:0": h0, "c0:0": c0},
                                            return_elements=["output:0"])
        embedding2_input = tf.reshape(lstm2_output, shape=[-1, 512])
        embedding2_graph_def = tf.compat.v1.GraphDef()
        with open(model_path_dict["embedding2"], "rb") as f:
            embedding2_graph_def.ParseFromString(f.read())
        embedding2_output, = tf.import_graph_def(embedding2_graph_def, name="embedding2",
                                                 input_map={"input:0": embedding2_input},
                                                 return_elements=["output:0"])
        preds = tf.transpose(tf.reshape(embedding2_output, [time_pos, batch_size, -1]), perm=[1, 0, 2])
        # preds = tf.transpose(tf.nn.softmax(tf.reshape(embedding2_output, [time_pos, batch_size, -1]), axis=2), perm=[1, 0, 2])
        preds_in = tf.math.argmax(preds, axis=2)
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
    
    pytorch_tensor = torch.load("input_tensor.pt")
    np_tensor = pytorch_tensor.cpu().numpy()
    with tf.compat.v1.Session(graph=graph, config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
        for i in range(10):
            t0 = time.time()
            output_tf_pb = sess.run(lstm_input, feed_dict={"cnn_input:0": np_tensor})
            print("preds cost:{}".format(time.time() - t0))
            # t0 = time.time()
            # pred_texts, time_pos_list = decode_time_pos(output_tf_pb, alphabet)
            # print("decode batch cost:{}".format(time.time() - t0))
    print("success test")
    
    # graph1 = tf.Graph()
    # with graph1.as_default():
    #     input_x = tf.compat.v1.placeholder(tf.float32, shape=[None, 1, 48, None], name="input")
    #     input_shape = tf.unstack(tf.shape(input_x), name="input_shape")
    #     batch_size = graph1.get_tensor_by_name("input_shape:0")
    #     zeros_output = tf.zeros((2, batch_size, 256), dtype=tf.float32, name="output")
    #
    # graph2 = tf.Graph()
    # with graph2.as_default():
    #
    # pytorch_tensor = torch.load("input_tensor.pt")
    # np_tensor = pytorch_tensor.cpu().numpy()
    # with tf.compat.v1.Session(graph=graph) as sess:
    #     input_x = graph.get_tensor_by_name("input:0")
    #     output_x = graph.get_tensor_by_name("output:0")
    #     batch_size_res = sess.run([output_x], feed_dict={input_x:np_tensor})
    # print("success graph compute")


if __name__ == '__main__':
    main()
