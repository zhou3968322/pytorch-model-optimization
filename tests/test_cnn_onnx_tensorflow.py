# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/7/8
import os
import sys
BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASEDIR)
import torch
import tensorflow as tf
import time
import onnx
import onnxruntime as rt
from onnx_tf.backend import prepare


def main():
    tensor_path = os.path.join(BASEDIR, "data/input_tensor.pt")
    pytorch_tensor = torch.load(tensor_path)
    np_tensor = pytorch_tensor.cpu().numpy()
    onnx_model_path = os.path.join(BASEDIR, "data/models/document_onnx_cnn.onnx")
    sess = rt.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    onnx_output = sess.run([label_name], {input_name: np_tensor})[0]
    onnx_model = onnx.load(onnx_model_path)
    # Prepare Tensorflow pb model
    model_dir = os.path.dirname(onnx_model_path)
    onnx_model_name = os.path.basename(onnx_model_path)
    tf_model_name = onnx_model_name.rsplit('.')[0] + "_graph.pb"
    tf_model_path = os.path.join(model_dir, tf_model_name)
    onnx.checker.check_model(onnx_model)
    tf_rep = prepare(onnx_model, strict=True)
    tf_rep.export_graph(tf_model_path)
    graph = tf.Graph()
    with graph.as_default():
        cnn_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 1, 48, None], name="cnn_input")
        cnn_graph_def = tf.compat.v1.GraphDef()
        with open(tf_model_path, "rb") as f:
            cnn_graph_def.ParseFromString(f.read())
        cnn_output = tf.import_graph_def(cnn_graph_def, name="cnn", input_map={"input:0": cnn_input},
                                         return_elements=["output:0"])
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
    with tf.compat.v1.Session(graph=graph, config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
        for i in range(10):
            t0 = time.time()
            output_tf_pb = sess.run(cnn_output, feed_dict={"cnn_input:0": np_tensor})
            print("preds cost:{}".format(time.time() - t0))
    print("success test")


if __name__ == '__main__':
    main()
