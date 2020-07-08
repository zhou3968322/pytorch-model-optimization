# -*- coding: utf-8 -*-
# email: zhoubingcheng@datagrand.com
# create  : 2020/7/2
import os
import tensorflow as tf
import numpy as np
import time
from tensorflow.python.saved_model import signature_constants, tag_constants


class Inference(object):
    
    def __init__(self, tf_model_path, **kwargs):
        gpu_fraction = kwargs.get('gpu_fraction', 0.1)
        name = kwargs.get("name", "")
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        self._model_path = tf_model_path
        self.graph = tf.Graph()
        self.name = name
        with self.graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with open(tf_model_path, "rb") as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name=name)
        self.session = tf.compat.v1.Session(graph=self.graph, config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        op = self.graph.get_operations()
        input_name = kwargs.get("input_name", "input:0")
        output_name = kwargs.get("output_name", "output:0")
        self._input_x = self.graph.get_tensor_by_name("{}/{}".format(self.name, input_name))  # input
        self._output_x = self.graph.get_tensor_by_name("{}/{}".format(self.name, output_name))  # output
        for m in op:
            if "input" in m.name:
                print(m.values())
            elif "output" in m.name:
                print(m.values())
        print('-------------ops done.---------------------')
    
    def save_builders(self, export_dir=None):
        if export_dir is None:
            tf_model_dir = os.path.dirname(self._model_path)
            export_dir = os.path.join(tf_model_dir, './{}'.format(self.name))
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
        sigs = {}
        with self.session as sess:
            # name="" is important to ensure we don't get spurious prefixing
            sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
                tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
                    {"input": self._input_x}, {"output": self._output_x})
            
            builder.add_meta_graph_and_variables(sess,
                                                 [tag_constants.SERVING],
                                                 signature_def_map=sigs)
        
        builder.save()
    
    def inference(self, input_np, input_name="input:0", output_name="output:0"):
        t0 = time.time()
        output_tf_pb = self.session.run([self._output_x], feed_dict={self._input_x: input_np})
        print("layer:{} cost:{}".format(self.name, time.time() - t0))
        return output_tf_pb[0]


class LstmInference(Inference):
    
    def inference(self, input_np, input_name="input:0", output_name="output:0"):
        input_h0 = self.graph.get_tensor_by_name("{}/h0:0".format(self.name))
        input_c0 = self.graph.get_tensor_by_name("{}/c0:0".format(self.name))
        input_h0_np = np.zeros(shape=(2, input_np.shape[1], 256), dtype=np.float32)
        input_c0_np = np.zeros(shape=(2, input_np.shape[1], 256), dtype=np.float32)
        t0 = time.time()
        output_tf_pb = self.session.run([self._output_x], feed_dict={self._input_x: input_np, input_h0: input_h0_np,
                                                                     input_c0: input_c0_np})
        print("layer:{} cost:{}".format(self.name, time.time() - t0))
        #  Post - Processing
        return output_tf_pb[0]



