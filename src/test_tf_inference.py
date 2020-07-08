# -*- coding: utf-8 -*-
# email: zhoubingcheng@datagrand.com
# create  : 2020/7/6
import os
import sys

BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(BASEDIR)
from scipy.special import softmax
import torch
import time
from scripts.utils import load_charsets, decode_time_pos
from scripts.inference import Inference, LstmInference


def get_crnn_output(model_dict, np_tensor):
    cnn_output = model_dict["cnn"].inference(np_tensor)
    cnn_output = cnn_output.squeeze().transpose(2, 0, 1)
    lstm1_output = model_dict["lstm1"].inference(cnn_output)
    embedding1_input = lstm1_output.reshape(lstm1_output.shape[0] * lstm1_output.shape[1], lstm1_output.shape[2])
    embedding1_output = model_dict["embedding1"].inference(embedding1_input)
    lstm2_input = embedding1_output.reshape(lstm1_output.shape[0], lstm1_output.shape[1], -1)
    lstm2_output = model_dict["lstm2"].inference(lstm2_input)
    embedding2_input = lstm2_output.reshape(lstm2_output.shape[0] * lstm2_output.shape[1], lstm2_output.shape[2])
    embedding2_output = model_dict["embedding2"].inference(embedding2_input)
    preds = embedding2_output.reshape(lstm2_output.shape[0], lstm2_output.shape[1], -1)
    return preds


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


def inference_test():
    charsets_path = os.path.join(BASEDIR, "recognition/data/recognition_model/document/doc_charset.txt")
    alphabet = load_charsets(charsets_path) + '卐'
    model_dict = init_models()
    pytorch_tensor = torch.load("input_tensor.pt")
    np_tensor = pytorch_tensor.cpu().numpy()
    for i in range(10):
        t0 = time.time()
        preds = get_crnn_output(model_dict, np_tensor)
        preds_prob = softmax(preds, axis=2).transpose(1, 0, 2)
        preds_in = preds_prob.argmax(2)
        pred_texts, time_pos_list = decode_time_pos(preds_in, alphabet)
        print("decode batch cost:{}".format(time.time() - t0))
    print('debugging')


def onnx_inference_test():
    import onnxruntime as rt
    charsets_path = os.path.join(BASEDIR, "recognition/data/recognition_model/document/doc_charset.txt")
    alphabet = load_charsets(charsets_path) + '卐'
    pytorch_tensor = torch.load("input_tensor.pt")
    cnn_out_tensor = torch.load("cnn_output_tensor.pt")
    cnn_out_tensor = cnn_out_tensor.cpu().numpy()
    onnx_model_path = "document_onnx_cnn.onnx"
    sess = rt.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    cnn_output = sess.run([label_name], {input_name: to_numpy(pytorch_tensor)})[0]
    print("finished test")


if __name__ == '__main__':
    # onnx_inference_test()
    inference_test()