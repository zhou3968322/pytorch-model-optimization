# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/7/8
import argparse
from src.torch2onnx import convert_crnn2onnx, convert_onnx2tf


def main():
    parser = argparse.ArgumentParser()
    document_model_path = "../data/models/document/netCRNN-epoch1-step1297999-lr1e-05-loss4.8294-acc0.7528.pth"
    default_onnx_pref = "data/models/document_onnx"
    parser.add_argument('--torch_model_path', help='torch model path', default=document_model_path)
    parser.add_argument('--onnx_model_pref', help='onnx model path', default=default_onnx_pref)
    opt = parser.parse_args()
    model_path_dict = convert_crnn2onnx(opt.torch_model_path, opt.onnx_model_pref)
    for key in model_path_dict:
        tf_model_path = convert_onnx2tf(model_path_dict[key])
        print("generate tensorflow model path:{}".format(tf_model_path))
    # cnn_onnx_model_path, rnn_onnx_model_path = convert_crnn2onnx(opt.torch_model_path, opt.onnx_model_pref)
    # print("generate  cnn onnx model path:{}, rnn model path:{}".format(cnn_onnx_model_path, rnn_onnx_model_path))
    # cnn_tf_model_path = convert_onnx2tf(cnn_onnx_model_path)
    # rnn_tf_model_path = convert_onnx2tf(rnn_onnx_model_path)
    # print("generate cnn tensorflow model path:{}, rnn model path:{}".format(cnn_tf_model_path, rnn_tf_model_path))


if __name__ == '__main__':
    main()