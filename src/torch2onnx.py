# -*- coding: utf-8 -*-
# email: zhoubingcheng@datagrand.com
# create  : 2020/7/2
import os
import torch
import onnx
from torch.autograd import Variable
from src.torch_module import CRNN
from onnx_tf.backend import prepare


def convert_crnn2onnx(torch_model_path, onnx_pref):
    imgH = 48
    imgW = 2044
    channel = 1
    nclass = 5540
    nh = 256
    batch_size = 64
    torch_crnn_model = CRNN(imgH, channel, nclass, nh)
    torch_crnn_model.load_state_dict(torch.load(torch_model_path))
    # cnn_model.load_state_dict(torch_crnn_model.cnn.state_dict())
    # b, c, h, w
    cnn_onnx_model_path = "{}_cnn.onnx".format(onnx_pref)
    cnn_model = torch_crnn_model.cnn
    cnn_model.eval()
    with torch.no_grad():
        dummy_cnn_input = Variable(torch.randn(batch_size, channel, imgH, imgW))
        torch.onnx.export(cnn_model,
                          dummy_cnn_input,
                          cnn_onnx_model_path,
                          input_names=['input'],
                          output_names=['output'],
                          # dynamic_axes={"input": {3: 'width'},
                          #               "output": {3: 'sequence'}})
                          dynamic_axes={"input": {0: 'batch_size',
                                                  3: 'width'},
                                        "output": {0: 'batch_size',
                                                   3: 'sequence'}})
    lstm1 = torch_crnn_model.rnn[0].rnn
    lstm1_onnx_model_path = "{}_lstm1.onnx".format(onnx_pref)
    with torch.no_grad():
        input = torch.randn(512, batch_size, 512)
        h0 = torch.randn(2, batch_size, 256)
        c0 = torch.randn(2, batch_size, 256)
        output, (hn, cn) = lstm1(input, (h0, c0))
        torch.onnx.export(lstm1, (input, (h0, c0)), lstm1_onnx_model_path,
                          input_names=['input', 'h0', 'c0'],
                          output_names=['output', 'hn', 'cn'],
                          dynamic_axes={'input': {0: 'sequence', 1: 'batch_size'},
                                        'h0': {1: 'batch_size'},
                                        'c0': {1: 'batch_size'},
                                        'output': {0: 'sequence', 1: 'batch_size'}})
        onnx_model = onnx.load(lstm1_onnx_model_path)
        # input shape ['sequence', 3, 10]
        print(onnx_model.graph.input[0])
    embedding = torch_crnn_model.rnn[0].embedding
    embedding1_onnx_model_path = "{}_embedding1.onnx".format(onnx_pref)
    with torch.no_grad():
        embedding_input = torch.randn(1, 512)
        torch.onnx.export(embedding, embedding_input, embedding1_onnx_model_path,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'embedding1_in_feature'},
                                        'output': {0: 'embedding1_out_feature'}})
        onnx_model = onnx.load(embedding1_onnx_model_path)
        # input shape ['sequence', 3, 10]
        print(onnx_model.graph.input[0])
    lstm2 = torch_crnn_model.rnn[1].rnn
    lstm2_onnx_model_path = "{}_lstm2.onnx".format(onnx_pref)
    with torch.no_grad():
        input = torch.randn(512, batch_size, 256)
        h0 = torch.randn(2, batch_size, 256)
        c0 = torch.randn(2, batch_size, 256)
        output, (hn, cn) = lstm2(input, (h0, c0))
        torch.onnx.export(lstm2, (input, (h0, c0)), lstm2_onnx_model_path,
                          input_names=['input', 'h0', 'c0'],
                          output_names=['output', 'hn', 'cn'],
                          dynamic_axes={'input': {0: 'sequence', 1: 'batch_size'},
                                        'h0': {1: 'batch_size'},
                                        'c0': {1: 'batch_size'},
                                        'output': {0: 'sequence', 1: 'batch_size'}})
        onnx_model = onnx.load(lstm2_onnx_model_path)
        # input shape ['sequence', 3, 10]
        print(onnx_model.graph.input[0])
    embedding = torch_crnn_model.rnn[1].embedding
    embedding2_onnx_model_path = "{}_embedding2.onnx".format(onnx_pref)
    with torch.no_grad():
        embedding_input = torch.randn(1, 512)
        torch.onnx.export(embedding, embedding_input, embedding2_onnx_model_path,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'embedding2_in_feature'},
                                        'output': {0: 'embedding2_out_feature'}})
        onnx_model = onnx.load(embedding2_onnx_model_path)
        # input shape ['sequence', 3, 10]
        print(onnx_model.graph.input[0])
    return {"cnn": cnn_onnx_model_path,
            "lstm1": lstm1_onnx_model_path,
            "embedding1": embedding1_onnx_model_path,
            "lstm2": lstm2_onnx_model_path,
            "embedding2": embedding2_onnx_model_path}


def convert_onnx2tf(onnx_model_path, tf_model_path=None):
    onnx_model = onnx.load(onnx_model_path)
    # Prepare Tensorflow pb model
    if tf_model_path is None:
        model_dir = os.path.dirname(onnx_model_path)
        onnx_model_name = os.path.basename(onnx_model_path)
        tf_model_name = onnx_model_name.rsplit('.')[0] + "_graph.pb"
        tf_model_path = os.path.join(model_dir, tf_model_name)
    onnx.checker.check_model(onnx_model)
    tf_rep = prepare(onnx_model, strict=True)
    tf_rep.export_graph(tf_model_path)
    return tf_model_path



