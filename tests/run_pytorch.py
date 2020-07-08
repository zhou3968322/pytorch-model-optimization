# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/7/8
import torch
from src.torch_module import CRNN
from src.utils import load_charsets, decode_time_pos
import torch.nn.functional as F
import time


def test_batch():
    imgH = 48
    channel = 1
    nclass = 5540
    nh = 256
    torch_model_path = "../data/models/document/netCRNN-epoch1-step1297999-lr1e-05-loss4.8294-acc0.7528.pth"
    charsets_path = "../data/models/document/doc_charset.txt"
    alphabet = load_charsets(charsets_path) + '卐'
    torch_crnn_model = CRNN(imgH, channel, nclass, nh)
    device = torch.device('cuda')
    torch_crnn_model.load_state_dict(torch.load(torch_model_path))
    torch_crnn_model.to(device)
    tensor_path = '../data/input_tensor.pt'
    input_tensor = torch.load(tensor_path)
    t0 = time.time()
    with torch.no_grad():
        preds = torch_crnn_model(input_tensor.to(device))
        preds_softmax = F.softmax(preds, dim=2)
        preds_tensor = preds_softmax.transpose(1, 0).contiguous()
        preds_in = preds_tensor.argmax(2)
    pred_texts, time_pos_list = decode_time_pos(preds_in, alphabet)
    print("decode batch cost:{}".format(time.time() - t0))


if __name__ == '__main__':
    test_batch()
