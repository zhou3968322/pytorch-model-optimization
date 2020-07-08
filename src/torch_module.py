# -*- coding: utf-8 -*-
# email: zhoubingcheng@datagrand.com
# create  : 2020/7/2
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)  # [T, b, h * 2]

        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, (imgH // 16, 2)]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
        
        # 假设input为64x1x48x817

        convRelu(0)  # conv_output: 64x64x48x817, relu_output:64x64x48x817
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # output:64x64x24x408
        convRelu(1)  # conv_output: 64x128x24x408, relu_output:64x128x24x408
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # output:64x128x12x204
        convRelu(2, True)  # conv_output:64x256x12x204, batch_norm_output:64x256x12x204, relu_output:64x256x12x204
        convRelu(3)  # conv_output:64x256x12x204, relu_output:64x256x12x204
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # output:64x256x6x205
        convRelu(4, True)  # conv_output:64x512x6x205, batch_norm_output:64x512x6x205, relu_output:64x512x6x205
        convRelu(5) # conv_output:64x512x6x205, relu_output:64x512x6x205
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # output:64x512x3x206
        convRelu(6, True)  # conv_output:64x512x1x205, batch_norm_output:64x512x1x205, relu_output:64x512x1x205

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        # input [1, 3, 40, 800] -> [3, 1, 40, 800]
        #input = input.permute(1,0,2,3)
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output

