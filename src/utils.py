# -*- coding: utf-8 -*-
# email: zhoubingcheng@datagrand.com
# create  : 2020/7/7


def load_charsets(charset_path):
    with open(charset_path, 'r') as f:
        result = f.read().strip('\n')
    return result


def decode_time_pos(preds_in, alphabet):
    texts = []
    idx_pos_list = []
    preds_cpu = preds_in
    for i in range(preds_cpu.shape[0]):
        char_list = []
        idx_pos = []
        for j in range(preds_cpu.shape[1]):
            if preds_cpu[i, j] != 0 and (not (j > 0 and preds_cpu[i, j - 1] == preds_cpu[i, j])):
                char_list.append(alphabet[preds_cpu[i, j] - 1])
                idx_pos.append(j)
        texts.append(''.join(char_list))
        idx_pos_list.append(idx_pos)
    return texts, idx_pos_list
