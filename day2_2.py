#encding:utf-8
import os
path = os.path.dirname(__file__)
#语言模型与数据集
#一段自然语言文本可以看作是一个离散时间序列，给定一个长度为的词的序列
#语言模型的目标就是评估该序列是否合理，即计算该序列的概率：主要涉及n元语法（n-gram）


#随着序列长度的增加，计算与储存的复杂度会指数级上升
#可以按照马尔克夫链花间模型

#读取数据集
def load()
    with open(path+'\\2.txt') as f:
        corpus_chars = f.read()
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')#去掉空行一类的
    corpus_chars = corpus_chars[: 10000]
    idx_to_char = list(set(corpus_chars))#排序后list
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])#字典化
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]#建立索引
    return corpus_indices, char_to_idx, idx_to_char, vocab_size

#采样分为
    #时序数据采样
        #在训练中我们需要每次随机读取小批量样本和标签。
        # 与之前章节的实验数据不同的是，时序数据的一个样本通常包含连续的字符。
        # 假设时间步数为5，样本序列为5个字符，即“想”“要”“有”“直”“升”。
        # 该样本的标签序列为这些字符分别在训练集中的下一个字符，
        # 即“要”“有”“直”“升”“机”，即X=“想要有直升”，=“Y要有直升机”。
    #简单来说就是按照固定长度遍历

    #随机采样
        #简单来说就是随机从句子里面抽
    
import torch
import random

def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为对于长度为n的序列，X最多只有包含其中的前n - 1个字符
    num_examples = (len(corpus_indices) - 1) // num_steps  # 下取整，得到不重叠情况下的样本个数
    example_indices = [i * num_steps for i in range(num_examples)]  # 每个样本的第一个字符在corpus_indices中的下标
    random.shuffle(example_indices)#随机打乱顺序

    def _data(i):
        # 返回从i开始的长为num_steps的序列
        return corpus_indices[i: i + num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(0, num_examples, batch_size):
        # 每次选出batch_size个随机样本
        batch_indices = example_indices[i: i + batch_size]  # 当前batch的各个样本的首字符的下标
        X = [_data(j) for j in batch_indices]
        Y = [_data(j + 1) for j in batch_indices]
        yield torch.tensor(X, device=device), torch.tensor(Y, device=device)


my_seq = list(range(30))#查看随机采样的样本标签
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')


    #相邻采样
        #即相邻的两个随机小批量在原始序列上的位置相毗邻
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_len = len(corpus_indices) // batch_size * batch_size  # 保留下来的序列的长度
    corpus_indices = corpus_indices[: corpus_len]  # 仅保留前corpus_len个字符
    indices = torch.tensor(corpus_indices, device=device)
    indices = indices.view(batch_size, -1)  # resize成(batch_size, )
    batch_num = (indices.shape[1] - 1) // num_steps
    for i in range(batch_num):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')















