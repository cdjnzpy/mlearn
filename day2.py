#encding:utf-8
#文本预处理部分
#读入文本
import collections
import re
import os
path = os.path.dirname(__file__)


def read_book():
    with open(path+'\\JaneEyre.txt', 'r') as f:
        lines = [re.sub('[^a-z]+', ' ', line.strip().lower()) for line in f]#去除掉所有符号，只保留英文
    return lines#输出为句子，列表

#随后进行分词
def char_s(lines,token='word'):
    if token == 'word':
        return [sentence.split(' ') for sentence in lines]#如果按照词语分，就按照空格分列
    elif token == 'char':
        return [list(sentence) for sentence in lines]#按照字母分，直接返回各个字母（list（str））
    else:
        print('ERROR: unkown token type '+token)#错误的选择
#建立字典储存
class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        counter = count_corpus(tokens)  # : 
        self.token_freqs = list(counter.items())
        self.idx_to_token = []
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            self.idx_to_token += ['', '', '', '']
        else:
            self.unk = 0
            self.idx_to_token += ['']
        self.idx_to_token += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in self.idx_to_token]
        self.token_to_idx = dict()
        for idx, token in enumerate(self.idx_to_token):
            self.token_to_idx[token] = idx

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(sentences):
    tokens = [tk for st in sentences for tk in st]
    return collections.Counter(tokens)  # 返回一个字典，记录每个词的出现次数


#另外的建立字典方式
lines = read_book()
words = (char_s(lines,token='word'))
dirs={}
for sen in words:
    for cha in sen:
        if cha == "":
            pass
        else:
            if cha in dirs:
                dirs[cha]+=1
            else:
                dirs[cha]=1
print(dirs)



#将单词转换为索引
##现有的工具的分词
#中文jieba，snownlp
#英文spaCY,NLTK
import spacy
#先变为spacy形式，然后采用x.text模式分词，类似BS中先soup再标签



