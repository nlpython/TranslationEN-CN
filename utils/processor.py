from collections import Counter
from nltk import word_tokenize
from model.utils import cht_to_chs


class Processor(object):

    def __init__(self, args):
        self.args = args
        self.train_file = args.train_file
        self.dev_file = args.dev_file

        # 读取数据、分词
        self.train_en, self.train_cn = self.load_and_cache_file(mode='train')
        self.dev_en, self.dev_cn = self.load_and_cache_file(mode='dev')
        # 构建词表
        self.en_word_dict, self.en_total_words, self.en_index_dict = \
            self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = \
            self.build_dict(self.train_cn)
        # 单词映射为索引
        self.train_en, self.train_cn = self.word2id(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn = self.word2id(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)

        self.BOS = self.en_word_dict['<BOS>']
        self.EOS = self.en_word_dict['<EOS>']

    def build_dataset(self, mode='train'):
        """
        构建数据集
        """
        if mode == 'train':
            return self.train_en, self.train_cn
        elif mode == 'dev':
            return self.dev_en, self.dev_cn
        else:
            raise ValueError('mode must be train or dev')


    def load_and_cache_file(self, mode='train'):
        if mode == 'train':
            file = self.train_file
        elif mode == 'dev':
            file = self.dev_file
        else:
            raise ValueError('mode must be train or dev')

        en, cn = [], []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                sent_en = line[0].lower()
                sent_cn = cht_to_chs(line[1])

                sent_en = ['<BOS>'] + word_tokenize(sent_en) + ['<EOS>']
                sent_cn = ['<BOS>'] + [char for char in sent_cn] + ['<EOS>']

                en.append(sent_en)
                cn.append(sent_cn)

        return en, cn

    def build_dict(self, sentences):
        """
        构造分词后的列表数据
        构建单词-索引映射（key为单词，value为id值）
        """

        # 统计数据集中单词词频
        word_count = Counter([word for sent in sentences for word in sent])
        # 按词频保留前max_words个单词构建词典
        # 添加UNK和PAD两个单词
        ls = word_count.most_common(int(self.args.max_vocab_size))
        total_words = len(ls) + 2
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = self.args.UNK
        word_dict['PAD'] = self.args.PAD

        # 构建id2word映射
        index_dict = {v: k for k, v in word_dict.items()}
        return word_dict, total_words, index_dict

    def word2id(self, en, cn, en_dict, cn_dict, sort=True):
        """
        将英文、中文单词列表转为单词索引列表
        `sort=True`表示以英文语句长度排序，以便按批次填充时，同批次语句填充尽量少
        """
        length = len(en)
        # 单词映射为索引
        out_en_ids = [[en_dict.get(word, self.args.UNK) for word in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(word, self.args.UNK) for word in sent] for sent in cn]

        # 按照语句长度排序
        def len_argsort(seq):
            """
            传入一系列语句数据(分好词的列表形式)，
            按照语句长度排序后，返回排序后原来各语句在数据中的索引下标
            """
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        # 按相同顺序对中文、英文样本排序
        if sort:
            # 以英文语句长度排序
            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[idx] for idx in sorted_index]
            out_cn_ids = [out_cn_ids[idx] for idx in sorted_index]
        return out_en_ids, out_cn_ids

    def vocab_size(self):
        return len(self.en_word_dict), len(self.cn_word_dict)
