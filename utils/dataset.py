from torch.utils.data import Dataset
import numpy as np
from utils.features import Batch

class TranslateDataset(Dataset):

    def __init__(self, en, cn):
        super(TranslateDataset, self).__init__()

        self.en = en
        self.cn = cn

    def __len__(self):
        assert len(self.en) == len(self.cn), 'en and cn must have the same length!'
        return len(self.en)

    def __getitem__(self, index):
        return self.en[index], self.cn[index]

    @staticmethod
    def collate_fn(batch):
        en, cn = zip(*batch)
        batch_cn = seq_padding(cn)
        batch_en = seq_padding(en)

        return Batch(batch_en, batch_cn)


def seq_padding(X, padding=0):
    """
    按批次（batch）对数据填充、长度对齐
    """
    # 计算该批次各条样本语句长度
    L = [len(x) for x in X]
    # 获取该批次样本中语句长度最大值
    ML = max(L)
    # 遍历该批次样本，如果语句长度小于最大长度，则用padding填充
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])
