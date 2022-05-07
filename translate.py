import time
from utils.tools import *
import json
import os
import pickle
from evaluate import evaluate, greedy_decode
from utils.processor import Processor
from utils.dataset import TranslateDataset
from torch.utils.data import DataLoader
from loguru import logger
from nltk import word_tokenize

def translate():
    config = json.loads(open('./config.json', 'r').read())
    args = Config(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device

    logger.info(args.__str__())

    # 数据加载
    if os.path.exists(args.processor_path):
        processor = pickle.load(open(args.processor_path, 'rb'))
        logger.info('Successfully load processor...')

    else:
        logger.info('Saved processor to %s' % args.processor_path)
        processor = Processor(args)

    src_vocab_size, tgt_vocab_size = processor.vocab_size()

    # 初始化模型
    model = make_model(
        src_vocab_size,
        tgt_vocab_size,
        args.layers,
        args.d_model,
        args.d_ff,
        args.heads,
        args.dropout,
        device=device
    )

    # 预测
    # 加载模型
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    # 开始预测
    # logger.info(">>>>>>> start evaluate")
    # evaluate_start = time.time()
    # evaluate(processor, model, device=device)
    # logger.info(f"<<<<<<< finished evaluate, cost {time.time() - evaluate_start:.4f} seconds")

    BOS = processor.BOS
    EOS = processor.EOS
    UNK = args.UNK

    with torch.no_grad():
        while True:
            text = input("Please input English sentence: ")

            # 将当前以单词id表示的英文语句数据转为tensor，并放如DEVICE中
            src = torch.from_numpy(
                np.array([BOS]+[processor.en_word_dict.get(word, UNK) for word in word_tokenize(text.lower())]+[EOS])).long().to(device)
            # 增加一维
            src = src.unsqueeze(0)
            # 设置attention mask
            src_mask = (src != 0).unsqueeze(-2)
            # 用训练好的模型进行decode预测
            out = greedy_decode(model, src, src_mask, max_len=30, start_symbol=processor.cn_word_dict["<BOS>"])
            # 初始化一个用于存放模型翻译结果语句单词的列表
            translation = []
            # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
            for j in range(1, out.size(1)):
                # 获取当前下标的输出字符
                sym = processor.cn_index_dict[out[0, j].item()]
                # 如果输出字符不为'EOS'终止符，则添加到当前语句的翻译结果列表
                if sym != '<EOS>':
                    translation.append(sym)
                # 否则终止遍历
                else:
                    break
            # 打印模型翻译输出的中文语句结果
            print("translation: %s\n" % " ".join(translation))


if __name__ == '__main__':
    translate()