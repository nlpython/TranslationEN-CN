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

def test():
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
    logger.info(">>>>>>> start evaluate")
    evaluate_start = time.time()
    evaluate(processor, model, device=device)
    logger.info(f"<<<<<<< finished evaluate, cost {time.time() - evaluate_start:.4f} seconds")

if __name__ == '__main__':
    test()