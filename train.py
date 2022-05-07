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

def run_epoch(train_loader, model, loss_compute, epoch):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.

    for i, batch in enumerate(train_loader):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            logger.info("Epoch %d Batch: %d Loss: %f Tokens per Sec: %fs" % (
            epoch, i - 1, loss / batch.ntokens, (tokens.float() / elapsed / 1000.)))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


def train(train_loader, dev_loader, model, epochs, criterion, optimizer):
    """
    训练并保存模型
    """
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_dev_loss = 1e5

    for epoch in range(epochs):
        # 模型训练
        model.train()
        run_epoch(train_loader, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)
        model.eval()

        # 在dev集上进行loss评估
        logger.info('>>>>> Evaluate')
        dev_loss = run_epoch(dev_loader, model, SimpleLossCompute(model.generator, criterion, None), epoch)
        logger.info('<<<<< Evaluate loss: %f' % dev_loss)

        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if dev_loss < best_dev_loss:
            torch.save(model.state_dict(), './checkpoints/model.pth')
            best_dev_loss = dev_loss
            logger.info('****** Save model done... ******')

        logger.info("Epoch %d Done\n" % epoch)


if __name__ == '__main__':

    config = json.loads(open('./config.json', 'r').read())
    args = Config(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device

    logger.add('./logs/{time}.log')
    logger.info(args.__str__())

    # 数据加载
    if os.path.exists(args.processor_path):
        processor = pickle.load(open(args.processor_path, 'rb'))
        logger.info('Successfully load processor...')
    else:
        processor = Processor(args)
        pickle.dump(processor, open(args.processor_path, 'wb'))
        logger.info('Saved processor to %s' % args.processor_path)

    train_set = TranslateDataset(*processor.build_dataset('train'))
    dev_set = TranslateDataset(*processor.build_dataset('dev'))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=TranslateDataset.collate_fn)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, collate_fn=TranslateDataset.collate_fn)

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
    # 训练
    logger.info(">>>>>>> start train")
    train_start = time.time()
    criterion = LabelSmoothing(tgt_vocab_size, padding_idx=0, smoothing=0.0)
    optimizer = NoamOpt(args.d_model, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    train(train_loader, dev_loader, model, args.epochs, criterion, optimizer)
    logger.info(f"<<<<<<< finished train, cost {time.time() - train_start:.4f} seconds")

    # 预测
    # 加载模型
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    # 开始预测
    logger.info(">>>>>>> start evaluate")
    evaluate_start = time.time()
    evaluate(processor, model, device=device)
    logger.info(f"<<<<<<< finished evaluate, cost {time.time() - evaluate_start:.4f} seconds")

