import os
import argparse
from tqdm.auto import tqdm
from bert4torch.model import *
from transformers import MT5ForConditionalGeneration, BertTokenizer

from utils.data_process import T5PegasusTokenizer,prepare_data
from utils.eval import compute_rouge,compute_rouges

from utils.models import net


def train_model(model, adam, train_data, dev_data, tokenizer, device, args):
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    best = 0
    for epoch in range(args.num_epoch):
        model.train()
        for i, cur in enumerate(tqdm(train_data)):
            cur = {k: v.to(device) for k, v in cur.items()}
            # model(**cur) == model(input_ids, decoder_input_ids, attention_mask, decoder_attention_mask)
            # prob:T5预测出来结果
            prob = model(**cur)[0]

            # 获得网络的预测值和labels
            prob,labels = net(cur,prob)

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prob, labels)
            if i % 100 == 0:
                print(i, loss.item())
            loss.backward()
            adam.step()
            adam.zero_grad()

        # 测试
        model.eval()
        gens = []
        summaries = []
        for feature in tqdm(dev_data):
            title = feature['title']
            content = {k: v for k, v in feature.items() if k != 'title'}
            if args.data_parallel and torch.cuda.is_available():
                gen = model.module.generate(max_length=args.max_len_generate,
                                            eos_token_id=tokenizer.sep_token_id,
                                            decoder_start_token_id=tokenizer.cls_token_id,
                                            **content)
            else:
                gen = model.generate(max_length=args.max_len_generate,
                                     eos_token_id=tokenizer.sep_token_id,
                                     decoder_start_token_id=tokenizer.cls_token_id,
                                     **content)
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            gen = [item.replace(' ', '') for item in gen]
            # print(title)
            # print(gen)
            gens.extend(gen)
            summaries.extend(title)
        scores = compute_rouges(gens, summaries)
        print(scores)
        rouge_l = scores['rouge-l']
        if rouge_l > best:
            best = rouge_l
            if args.data_parallel and torch.cuda.is_available():
                torch.save(model.module, os.path.join(args.model_dir, 'summary_model'))
            else:
                torch.save(model, os.path.join(args.model_dir, 'summary_model'))
        # torch.save(model, os.path.join(args.model_dir, 'summary_model_epoch_{}'.format(str(epoch))))


def init_argument():
    parser = argparse.ArgumentParser(description='t5-pegasus-chinese')
    parser.add_argument('--train_data', default='./data/train.tsv')
    parser.add_argument('--dev_data', default='./data/dev.tsv')
    parser.add_argument('--pretrain_model', default='./t5_pegasus_pretrain')
    parser.add_argument('--model_dir', default='./saved_model')

    parser.add_argument('--num_epoch', default=15, help='number of epoch')
    parser.add_argument('--batch_size', default=4, help='batch size')
    parser.add_argument('--lr', default=2e-4, help='learning rate')
    parser.add_argument('--data_parallel', default=False)
    parser.add_argument('--max_len', default=512, help='max length of inputs')
    parser.add_argument('--max_len_generate', default=40, help='max length of generated text')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # step 1. 初始化参数
    args = init_argument()

    # step 2. 准备训练数据和验证数据
    tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model)
    train_data = prepare_data(args, args.train_data, tokenizer, term='train')
    dev_data = prepare_data(args, args.dev_data, tokenizer, term='dev')

    # step 3. 加载预训练模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MT5ForConditionalGeneration \
        .from_pretrained(args.pretrain_model).to(device)
    if args.data_parallel and torch.cuda.is_available():
        device_ids = range(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # step 4. 微调
    adam = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_model(model, adam, train_data, dev_data, tokenizer, device, args)
