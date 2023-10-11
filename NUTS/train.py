from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import datetime

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter

from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForTokenClassification, BertTokenizer)

from utils_ner import Processor, eval

from model import PartialPCFG

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from torch.nn.utils.rnn import pad_sequence
import os
import random
import numpy as np

from conlleval_pro5 import evaluate_conll_file
import torch.nn.functional as F
import pickle
import time

import os
import json

from pytorch_pretrained_bert import BertAdam

from bert_neg_utils import UnitAlphabet, LabelAlphabet
from bert_neg_model import PhraseClassifier
from misc import fix_random_seed
from bert_neg_utils import corpus_to_iterator, Procedure

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_num_threads(4)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--train_file", default=None, type=str, required=True,
                    help="training data")
parser.add_argument("--predict_file", default=None, type=str, required=True,
                    help="development data")
parser.add_argument("--test_file", default=None, type=str, required=True,
                    help="test data")
parser.add_argument("--model_type", default=None, type=str, required=True,
                    help="Model type -- BERT")
parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                    help="Path to pre-trained model or shortcut name")
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model checkpoints and predictions will be written.")
parser.add_argument("--lambda_ent", default=1e-2, type=float,
                    help="lambda_ent")
parser.add_argument("--dataset", default="ACE05", type=str,
                    help="ACE04,ACE05,GENIA,NNE")
parser.add_argument("--latent_size", default=7, type=int,
                    help="latent_size")

## Other parameters
parser.add_argument("--config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--tokenizer_name", default="", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--cache_dir", default="", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--max_seq_length", default=384, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                         "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--do_train", action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_predict", action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--evaluate_during_training", action='store_true',
                    help="Rul evaluation during training at each logging step.")
parser.add_argument("--do_lower_case", action='store_true',
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=10, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument('--logging_steps', type=int, default=50,
                    help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=50,
                    help="Save checkpoint every X updates steps.")
parser.add_argument("--eval_all_checkpoints", action='store_true',
                    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
parser.add_argument("--no_cuda", action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--overwrite_output_dir', action='store_true',
                    help="Overwrite the content of the output directory")
parser.add_argument('--overwrite_cache', action='store_true',
                    help="Overwrite the cached training and evaluation sets")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")

parser.add_argument('--parser_type', type=str, default='bilinear',
                    help="bilinear, biaffine, or deepbiaffine")
parser.add_argument('--parser_dropout', type=float, default=0.,
                    help="parser dropout probability")
parser.add_argument('--state_dropout_p', type=float, default=0.0,
                    help="state dropout probability")
parser.add_argument('--state_dropout_mode', type=str, default='latent',
                    help="state dropout mode, latent or full")
parser.add_argument('--structure_smoothing_p', type=float, default=1.0,
                    help="structure smoothing ratio")
parser.add_argument('--potential_normalization', type=str2bool,
                    nargs='?', const=True, default=False,
                    help="if use potential normalization")
parser.add_argument('--use_vanilla_crf', type=str2bool,
                    nargs='?', const=True, default=False,
                    help="if use vanilla crf implementation, slow speed")
parser.add_argument('--use_crf', type=str2bool,
                    nargs='?', const=True, default=True,
                    help="if use crf for parsing, else use local normalization")
parser.add_argument('--decode_method', type=str, default='argmax',
                    help="argmax or marginal")
parser.add_argument('--no_batchify', type=str2bool,
                    nargs='?', const=True, default=False,
                    help="if do not batchify the inside algorithm")
parser.add_argument('--full_print', type=str2bool,
                    nargs='?', const=True, default=False,
                    help="if full print for tree, else only print not latent node")

parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--fp16', action='store_true',
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
parser.add_argument('--gpu_id', type=int, default=-1, help="gpu_id")

#lwc添加
parser.add_argument("--check_dir", "-cd", type=str, required=True)
parser.add_argument("--resource_dir", "-rd", type=str, required=True)
parser.add_argument("--miu", type=int, default=8)
parser.add_argument("--T", type=float, default=2)
parser.add_argument("--warmup_proportion", "-wp", type=float, default=0.1)
parser.add_argument("--hidden_dim", "-hd", type=int, default=256)
parser.add_argument("--dropout_rate", "-dr", type=float, default=0.1)
parser.add_argument("--trained_models", default=None, type=str,
                    help="trained_models")
parser.add_argument("--save_predict_path", default=None, type=str,
                    help="save_predict_path")
parser.add_argument("--batch_size",type=int,default=16)
parser.add_argument("--lr",type=float,default=0.00001)
parser.add_argument("--n_epochs",type=int,default=10)
parser.add_argument("--finetuning",dest="finetuning",action="store_true")
parser.add_argument("--top_rnns",dest="top_rnns",action="store_true")
parser.add_argument("--logdir",type=str,default="checkpoints/02")
parser.add_argument("--trainset",type=str,default="data/weibo_++/train.txt")
parser.add_argument("--validset",type=str,default="data/weibo_++/dev.txt")
parser.add_argument("--testset",type=str,default="data/weibo_++/test.txt")

parser.add_argument("--p",type=float,default=0.7)
parser.add_argument("--new_trainset",type=str,default="new_data/weibo_++/train.txt")
parser.add_argument("--new_validset",type=str,default="new_data/weibo_++/dev.txt")
parser.add_argument("--new_testset",type=str,default="new_data/weibo_++/test.txt")
parser.add_argument("--relabeled_train",type=str,default="data/weibo_++/train.txt")
parser.add_argument("--predict_file_bert",type=str,default=None)
parser.add_argument("--bert_probs_file",type=str,default=None)
parser.add_argument("--bert_dropout_file",type=str,default=None)
parser.add_argument("--bert_preds_file",type=str,default=None)
parser.add_argument("--interpolation_ori",type=float,default=1.0)
parser.add_argument("--interpolation_high",type=float,default=1.0)
parser.add_argument("--dropout_num",type=int,default=11)
parser.add_argument('--method', type=int, default=0,
                    help="0:drop,1:preds,2:probs,3:preds entropy,4:preds entropy,5:drop entropy")


args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
log_file_name = time.strftime("%Y%m%d-%H%M%S") + '.log.txt'
log_file_name = os.path.join(args.output_dir, log_file_name)
file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
logging.root.addHandler(file_handler)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForTokenClassification, BertTokenizer)
}

best_F = -1.0

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, dev_dataset, test_dataset, model, tokenizer, processor,index):
    """ Train the model """

    global best_F

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    now_epoch=0
    for _ in train_iterator:
        '''lwc添加'''
        if _ >30:
            now_epoch=_
            break
        loss_ = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'gather_ids': batch[3],
                      'gather_masks': batch[4],
                      'partial_masks': batch[5],
                      'eval_masks':batch[6]
                      }
            # torch.set_printoptions(profile="full")
            # print('batch5',batch[5])
            # print('batch6',batch[6])
            # stop
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            loss_ += loss.item()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                # optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, dev_dataset, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model.module if hasattr(model,
                #                                             'module') else model  # Take care of distributed/parallel training
                #     model_to_save.save_pretrained(output_dir)
                #     torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                #     logger.info("Saving model checkpoint to %s", output_dir)
                #     best_F = evaluate(args, model, dev_dataset, tokenizer, processor, global_step, _, best_F, "dev")
                #     evaluate(args, model, test_dataset, tokenizer, processor, global_step, _, best_F, "test")

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        # output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # model_to_save = model.module if hasattr(model,
        #                                         'module') else model  # Take care of distributed/parallel training
        # model_to_save.save_pretrained(output_dir)
        # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        # logger.info("Saving model checkpoint to %s", output_dir)
        
        evaluate(args, model, dev_dataset, tokenizer, processor, global_step, str(index)+'_'+ str(_), "dev")
        evaluate(args, model, train_dataset, tokenizer, processor, global_step,str(index)+'_'+ str(_), "train")
        evaluate(args, model, test_dataset, tokenizer, processor, global_step, str(index)+'_'+ str(_), "test")

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step,now_epoch

def predict2file(args, outputs, partial_masks, labels, gather_masks):
    predicts=[]
    goldens=[]
    label_size=len(labels)
    gather_masks = gather_masks.sum(1).cpu().numpy()
    outputs = outputs.cpu().numpy()
    partial_masks = partial_masks.cpu().numpy()
    for output,partial_mask,l in zip(outputs,partial_masks,gather_masks):
        tmp_predicts=[]
        tmp_goldens=[]
        for i in range(l):
            for j in range(l):
                if output[i][j]>=0:
                    if output[i][j]<label_size:
                        tmp_predicts.append([str(i),str(j),labels[int(output[i][j])]])
                for k in range(label_size):
                    if partial_mask[i][j][k]==1:
                        tmp_goldens.append([str(i),str(j),labels[k]])
        predicts.append(tmp_predicts)
        goldens.append(tmp_goldens)
    return predicts,goldens

def bert2file(model,Words, Is_heads, Tags, Y_hat,file):
    sentences=[]
    labels=[]
    for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):

        y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
        preds = [idx2tag[hat] for hat in y_hat]

        # assert len(preds) == len(words.split()) == len(tags.split())
        for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
            word.append(w)
            pred.append(p)
        sentences.append(word)
        labels.append(pred)
    with open(file,'w',encoding='utf-8') as f:
        for sen,tags in zip(sentences,labels):
            for word,tag in zip(sen,tags):
                f.write(word+' '+tag+'\n')
            f.write('\n')

    return sentences,labels

def all_predict2file(predicts,goldens,file):
    with open(file,'w',encoding='utf-8') as f:
        for golden,predict in zip(goldens,predicts):
            for index in range(len(golden)):
                if index==len(golden)-1:
                    f.write(','.join((golden[index])))
                else:
                    f.write(','.join((golden[index])))
                    f.write('|')
            f.write('\n')

            for index in range(len(predict)):
                if index==len(predict)-1:
                    f.write(','.join((predict[index])))
                else:
                    f.write(','.join((predict[index])))
                    f.write('|')
            f.write('\n')
            f.write('\n')

def dropout_infer(model,inputs,dropout_num):
    results=[]
    shapes=0
    for i in range(dropout_num):
        if isinstance(model, torch.nn.DataParallel):
            outputs = model.module.infer(**inputs)
        else:
            outputs = model.infer(**inputs)
        shapes=outputs[0].shape

        if i==0:
            result=outputs[0]

        else:
            result=torch.cat((result,outputs[0]),dim=2)

    result=result.view(shapes[0],shapes[1],shapes[2],-1).int()
    result = torch.where(result == -1, 100000, result)

    for i in range(shapes[0]):
        for j in range(shapes[1]):
            for k in range(shapes[2]):
                
                # print(result[i][j][k])

                max_count,max_index=torch.topk(torch.bincount(result[i][j][k]),1)
                # print(max_count,max_index)
                outputs[0][i][j][k]=max_index
                # print(outputs[0][i][j][k])
                # stop
                # print(outputs[0][i][j][k])
    outputs[0] = torch.where(outputs[0] == 100000, -1, outputs[0])
    
    return outputs

def evaluate(args, model, dataset, tokenizer, processor, step, _, prefix):
    global best_F

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    nb_eval_steps, nb_eval_examples = 0, 0
    correct, precision, recall = 0, 0, 0
    correct_alls,precision_alls,recall_alls,F_alls=dict(),dict(),dict(),dict()
    all_predicts=[]
    all_goldens=[]

    for lab in processor.labels:
        correct_alls[lab]=0
        precision_alls[lab]=0
        recall_alls[lab]=0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        

        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'gather_ids': batch[3],
                      'gather_masks': batch[4]
                      }
            if isinstance(model, torch.nn.DataParallel):
                outputs = model.module.infer(**inputs)
            else:
                outputs = model.infer(**inputs)
            #lwc修改，使用eval_mask进行评估
            correct_count, pred_count, gold_count, correct_all,pred_all,gold_all = eval(args, outputs[0], batch[6], len(processor.labels),
                                                                  batch[4])

            # outputs=dropout_infer(model,inputs,5)

            tmp_predicts,tmp_goldens=predict2file(args, outputs[0], batch[6], processor.labels,
                                                                  batch[4])
            all_predicts+=tmp_predicts
            all_goldens+=tmp_goldens

            correct += correct_count
            precision += pred_count
            recall += gold_count
            for idx in range(len(processor.labels)):
                correct_alls[processor.labels[idx]]+=correct_all[idx]
                precision_alls[processor.labels[idx]]+=pred_all[idx]
                recall_alls[processor.labels[idx]]+=gold_all[idx]
            


        nb_eval_examples += inputs['input_ids'].size(0)
        nb_eval_steps += 1
    all_predict2file(all_predicts,all_goldens,args.save_predict_path+'_'+str(_)+'_'+prefix+'.txt')
    if precision > 0 and recall > 0 and correct > 0:
        precision = correct / precision
        recall = correct / recall
        F = 2 * precision * recall / (precision + recall)
    else:
        precision = 0
        recall = 0
        F = 0
    result_all=dict()
    for lab in processor.labels:
        if precision_alls[lab]>0 and recall_alls[lab]>0 and correct_alls[lab]>0:
            precision_alls[lab]=correct_alls[lab]/precision_alls[lab]
            recall_alls[lab]=correct_alls[lab]/recall_alls[lab]
            F_alls[lab]=2*precision_alls[lab]*recall_alls[lab]/(precision_alls[lab]+recall_alls[lab])
        else:
            precision_alls[lab]=0
            recall_alls[lab]=0
            F_alls[lab]=0
        result_all[lab]='precision:  '+str(precision_alls[lab])+'  recall:  '+str(recall_alls[lab])+'  F:  '+str(F_alls[lab])

    best_flag=False
    if prefix == 'dev' and F > best_F:
        best_flag=True
        best_F = F
        output_dir=args.output_dir
        # output_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}-{}'.format(_, step, int(F * 10000)))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)
    result = {'prefix': prefix,
              'epoch': _,
              'eval_precision': precision,
              'eval_recall': recall,
              'eval_F': F,
              'best_flag':best_flag
              }
    result.update(result_all)
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        # writer.write("Epoch = %d Step = %d Total Loss = %d \n" % _, step, tr_loss)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write("\n")

    model.train()

    return best_flag

def get_results(correct,precision,recall):
    if precision > 0 and recall > 0 and correct > 0:
        precision = correct / precision
        recall = correct / recall
        F = 2 * precision * recall / (precision + recall)
    else:
        precision = 0
        recall = 0
        F = 0
    return precision,recall,F

def load_and_cache_examples(args, tokenizer, processor, index,evaluate=False, output_examples=False):
    # Load data features from cache or dataset file

    logger.info("Creating features from dataset file at %s", args.train_file)
    train_examples = processor.get_train_examples(input_file=args.train_file)
    logger.info("Creating features from dataset file at %s", args.predict_file)
    dev_examples = processor.get_dev_examples(input_file=args.predict_file)
    logger.info("Creating features from dataset file at %s", args.test_file)
    test_examples = processor.get_dev_examples(input_file=args.test_file)
    logger.info("Creating features from dataset file at %s", args.relabeled_train)
    relabeled_examples = processor.get_dev_examples(input_file=args.relabeled_train)

    train_features = processor.convert_examples_to_features(train_examples, args.max_seq_length,
                                                            tokenizer,'train',index)
    dev_features = processor.convert_examples_to_features(dev_examples, args.max_seq_length,
                                                          tokenizer,'dev',index)
    test_features = processor.convert_examples_to_features(test_examples, args.max_seq_length,
                                                           tokenizer,'test',index)
    relabeled_features = processor.convert_examples_to_features(relabeled_examples, args.max_seq_length,
                                                           tokenizer,'relabeled_train',index)

    # if args.local_rank in [-1, 0]:
    #     logger.info("Saving features into cached file %s", dev_cached_features_file)
    #     torch.save(dev_features, dev_cached_features_file)

    # Convert to Tensors and build dataset

    def getTensorDataset(features):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_gather_ids = torch.tensor([f.gather_ids for f in features], dtype=torch.long)
        all_gather_masks = torch.tensor([f.gather_masks for f in features], dtype=torch.long)
        all_partial_masks = torch.tensor([f.partial_masks for f in features], dtype=torch.float)
        all_eval_masks = torch.tensor([f.eval_masks for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_gather_ids, all_gather_masks, all_partial_masks,all_eval_masks)
        return dataset

    train_dataset = getTensorDataset(train_features)
    dev_dataset = getTensorDataset(dev_features)
    test_dataset = getTensorDataset(test_features)
    relabeled_dataset = getTensorDataset(relabeled_features)

    return train_dataset, dev_dataset, test_dataset, relabeled_dataset


def main(pt_file,index,lexical_vocab, label_vocab):
    args.trained_models=pt_file
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case, do_basic_tokenize=False)
    # if index!=0:
    #     if inter_boost:
    #         args.interpolation_ori=max(0,args.interpolation_ori-0.1)
    #         args.interpolation_high=max(0,args.interpolation_high-0.1)
    #     else:
    #         args.interpolation_ori=min(1,args.interpolation_ori+0.1)
    #         args.interpolation_high=min(1,args.interpolation_high+0.1)
    processor = Processor(logger, args.dataset, args.latent_size,args.trained_models,tokenizer,args.bert_probs_file, \
                            args.bert_dropout_file,args.bert_preds_file,args.interpolation_ori,args.interpolation_high, \
                            args.dropout_num,args.method,lexical_vocab, label_vocab,args.config_name, \
                            args.hidden_dim, args.dropout_rate, args.p)

    config.label_size = len(processor.labels) + args.latent_size
    config.observed_label_size = len(processor.labels)
    config.latent_label_size = config.label_size - config.observed_label_size
    config.parser_type = args.parser_type
    config.parser_dropout = args.parser_dropout
    config.state_dropout_p = args.state_dropout_p
    config.state_dropout_mode = args.state_dropout_mode
    config.lambda_ent = args.lambda_ent
    config.structure_smoothing_p = args.structure_smoothing_p
    config.potential_normalization = args.potential_normalization
    config.decode_method = args.decode_method
    config.use_vanilla_crf = args.use_vanilla_crf
    config.use_crf = args.use_crf
    config.full_print = args.full_print
    config.no_batchify = args.no_batchify

    model = PartialPCFG.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("Training/evaluation parameters %s", args)

    train_dataset, dev_dataset, test_dataset, relabeled_dataset = load_and_cache_examples(args, tokenizer, processor, index, evaluate=False,
                                                                       output_examples=False)

    # Training

    if args.do_train:
        global_step, tr_loss,now_epoch = train(args, train_dataset, dev_dataset, test_dataset, model, tokenizer, processor,index)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_predict and args.local_rank in [-1, 0]:
        checkpoint = args.output_dir
        # path='outputs_ee/checkpoint-94-38095-7869'
        model = PartialPCFG.from_pretrained(checkpoint, config=config)
        model.to(args.device)
        evaluate(args, model, test_dataset, tokenizer, processor, 0, index, "eval_test")
        evaluate(args, model, train_dataset, tokenizer, processor, 0, index, "eval_train")
        evaluate(args, model, dev_dataset, tokenizer, processor, 0, index, "eval_dev")
        evaluate(args, model, relabeled_dataset, tokenizer, processor, 0, index, "eval_relabel_train")

    return args.relabeled_train,args.test_file,args.predict_file,args.save_predict_path+'_'+str(index)+'_eval_relabel_train.txt',args.save_predict_path+'_'+str(index)+'_eval_test.txt',args.save_predict_path+'_'+str(index)+'_eval_dev.txt'


def seed_torch(seed=42):
    seed=int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=True

def general_train_all(i_flag):
    hp=parser.parse_args()
    for k in hp.__dict__:
        print(k+": "+str(hp.__dict__[k]))

    device='cuda' if torch.cuda.is_available() else 'cpu'
    seed_torch(hp.seed)

    if i_flag>0:
        hp.trainset=hp.new_trainset
        # hp.testset=hp.new_testset
        # hp.validset=hp.new_validset
        print('trainset',hp.trainset)
        print('testset',hp.testset)
        print('validset',hp.validset)
    final_pt_file=''

    lexical_vocab = UnitAlphabet(os.path.join(hp.resource_dir, hp.config_name, "vocab.txt"))
    label_vocab = LabelAlphabet()

    train_loader = corpus_to_iterator(hp.trainset, hp.batch_size, True, label_vocab)
    dev_loader = corpus_to_iterator(hp.validset, hp.batch_size, False, label_vocab)
    test_loader = corpus_to_iterator(hp.testset, hp.batch_size, False, label_vocab)

    bert_path = os.path.join(hp.resource_dir, hp.config_name)
    model = PhraseClassifier(lexical_vocab, label_vocab, hp.hidden_dim,
                             hp.dropout_rate, hp.p, bert_path)
    model = model.cuda() if torch.cuda.is_available() else model.cpu()

    all_parameters = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_param = [{'params': [p for n, p in all_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                     {'params': [p for n, p in all_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    total_steps = int(len(train_loader) * (hp.n_epochs + 1))
    optimizer = BertAdam(grouped_param, hp.lr, warmup=hp.warmup_proportion, t_total=total_steps)

    if not os.path.exists(hp.check_dir):
        os.makedirs(hp.check_dir)
    best_dev = -1.0
    best_test=0.0
    best_epoch=0
    script_path = os.path.join(hp.resource_dir, "conlleval.pl")
    checkpoint_path = os.path.join(hp.check_dir, "pytorch_model.bin")

    for epoch in range(1,hp.n_epochs+1):
        loss, train_time = Procedure.train(model, train_loader, optimizer)
        print("[Epoch {:3d}] loss on train set is {:.5f} using {:.3f} secs".format(epoch, loss, train_time))

        dev_best, dev_f1, dev_time = Procedure.test(model, dev_loader, script_path)
        print("(Epoch {:3d}) f1 score on dev set is {:.5f} using {:.3f} secs".format(epoch, dev_f1, dev_time))

        if dev_f1 > best_dev:
            best_dev = dev_f1
            best_epoch=epoch
            test_best, test_f1, test_time = Procedure.test(model, test_loader, script_path)
            print("{{Epoch {:3d}}} f1 score on test set is {:.5f} using {:.3f} secs".format(epoch, test_f1, test_time))

            print("\n<Epoch {:3d}> save best model with score: {:.5f} in terms of test set".format(epoch, test_f1))
            torch.save(model, checkpoint_path)
            best_test=test_best
            fname=os.path.join(hp.logdir,str(epoch))
            torch.save(model.state_dict(),f"{fname}.pt")
            print(f"weights were saved to {fname}.pt")
            final_pt_file=fname+'.pt'
        print(end="\n\n")

    print('第几次迭代：',i_flag)
    print(best_epoch)
    print(best_test)
    print(hp.trainset)
    print('ending\n\n')

    return hp.new_trainset,hp.new_testset,hp.new_validset,final_pt_file,lexical_vocab, label_vocab

def predict_main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument("--lr",type=float,default=0.0001)
    parser.add_argument("--n_epochs",type=int,default=20)
    parser.add_argument("--finetuning",dest="finetuning",action="store_true")
    parser.add_argument("--top_rnns",dest="top_rnns",action="store_true")
    parser.add_argument("--logdir",type=str,default="checkpoints/02")
    parser.add_argument("--trainset",type=str,default="data/youku_liujian/train.txt")
    parser.add_argument("--validset",type=str,default="data/youku_liujian/dev.txt")
    parser.add_argument("--testset",type=str,default="data/youku_liujian/test.txt")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--p",type=float,default=0.5)
    parser.add_argument("--path",type=str,default='./checkpoints/02/20.pt')
    parser.add_argument("--save_path",type=str,default='./pickle_file/youku')
    hp=parser.parse_args()
    for k in hp.__dict__:
        print(k+": "+str(hp.__dict__[k]))

    device='cuda' if torch.cuda.is_available() else 'cpu'
    seed_torch(hp.seed)

    train_dataset=NerDataset(hp.trainset,hp.p)
    eval_dataset=NerDataset(hp.validset,hp.p)
    test_dataset=NerDataset(hp.testset,hp.p)

    train_iter=data.DataLoader(dataset=train_dataset,
                               batch_size=hp.batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=pad)
    eval_iter=data.DataLoader(dataset=eval_dataset,
                              batch_size=hp.batch_size,
                              shuffle=False,
                              num_workers=4,
                              collate_fn=pad)
    test_iter=data.DataLoader(dataset=test_dataset,
                               batch_size=hp.batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=pad)
    predict(hp.path,test_iter,hp.save_path+'_test')

def pot2bio(ori_file,predict_file,new_file):
    sentences=[]
    ori_tuples=[]
    tri_tuples=[]
    new_tuples_=[]
    bios=[]
    true_sentences=[]
    with open(ori_file,'r',encoding='utf-8') as f:
        lines=[]
        for line in f:
            lines.append(line.strip())
        for i in range(0,len(lines),4):
            sentences.append(lines[i])
            ori_tuples.append(lines[i+2].split('|'))
    with open(predict_file,'r',encoding='utf-8') as f:
        lines=[]
        for line in f:
            lines.append(line.strip())
        for i in range(0,len(lines),3):
            tri_tuples.append(lines[i+1].split('|'))
    for ori_sen,ori,tri in zip(sentences,ori_tuples,tri_tuples):
        i=0
        j=0
        new_tuples=[]
        while i<len(ori) and j<len(tri):
            if len(ori[i])==0:
                i+=1
                continue
            ori_pos,ori_tag=ori[i].split(' ')
            ori_start,ori_end=ori_pos.split(',')
            ori_start=int(ori_start)
            ori_end=int(ori_end)-1

            if len(tri[j])==0:
                # new_tuples.append(str(ori_start)+','+str(ori_end)+','+str(ori_tag))
                j+=1
                continue

            tri_start,tri_end,tri_tag=tri[j].split(',')
            tri_start=int(tri_start)
            tri_end=int(tri_end)

            if tri_end-tri_start>10:
                j+=1
                continue

            if tri_start<=ori_start and tri_end>=ori_end:
                i+=1
                # print(1)
            elif tri_end<ori_start:
                new_tuples.append(str(tri_start)+','+str(tri_end)+','+str(tri_tag))
                j+=1
                # print(2)
            elif tri_end>ori_end and tri_start<ori_end:
                new_tuples.append(str(tri_start)+','+str(tri_end)+','+str(tri_tag))
                i+=1
                j+=1
                # print(3)
            elif ori_end<=tri_start:
                new_tuples.append(str(ori_start)+','+str(ori_end)+','+str(ori_tag))
                i+=1
                # print(4)
            elif tri_start>ori_end:
                new_tuples.append(str(tri_start)+','+str(tri_end)+','+str(tri_tag))
                j+=1
                # print(5)
            elif tri_end<=ori_end and tri_start>=ori_start:
                new_tuples.append(str(ori_start)+','+str(ori_end)+','+str(ori_tag))
                i+=1
                j+=1
                # print(6)
            elif tri_end<ori_end and tri_start<ori_start:
                new_tuples.append(str(ori_start)+','+str(ori_end)+','+str(ori_tag))
                i+=1
                j+=1
            # print(i,j,ori,tri)

        while i<len(ori):
            ori_pos,ori_tag=ori[i].split(' ')
            ori_start,ori_end=ori_pos.split(',')
            ori_start=int(ori_start)
            ori_end=int(ori_end)-1
            new_tuples.append(str(ori_start)+','+str(ori_end)+','+str(ori_tag))
            i+=1
        while j<len(tri):
            if len(tri[j])==0:
                j+=1
                continue
            tri_start,tri_end,tri_tag=tri[j].split(',')
            new_tuples.append(str(tri_start)+','+str(tri_end)+','+str(tri_tag))
            j+=1
        
        if len(new_tuples)!=0:
            new_tuples_.append(list(set(new_tuples)))
            true_sentences.append(ori_sen)

    with open(new_file,'w',encoding='utf-8') as f:
        print('写入新的文件:',new_file)
        for sen,tri in zip(true_sentences,new_tuples_):
            sens=sen.split(' ')
            bios=['O' for i in range(len(sens))]
            for tri_tuple in tri:
                tris=tri_tuple.split(',')
                if len(tris)==3:
                    bios[int(tris[0])]='B-'+tris[-1]
                    for index in range(int(tris[0])+1,int(tris[1])+1):
                        bios[index]='I-'+tris[-1]
            for i in range(len(sens)):
                f.write(sens[i]+' '+bios[i])
                f.write('\n')
            f.write('\n')

def bio2pot(bio_path,pot_file):
    entries = open(bio_path, 'r').read().strip().split("\n\n")
    sents, tags_li = [], [] # list of lists
    for entry in entries:
        words = [line.split()[0] for line in entry.splitlines()]
        tags = ([line.split()[-1] for line in entry.splitlines()])

        sents.append(words)
        tags_li.append(tags)

    def get_entity(tags):
        i=0
        res=[]
        labels=[]
        while i<len(tags):
            if 'B' in tags[i]:
                start=i
                tag=tags[i].split('-')[1]
                i+=1
                while i<len(tags) and len(tags[i].split('-'))==2:
                    if tags[i].split('-')[0]=='I' and  tag in tags[i]:
                        i+=1
                    elif tags[i].split('-')[0]=='E' and  tag in tags[i]:
                        i+=1
                    else:
                        break
                end=i-1
                res.append([start,end,tag])
                labels.append(tag)
            else:
                i+=1
        return res

    print("写入")
    with open(pot_file,'w',encoding='utf-8') as f:
        for words,tags in zip(sents,tags_li):
            # print(len(words),len(tags))
            if len(words)==0:
                continue
            span=get_entity(tags)
            for idx in range(len(words)):
                if idx==len(words)-1:
                    f.write(words[idx]+'\n')
                else:
                    f.write(words[idx]+' ')
            for idx in range(len(words)):
                if idx==len(words)-1:
                    f.write(words[idx]+'\n')
                else:
                    f.write(words[idx]+' ')
            if len(span)==0:
                f.write('\n\n')
                continue
            for idx in range(len(span)):
                if idx==len(span)-1:
                    f.write(str(span[idx][0])+','+str(span[idx][1]+1)+' '+span[idx][-1]+'\n')
                else:
                    f.write(str(span[idx][0])+','+str(span[idx][1]+1)+' '+span[idx][-1]+'|')
            f.write('\n')

def pot2neg(path,filename):
    tags=[]
    sentences=[]
    with open(path,'r',encoding='utf-8') as f:
        lines=[]
        for line in f:
            lines.append(line.strip())
        for i in range(0,len(lines),4):
            sentences.append(lines[i])
            tags.append(lines[i+2])

    res=[]
    for sen,tag in zip(sentences,tags):
        data=dict()
        data['sentence']=str(sen.strip().split())
        entities=[]

        tag_tmp=tag.split('|')


        for tag_tm in tag_tmp:
            if len(tag_tm.split())<2:
                break
            pos,label=tag_tm.split()
            start,end=pos.split(',')
            start=int(start)
            end=int(end)-1
            entities.append((start, end, label))

        data['labeled entities']=str(entities)
        res.append(data)
    
    with open(filename, "w",encoding='utf-8') as outfile:
        json.dump(res, outfile,indent=4,ensure_ascii=False)

if __name__ == "__main__":
    # 完整交替训练
    for i in range(5):
        print('..........................')
        print('开始迭代',i)
        print('..........................')
        print('bert训练')
        new_train_file,new_test_file,new_dev_file,pt_file,lexical_vocab, label_vocab=general_train_all(i)
        print('partial tree 训练')

        ori_train_file,ori_test_file,ori_dev_file,predict_train_file,predict_test_file,predict_dev_file=main(pt_file,i,lexical_vocab, label_vocab)

        print('标记新的文件')
        pot2bio(ori_train_file,predict_train_file,new_train_file+'_bio')
        bio2pot(new_train_file+'_bio',new_train_file+'_pot')
        pot2neg(new_train_file+'_pot',new_train_file)
        print(ori_train_file,predict_train_file,new_train_file)
        print('标注完成')
        # pot2bio(ori_test_file,predict_test_file,new_test_file)
        # pot2bio(ori_dev_file,predict_dev_file,new_dev_file)
    print('训练结束')

