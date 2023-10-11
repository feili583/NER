import argparse
import os
import json

import torch
from pytorch_pretrained_bert import BertAdam

from bert_neg_utils import UnitAlphabet, LabelAlphabet
from bert_neg_model import PhraseClassifier
from misc import fix_random_seed
from bert_neg_utils import corpus_to_iterator, Procedure

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", "-dd", type=str, required=True)
    parser.add_argument("--check_dir", "-cd", type=str, required=True)
    parser.add_argument("--resource_dir", "-rd", type=str, required=True)
    parser.add_argument("--random_state", "-rs", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",type=float,default=0.00001)
    parser.add_argument("--miu", type=int, default=8)
    parser.add_argument("--T", type=float, default=2)

    parser.add_argument("--p", type=float, default=0.7)
    parser.add_argument("--warmup_proportion", "-wp", type=float, default=0.1)
    parser.add_argument("--hidden_dim", "-hd", type=int, default=256)
    parser.add_argument("--dropout_rate", "-dr", type=float, default=0.1)
    parser.add_argument("--config_name", "-pm", type=str, required=True)
    parser.add_argument("--trainset", type=str, required=True)
    parser.add_argument("--testset", type=str, required=True)
    parser.add_argument("--validset", type=str, required=True)

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=True), end="\n\n")

    fix_random_seed(args.random_state)


    lexical_vocab = UnitAlphabet(os.path.join(hp.resource_dir, hp.config_name, "vocab.txt"))
    label_vocab = LabelAlphabet()

    train_loader = corpus_to_iterator(hp.trainset, hp.batch_size, True, label_vocab)
    dev_loader = corpus_to_iterator(hp.validset, hp.batch_size, False)
    test_loader = corpus_to_iterator(hp.testset, hp.batch_size, False)

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
    best_dev = 0.0
    best_test=0.0
    best_epoch=0
    script_path = os.path.join(hp.resource_dir, "conlleval.pl")
    checkpoint_path = os.path.join(hp.check_dir, "pytorch_model.bin")

    for epoch in range(0, hp.n_epochs + 1):
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

    print(best_epoch)
    print(best_test)
    print(hp.trainset)
    print('ending\n\n')
