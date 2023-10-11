## Dependency:

* [torch-struct](https://github.com/harvardnlp/pytorch-struct)
* [transformers](https://github.com/huggingface/transformers)

## Preparation
Firstly, put your dataset under the data folder.

Secondly, pretrained LM (i.e., [BERT](https://www.aclweb.org/anthology/N19-1423/))
Create a dir. named "resource" and arrange them as
- resource
    - bert-base-cased
        - model.pt
        - vocab.txt
    - conlleval.pl

## Running our approaches

Train:
```bash
python train.py --output_dir outputs_conll --model_type bert --config_name ./resource/bert-base-cased \
--model_name_or_path ./resource/bert-base-cased --train_file data/conll2003++/pot/train.txt \
--predict_file data/conll2003++/pot/dev_full.txt --test_file data/conll2003++/pot/test_full.txt --max_seq_length 64 \
--per_gpu_train_batch_size 96 --per_gpu_eval_batch_size 96 --do_train --do_predict --learning_rate 3e-5 \
--num_train_epochs 20 --overwrite_output_dir --save_steps 1000 --dataset CONLL --potential_normalization True \
--structure_smoothing_p 0.98 --parser_type deepbiaffine --latent_size 1 --seed 12345 \
--trained_models train_models/1.pt --save_predict_path predict_file/conll \
--trainset data/conll2003++/neg/train.json --testset data/conll2003++/neg/test.json \
--validset data/conll2003++/neg/dev.json --finetuning \
--logdir pt_file/conll --batch_size 16 --n_epochs 40 --p 0.5 --new_trainset new_data/conll++2003/train.txt \
--relabeled_train data/conll2003++/pot/train_full.txt \
--predict_file_bert predict_file_bert/conll --bert_probs_file bert_probs_file/conll \
--bert_dropout_file bert_probs_file/conll --interpolation_ori 0.95 --dropout_num 21 \
--interpolation_high 0.95 -cd checkpoints -rd resource --bert_preds_file bert_preds_file/conll --method 5
