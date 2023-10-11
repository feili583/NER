import torch
from tqdm import tqdm
import sys
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer
import pickle
from bert_neg_model import PhraseClassifier


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, pos, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.pos = pos
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, gather_ids, gather_masks, partial_masks,eval_masks):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.gather_ids = gather_ids
        self.gather_masks = gather_masks
        self.partial_masks = partial_masks
        self.eval_masks = eval_masks


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, logger):
        self.logger = logger

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines


class Processor(DataProcessor):
    """Processor NQG data set."""

    def __init__(self, logger, dataset, latent_size,trained_models,tokenizer,bert_probs_file,bert_dropout_file,bert_preds_file, \
    interpolation_ori,interpolation_high,dropout_num,method,lexical_vocab, label_vocab,config_name,hidden_dim, dropout_rate, p):
        self.logger = logger
        if dataset == "ACE04" or dataset == "ACE05":
            self.labels = ['PER', 'LOC', 'ORG', 'GPE', 'FAC', 'VEH', 'WEA']
        elif dataset == "GENIA":
            self.labels = ['None', 'G#RNA', 'G#protein', 'G#DNA', 'G#cell_type', 'G#cell_line']
        elif dataset == "CONLL":
            self.labels = ['LOC', 'PER', 'ORG', 'MISC']
        elif dataset=='WEIBO_++':
            self.labels=['LOC.NAM','PER.NAM','PER.NOM','LOC.NOM','GPE.NAM','ORG.NOM','ORG.NAM','GPE.NOM']
        elif dataset=='WEIBO_++MERGE':
            self.labels=['LOC','PER','GPE','ORG']
        elif dataset=='YOUKU':
            self.labels=['TV','PER','NUM']
        elif dataset=='ECOMMERCE':
            self.labels=['HCCX','MISC','XH','HPPX']
        elif dataset=='ACE':
            self.labels=['Nation', 'rPHYS:Located', 'Plant', 'rORG-AFF:Founder', 'ORG-AFF:Student-Alum', 'trigger', 'Water-Body', 'ART:User-Owner-Inventor-Manufacturer', 'rJustice:Pardon', 'Life:Divorce', 'Land-Region-Natural', 'Celestial', 'rPHYS:Near', 'rBusiness:End-Org', 'Justice:Acquit', 'rPersonnel:End-Position', 'rJustice:Convict', 'Medical-Science', 'rORG-AFF:Student-Alum', 'rLife:Marry', 'Religious', 'rORG-AFF:Investor-Shareholder', 'rJustice:Sentence', 'PER-SOC:Family', 'Justice:Sentence', 'rGEN-AFF:Org-Location', 'Path', 'Contact:Phone-Write', 'Air', 'rART:User-Owner-Inventor-Manufacturer', 'County-or-District', 'rPER-SOC:Business', 'Life:Injure', 'rORG-AFF:Ownership', 'rLife:Injure', 'ORG-AFF:Membership', 'Land', 'Justice:Charge-Indict', 'Region-International', 'Special', 'GEN-AFF:Citizen-Resident-Religion-Ethnicity', 'rORG-AFF:Sports-Affiliation', 'Life:Die', 'rPersonnel:Start-Position', 'Population-Center', 'rConflict:Attack', 'Transaction:Transfer-Money', 'rPersonnel:Nominate', 'Personnel:End-Position', 'Exploding', 'Group', 'Justice:Convict', 'Life:Be-Born', 'rJustice:Acquit', 'rLife:Divorce', 'Commercial', 'PART-WHOLE:Geographical', 'Business:Start-Org', 'Justice:Trial-Hearing', 'rPersonnel:Elect', 'rGEN-AFF:Citizen-Resident-Religion-Ethnicity', 'rConflict:Demonstrate', 'ORG-AFF:Ownership', 'Region-General', 'rTransaction:Transfer-Ownership', 'Justice:Pardon', 'PER-SOC:Business', 'Conflict:Attack', 'rORG-AFF:Employment', 'Justice:Execute', 'rLife:Die', 'Movement:Transport', 'Subarea-Vehicle', 'rJustice:Arrest-Jail', 'Subarea-Facility', 'rORG-AFF:Membership', 'ORG-AFF:Employment', 'ORG-AFF:Sports-Affiliation', 'Projectile', 'Sports', 'Conflict:Demonstrate', 'rLife:Be-Born', 'Justice:Sue', 'Airport', 'rJustice:Release-Parole', 'Personnel:Start-Position', 'Business:Merge-Org', 'rPER-SOC:Family', 'rPART-WHOLE:Geographical', 'rJustice:Appeal', 'GEN-AFF:Org-Location', 'rTransaction:Transfer-Money', 'Life:Marry', 'Justice:Release-Parole', 'Indeterminate', 'Blunt', 'PART-WHOLE:Subsidiary', 'rJustice:Trial-Hearing', 'Media', 'Boundary', 'rPART-WHOLE:Subsidiary', 'rPART-WHOLE:Artifact', 'Justice:Extradite', 'Justice:Appeal', 'Sharp', 'rJustice:Charge-Indict', 'rJustice:Fine', 'PHYS:Located', 'Contact:Meet', 'Personnel:Nominate', 'Individual', 'rJustice:Execute', 'Business:Declare-Bankruptcy', 'Business:End-Org', 'rMovement:Transport', 'rContact:Phone-Write', 'Continent', 'Biological', 'Justice:Arrest-Jail', 'Educational', 'rJustice:Extradite', 'rContact:Meet', 'ORG-AFF:Founder', 'PART-WHOLE:Artifact', 'rBusiness:Start-Org', 'Personnel:Elect', 'Non-Governmental', 'State-or-Province', 'Nuclear', 'Water', 'rPER-SOC:Lasting-Personal', 'ORG-AFF:Investor-Shareholder', 'Building-Grounds', 'Government', 'Underspecified', 'rBusiness:Declare-Bankruptcy', 'GPE-Cluster', 'Justice:Fine', 'Chemical', 'rBusiness:Merge-Org', 'PER-SOC:Lasting-Personal', 'Entertainment', 'PHYS:Near', 'Transaction:Transfer-Ownership', 'Address', 'rJustice:Sue', 'Shooting']
        elif dataset=='JSON_ACE':
            self.labels=['Victim', 'FAC', 'rAttacker', 'rProsecutor', 'rORG-AFF.Founder', 'rInstrument', 'ORG-AFF.Student-Alum', 'rAgent', 'ORG-AFF.Ownership', 'PER-SOC.Lasting-Personal', 'rOrigin', 'trigger', 'GEN-AFF.Citizen-Resident-Religion-Ethnicity', 'rORG-AFF.Membership', 'ART.User-Owner-Inventor-Manufacturer', 'Giver', 'rGEN-AFF.Org-Location', 'Artifact', 'rPART-WHOLE.Subsidiary', 'rPlace', 'rPHYS.Near', 'Org', 'PER-SOC.Business', 'rSeller', 'rART.User-Owner-Inventor-Manufacturer', 'VEH', 'PER-SOC.Family', 'GPE', 'Place', 'Entity', 'rOrg', 'rGEN-AFF.Citizen-Resident-Religion-Ethnicity', 'LOC', 'Agent', 'rEntity', 'rDestination', 'ORG-AFF.Founder', 'rTarget', 'rVehicle', 'rPlaintiff', 'rORG-AFF.Sports-Affiliation', 'Defendant', 'Attacker', 'rPerson', 'Vehicle', 'PER', 'rGiver', 'rAdjudicator', 'rORG-AFF.Employment', 'Instrument', 'ORG-AFF.Sports-Affiliation', 'rBuyer', 'PART-WHOLE.Artifact', 'Person', 'Beneficiary', 'Adjudicator', 'rPER-SOC.Lasting-Personal', 'Plaintiff', 'rORG-AFF.Ownership', 'ORG-AFF.Investor-Shareholder', 'rPER-SOC.Family', 'ORG-AFF.Membership', 'GEN-AFF.Org-Location', 'rPART-WHOLE.Artifact', 'PART-WHOLE.Geographical', 'Target', 'rPART-WHOLE.Geographical', 'rDefendant', 'WEA', 'rORG-AFF.Investor-Shareholder', 'PART-WHOLE.Subsidiary', 'ORG-AFF.Employment', 'Seller', 'Origin', 'PHYS.Located', 'rVictim', 'Prosecutor', 'rRecipient', 'Buyer', 'Destination', 'rArtifact', 'rBeneficiary', 'ORG', 'rPER-SOC.Business', 'rPHYS.Located', 'Recipient', 'PHYS.Near', 'rORG-AFF.Student-Alum']
        else:
            raise NotImplementedError()

        if dataset == "ACE05" or dataset == "GENIA" or dataset == "ACE04" or dataset == "CONLL" \
                or dataset=='WEIBO_++' or dataset=='WEIBO_++MERGE' or dataset=='YOUKU' or dataset=='ECOMMERCE' \
                or dataset=='ACE' or dataset=='JSON_ACE':
            self.interval = 4
        else:
            raise NotImplementedError()

        self.trained_model_path=trained_models
        self.true_mask=dataset
        self.latent_size = latent_size
        self.trained_tokenizer=tokenizer
        self.bert_probs_file=bert_probs_file
        self.bert_dropout_file=bert_dropout_file
        self.bert_preds_file=bert_preds_file
        self.interpolation_ori=interpolation_ori
        self.dropout_num=dropout_num
        self.interpolation_high=interpolation_high
        self.method=method
        self.lexical_vocab=lexical_vocab
        self.label_vocab=label_vocab
        self.config_name=config_name
        self.hidden_dim=hidden_dim
        self.dropout_rate=dropout_rate
        self.p=p

        with torch.no_grad():
            model = PhraseClassifier(self.lexical_vocab, self.label_vocab, self.hidden_dim,
                             self.dropout_rate, self.p, self.config_name)
            self.trained_model = model.cuda() if torch.cuda.is_available() else model.cpu()
            self.trained_model.load_state_dict(torch.load(self.trained_model_path))
            self.trained_model.eval()
      

    def get_train_examples(self, input_file):
        """See base class."""
        self.logger.info("LOOKING AT {}".format(input_file))
        return self._create_examples(
            self._read(input_file), "train")

    def get_dev_examples(self, input_file):
        """See base class."""
        self.logger.info("LOOKING AT {}".format(input_file))
        return self._create_examples(
            self._read(input_file), "dev")

    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        examples = []

        for i in range(0, len(lines), self.interval):
            text_a = lines[i]
            label = lines[i + 2]

            examples.append(
                InputExample(guid=len(examples), text_a=text_a, pos=None, label=label))
        return examples

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer,flag,index):
        """Loads a data file into a list of `InputBatch`s."""

        features = []

        for (ex_index, example) in enumerate(tqdm(examples)):

            tokens = tokenizer.tokenize(example.text_a)

            gather_ids = list()
            for (idx, token) in enumerate(tokens):
                if (not token.startswith("##") and idx < max_seq_length - 2):
                    gather_ids.append(idx + 1)

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:max_seq_length - 2]

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            
            #is_head
            gather_padding = [0] * (max_seq_length - len(gather_ids))
            gather_masks = [1] * len(gather_ids) + gather_padding
            #is_head在句子中的索引
            gather_ids += gather_padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(gather_ids) == max_seq_length
            assert len(gather_masks) == max_seq_length

            eval_masks,partial_masks = self.generate_partial_masks(example.text_a.split(' '), max_seq_length, example.label,
                                                            self.labels,flag,index)

            if ex_index < 2:
                self.logger.info("*** Example ***")
                self.logger.info("guid: %s" % (example.guid))
                self.logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                self.logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                self.logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                self.logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                self.logger.info(
                    "gather_ids: %s" % " ".join([str(x) for x in gather_ids]))
                self.logger.info(
                    "gather_masks: %s" % " ".join([str(x) for x in gather_masks]))
                # self.logger.info("label: %s (id = %s)" % (example.label, " ".join([str(x) for x in label_ids])))

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              partial_masks=partial_masks,
                              gather_ids=gather_ids,
                              gather_masks=gather_masks,
                              eval_masks=eval_masks))

        return features

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def get_probs(self,x,is_heads,y,tokens,flag,index):
        with torch.no_grad():
            self.trained_model.eval()
                # self.trained_tokenizer=BertTokenizer.from_pretrained('./bert-chinese-wwm')
            if self.true_mask=='CONLL':
                _,logits,y,y_hat=self.trained_model(x.cuda(),y.cuda())
            else:
                logits,y,y_hat=self.trained_model(x.cuda(),y.cuda())
            logits=torch.softmax(logits,-1)
            probs=[logit.numpy().tolist() for logit,is_head in zip(logits.cpu()[0],is_heads[0]) if is_head==1]

        with open(self.bert_probs_file+'_'+str(index)+'_'+flag+'_'+'prob.txt','ab') as fw:
                pickle.dump(probs,fw)
        with open(self.bert_probs_file+'_'+str(index)+'_'+flag+'_'+'prob_sen.txt','a') as fw:
            fw.write(' '.join(tokens)+'\n')
        return probs

    def get_distri(self,x,is_heads,y,tokens,flag,index):
        dropout_num=self.dropout_num
        with torch.no_grad():
            self.trained_model.train()
            for num in range(dropout_num):
                if self.true_mask=='CONLL':
                    _,logits,y,y_hat=self.trained_model(x.cuda(),y.cuda())
                else:
                    logits,y,y_hat=self.trained_model(x.cuda(),y.cuda())

                one_matrix=torch.ones_like(logits[:,1:-1,:],)
                logits=torch.softmax(logits,-1)[:,1:-1]
                mask=(logits==logits.max(dim=-1,keepdim=True)[0])
                
                result=torch.mul(mask,one_matrix)
                if num==0:
                    Y=result
                else:
                    Y=torch.add(Y,result)

            one_matrix=torch.ones_like(is_heads[:,1:-1])
            one_matrix=one_matrix==is_heads[:,1:-1]

            Y=(Y[one_matrix]/dropout_num).cpu().numpy().tolist()

            with open(self.bert_dropout_file+'_'+str(index)+'_'+flag+'_'+'drop_prob.txt','ab') as fw:
                pickle.dump(Y,fw)
            with open(self.bert_dropout_file+'_'+str(index)+'_'+flag+'_'+'drop_sen.txt','a') as fw:
                fw.write(' '.join(tokens)+'\n')
        return Y

    def get_distri_span(self,tokens,flag,index):
        dropout_num=self.dropout_num
        with torch.no_grad():
            self.trained_model.train()

            for num in range(dropout_num):
                entities,scores=self.trained_model.inference([tokens])
                one_matrix=torch.ones_like(scores)
                mask=(scores==scores.max(dim=-1,keepdim=True)[0])

                result=torch.mul(mask,one_matrix)
                if num==0:
                    Y=result
                else:
                    Y=torch.add(Y,result)

            Y=(Y.squeeze(0)/dropout_num).cpu().numpy().tolist()

            with open(self.bert_dropout_file+'_'+str(index)+'_'+flag+'_'+'drop_prob.txt','ab') as fw:
                pickle.dump(Y,fw)
            with open(self.bert_dropout_file+'_'+str(index)+'_'+flag+'_'+'drop_sen.txt','a') as fw:
                fw.write(' '.join(tokens)+'\n')
        return Y

    def get_preds(self,x,is_heads,y,tokens,flag,index):
        with torch.no_grad():
            self.trained_model.eval()

            if self.true_mask=='CONLL':
                _,logits,y,y_hat=self.trained_model(x.cuda(),y.cuda())
            else:
                logits,y,y_hat=self.trained_model(x.cuda(),y.cuda())

            one_matrix=torch.ones_like(logits[:,:,:],)
            logits=torch.softmax(logits,-1)
            mask=(logits==logits.max(dim=-1,keepdim=True)[0])
            result=torch.mul(mask,one_matrix)

            preds=[logit.numpy().tolist() for logit,is_head in zip(result.cpu()[0],is_heads[0]) if is_head==1]

        with open(self.bert_preds_file+'_'+str(index)+'_'+flag+'_'+'preds.txt','ab') as fw:
                pickle.dump(preds,fw)
        with open(self.bert_preds_file+'_'+str(index)+'_'+flag+'_'+'preds_sen.txt','a') as fw:
            fw.write(' '.join(tokens)+'\n')
        return preds

    def get_preds_span(self,tokens,flag,index):

        with torch.no_grad():
            entities,scores=self.trained_model.inference([tokens])
            scores=torch.exp(scores)
            Y=scores.squeeze(0).cpu().numpy().tolist()

            with open(self.bert_dropout_file+'_'+str(index)+'_'+flag+'_'+'preds_prob.txt','ab') as fw:
                pickle.dump(Y,fw)
            with open(self.bert_dropout_file+'_'+str(index)+'_'+flag+'_'+'preds_sen.txt','a') as fw:
                fw.write(' '.join(tokens)+'\n')
        return Y

    def get_preds_one_span(self,tokens,flag,index):

        with torch.no_grad():
            entities,scores=self.trained_model.inference([tokens])
            scores=scores.squeeze(0)

            one_matrix=torch.ones_like(scores[:,:,:],)
            mask=(scores==scores.max(dim=-1,keepdim=True)[0])
            preds=torch.mul(mask,one_matrix).cpu().numpy().tolist()

        with open(self.bert_preds_file+'_'+str(index)+'_'+flag+'_'+'preds_one.txt','ab') as fw:
                pickle.dump(preds,fw)
        with open(self.bert_preds_file+'_'+str(index)+'_'+flag+'_'+'preds_one_sen.txt','a') as fw:
            fw.write(' '.join(tokens)+'\n')
        return preds

    def get_preds_span_entropy(self,tokens,flag,index):

        with torch.no_grad():
            entities,scores=self.trained_model.inference([tokens])
            scores=torch.exp(scores)
            scores=torch.softmax(scores,-1)

            
            entropy=-torch.sum(torch.mul(scores,torch.log(scores)),-1)

            all_num=entropy.shape[0]*entropy.shape[1]
            
            mask = entropy < torch.topk(entropy, int(all_num*0.3))[0][..., -1, None]
            mask=mask.unsqueeze(3).repeat(1,1,1,scores.shape[-1]).squeeze(0).cpu().numpy().tolist()

            scores=scores.squeeze(0)
            one_matrix=torch.ones_like(scores[:,:,:],)
            scores_mask=(scores==scores.max(dim=-1,keepdim=True)[0])
            Y=torch.mul(scores_mask,one_matrix).cpu().numpy().tolist()
            
            with open(self.bert_dropout_file+'_'+str(index)+'_'+flag+'_'+'preds_prob_entropy_one.txt','ab') as fw:
                pickle.dump(Y,fw)
            with open(self.bert_dropout_file+'_'+str(index)+'_'+flag+'_'+'preds_sen_entropy_one.txt','a') as fw:
                fw.write(' '.join(tokens)+'\n')
        return Y,mask

    def get_distri_span_entropy(self,tokens,flag,index):

        dropout_num=self.dropout_num
        with torch.no_grad():
            self.trained_model.train()

            for num in range(dropout_num):
                entities,scores=self.trained_model.inference([tokens])
                one_matrix=torch.ones_like(scores)
                mask=(scores==scores.max(dim=-1,keepdim=True)[0])

                result=torch.mul(mask,one_matrix)
                if num==0:
                    Y=result
                else:
                    Y=torch.add(Y,result)
            Y=Y.squeeze(0)/dropout_num

            one_matrix=torch.ones_like(Y)
            mask=Y>0

            Y_mask=torch.mul(mask,one_matrix)

            mask=torch.sum(Y_mask,-1).unsqueeze(2).repeat(1,1,scores.shape[-1]).cpu().numpy().tolist()
            Y=Y.cpu().numpy().tolist()

            with open(self.bert_dropout_file+'_'+str(index)+'_'+flag+'_'+'drop_entropy.txt','ab') as fw:
                pickle.dump(Y,fw)
            with open(self.bert_dropout_file+'_'+str(index)+'_'+flag+'_'+'drop_sen_entropy.txt','a') as fw:
                fw.write(' '.join(tokens)+'\n')
        return Y,mask

    def get_preds_span_max_sec(self,tokens,flag,index):

        with torch.no_grad():
            entities,scores=self.trained_model.inference([tokens])
            scores=torch.exp(scores)
            scores=torch.softmax(scores,-1).squeeze(0)

            one_matrix=torch.ones_like(scores[:,:,:],)
            scores_mask=(scores==scores.max(dim=-1,keepdim=True)[0])

            Y=torch.mul(scores_mask,one_matrix).cpu().numpy().tolist()

            max_score=scores.max(dim=-1)[0]
            mask=(max_score>0.6).unsqueeze(2).repeat(1,1,scores.shape[-1])

            with open(self.bert_dropout_file+'_'+str(index)+'_'+flag+'_'+'preds_prob_span_max.txt','ab') as fw:
                pickle.dump(Y,fw)
            with open(self.bert_dropout_file+'_'+str(index)+'_'+flag+'_'+'preds_sen_span_max.txt','a') as fw:
                fw.write(' '.join(tokens)+'\n')
        return Y,mask


    def generate_partial_masks(self, tokens, max_seq_length, labels, tags,flag,index):

        #所有标签
        total_tags_num = len(tags) + self.latent_size
        #L0，已知标签序列
        labels = labels.split('|')
        label_list = list()

        for label in labels:
            if not label:
                continue
            sp = label.strip().split(' ')
            start, end = sp[0].split(',')[:2]
            start = int(start)
            end = int(end) - 1
            label_list.append((start, end, sp[1]))

        #初始化所有节点为隐藏节点,生成评估矩阵
        mask = [[[2 for x in range(total_tags_num)] for y in range(max_seq_length)] for z in range(max_seq_length)]
        l = min(len(tokens), max_seq_length)

        # 2 marginalization
        # 1 evaluation
        # 0 rejection

        for start, end, tag in label_list:
            if start < max_seq_length and end < max_seq_length:
                tag_idx = tags.index(tag)
                mask[start][end][tag_idx] = 1
                for k in range(total_tags_num):
                    if k != tag_idx:
                        mask[start][end][k] = 0

            for i in range(l):
                if i > end:
                    continue
                for j in range(i, l):
                    if j < start:
                        continue
                    if (i > start and i <= end and j > end) or (i < start and j >= start and j < end):
                        for k in range(total_tags_num):
                            mask[i][j][k] = 0

        for i in range(l):
            for j in range(0, i):
                for k in range(total_tags_num):
                    mask[i][j][k] = 0

        for i in range(l):
            for j in range(i, l):
                for k in range(total_tags_num):
                    if mask[i][j][k] == 2:
                        if k < len(tags):
                            mask[i][j][k] = 0
                        else:
                            mask[i][j][k] = 1

        for i in range(max_seq_length):
            for j in range(max_seq_length):
                for k in range(total_tags_num):
                    if mask[i][j][k] == 2:
                        mask[i][j][k] = 0
        
        true_mask = [[[2 for x in range(total_tags_num)] for y in range(max_seq_length)] for z in range(max_seq_length)]
        l = min(len(tokens), max_seq_length)

        # 2 marginalization
        # 1 evaluation
        # 0 rejection

        if self.true_mask=='CONLL' or self.true_mask=='ECOMMERCE' or self.true_mask=='WEIBO_++' \
            or self.true_mask=='WEIBO_++MERGE' or self.true_mask=='YOUKU':
            if self.method==0:
                probs=self.get_distri_span(tokens,flag,index)
            elif self.method==1:
                probs=self.get_preds_one_span(tokens,flag,index)
            elif self.method==2:
                probs=self.get_preds_span(tokens,flag,index)
            elif self.method==3:
                probs,bool_mask=self.get_preds_span_entropy(tokens,flag,index)
            elif self.method==4:
                probs,bool_mask=self.get_preds_span_max_sec(tokens,flag,index)
            elif self.method==5:
                probs,bool_mask=self.get_distri_span_entropy(tokens,flag,index)

            for start, end, tag in label_list:
                if start < max_seq_length and end < max_seq_length:
                    probs_sum=0

                    tag_idx = tags.index(tag)
                    true_mask[start][end][tag_idx] = 1

                    for k in range(total_tags_num):
                        if self.method==5:
                            if bool_mask[start][end][k]==1:
                                true_mask[start][end][k]=mask[start][end][k]
                            else:
                                if k<len(tags):
                                    true_mask[start][end][k]=0
                                else:
                                    true_mask[start][end][k]=1
                            
                        else:
                            if k<len(tags):
                                true_mask[start][end][k] = self.interpolation_ori*mask[start][end][k]+(1-self.interpolation_ori)*probs[start][end][self.label_vocab.index(tags[k])]
                                probs_sum+=true_mask[start][end][k]
                            else:
                                true_mask[start][end][k]=1-probs_sum

                for i in range(l):
                    if i > end:
                        continue
                    for j in range(i, l):
                        if j < start:
                            continue
                        if (i > start and i <= end and j > end) or (i < start and j >= start and j < end):
                            for k in range(total_tags_num):
                                true_mask[i][j][k] = 0

            for i in range(l):
                for j in range(0, i):
                    for k in range(total_tags_num):
                        true_mask[i][j][k] = 0

            #L0=0,L1=1
            for i in range(l):
                for j in range(i, l):
                    probs_sum=0
                    for k in range(total_tags_num):
                        if true_mask[i][j][k] == 2:
                            if self.method==5:
                                true_mask[i][j][k]=mask[i][j][k]
                            else:
                                if k < len(tags):
                                    true_mask[i][j][k] = self.interpolation_high*mask[i][j][k]+(1-self.interpolation_high)*probs[i][j][self.label_vocab.index(tags[k])]
                                    probs_sum+=true_mask[i][j][k]
                                else:
                                    true_mask[i][j][k] = 1-probs_sum

            for i in range(max_seq_length):
                for j in range(max_seq_length):
                    for k in range(total_tags_num):
                        if true_mask[i][j][k] == 2:
                            true_mask[i][j][k] = 0
        else:
            true_mask=mask

        return mask,true_mask

class MultitasksResultItem():

    def __init__(self, id, start_prob, end_prob, span_prob, label_id, position_id, start_id, end_id):
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.span_prob = span_prob
        self.id = id
        self.label_id = label_id
        self.position_id = position_id
        self.start_id = start_id
        self.end_id = end_id

def eval(args, outputs, partial_masks, label_size, gather_masks):

    correct, pred_count, gold_count = 0, 0, 0
    gather_masks = gather_masks.sum(1).cpu().numpy()
    outputs = outputs.cpu().numpy()
    partial_masks = partial_masks.cpu().numpy()
    correct_all,pred_all,gold_all=dict(),dict(),dict()
    for id in range(0,label_size):
        correct_all[id]=0
        pred_all[id]=0
        gold_all[id]=0

    for output, partial_mask, l in zip(outputs, partial_masks, gather_masks):

        golds = list()
        preds = list()
        golds_set=dict()
        preds_set=dict()
        for id in range(0,label_size):
            golds_set[id]=[]
            preds_set[id]=[]

        for i in range(l):
            for j in range(l):
                if output[i][j] >= 0:
                    if output[i][j] < label_size:
                        preds.append("{}_{}_{}".format(i, j, int(output[i][j])))
                        preds_set[int(output[i][j])].append("{}_{}_{}".format(i, j, int(output[i][j])))
                for k in range(label_size):
                    if partial_mask[i][j][k] == 1:
                        golds.append("{}_{}_{}".format(i, j, k))
                        golds_set[k].append("{}_{}_{}".format(i, j, k))
        pred_count += len(preds)
        gold_count += len(golds)
        correct += len(set(preds).intersection(set(golds)))

        for key in golds_set.keys():
            pred_all[key]+=len(preds_set[key])
            gold_all[key]+=len(golds_set[key])
            correct_all[key]+=len(set(preds_set[key]).intersection(set(golds_set[key])))
    return correct, pred_count, gold_count,correct_all,pred_all,gold_all