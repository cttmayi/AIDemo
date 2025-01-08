import os
import json
import copy
from tqdm.notebook import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMaskedLM, BertTokenizerFast


class Config:
    def __init__(self):
        pass
    
    def mlm_config(
        self, 
        mlm_probability=0.15, 
        special_tokens_mask=None,
        prob_replace_mask=0.8,
        prob_replace_rand=0.1,
        prob_keep_ori=0.1,
    ):
        """
        :param mlm_probability: 被mask的token总数
        :param special_token_mask: 特殊token
        :param prob_replace_mask: 被替换成[MASK]的token比率
        :param prob_replace_rand: 被随机替换成其他token比率
        :param prob_keep_ori: 保留原token的比率
        """
        assert sum([prob_replace_mask, prob_replace_rand, prob_keep_ori]) == 1,                 ValueError("Sum of the probs must equal to 1.")
        self.mlm_probability = mlm_probability
        self.special_tokens_mask = special_tokens_mask
        self.prob_replace_mask = prob_replace_mask
        self.prob_replace_rand = prob_replace_rand
        self.prob_keep_ori = prob_keep_ori
        
    def training_config(
        self,
        batch_size,
        epochs,
        learning_rate,
        weight_decay,
        device,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        
    def io_config(
        self,
        from_path,
        save_path,
    ):
        self.from_path = from_path
        self.save_path = save_path


config = Config()
config.mlm_config()
config.training_config(batch_size=4, epochs=10, learning_rate=1e-5, weight_decay=0, device='cuda:0')
config.io_config(from_path='/data/BERTmodels/huggingface/chinese_wwm/', 
                 save_path='./finetune_embedding_model/mlm/')


bert_tokenizer = BertTokenizerFast.from_pretrained(config.from_path)
bert_mlm_model = BertForMaskedLM.from_pretrained(config.from_path)


class TrainDataset(Dataset):
    """
    注意：由于没有使用data_collator，batch放在dataset里边做，
    因而在dataloader出来的结果会多套一层batch维度，传入模型时注意squeeze掉
    """
    def __init__(self, input_texts, tokenizer, config):
        self.input_texts = input_texts
        self.tokenizer = tokenizer
        self.config = config
        self.ori_inputs = copy.deepcopy(input_texts)
        
    def __len__(self):
        return len(self.input_texts) // self.config.batch_size
    
    def __getitem__(self, idx):
        batch_text = self.input_texts[: self.config.batch_size]
        features = self.tokenizer(batch_text, max_length=512, truncation=True, padding=True, return_tensors='pt')
        inputs, labels = self.mask_tokens(features['input_ids'])
        batch = {"inputs": inputs, "labels": labels}
        self.input_texts = self.input_texts[self.config.batch_size: ]
        if not len(self):
            self.input_texts = self.ori_inputs
        
        return batch
        
    def mask_tokens(self, inputs):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.config.mlm_probability)
        if self.config.special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = self.config.special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.config.prob_replace_mask)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        current_prob = self.config.prob_replace_rand / (1 - self.config.prob_replace_mask)
        indices_random = torch.bernoulli(torch.full(labels.shape, current_prob)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


training_texts = ["你好，我是中国人", "我是中国人", "我是中国人", "我是中国人", "我是中国人", "我是中国人", "我是中国人", "我是中国人", "我是中国人", "我是中国人"]

train_dataset = TrainDataset(training_texts, bert_tokenizer, config)
train_dataloader = DataLoader(train_dataset)


def train(model, train_dataloader, config):
    """
    训练
    :param model: nn.Module
    :param train_dataloader: DataLoader
    :param config: Config
    ---------------
    ver: 2021-11-08
    by: changhongyu
    """
    assert config.device.startswith('cuda') or config.device == 'cpu', ValueError("Invalid device.")
    device = torch.device(config.device)
    
    model.to(device)
    
    if not len(train_dataloader):
        raise EOFError("Empty train_dataloader.")
        
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    
    for cur_epc in tqdm(range(int(config.epochs)), desc="Epoch"):
        training_loss = 0
        print("Epoch: {}".format(cur_epc+1))
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc='Step')):
            input_ids = batch['inputs'].squeeze(0).to(device)
            labels = batch['labels'].squeeze(0).to(device)
            loss = model(input_ids=input_ids, labels=labels).loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            training_loss += loss.item()
        print("Training loss: ", training_loss)


train(model=bert_mlm_model, train_dataloader=train_dataloader, config=config)


torch.save(bert_mlm_model.bert.embeddings.state_dict(), os.path.join(config.save_path, 'bert_mlm_ep_{}_eb.bin'.format(config.epochs)))
torch.save(bert_mlm_model.bert.encoder.state_dict(), os.path.join(config.save_path, 'bert_mlm_ep_{}_ec.bin'.format(config.epochs)))
