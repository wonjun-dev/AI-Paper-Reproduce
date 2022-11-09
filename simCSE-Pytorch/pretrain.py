"""
simCSE(https://arxiv.org/pdf/2104.08821.pdf) 사전 학습을 위한 모델과 학습 구현체 입니다.
simCLR 구현체 (https://theaisummer.com/simclr/) 를 참고하여 simCSE 세팅에 맞게 변형 하였습니다.
"""

import enum
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from sklearn.model_selection import StratifiedKFold

from load_data import *
from train import label_to_num

def device_as(t1, t2):
    return t1.to(t2.device)


class SimCSELoss(nn.Module):
    def __init__(self, batch_size, temperature=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()    # negative pair를 indexing 하는 마스크입니다. (자기 자신(대각 성분)을 제외한 나머지) 

    def calc_sim_batch(self, a, b):
        reprs = torch.cat([a, b], dim=0)
        return F.cosine_similarity(reprs.unsqueeze(1), reprs.unsqueeze(0), dim=2)   # 두 representation의 cosine 유사도를 계산합니다.
    
    def calc_align(self, x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean().detach()  # 두 representation의 alignment를 계산하고 반환합니다.

    def calc_unif(self, x, t=2):
        sp_pdist = torch.pdist(x, p=2).pow(2)
        return sp_pdist.mul(-t).exp().mean().log().detach()  # 미니 배치 내의 represenation의 uniformity를 계산하고 반환합니다.
    
    
    def forward(self, proj_1, proj_2):
        batch_size = proj_1.shape[0]
        if batch_size != self.batch_size:
            mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float() # 에폭 안에서 마지막 미니 배치를 위해서 마스크를 새롭게 정의 합니다.
        else:
            mask = self.mask
            
        z_i = F.normalize(proj_1, p=2, dim=1)   # 모델의 [CLS] represenation을 l2 nomalize 합니다.
        z_j = F.normalize(proj_2, p=2, dim=1)

        sim_matrix = self.calc_sim_batch(z_i, z_j)  # 배치 단위로 두 representation의 cosine 유사도를 계산합니다.

        sim_ij = torch.diag(sim_matrix, batch_size) # sim_matrix에서 positive pair의 위치를 인덱싱 합니다. (대각 성분에서 배치 사이즈만큼 떨어져 있습니다.)
        sim_ji = torch.diag(sim_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = device_as(mask, sim_matrix) * torch.exp(sim_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))  # constrastive loss
        loss = torch.sum(all_losses) / (2 * batch_size) # 샘플 갯수로 나누어 평균 내줍니다.

        lalign = self.calc_align(z_i, z_j)
        lunif = (self.calc_unif(z_i[:batch_size//2]) + self.calc_unif(z_i[batch_size//2:])) / 2

        return loss, lalign, lunif


class AddProjection(nn.Module):
    def __init__(self, device, backbone_name=None):
        super(AddProjection, self).__init__()
        backbone_config = AutoConfig.from_pretrained(backbone_name)
        self.backbone = AutoModelForSequenceClassification.from_pretrained(backbone_name, config=backbone_config)
        
        if backbone_name == "klue/bert-base":
            # [CLS] represenation을 얻기 위해서 BERT encoder 이후의 classifier를 제거합니다.
            self.backbone.dropout = nn.Identity()
            self.backbone.classifier = nn.Identity()
        elif backbone_name == "klue/roberta-large":
            #  [CLS] represenation을 얻기 위해서 RoBERTa encoder 이후의 classifier를 제거합니다.
            self.backbone.classifier.dense = nn.Identity()
            self.backbone.classifier.dropout = nn.Identity()
            self.backbone.classifier.out_proj = nn.Identity()

        # projection head 추가해줍니다
        self.projection = nn.Sequential(
            nn.Linear(backbone_config.hidden_size, backbone_config.hidden_size),
            nn.Tanh()
        )

        self.device = device

    def forward(self, x):
        outputs = self.backbone(
            input_ids=x['input_ids'].to(self.device),
            attention_mask=x['attention_mask'].to(self.device),
            token_type_ids=x['token_type_ids'].to(self.device)
            )
        pooled_cls = outputs.logits[:]  # (batch_size, hidden_size)
        
        return self.projection(pooled_cls)



class BertSimCSE(pl.LightningModule):
    def __init__(self, device, model_name, batch_size, temperature, lr, expr_name):
        super().__init__()
        self.save_hyperparameters()
        self.model = AddProjection(device, backbone_name=model_name)
        self.loss = SimCSELoss(batch_size=batch_size, temperature=temperature)
        self.lr = lr

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x = batch
        z1 = self.model(x)  # dropout 마스크가 다르게 적용된 positive pair를 구성하기 위햇 같은 미니배치를 모델에 두번 forward 합니다.
        z2 = self.model(x)
        loss, lalign, lunif = self.loss(z1, z2)
        self.log('simCSE loss(unsup.)', loss)   # tensorboard --logdir=./pretrain_backbone/bert-base-balance/expr_name
        self.log('measure_align', lalign)
        self.log('measure_unif', lunif)

        return loss
    
    def on_train_end(self):
        expr_name = self.hparams['expr_name']
        self.model.backbone.save_pretrained(f'./pretrain_backbone/bert-base-balance/{expr_name}')
        # self.model.backbone.save_pretrained(f'./pretrain_backbone/roberta-large-balance/{expr_name}')
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return [optimizer]


def main():
    np.random.seed(2022)
    random.seed(2022) 
    torch.manual_seed(2022)
    pl.seed_everything(2022)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_NAME = "klue/bert-base"
    # MODEL_NAME = "klue/roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    learning_rate = 3e-5
    temper = [1.0]
    for temp in temper:
        print('Temperature: ', temp)
        expr_name = f'{temp}'
        model = BertSimCSE(device, MODEL_NAME, batch_size=32, temperature=temp, lr=learning_rate, expr_name=expr_name)
        # model = BertSimCSE(device, MODEL_NAME, batch_size=8, temperature=temp, lr=learning_rate, expr_name=expr_name)

        # load dataset
        dataset = load_data("../dataset/train/train.csv")
        
        train_dataset = dataset
        train_label = label_to_num(train_dataset['label'].values)

        # long tail distribution 를 고려해서 label이 균등한 미니배치를 구성하기 위해 weighted sampler 정의
        class_sample_count = np.array([len(np.where(train_label == t)[0]) for t in np.unique(train_label)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_label])
        samples_weight = torch.from_numpy(samples_weight).double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        tokenized_train = tokenized_dataset(train_dataset, tokenizer)
        RE_train_dataset = RE_Dataset(tokenized_train, train_label)
        dataloader = DataLoader(RE_train_dataset, batch_size=32, sampler=sampler)

        trainer = pl.Trainer(default_root_dir=f'./pretrain_backbone/bert-base-balance/{expr_name}', devices=[0], accelerator='gpu', max_epochs=1, log_every_n_steps=10, accumulate_grad_batches=2, enable_checkpointing=False)
        # trainer = pl.Trainer(default_root_dir=f'./pretrain_backbone/roberta-large-balance/whole/{temp}', devices=[0], accelerator='gpu', max_epochs=2, log_every_n_steps=10, accumulate_grad_batches=64, enable_checkpointing=False)
        trainer.fit(model, dataloader)


if __name__ == '__main__':
    main()