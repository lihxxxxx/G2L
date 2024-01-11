import torch
from torch import nn
from transformers import DistilBertModel,BertModel
import numpy as np
import pdb
class DistilBert(nn.Module):
    def __init__(self, joint_space_size, dataset,aggregation,text_encoder,norm_mode):
        super().__init__()
        # self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        if text_encoder=='BERT':
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            # self.bert = BertModel.from_pretrained(pretrained_model_name_or_path='./bert-base-uncased')
        elif text_encoder=='DistilBERT':
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.norm_mode=norm_mode
        self.n_components= joint_space_size
        if self.norm_mode=="layernorm":
            hidden=768
        elif self.norm_mode=="whitening":
            hidden=self.n_components
        self.fc_out1 = nn.Linear(hidden, joint_space_size)
        self.fc_out2 = nn.Linear(hidden, joint_space_size)
        self.dataset = dataset
        self.layernorm = nn.LayerNorm(768)
        self.aggregation = aggregation  # cls, avg   
        
    def forward(self, queries, wordlens,device):
        '''
        Average pooling over bert outputs among words to be sentence feature
        :param queries:
        :param wordlens:
        :param vid_avg_feat: B x C
        :return: list of [num_sent, C], len=Batch_size
        '''
        sent_feat = []
        sent_feat_iou = []
        vecs = []
        for query, word_len in zip(queries, wordlens):  # each sample (several sentences) in a batch (of videos)

            N, word_length = query.size(0), query.size(1)
            attn_mask = torch.zeros(N, word_length, device=query.device)
            for i in range(N):
                attn_mask[i, :word_len[i]] = 1  # including [CLS] (first token) and [SEP] (last token)
            # pdb.set_trace()
            bert_encoding = self.bert(query, attention_mask=attn_mask,return_dict=True, output_hidden_states=True).hidden_states  # [N, max_word_length, C]  .permute(2, 0, 1)
            # bert_encoding = self.bert(query, attention_mask=attn_mask)[0]

            if self.aggregation == "cls":
                query = bert_encoding[-1][:, 0, :]  # [N, C], use [CLS] (first token) as the whole sentence feature
                # query = bert_encoding[:, 0, :]
            elif self.aggregation == "avg":
                avg_mask = torch.zeros(N, word_length, device=query.device)
                for i in range(N):
                    avg_mask[i, :word_len[i]] = 1       # including [CLS] (first token) and [SEP] (last token)
                avg_mask = avg_mask / (word_len.unsqueeze(-1))
                bert_encoding = bert_encoding[-1].permute(2, 0, 1) * avg_mask  # use avg_pool as the whole sentence feature
                query = bert_encoding.sum(-1).t()  # [N, C]
            elif self.aggregation == 'first_last_avg':
                query = (bert_encoding[-1] + bert_encoding[1]).mean(dim=1)
            elif self.aggregation == 'last_avg':
                query = (bert_encoding[-1]).mean(dim=1)
            elif self.aggregation == 'last2avg':
                query = (bert_encoding[-1] + bert_encoding[-2]).mean(dim=1)
            elif self.aggregation == 'last3avg':
                query = (bert_encoding[-1] + bert_encoding[-2] + bert_encoding[-3]).mean(dim=1)
            elif self.aggregation == 'last4avg':
                query = (bert_encoding[-1] + bert_encoding[-2] + bert_encoding[-3] + bert_encoding[-4]).mean(dim=1)
            else:
                raise NotImplementedError
            # vec = query.cpu().numpy()[0]
            vecs.append(query)
        if self.norm_mode=="whitening":
            kernel, bias = self.whitening(vecs)
            # import pdb;pdb.set_trace()
            kernel, bias = torch.from_numpy(kernel).float().to(device), torch.from_numpy(bias).float().to(device)
        for vec in vecs:
            if self.norm_mode=="layernorm":
                vec_norm = self.layernorm(vec)
            elif self.norm_mode=="whitening":
                vec_norm = self.transform_and_normalize(vec, 
                    kernel=kernel,
                    bias=bias
                )

            out_iou = self.fc_out1(vec_norm)
            out = self.fc_out2(vec_norm)
            sent_feat.append(out)
            sent_feat_iou.append(out_iou)
            # import pdb;pdb.set_trace()
        return sent_feat, sent_feat_iou

    def whitening(self,vecs):
        # import pdb;pdb.set_trace()
        vecs =[i.cpu().numpy() for i in vecs]
        vecs = np.concatenate(vecs)
        kernel, bias = self.compute_kernel_bias(vecs)
        kernel = kernel[:, :self.n_components]
        return kernel, bias

    def transform_and_normalize(self,vecs, kernel, bias):
        if not (kernel is None or bias is None):
            vecs = torch.mm((vecs + bias),kernel)
        return self.normalize(vecs)


    def normalize(self,vecs):
        return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

    def compute_cov(self, m):#compute_cov(x) = np.cov(x.T)
        x = m - m.mean()
        cov_m = torch.mm(x.T, x) / (x.shape[0]-1)
        return cov_m

    def compute_kernel_bias(self,vecs):
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        from scipy import linalg
        u, s, vh = linalg.svd(cov)
        W = np.dot(u, np.diag(1/np.sqrt(s)))
        return W, -mu

def build_text_encoder(cfg):
    joint_space_size = cfg.MODEL.G2L.JOINT_SPACE_SIZE
    dataset_name = cfg.DATASETS.NAME
    aggregation = cfg.MODEL.G2L.TEXT_ENCODER.AGGREGATION
    text_encoder= cfg.MODEL.G2L.TEXT_ENCODER.NAME
    norm_mode=cfg.MODEL.G2L.TEXT_ENCODER.NORM_MODE
    return DistilBert(joint_space_size, dataset_name,aggregation,text_encoder,norm_mode)
