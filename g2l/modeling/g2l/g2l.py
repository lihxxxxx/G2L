import pdb
import torch
from torch import nn
from torch._C import device
from torch.functional import F
from .featpool import build_featpool  # downsample 1d temporal features to desired length
from .feat2d import build_feat2d  # use MaxPool1d/Conv1d to generate 2d proposal-level feature map from 1d temporal features
import copy
from .loss import build_contrastive_loss
from .loss import build_bce_loss

from .text_encoder import build_text_encoder
from .proposal_conv import build_proposal_conv
from copy import deepcopy
import math

class G2L(nn.Module):
    def __init__(self, cfg):
        super(G2L, self).__init__()
        self.only_iou_loss_epoch = cfg.SOLVER.ONLY_IOU
        self.featpool = build_featpool(cfg) 
        self.feat2d = build_feat2d(cfg)
        self.contrastive_loss = build_contrastive_loss(cfg, self.feat2d.mask2d)
        self.iou_score_loss = build_bce_loss(cfg, self.feat2d.mask2d)
        # self.sa_loss=build_sa_loss(cfg, self.feat2d.mask2d)
        self.text_encoder = build_text_encoder(cfg)
        self.proposal_conv = build_proposal_conv(cfg, self.feat2d.mask2d)
        self.joint_space_size = cfg.MODEL.G2L.JOINT_SPACE_SIZE
        self.encoder_name = cfg.MODEL.G2L.TEXT_ENCODER.NAME

        self.mix_w = cfg.MODEL.MIXW
        self.mix_mode=cfg.MODEL.MIX
        self.use_feat=cfg.MODEL.USE_FEAT
        self.use_feat_att_bce=cfg.MODEL.USE_FEAT_BCE
        self.use_feat_att_cl=cfg.MODEL.USE_FEAT_CL
        self.add_time_bce=cfg.MODEL.ADD_TIME_BCE
        self.add_time_cl=cfg.MODEL.ADD_TIME_CL
   
        self.mha=nn.MultiheadAttention(embed_dim=256,num_heads=4)
        self.conv1d=nn.Conv1d(256,64,1)

        d_model,dim_feedforward,dropout=256,1024,0.1

        self.activation = F.relu
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)


    def get_iou_scores(self,map2d_iou,sent_feat_iou):
        # inference

        iou_scores = []
        _, T, _ = map2d_iou[0].size()
        for i, sf_iou in enumerate(sent_feat_iou):  # sent_feat_iou: [num_sent x C] (len=B)
            # iou part
            vid_feat_iou = map2d_iou[i]  # C x T x T T=64
            vid_feat_iou_norm = F.normalize(vid_feat_iou, dim=0)
            sf_iou_norm = F.normalize(sf_iou, dim=1)
            iou_score = torch.mm(sf_iou_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.size(0), -1)).reshape(-1, T, T)  # num_sent x T x T
            iou_scores.append((iou_score*10).sigmoid() * self.feat2d.mask2d)
            # iou_scores.append((iou_score).sigmoid() * self.feat2d.mask2d)

        return iou_scores


    def embedding(self,batches,device): # ,video_embedding,text_embedding
        ious2d = batches.all_iou2d
        moments=batches.moments
        assert len(ious2d) == batches.feats.size(0)
        for idx, (iou, sent) in enumerate(zip(ious2d, batches.queries)):
            assert iou.size(0) == sent.size(0)
            # assert iou.size(0) == batches.num_sentence[idx]
        # pdb.set_trace()
        feats = self.featpool(batches.feats)  # from pre_num_clip to num_clip with overlapped average pooling, e.g., 256 -> 128 [48, 256, 500] -> [48, 512, 64]
        map2d = self.feat2d(feats)  # use MaxPool1d to generate 2d proposal-level feature map from 1d temporal features #  -> [48, 512, 64, 64]

        map2d, map2d_iou = self.proposal_conv(map2d) # -> [48, 256, 64, 64]
        sent_feat, sent_feat_iou = self.text_encoder(batches.queries, batches.wordlens,device)

        ious2d_bce,ious2d_cl=deepcopy(ious2d),deepcopy(ious2d)

        return map2d,map2d_iou,sent_feat,sent_feat_iou,ious2d_bce,ious2d_cl,moments

    def forward(self, batches, cur_epoch=1,device=None,idx=None):
        """
        Arguments:
            batches.all_iou2d: list(B) num_sent x T x T
            feat2ds: B x C x T x T
            sent_feats: list(B) num_sent x C
        """
        # backbone
        map2d,map2d_iou,sent_feat,sent_feat_iou,ious2d_bce,ious2d_cl,moments=self.embedding(batches,device)

        # inference
        iou_scores=self.get_iou_scores(map2d_iou,sent_feat_iou)
        
        # loss
        if self.training:
            loss_iou = self.iou_score_loss(map2d_iou,sent_feat_iou, torch.cat(iou_scores, dim=0), torch.cat(ious2d_bce, dim=0), cur_epoch)
            loss_sa,loss_vid, loss_hard_sent, loss_sent = self.contrastive_loss(map2d, sent_feat, ious2d_cl, moments)

            # return loss_sa, loss_iou
            return -loss_sa, loss_vid, loss_hard_sent, loss_sent ,loss_iou
        else:
            contrastive_scores=[]
            for i, sf in enumerate(sent_feat):
                # contrastive part
                vid_feat = map2d[i, ...]  # C x T x T
                vid_feat_norm = F.normalize(vid_feat, dim=0)
                sf_norm = F.normalize(sf, dim=1)  # num_sent x C
                _, T, _ = vid_feat.size()
                contrastive_score = torch.mm(sf_norm, vid_feat_norm.reshape(vid_feat.size(0), -1)).reshape(-1, T, T) * self.feat2d.mask2d  # num_sent x T x T
                contrastive_scores.append(contrastive_score)
            return map2d_iou, sent_feat_iou, contrastive_scores, iou_scores  # first two maps for visualization
