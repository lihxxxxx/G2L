# from ctypes import alignment
import pdb
# import ipdb
import torch
# from torch._C import Graph, short
# from torch._C import device, float64, merge_type_from_type_comment
import torch.nn as nn
from torch.functional import F
from g2l.data.datasets.utils import box_iou
from itertools import combinations, permutations
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, johnson,shortest_path,reconstruct_path
from torch_geometric.nn import knn_graph


class BceLoss(object):
    def __init__(self, min_iou, max_iou, mask2d):
        self.min_iou, self.max_iou = min_iou, max_iou
        self.mask2d = mask2d
        self.bceloss = torch.nn.BCELoss(reduction='none')
        self.hinge_loss = False

    def linear_scale(self, iou):
        return (iou - self.min_iou) / (self.max_iou - self.min_iou)

    def __call__(self, feat2ds,sent_feats, scores2d, ious2d, epoch):
        # iou BCE loss
        # iou1d = torch.cat(ious2d, dim=0).masked_select(self.mask2d)
        iou1d = ious2d.masked_select(self.mask2d)
        scores1d = scores2d.masked_select(self.mask2d)
        loss = 0
        iou1d = self.linear_scale(iou1d).clamp(0, 1)
        loss += self.bceloss(scores1d, iou1d).mean()

        return loss



def build_bce_loss(cfg, mask2d):
    min_iou = cfg.MODEL.G2L.LOSS.MIN_IOU 
    max_iou = cfg.MODEL.G2L.LOSS.MAX_IOU
    return BceLoss(min_iou, max_iou, mask2d)



def inter(sent,map2d,i,j,N,M,device):
    maskj=torch.LongTensor([j]).to(device)
    maski=torch.LongTensor([i]).to(device)
    shap_value = (
        # sim_score(sent, map2d) -
        sim_score(sent, map2d.index_fill(1, maski, -1e9) ) -
        sim_score(sent.index_fill(0, maskj, -1e9), map2d ) +
        sim_score(sent.index_fill(0, maskj, -1e9), map2d.index_fill(1, maski, -1e9) )
        )
    return shap_value

def sim_score(sent_feat,map2d_feat):

    a = F.softmax(torch.mm(sent_feat, map2d_feat), dim=1)
    p1,_=torch.max(a,0)
    p2,_=torch.max(a,1)

    return (p1.mean()+p2.mean())/2

def shaploss(sent, feat1d, iou1d, shape_top_k, C):   
    topk_index = shape_top_k
    selected_feat1d = feat1d.index_select(dim=1, index=topk_index.reshape(-1)).reshape(C, -1)     # C x num_sent x top_k
    selected_feat1d = F.normalize(selected_feat1d, dim=0)

    sa_loss=0
    N = sent.size()[0]
    M = selected_feat1d.size()[1]

    alignment_matrix = F.softmax(torch.mm(sent, selected_feat1d), dim=1).reshape(-1)
    full_score = sim_score(sent, selected_feat1d)
    shap_value = []
    for i in range(M):
        for j in range(N):
            soft_label = full_score - inter(sent, selected_feat1d, i, j, N, M, device=sent.device)
            shap_value.append(soft_label)

    shap_value = F.normalize(torch.stack(shap_value).unsqueeze(0)).reshape(-1)
    for i, aij in enumerate(alignment_matrix):
        sa_loss += shap_value[i]*torch.log(aij)
    return -sa_loss/M*N

class ContrastiveLoss(object):
    def __init__(self, cfg, mask2d):
        self.mask2d = mask2d
        self.T_v = cfg.MODEL.G2L.LOSS.TAU_VIDEO
        self.T_s = cfg.MODEL.G2L.LOSS.TAU_SENT
        self.cri = torch.nn.CrossEntropyLoss()
        self.neg_iou = cfg.MODEL.G2L.LOSS.NEGATIVE_VIDEO_IOU
        self.top_k = cfg.MODEL.G2L.LOSS.NUM_POSTIVE_VIDEO_PROPOSAL
        self.sent_removal_iou = cfg.MODEL.G2L.LOSS.SENT_REMOVAL_IOU
        self.margin = cfg.MODEL.G2L.LOSS.MARGIN
        self.eps = 1e-6
        # self.geo_eps = 0.5
        self.dataset = cfg.DATASETS.NAME

        self.as_cl_weight= cfg.MODEL.G2L.LOSS.AS_CL_WEIGHT
        self.use_as_cl=cfg.MODEL.G2L.LOSS.AS_CL
        self.use_dense_neg=cfg.MODEL.G2L.LOSS.DENSE_NEG
        self.Dijkstra = dijkstra
        self.pdist = nn.PairwiseDistance(p=2)
        self.geo_topk = cfg.MODEL.G2L.LOSS.GEO_TOPK
        self.geo_k = cfg.MODEL.G2L.LOSS.GEO_K
        self.num_sa = cfg.MODEL.G2L.LOSS.SHAPLEY_NUM
        self.shapley = cfg.MODEL.G2L.LOSS.SHAPLEY


    def geodesic_distance(self, text_feat, vis_feat, indexs, k=5, limit = 0.5):
        # vis B C N
        N = vis_feat.size()[1]
        # adj_matrix = torch.eye(feat1d.size()[1],feat1d.size()[1], device=feat1d.device)
        # print("cosine=False")
        # edge_indexs = knn_graph(vis_feat.permute(1,0), k=k, loop=False, flow='target_to_source', cosine=True).permute(1,0).to(vis_feat.device)
        edge_indexs = knn_graph(vis_feat.permute(1,0), k=k, loop=False, flow='target_to_source', cosine=False).permute(1,0).to(vis_feat.device)
        # edge_indexs = knn_graph(torch.mm(text_feat, vis_feat).permute(1,0), k=k, loop=False, flow='target_to_source',num_workers=0).permute(1,0).to(vis_feat.device)

        vis_feat = F.normalize(vis_feat,dim=0)
        edge_weight = torch.mm(vis_feat.permute(1,0), vis_feat)
        # edge_weight = 1.0 - edge_weight
        edge_weight = edge_weight.max(1)[0] - edge_weight
        # geo_dis = []
        inf = 1.0
        weighted_graph = torch.ones(N, N) * inf - torch.eye(N, N) * inf
        for idx in edge_indexs:
            weighted_graph[idx[0]][idx[1]] = edge_weight[idx[0]][idx[1]]
        weighted_graph.to(vis_feat.device)
        graph = csr_matrix(weighted_graph.detach().numpy())

        # shortest_path
        dist_matrix = dijkstra(csgraph=graph, directed=True, indices=indexs.cpu(), return_predecessors=False,limit=limit, min_only=False)
        dist_matrix[dist_matrix >= inf] = 1.0
        # dist_matrix[dist_matrix == 0 ] = 0.0
        geo_dis = torch.from_numpy(dist_matrix).squeeze().to(vis_feat.device)
        return 1.0 - geo_dis


    def __call__(self, feat2ds, sent_feats, iou2ds, gt_proposals):
        """
            feat2ds: B x C x T x T
            sent_feats: list(B) num_sent x C
            iou2ds: list(B) num_sent x T x T
            gt_proposals: list(B) num_sent x 2, with format [start, end], unit being seconds (frame/fps)
        """
        # prepare tensors

        B, C, _, _ = feat2ds.size()
        feat1ds = feat2ds.masked_select(self.mask2d).reshape(B, C, -1)
        feat1ds_norm = F.normalize(feat1ds, dim=1)  # B x C x num_sparse_selected_proposal

        sent_feat_cat = torch.cat(sent_feats, 0)  # sum(num_sent) x C, whole batch
        sum_num_sent = sent_feat_cat.size(0)
        sent_feat_cat_norm = F.normalize(sent_feat_cat, dim=1)  # sum(num_sent) x C, whole batch
        sent_mask = torch.ones(sum_num_sent, sum_num_sent, device=feat2ds.device)
    
        all_num_sent = [0]
        curr_num_sent = 0
        for i in range(len(sent_feats)):
            curr_num_sent += sent_feats[i].size(0)
            all_num_sent.append(curr_num_sent)
        for i, gt_per_video in enumerate(gt_proposals):
            iou_map_per_video = box_iou(gt_per_video, gt_per_video)
            iou_mask = iou_map_per_video < self.sent_removal_iou  # remove high iou sentence, keep low iou sentence
            sent_mask[all_num_sent[i]:all_num_sent[i+1], all_num_sent[i]:all_num_sent[i+1]] = iou_mask.float()

        sent_mask += torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)).float()  # add the sentence itself to the denominator in the loss
        margin_mask = torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)) * self.margin
        
        vid_pos_list = []
        vid_neg_list = []
        sent_pos_list = []
        sent_neg_list = []

        hard_sent_pos_list = []
        hard_sent_neg_list = []

        v2v_pos_list = []
        v2v_neg_list = []
        if self.shapley:
            loss_sa=0
        else:
            loss_sa=torch.tensor(0)
        
        for i, (sent_feat, iou2d) in enumerate(zip(sent_feats, iou2ds)):  # each video in the batch
            # select positive samples
            num_sent_this_batch = sent_feat.size(0)
            feat1d = feat1ds_norm[i, :, :]                                                                          # C x num_sparse_selected_proposal
            sent_feat = F.normalize(sent_feat, dim=1)                                                               # num_sent x C
            iou1d = iou2d.masked_select(self.mask2d).reshape(sent_feat.size(0), -1)                                 # num_sent x num_sparse_selected_proposal

            topk_index = torch.topk(iou1d, self.top_k, dim=-1)[1]                                                   # num_sent x top_k
            selected_feat = feat1d.index_select(dim=1, index=topk_index.reshape(-1)).reshape(C, -1, self.top_k)     # C x num_sent x top_k
            selected_feat = selected_feat.permute(1, 2, 0)                                                          # num_sent x top_k x C

            # positive video proposal with pos/neg sentence samples
            vid_pos = torch.bmm(selected_feat,
                                sent_feat.unsqueeze(2)).reshape(-1, self.top_k) - self.margin                       # num_sent x top_k, bmm of (num_sent x top_k x C) and (num_sent x C x 1)
            vid_neg = torch.mm(selected_feat.view(-1, C),
                               sent_feat_cat_norm.t()).reshape(-1, self.top_k, sum_num_sent)                        # num_sent x topk x sum(num_sent), mm of (num_sent*top_k x C) and (C x sum(num_sent))

            vid_pos_list.append(vid_pos)
            vid_neg_list.append(vid_neg)
            
            # positive sentence with pos/neg video proposals
            
            # semantic pairs  
            
            geo_dis_intra = self.geodesic_distance(sent_feat, feat1ds[i, :, :] , topk_index[:,0], k=self.geo_k)

            if len(geo_dis_intra.size()) == 1:
                geo_dis_intra = geo_dis_intra.unsqueeze(0)

            if self.shapley:
                sa_topk_index = torch.cat([
                    torch.topk(geo_dis_intra, self.num_sa, -1)[1],
                    torch.topk(torch.mm(sent_feat,feat1d),self.num_sa,dim=-1)[1],
                    # torch.topk(iou1d, self.num_sa, dim=-1)[1]
                ], 1) 

                loss_sa += shaploss(sent_feat, feat1ds[i, :, :], iou1d, sa_topk_index, C)
                # loss_sa += shaploss(sent_feat, feat1d, iou1d, sa_topk_index, C)  


            # intra- vido
            # weak pairs
            sent_same_video_feat = torch.exp(torch.mm(sent_feat, feat1d))                                                   # num_sent x num_sparse_selected_proposal

            pos_geo_topk = torch.topk(geo_dis_intra, self.geo_topk, -1)
            # pos_geo_topk[0] [pos_geo_topk[0] < 1.0 ] = 0.5
            pos_same_video = torch.zeros(num_sent_this_batch, self.geo_topk).to(feat2ds.device)
            for j in range(num_sent_this_batch):
                pos_same_video[j] = sent_same_video_feat[j].index_select(dim=0, index=pos_geo_topk[1][j]) * pos_geo_topk[0][j]

            sent_pos_list.append(pos_same_video)

            # sent_pos_list.append(vid_pos.clone())
            iou_neg_mask = (iou1d < self.neg_iou).float()                                                       # only keep the low iou proposals as negative samples in the same video
            sent_neg_same_video = iou_neg_mask * sent_same_video_feat                                       # num_sent x num_sparse_selected_proposal
            feat1d_other_video = feat1ds_norm.index_select(dim=0, index=torch.arange(
                B, device=feat2ds.device)[torch.arange(B, device=feat2ds.device) != i])                         # (B-1) x C x num_sparse_selected_proposal
            feat1d_other_video = feat1d_other_video.transpose(1, 0).reshape(C, -1)                              # C x ((B-1) x num_sparse_selected_proposal)
            sent_neg_other_video = torch.mm(sent_feat, feat1d_other_video)                                                  # num_sent x ((B-1) x num_sparse_selected_proposal)
            
            
            # weak_neg_same = sent_same_video_feat * (geo_dis_intra + 1.0)
           
            intra_v2v = torch.exp(torch.mm(selected_feat.view(-1, C), feat1d))
            hard_sent_pos_list.append(pos_same_video.clone())
            

            geo_dis_intra[geo_dis_intra == 1.0] = 0.0

            hard_same_video = sent_same_video_feat * (geo_dis_intra+ 1.0 ) * intra_v2v * iou_neg_mask 

            
            hard_other_sent2video = F.softmax(sent_neg_other_video, dim=1) * 1000
            inter_v2v = torch.mm(selected_feat.view(-1, C), feat1d_other_video)

            inter_v2v_mask = (hard_other_sent2video> hard_other_sent2video.mean()).float()*inter_v2v + (inter_v2v > inter_v2v.mean()).float() * hard_other_sent2video


            hard_other_neg = sent_neg_other_video * inter_v2v_mask

            hard_sent_neg_all = [
                                pos_same_video.clone().unsqueeze(1).repeat(1, self.top_k, 1),
                                hard_same_video.clone().unsqueeze(1).repeat(1, self.top_k, 1),
                                hard_other_neg.clone().unsqueeze(1).repeat(1, self.top_k, 1)]
            hard_sent_neg_list.append(torch.cat(hard_sent_neg_all, dim=2))   

            sent_neg_all = [
                pos_same_video.clone().unsqueeze(1).repeat(1, self.top_k, 1),
                sent_neg_same_video.clone().unsqueeze(1).repeat(1, self.top_k, 1),
                sent_neg_other_video.clone().unsqueeze(1).repeat(1, self.top_k, 1),
            ]
        
            sent_neg_list.append(torch.cat(sent_neg_all, dim=2))                                # num_sent x topk x (1 + num_same + num_other)   


            pos_v2v = torch.zeros(num_sent_this_batch, self.geo_topk).to(feat2ds.device)
            for j in range(num_sent_this_batch):
                pos_v2v[j] = inter_v2v[j].index_select(dim=0, index=pos_geo_topk[1][j]) * pos_geo_topk[0][j]
            v2v_pos_list.append(pos_v2v)

        vid_pos = (torch.cat(vid_pos_list, dim=0).transpose(0, 1)) / self.T_v                   # top_k x num_sent
        vid_neg = torch.cat(vid_neg_list, dim=0).permute(1, 0, 2)                               # top_k x this_cat_to_be_sum(num_sent) x sum(num_sent)
        vid_neg = (vid_neg - margin_mask) / self.T_v                                            # top_k x this_cat_to_be_sum(num_sent) (positive) x sum(num_sent) (negative)
        sent_mask += torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)).float()
        vid_neg_exp = torch.exp(vid_neg) * sent_mask.clamp(min=0, max=1)
        loss_vid = -(vid_pos - torch.log(vid_neg_exp.sum(dim=2, keepdim=False))).mean()
        
        sent_pos = torch.cat(sent_pos_list, dim=0) / self.T_s
        sent_neg = torch.cat(sent_neg_list, dim=0) / self.T_s
        sent_neg_exp = torch.exp(sent_neg)
        loss_sent = -(sent_pos - torch.log(sent_neg_exp.sum(dim=2, keepdim=False) + self.eps)).mean()

        hard_sent_pos = torch.cat(hard_sent_pos_list, dim=0) / self.T_s
        hard_sent_neg = torch.cat(hard_sent_neg_list, dim=0) / self.T_s
        hard_sent_neg_exp = torch.exp(hard_sent_neg)
        loss_hard_sent = -(hard_sent_pos - torch.log(hard_sent_neg_exp.sum(dim=2, keepdim=False) + self.eps)).mean()

        return  loss_sa, loss_vid, loss_hard_sent , loss_sent

def build_contrastive_loss(cfg, mask2d):
    return ContrastiveLoss(cfg, mask2d)
