import json
import logging
import torch
from .utils import moment_to_iou2d, bert_embedding, get_vid_feat
from transformers import DistilBertTokenizer,BertTokenizer


class TACoSDataset(torch.utils.data.Dataset):

    def __init__(self, ann_file, feat_file, num_pre_clips, num_clips,cfg):
        super(TACoSDataset, self).__init__()
        self.feat_file = feat_file
        self.num_pre_clips = num_pre_clips
        with open(ann_file,'r') as f:
            annos = json.load(f)

        self.annos = []
        text_encoder=cfg.MODEL.G2L.TEXT_ENCODER.NAME
        if text_encoder=='BERT':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif text_encoder=='DistilBERT':
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        logger = logging.getLogger("g2l.trainer")
        logger.info("Preparing data, please wait...")
        logger.info("ann_file:{}".format(ann_file))
        # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        text_encoder=cfg.MODEL.G2L.TEXT_ENCODER.NAME
        if text_encoder=='BERT':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif text_encoder=='DistilBERT':
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        for vid, anno in annos.items():
            duration = anno['num_frames']/anno['fps']  # duration of the video
            # Produce annotations
            moments = []
            all_iou2d = []
            sentences = []
            for timestamp, sentence in zip(anno['timestamps'], anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    moment = torch.Tensor([max(timestamp[0]/anno['fps'], 0), min(timestamp[1]/anno['fps'], duration)])
                    moments.append(moment)
                    iou2d = moment_to_iou2d(moment, num_clips, duration)
                    all_iou2d.append(iou2d)
                    sentences.append(sentence)

            moments = torch.stack(moments)
            all_iou2d = torch.stack(all_iou2d)
            queries, word_lens = bert_embedding(sentences, tokenizer)  # padded query of N*word_len, tensor of size = N

            assert moments.size(0) == all_iou2d.size(0)
            assert moments.size(0) == queries.size(0)
            assert moments.size(0) == word_lens.size(0)


            self.annos.append(
                {
                    'vid': vid,
                    'moment': moments,
                    'iou2d': all_iou2d,
                    'sentence': sentences,
                    'query': queries,
                    'wordlen': word_lens,
                    'duration': duration,
                }
             )


    def __getitem__(self, idx):
        #feat = self.feats[self.annos[idx]['vid']]
        feat = get_vid_feat(self.feat_file, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="activitynet")


        return feat, self.annos[idx]['query'], self.annos[idx]['wordlen'], self.annos[idx]['iou2d'], self.annos[idx]['moment'],\
                len(self.annos[idx]['sentence']), idx #, self.annos[idx]['dense_ious2d'], self.annos[idx]['dense_moments']


    def __len__(self):
        return len(self.annos)

    def get_duration(self, idx):
        return self.annos[idx]['duration']

    def get_sentence(self, idx):
        return self.annos[idx]['sentence']

    def get_moment(self, idx):
        return self.annos[idx]['moment']

    def get_vid(self, idx):
        return self.annos[idx]['vid']
