import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
import os
import numpy as np
from pathlib import Path
from utils.metric_stats.phlvl_metric_stats import phlvlPMDMetricStats
from utils.metric_stats.loss_metric_stats import LossMetricStats
from utils.metric_stats.domain_metric_stats import DomainMetricStats
from utils.c2f_postprocess import get_c2f_ensemble_output, f2phlvl_seqs
from models.adda_model import ADDAModel


logger = logging.getLogger(__name__)


class SBModel(ADDAModel):

    def on_fit_start(self, stage):
        super(SBModel, self).on_fit_start()
        # load pretrained 'C2F_Encoder' and 'C2F_Classifier'
        if stage == sb.Stage.TRAIN:
            encoder_savepath = os.path.join(self.hparams.savept_path, 'C2F_Encoder.pt')
            classifier_savepath = os.path.join(self.hparams.savept_path, 'C2F_Classifier.pt')
            self.modules['C2F_Encoder'].load_state_dict(torch.load(encoder_savepath))
            self.modules['C2F_Classifier'].load_state_dict(torch.load(classifier_savepath))
            # freeze C2F_Encoder model
            for param in self.modules['C2F_Encoder'].parameters():
                param.requires_grad = False

    def on_stage_start(self, stage, epoch=None):
        super(SBModel, self).on_stage_start(stage, epoch)

        # define the target
        if stage == sb.Stage.TRAIN or stage == sb.Stage.VALID:
            assert epoch is not None
            train_targets = ['Encoder', 'Discriminator']
            self.target = train_targets[(epoch) % 2]
        elif stage == sb.Stage.TEST:
            self.target = 'Eval'
        else:
            raise ValueError(f'invalid stage {stage}')
        logger.info(f'Epoch {epoch}, {stage}: target is {self.target}')

        # initialize metric stats
        if self.target== 'Encoder' or self.target == 'Eval':
            self.stats_loggers['phlvl_stats'] = phlvlPMDMetricStats(self.hparams)
        if self.target == 'Discriminator' or self.target == 'Eval':
            self.stats_loggers['domain_stats'] = DomainMetricStats()

        # initialize metric stats for losses: D, E_class, E_domain
        for loss_key in self.hparams.metric_keys:
            if loss_key.endswith('_loss'):
                stats_key = loss_key.lower() + '_stats'
                self.stats_loggers[stats_key] = LossMetricStats(loss_key)

    
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        feats, feat_lens = batch['feat']
        batch_size = feats.shape[0]

        # separate source data
        indices_ssinger = torch.tensor(np.arange(0, batch_size, 2)).to(self.device)
        feats_ssinger = torch.index_select(feats, 0, indices_ssinger)
        feat_lens_ssinger = torch.index_select(feat_lens, 0, indices_ssinger)
        # separate target data
        indices_tsinger = torch.tensor(np.arange(1, batch_size, 2)).to(self.device)
        feats_tsinger = torch.index_select(feats, 0, indices_ssinger)
        feat_lens_tsinger = torch.index_select(feat_lens, 0, indices_tsinger)
            
        # computer embeddings of source and target
        feats_ssinger = torch.permute(feats_ssinger, (0, 2, 1))
        embedding_list_ssinger = self.modules['C2F_Encoder'](feats_ssinger)
        embedding_dim = embedding_list_ssinger[0].shape[1]
        feats_tsinger = torch.permute(feats_tsinger, (0, 2, 1))
        embedding_list_tsinger = self.modules['C2F_Encoder'](feats_tsinger)
        # [x_5, x_4, x_3, x_2, x_1, x] in embedding_list_ssinger
        # 0 torch.Size([8, 128, 45])
        # 1 torch.Size([8, 128, 91])
        # 2 torch.Size([8, 128, 183])
        # 3 torch.Size([8, 128, 366])
        # 4 torch.Size([8, 128, 732])
        # 5 torch.Size([8, 384, 732])
        if self.target == 'Eval':
            ### Concatenate source domain and target domain features ###
            # only take last layer's output as domain pred
            feat_concat = torch.cat((embedding_list_ssinger[0], embedding_list_tsinger[0]), 0) # [batch_size*2, 128, T]
            feat_concat_in = feat_concat.reshape((-1, embedding_dim))  # [batch_size*T, 128]
            ### Forward concatenated features through Discriminator ###
            domain_pred_concat = self.modules['discriminator'](feat_concat_in.detach())  # [batch_size*T, 2]
            domain_pred_concat = domain_pred_concat.reshape((batch_size, -1, 2))  # [batch_size, T, 2]
            # domain_label_concat: [batch_size, T]
            ### prepare source domain labels (1) and target domain labels (0) ###
            domain_label_src = torch.ones((embedding_list_ssinger[0].size(0), embedding_list_ssinger[0].size(-1))).type(torch.LongTensor)
            domain_label_tgt = torch.zeros(embedding_list_tsinger[0].size(0), embedding_list_tsinger[0].size(-1)).type(torch.LongTensor)
            domain_label_concat = torch.cat((domain_label_src, domain_label_tgt), 0)
            domain_label_concat = domain_label_concat.reshape((batch_size, -1)).squeeze().to(self.device)
            feat_lens_concat =  torch.cat((feat_lens_ssinger, feat_lens_tsinger), 0)

            outputs_list_ssinger = self.modules['C2F_Classifier'](embedding_list_ssinger)
            ensem_weights = self.hparams.ensem_weights
            outputs_ensemble_ssinger = get_c2f_ensemble_output(outputs_list_ssinger, ensem_weights)
            outputs_ensemble_ssinger = torch.permute(outputs_ensemble_ssinger, (0, 2, 1))
            ### Forward only TARGET DOMAIN features through Discriminator ###
            emb_tsinger = embedding_list_tsinger[0].reshape((-1, embedding_dim))  # [batch_size*T, 128]
            domain_pred_tsinger = self.modules['discriminator'](emb_tsinger)     
            domain_label_tsinger = torch.ones(domain_pred_tsinger.size(0)).squeeze().type(torch.LongTensor).to(self.device)  # prepare fake labels

            predictions = {
                'domain_pred_concat': domain_pred_concat,
                'domain_label_concat': domain_label_concat,
                'feat_lens_concat': feat_lens_concat,
                'out_ssinger': outputs_ensemble_ssinger,
                'domain_pred_tsinger': domain_pred_tsinger,
                'domain_label_tsinger':domain_label_tsinger,
            }

        if self.target == 'Discriminator':
            ##########################################################################   
            #############  1. Train Encoder & Discriminator with domain labels  ################
            ##########################################################################
            ### Concatenate source domain and target domain features ###
            # only take last layer's output as domain pred
            feat_concat = torch.cat((embedding_list_ssinger[0], embedding_list_tsinger[0]), 0) # [batch_size*2, 128, T]
            feat_concat_in = feat_concat.reshape((-1, embedding_dim))  # [batch_size*T, 128]

            ### Forward concatenated features through Discriminator ###
            domain_pred_concat = self.modules['discriminator'](feat_concat_in.detach())  # [batch_size*T, 2]
            domain_pred_concat = domain_pred_concat.reshape((batch_size, -1, 2))  # [batch_size, T, 2]
    
            ### prepare source domain labels (1) and target domain labels (0) ###
            # domain_label_concat: [batch_size, T]
            domain_label_src = torch.ones((embedding_list_ssinger[0].size(0), embedding_list_ssinger[0].size(-1))).type(torch.LongTensor)
            domain_label_tgt = torch.zeros(embedding_list_tsinger[0].size(0), embedding_list_tsinger[0].size(-1)).type(torch.LongTensor)
            domain_label_concat = torch.cat((domain_label_src, domain_label_tgt), 0)
            domain_label_concat = domain_label_concat.reshape((batch_size, -1)).squeeze().to(self.device)
            feat_lens_concat =  torch.cat((feat_lens_ssinger, feat_lens_tsinger), 0)

            predictions = {
                'domain_pred_concat': domain_pred_concat,
                'domain_label_concat': domain_label_concat,
                'feat_lens_concat': feat_lens_concat,
            }

        if self.target == 'Encoder':
            ##########################################################################
            ######## 2. Train Source Encoder & Classifier with class labels ##########
            ##########################################################################
            ### Forward only SOURCE DOMAIN audios through Encoder & Classifier #######
            # C2F_TCN input size:(BATCH, F, T)  ([8, 384, 732]) feats
            outputs_list_ssinger = self.modules['C2F_Classifier'](embedding_list_ssinger)
            ensem_weights = self.hparams.ensem_weights
            outputs_ensemble_ssinger = get_c2f_ensemble_output(outputs_list_ssinger, ensem_weights)
            outputs_ensemble_ssinger = torch.permute(outputs_ensemble_ssinger, (0, 2, 1))

            ##########################################################################
            ############  3. Train Source Encoder w/ FAKE domain label  ##############
            ##########################################################################
            ### Forward only TARGET DOMAIN features through Discriminator ###
            emb_tsinger = embedding_list_tsinger[0].reshape((-1, embedding_dim))  # [batch_size*T, 128]
            domain_pred_tsinger = self.modules['discriminator'](emb_tsinger)     
            domain_label_tsinger = torch.ones(domain_pred_tsinger.size(0)).squeeze().type(torch.LongTensor).to(self.device)  # prepare fake labels     

            predictions = {
                'out_ssinger': outputs_ensemble_ssinger,
                'domain_pred_tsinger': domain_pred_tsinger,
                'domain_label_tsinger':domain_label_tsinger,
            }

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        # get model outputs
        # out = predictions['out']  # (batch, time, pred_class)
        feats, feat_lens = batch['feat']
        batch_size = feats.shape[0]

        phn_mode_label_seqs, phlvl_phn_mode_label_len = batch['phn_mode_encoded']     
        flvl_phn_mode_label_seqs, flvl_phn_mode_label_len = batch['flvl_phn_mode_encoded']

        # get source gt labels
        indices = torch.tensor(np.arange(0, batch_size, 2)).to(self.device)
        phn_mode_label_seqs = torch.index_select(phn_mode_label_seqs, 0, indices)
        phlvl_phn_mode_label_len = torch.index_select(phlvl_phn_mode_label_len, 0, indices)
        flvl_phn_mode_label_seqs = torch.index_select(flvl_phn_mode_label_seqs, 0, indices)
        flvl_phn_mode_label_len = torch.index_select(flvl_phn_mode_label_len, 0, indices)

        if self.target == 'Eval':
            domain_pred_concat = predictions['domain_pred_concat']  # [batch_size, T, 2]
            domain_label_concat = predictions['domain_label_concat']  # [batch_size, T]
            feat_lens_concat = predictions['feat_lens_concat']

            # domain label for each sample
            criterion_DA  = nn.CrossEntropyLoss()
            loss_discriminator = criterion_DA(torch.permute(domain_pred_concat, (0, 2, 1)), domain_label_concat)

            domain_pred_concat_seqs = torch.argmax(torch.prod(domain_pred_concat, dim=1), dim=-1)  # [batch_size]
            domain_gt_concat_seqs = torch.prod(domain_label_concat, dim=1)  # [batch_size]

            s_domain_pred_seqs = domain_pred_concat_seqs[:int(batch_size/2)].tolist()  # source
            s_domain_gt_seqs = domain_gt_concat_seqs[:int(batch_size/2)].tolist()  # source
            t_domain_pred_seqs = domain_pred_concat_seqs[int(batch_size/2):].tolist()  # target
            t_domain_gt_seqs = domain_gt_concat_seqs[int(batch_size/2):].tolist()  # target

            self.stats_loggers['domain_stats'].append(
                batch['id'],
                s_domain_pred_seqs=s_domain_pred_seqs,
                s_domain_gt_seqs=s_domain_gt_seqs,
                t_domain_pred_seqs=t_domain_pred_seqs,
                t_domain_gt_seqs=t_domain_gt_seqs,
            )
            self.stats_loggers['e_domain_loss_stats'].append(
                loss_discriminator,
            )
            # Calculate class-classification loss for Encoder and Classifier ###
            out_ssinger = predictions['out_ssinger']
            flvl_phn_mode_label_onehot = F.one_hot(flvl_phn_mode_label_seqs.data, num_classes=self.hparams.n_phonation).to(out_ssinger.device)
            flvl_phn_mode_label = flvl_phn_mode_label_seqs.data

            ce = nn.CrossEntropyLoss()  # Frame wise binary cross entropy loss
            mse = nn.MSELoss(reduction='none')           # Migitating transistion loss 
            outp_wo_softmax = torch.log(out_ssinger + 1e-10)         # log is necessary because ensemble gives softmax output
            outp_wo_softmax = outp_wo_softmax.permute(0, 2, 1)

            if outp_wo_softmax.shape[-1] != flvl_phn_mode_label.shape[-1]:
                outp_wo_softmax = outp_wo_softmax[...,:flvl_phn_mode_label.shape[-1]]        
            # CE loss
            ce_loss = ce(outp_wo_softmax, flvl_phn_mode_label)
            # SM loss
            mse_loss = 0.15 * torch.mean(torch.clamp(mse(outp_wo_softmax[:, :, 1:], outp_wo_softmax.detach()[:, :, :-1]), 
                                                    min=0, max=16) )#* src_msk_send[:, :, 1:])
            loss_CLS = ce_loss + mse_loss

            # Calculate FAKE domain-classification loss for Encoder ###
            domain_pred_tsinger = predictions['domain_pred_tsinger']
            domain_label_tsinger = predictions['domain_label_tsinger']
            criterion_DA  = nn.CrossEntropyLoss()
            loss_DA = criterion_DA(domain_pred_tsinger, domain_label_tsinger)

            ids = [batch['id'][i] for i in indices]
            flvl_pred_pmd_lbl_seqs = torch.argmax(out_ssinger, dim=-1)
            flvl_pred_pmd_lbl_seqs_unpad = undo_padding(flvl_pred_pmd_lbl_seqs, flvl_phn_mode_label_len)
            phlvl_gt_seg_on_seqs = [batch['seg_on_list'][i] for i in indices]
            phlvl_gt_seg_off_seqs = [batch['seg_off_list'][i] for i in indices]
            phlvl_gt_pmd_lbl_seqs = undo_padding(phn_mode_label_seqs, phlvl_phn_mode_label_len)
            for i in range(len(phlvl_gt_pmd_lbl_seqs)):
                assert len(phlvl_gt_pmd_lbl_seqs[i]) > 0 

            sr = self.hparams.sample_rate
            hoplen = self.hparams.hop_length  # in ms
            phlvl_pred_pmd_lbl_seqs, phlvl_pred_seg_on_seqs, phlvl_pred_seg_off_seqs = f2phlvl_seqs(flvl_pred_pmd_lbl_seqs_unpad, sr, hoplen)

            self.stats_loggers['phlvl_stats'].append(
                ids = ids,
                sr = sr,
                n_class = self.hparams.n_phonation,
                pred_pmd_lbl_seqs = phlvl_pred_pmd_lbl_seqs,
                gt_pmd_lbl_seqs = phlvl_gt_pmd_lbl_seqs,
                pred_seg_on_seqs = phlvl_pred_seg_on_seqs,
                pred_seg_off_seqs = phlvl_pred_seg_off_seqs,
                gt_seg_on_seqs = phlvl_gt_seg_on_seqs,
                gt_seg_off_seqs = phlvl_gt_seg_off_seqs,
            )

            self.stats_loggers['d_loss_stats'].append(
                loss_DA, 
            )
            self.stats_loggers['e_class_loss_stats'].append(
                loss_CLS,
            )
            alpha_CLS = self.hparams.alpha_cls
            alpha_DA = self.hparams.alpha_da
            loss = alpha_CLS * loss_CLS + alpha_DA * loss_DA
            return loss


        if self.target == 'Discriminator':
        ##-----compute loss D--------
            ### 1. Calculate domain-classification loss for Discriminator ###
            domain_pred_concat = predictions['domain_pred_concat']  # [batch_size, T, 2]
            domain_label_concat = predictions['domain_label_concat']  # [batch_size, T]
            ### Define loss for domain-classification ###
            criterion_DA  = nn.CrossEntropyLoss()
            loss_discriminator = criterion_DA(torch.permute(domain_pred_concat, (0, 2, 1)), domain_label_concat)

        ##-----compute metrics D-----
            domain_pred_concat_seqs = torch.argmax(torch.prod(domain_pred_concat, dim=1), dim=-1)  # [batch_size]
            domain_gt_concat_seqs = torch.prod(domain_label_concat, dim=1)  # [batch_size]
            s_domain_pred_seqs = domain_pred_concat_seqs[:int(batch_size/2)].tolist()  # source
            s_domain_gt_seqs = domain_gt_concat_seqs[:int(batch_size/2)].tolist()  # source
            t_domain_pred_seqs = domain_pred_concat_seqs[int(batch_size/2):].tolist()  # target
            t_domain_gt_seqs = domain_gt_concat_seqs[int(batch_size/2):].tolist()  # target

            self.stats_loggers['domain_stats'].append(
                batch['id'],
                s_domain_pred_seqs=s_domain_pred_seqs,
                s_domain_gt_seqs=s_domain_gt_seqs,
                t_domain_pred_seqs=t_domain_pred_seqs,
                t_domain_gt_seqs=t_domain_gt_seqs,
            )
            self.stats_loggers['e_domain_loss_stats'].append(
                loss_discriminator,
            )
            return loss_discriminator

        if self.target == 'Encoder':
        ##-----compute loss E--------
            ### 2. Calculate class-classification loss for Encoder and Classifier ###
            out_ssinger = predictions['out_ssinger']
            flvl_phn_mode_label_onehot = F.one_hot(flvl_phn_mode_label_seqs.data, num_classes=self.hparams.n_phonation).to(out_ssinger.device)
            flvl_phn_mode_label = flvl_phn_mode_label_seqs.data

            ce = nn.CrossEntropyLoss()  # Frame wise binary cross entropy loss
            mse = nn.MSELoss(reduction='none')           # Migitating transistion loss 
            outp_wo_softmax = torch.log(out_ssinger + 1e-10)         # log is necessary because ensemble gives softmax output
            outp_wo_softmax = outp_wo_softmax.permute(0, 2, 1)

            if outp_wo_softmax.shape[-1] != flvl_phn_mode_label.shape[-1]:
                outp_wo_softmax = outp_wo_softmax[...,:flvl_phn_mode_label.shape[-1]]        
            ce_loss = ce(outp_wo_softmax, flvl_phn_mode_label)

            mse_loss = 0.15 * torch.mean(torch.clamp(mse(outp_wo_softmax[:, :, 1:], outp_wo_softmax.detach()[:, :, :-1]), 
                                                    min=0, max=16) )

            ### Calculate class-classification loss for Encoder and Classifier ###
            loss_CLS = ce_loss + mse_loss

            ### 3. Calculate FAKE domain-classification loss for Encoder ###
            domain_pred_tsinger = predictions['domain_pred_tsinger']
            domain_label_tsinger = predictions['domain_label_tsinger']
            criterion_DA  = nn.CrossEntropyLoss()
            loss_DA = criterion_DA(domain_pred_tsinger, domain_label_tsinger)

            ### For encoder and Classifier, 
            ### optimize class-classification & fake domain-classification losses together ###
            alpha_CLS = 1
            alpha_DA = 1
            loss = alpha_CLS * loss_CLS + alpha_DA * loss_DA

        ##-----compute metrics E-----
            ids = [batch['id'][i] for i in indices]
            ### frame level
            flvl_pred_pmd_lbl_seqs = torch.argmax(out_ssinger, dim=-1)
            flvl_pred_pmd_lbl_seqs_unpad = undo_padding(flvl_pred_pmd_lbl_seqs, flvl_phn_mode_label_len)

            ### phonation level
            phlvl_gt_seg_on_seqs = [batch['seg_on_list'][i] for i in indices]
            phlvl_gt_seg_off_seqs = [batch['seg_off_list'][i] for i in indices]

            phlvl_gt_pmd_lbl_seqs = undo_padding(phn_mode_label_seqs, phlvl_phn_mode_label_len)
            for i in range(len(phlvl_gt_pmd_lbl_seqs)):
                assert len(phlvl_gt_pmd_lbl_seqs[i]) > 0 

            sr = self.hparams.sample_rate
            hoplen = self.hparams.hop_length  # in ms
            phlvl_pred_pmd_lbl_seqs, phlvl_pred_seg_on_seqs, phlvl_pred_seg_off_seqs = f2phlvl_seqs(flvl_pred_pmd_lbl_seqs_unpad, sr, hoplen)
           
            self.stats_loggers['phlvl_stats'].append(
                ids = ids,
                sr = sr,
                n_class = self.hparams.n_phonation,
                pred_pmd_lbl_seqs = phlvl_pred_pmd_lbl_seqs,
                gt_pmd_lbl_seqs = phlvl_gt_pmd_lbl_seqs,
                pred_seg_on_seqs = phlvl_pred_seg_on_seqs,
                pred_seg_off_seqs = phlvl_pred_seg_off_seqs,
                gt_seg_on_seqs = phlvl_gt_seg_on_seqs,
                gt_seg_off_seqs = phlvl_gt_seg_off_seqs,
            )

            self.stats_loggers['d_loss_stats'].append(
                loss_DA, 
            )
            self.stats_loggers['e_class_loss_stats'].append(
                loss_CLS,
            )
            return loss

    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=self.hparams.max_key, min_key=self.hparams.min_key)

