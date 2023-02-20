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
from utils.c2f_postprocess import get_c2f_ensemble_output, f2phlvl_seqs
from models.pmd_model import PMDModel


logger = logging.getLogger(__name__)


class SBModel(PMDModel):
    def on_stage_start(self, stage, epoch=None):
        super(SBModel, self).on_stage_start(stage, epoch)
        # initialize metric stats
        self.stats_loggers['phlvl_stats'] = phlvlPMDMetricStats(self.hparams)

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

        if stage == sb.Stage.TEST:
            # only evaluate target singer on test stage
            feats = feats_tsinger
            feat_lens = feat_lens_tsinger
        else:
            feats = feats_ssinger
            feat_lens = feat_lens_ssinger           

        # C2F_TCN input size:(BATCH, F, T)
        feats = torch.permute(feats, (0, 2, 1))
        embedding_list = self.modules['C2F_Encoder'](feats)
        outputs_list = self.modules['C2F_Classifier'](embedding_list)
        ensem_weights = self.hparams.ensem_weights
        outputs_ensemble = get_c2f_ensemble_output(outputs_list, ensem_weights)
        outputs_ensemble = torch.permute(outputs_ensemble, (0, 2, 1))

        predictions = {
            'out': outputs_ensemble,
        }

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        # get model outputs
        out = predictions['out']  # (batch, time, pred_class)
        feats, feat_lens = batch['feat']
        batch_size = feats.shape[0]

        phn_mode_label_seqs, phlvl_phn_mode_label_len = batch['phn_mode_encoded']     
        flvl_phn_mode_label_seqs, flvl_phn_mode_label_len = batch['flvl_phn_mode_encoded']
        
        if self.hparams.pretrain_domain == self.hparams.ssinger:
            # train on source data
            indices = torch.tensor(np.arange(0, batch_size, 2)).to(self.device)
        elif self.hparams.pretrain_domain == self.hparams.tsinger:
            # train on target data
            indices = torch.tensor(np.arange(1, batch_size, 2)).to(self.device)

        phn_mode_label_seqs = torch.index_select(phn_mode_label_seqs, 0, indices)
        phlvl_phn_mode_label_len = torch.index_select(phlvl_phn_mode_label_len, 0, indices)
        flvl_phn_mode_label_seqs = torch.index_select(flvl_phn_mode_label_seqs, 0, indices)
        flvl_phn_mode_label_len = torch.index_select(flvl_phn_mode_label_len, 0, indices)

        ##-----compute loss--------
        flvl_phn_mode_label = flvl_phn_mode_label_seqs.data

        ce = nn.CrossEntropyLoss()  # Frame wise binary cross entropy loss
        mse = nn.MSELoss(reduction='none')           # Migitating transistion loss 

        outp_wo_softmax = torch.log(out + 1e-10)         # log is necessary because ensemble gives softmax output
        outp_wo_softmax = outp_wo_softmax.permute(0, 2, 1)

        if outp_wo_softmax.shape[-1] != flvl_phn_mode_label.shape[-1]:
            outp_wo_softmax = outp_wo_softmax[...,:flvl_phn_mode_label.shape[-1]]
        
        
        # CE loss
        ce_loss = ce(outp_wo_softmax, flvl_phn_mode_label)
        # SM loss
        mse_loss = 0.15 * torch.mean(torch.clamp(mse(outp_wo_softmax[:, :, 1:], outp_wo_softmax.detach()[:, :, :-1]), 
                                                 min=0, max=16))

        loss = ce_loss + mse_loss 

        
        ##-----compute metrics-----
        ### frame level
        flvl_pred_pmd_lbl_seqs = torch.argmax(out, dim=-1)
        # unpad sequences
        flvl_gt_pmd_lbl_seqs_unpad = undo_padding(flvl_phn_mode_label_seqs, flvl_phn_mode_label_len)
        flvl_pred_pmd_lbl_seqs_unpad = undo_padding(flvl_pred_pmd_lbl_seqs, flvl_phn_mode_label_len)

        for i in range(len(flvl_pred_pmd_lbl_seqs_unpad)):
            # if pred label length is larger than gt label length
            if len(flvl_pred_pmd_lbl_seqs_unpad[i]) != len(flvl_gt_pmd_lbl_seqs_unpad[i]):
                flvl_pred_pmd_lbl_seqs_unpad[i] = flvl_pred_pmd_lbl_seqs_unpad[i][:len(flvl_gt_pmd_lbl_seqs_unpad[i])]
        
        ids = [batch['id'][i] for i in indices]

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

        return loss

    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=self.hparams.max_key, min_key=self.hparams.min_key)

        encoder_savepath = os.path.join(self.hparams.savept_path, 'C2F_Encoder.pt')
        Path(encoder_savepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.modules['C2F_Encoder'].state_dict(), encoder_savepath)

        classifier_savepath = os.path.join(self.hparams.savept_path, 'C2F_Classifier.pt')
        Path(classifier_savepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.modules['C2F_Classifier'].state_dict(), classifier_savepath)
