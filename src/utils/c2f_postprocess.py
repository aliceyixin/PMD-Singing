import torch.nn.functional as F
from itertools import groupby

def get_c2f_ensemble_output(outp, weights):
    
    ensemble_prob = F.softmax(outp[0], dim=1) * weights[0] / sum(weights)

    for i, outp_ele in enumerate(outp[1]):
        upped_logit = F.upsample(outp_ele, size=outp[0].shape[-1], mode='linear', align_corners=True)
        ensemble_prob = ensemble_prob + F.softmax(upped_logit, dim=1) * weights[i + 1] / sum(weights)
    
    return ensemble_prob

def frame2sample(framelist, sr, hoplen):

    hoplen_insample = hoplen * sr / 1000
    samplelist = [int(i * hoplen_insample) for i in framelist]
    return samplelist

def f2phlvl_seqs(flvl_pred_pmd_lbl_seqs_unpad, sr, hoplen):
    
    phlvl_pred_pmd_lbl_seqs = []
    phlvl_pred_seg_on_seqs = []
    phlvl_pred_seg_off_seqs = []
    # if a segment has a steady pred phonation mode of more than 5 frames, it can be regarded as detecting out a PM.
    tolerance = 5  # 5 frames
    for flvl_pred_pmd_lbl_seq in flvl_pred_pmd_lbl_seqs_unpad:
        # for each seq in a batch
        phlvl_pred_pmd_lbl_seq = []
        f_phlvl_pred_seg_on_seq = []
        f_phlvl_pred_seg_off_seq = []
        
        index = 0
        for label, group in groupby(flvl_pred_pmd_lbl_seq):
            duration = len(list(group))
            if duration >= tolerance and label != 0:
                phlvl_pred_pmd_lbl_seq.append(label)
                f_phlvl_pred_seg_on_seq.append(index)
                f_phlvl_pred_seg_off_seq.append(index+duration)
            index += duration

        phlvl_pred_seg_on_seq = frame2sample(f_phlvl_pred_seg_on_seq, sr, hoplen)
        phlvl_pred_seg_off_seq = frame2sample(f_phlvl_pred_seg_off_seq, sr, hoplen)

        if len(phlvl_pred_pmd_lbl_seq) == 0:
            # if the pred label are all zeros
            phlvl_pred_pmd_lbl_seqs.append([0])
            phlvl_pred_seg_on_seqs.append([0])
            phlvl_pred_seg_off_seqs.append([0])
        else:
            phlvl_pred_pmd_lbl_seqs.append(phlvl_pred_pmd_lbl_seq)
            phlvl_pred_seg_on_seqs.append(phlvl_pred_seg_on_seq)
            phlvl_pred_seg_off_seqs.append(phlvl_pred_seg_off_seq)

    return phlvl_pred_pmd_lbl_seqs, phlvl_pred_seg_on_seqs, phlvl_pred_seg_off_seqs