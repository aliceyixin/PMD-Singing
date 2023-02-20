import torch
import numpy as np
from utils.metric_stats.base_metric_stats import BaseMetricStats


class phlvlPMDMetricStats(BaseMetricStats):
    def __init__(self, hparams):
        super(phlvlPMDMetricStats, self).__init__(metric_fn=batch_seq_pmd_scoring)
        self.saved_seqs = {}
        
    def append(self, ids, **kwargs):

        self.n_class = kwargs['n_class']

        if self.metric_fn is None:
            raise ValueError('No metric_fn has been provided')

        self.ids.extend(ids)  # save ID
        scores, seqs = self.metric_fn(**kwargs)
        self.scores_list.extend(scores)  # save metrics
        if len(self.metric_keys) == 0:  # save metric keys
            self.metric_keys = list(self.scores_list[0].keys())

        # update saved sequences
        seqs['utt_ids'] = ids

        if len(self.saved_seqs) == 0:
            self.saved_seqs = seqs
        else:
            for key in self.saved_seqs:
                if key != 'utt_ids':
                    self.saved_seqs[key].extend(seqs[key])

    def summarize(self, field=None):
        mean_scores = super(phlvlPMDMetricStats, self).summarize()

        eps = 1e-6
        ER = mean_scores['ER'] 

        ## class-based metrics
        # Macro average: calculate F1 for each class
        TP_class = np.zeros((self.n_class-1))
        FP_class = np.zeros((self.n_class-1))
        FN_class = np.zeros((self.n_class-1))
        TN_class = np.zeros((self.n_class-1))
        ACC_class = np.zeros((self.n_class-1))
        PRE_class = np.zeros((self.n_class-1))
        REC_class = np.zeros((self.n_class-1))
        F1_class = np.zeros((self.n_class-1))

        for scores in self.scores_list:
            TP_class = np.add(TP_class, scores['TP_class'])
            FP_class = np.add(FP_class, scores['FP_class'])
            FN_class = np.add(FN_class, scores['FN_class'])
            TN_class = np.add(TN_class, scores['TN_class'])

        eps = 1e-6
        for cl in range(self.n_class-1):            
            ACC_class[cl] = (TP_class[cl] + TN_class[cl]) / (TP_class[cl] + TN_class[cl] + FP_class[cl] + FN_class[cl] + eps) * 100
            PRE_class[cl] = TP_class[cl] / (TP_class[cl] + FP_class[cl] + eps) * 100
            REC_class[cl] = TP_class[cl] / (TP_class[cl] + FN_class[cl] + eps) * 100
            F1_class[cl] = np.round(2 * PRE_class[cl] * REC_class[cl] / (PRE_class[cl] + REC_class[cl] + eps), 4)  # 4 decimals 

        mean_scores['ACC_class'] = np.mean(ACC_class)
        mean_scores['PRE_class'] = np.mean(PRE_class)
        mean_scores['REC_class'] = np.mean(REC_class)
        mean_scores['F1_class'] = np.mean(F1_class)
        mean_scores['F1_breathy_class'] = F1_class[0]
        mean_scores['F1_neutral_class'] = F1_class[1]
        mean_scores['F1_pressed_class'] = F1_class[2]


        ## instance-based metrics
        # calculate F1 for each instance
        micro_ACC_class = (np.sum(TP_class) + np.sum(TN_class))/(np.sum(TP_class) + np.sum(TN_class) + np.sum(FP_class)+ np.sum(FN_class)+ eps) * 100
        micro_PRE_class = np.sum(TP_class)/(np.sum(TP_class) + np.sum(FP_class) + eps) * 100
        micro_REC_class = np.sum(TP_class)/(np.sum(TP_class) + np.sum(FN_class) + eps) * 100
        micro_F1_class = np.round(2 * micro_PRE_class * micro_REC_class / (micro_PRE_class + micro_REC_class + eps), 4) 

        mean_scores['ACC_instance'] = micro_ACC_class
        mean_scores['PRE_instance'] = micro_PRE_class
        mean_scores['REC_instance'] = micro_REC_class
        mean_scores['F1_instance'] = micro_F1_class

        for key in mean_scores:
            mean_scores[key] = round(mean_scores[key].item(), 2)

        if field is None:
            return mean_scores
        else:
            return mean_scores[field]

    def write_sample_results_to_file(self, path, label_encoder=None):
        # !!!
        with open(path, 'w') as f:
            batch_write_results(fp=f, scores_list=self.scores_list, label_encoder=label_encoder, **self.saved_seqs)


def convert_to_tensor(x):
    """
    Convert input to torch.LongTensor
    Parameters
    ----------
    x : list or np.ndarray or torch.tensor
        Binary one-dimension input.
    Returns
    -------
    converted_x : torch.LongTensor
        Converted input.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif isinstance(x, list):
        x = torch.Tensor(x)
    elif not isinstance(x, torch.Tensor):
        raise TypeError(f'Unsupported input type: {type(x).__name__}')
    x = x.int().squeeze()

    if x.ndim > 1:
        raise ValueError('Only one-dimension input is allowed')

    return x

def seq_pmd_scoring(sr, n_class, prediction, target, note_on_seq, note_off_seq, seg_on_seq, seg_off_seq):  # frame level scoring
    """
    Compute PMD scores of two binary sequences.
    Parameters
    ----------
    prediction : np.ndarray or torch.Tensor or list
        PMD results predicted by the model.
    target : torch.Tensor or list
        PMD ground truth.
    Returns
    -------
    pmd_scores : dict
        A dictionary of PMD scores.
    """

    # check input
    valid_input_types = (int, np.ndarray, torch.Tensor, list)
    if not isinstance(prediction, valid_input_types):
        raise TypeError(f'Unsupported input type: {type(prediction).__name__}')
    if not isinstance(target, valid_input_types):
        raise TypeError(f'Unsupported input type: {type(target).__name__}')
   
    def indices_to_one_hot(data, nb_classes):
        """Convert an iterable of indices to one-hot encoded labels."""
        targets = np.array(data).reshape(-1)
        return np.eye(nb_classes)[targets].reshape((-1, nb_classes))

    # Labels to onehot labels
    target_onehot = indices_to_one_hot(target, n_class)
    prediction_onehot = indices_to_one_hot(prediction, n_class)


    TPs = np.zeros((n_class))
    FPs = np.zeros((n_class))
    FNs = np.zeros((n_class))
    TNs = np.zeros((n_class))
    ACCs = np.zeros((n_class))
    PREs = np.zeros((n_class))
    RECs = np.zeros((n_class))
    F1_classes = np.zeros((n_class))

    for cl in range(1, n_class):  # ignore 'rest'
        TP = 0
        for i in range(len(note_on_seq)):
            note_on = note_on_seq[i]
            note_off = note_off_seq[i]
            onl = [(seg_on_seq[k]-note_on) for k in range(len(seg_on_seq))]
            offl = [(seg_off_seq[k]-note_off) for k in range(len(seg_off_seq))]
            d_on = np.min(np.abs(onl))
            d_off = np.min(np.abs(offl))
            index = np.argmin(np.abs(onl))

            delta = 0.25 * sr  # 250ms
            if ((d_on + d_off) < delta) and (target_onehot[i, cl] * prediction_onehot[index, cl]):
                TP+=1                

        FP = np.sum(prediction_onehot[...,cl]) - TP
        FN = np.sum(target_onehot[...,cl]) - TP
        TN = len(target_onehot) - TP - FP - FN
        eps = 1e-6
        ACCs[cl] = (TP + TN) / (TP + TN + FP + FN + eps) * 100
        PREs[cl] = TP / (TP + FP + eps) * 100
        RECs[cl] = TP / (TP + FN + eps) * 100
        F1_classes[cl] = np.round(2 * PREs[cl] * RECs[cl] / (PREs[cl] + RECs[cl] + eps), 4)  # 4 decimals 

        TPs[cl] = TP
        FPs[cl] = FP
        FNs[cl] = FN
        TNs[cl] = TN
    
    ## Instance-based averaging 
    TP = np.sum(TPs[1:])
    FP = np.sum(FPs[1:])
    FN = np.sum(FNs[1:])
    TN = np.sum(TNs[1:])

    ACC_instance = (TP + TN) / (TP + TN + FP + FN + eps) * 100
    PRE_instance = TP / (TP + FP + eps) * 100
    REC_instance = TP / (TP + FN + eps) * 100
    F1_instance = 2 * PRE_instance * REC_instance / (PRE_instance + REC_instance + eps)

    ## Class-based averaging   # Exclude 'rest'

    ACC_class = np.mean(ACCs[1:])
    PRE_class = np.mean(PREs[1:])
    REC_class = np.mean(RECs[1:])
    F1_class = np.mean(F1_classes[1:])

    # ERROR RATE
    S = min(FN, FP)  # substitutions
    D = max(0, FN-FP)  # deletions
    I = max(0, FP-FN)  # insertions
    N = len(note_on_seq)  # num of event in the reference
    ER = np.round((S + D + I)/N, n_class-1)

    pmd_scores = {
        'ACC_class': ACC_class,
        'PRE_class': PRE_class,
        'REC_class': REC_class,
        'F1_class': F1_class,
        'ACC_instance': ACC_instance,
        'PRE_instance': PRE_instance,
        'REC_instance': REC_instance,        
        'F1_instance': F1_instance,
        'ER': ER,
        'TP_class': TPs[1:],
        'FP_class': FPs[1:],
        'FN_class': FNs[1:],
        'TN_class': TNs[1:],
    }

    return pmd_scores


def batch_seq_pmd_scoring(
        sr=None,
        n_class=None,
        pred_pmd_lbl_seqs=None,
        gt_pmd_lbl_seqs=None,
        pred_seg_on_seqs=None,
        pred_seg_off_seqs=None,
        gt_seg_on_seqs=None,
        gt_seg_off_seqs=None,
        ):
    """
    Compute PMD scores for a batch.
    Parameters
    ----------
    pred_pmd_lbl_seqs : list
        List of predicted PMD labels.
    gt_pmd_lbl_seqs : list
        List of ground truth PMD labels.
    Returns
    -------
    batch_pmd_scores : list
        list of PMD scores
    """

    # check input
    for x in [pred_pmd_lbl_seqs, gt_pmd_lbl_seqs]:
        if x is not None and not isinstance(x, list):
            raise TypeError(f'Input type must be list, not {type(x).__name__}')

    # compute and save PMD scores for each sample in the batch
    if len(pred_pmd_lbl_seqs) != len(gt_pmd_lbl_seqs):
        raise ValueError(f'Inconsistent batch size: {len(pred_pmd_lbl_seqs)} != {len(gt_pmd_lbl_seqs)}')
    
    pmd_scores = []
    for i in range(len(pred_pmd_lbl_seqs)): # for each batch
        # PMD scores
        seq_pmd_scores = seq_pmd_scoring(sr, n_class, pred_pmd_lbl_seqs[i], gt_pmd_lbl_seqs[i],
                                        gt_seg_on_seqs[i], gt_seg_off_seqs[i], pred_seg_on_seqs[i], pred_seg_off_seqs[i])
        
        # save scores
        pmd_scores.append(seq_pmd_scores)

    # save sequences for writing to the file
    seqs_keys = ['gt_pmd_lbl_seqs', 'pred_pmd_lbl_seqs']
    seqs_dict = {key: [] for key in seqs_keys}
    for i in range(len(pmd_scores)):
        seqs_dict['gt_pmd_lbl_seqs'].append(gt_pmd_lbl_seqs)
        seqs_dict['pred_pmd_lbl_seqs'].append(pred_pmd_lbl_seqs)

 
    return pmd_scores, seqs_dict

# -------------------------------------------------------------------------------------------
def write_pmd_results(
        fp,
        scores,
        utt_id,
        gt_pmd_lbl_seq,
        pred_pmd_lbl_seq=None,
        label_encoder=None
):

    """
    Write PMD results to a file.
    Parameters
    ----------
    scores : dict
        PMD scores.
    fp : File
        File object to write the results.
    utt_id : str
        Utterance ID.
    gt_pmd_lbl_seq : list
        Ground truth PMD labels.
    pred_pmd_lbl_seq : list
        Predicted PMD labels.
    label_encoder : LabelEncoder
        The label encoder.
    """
    # input check
    if gt_pmd_lbl_seq is None and pred_pmd_lbl_seq is None:
        raise ValueError('gt_pmd_lbl_seq and pred_pmd_lbl_seq cannot be None at the same time.')
    length = len(gt_pmd_lbl_seq)

    # initialize lines with the first ID line
    lines = [f'ID: {utt_id}\n']

    # template for each line
    line_template = '{:11s}: |' + '|'.join(['{:^4s}'] * length) + '|\n'
    lines.append(line_template.format('gt_pmd_lbl', *[str(x) for x in gt_pmd_lbl_seq]))
    lines.append(line_template.format('pred_pmd_lbl', *[str(x) for x in pred_pmd_lbl_seq]))

    # scores
    for key, value in scores.items():
        lines.append(f'{key}: {value}\n')

    lines.append('\n')

    # write to file
    fp.writelines(lines)

def batch_write_results(
        fp,
        scores_list,
        utt_ids,
        gt_pmd_lbl_seqs=None,
        pred_pmd_lbl_seqs=None,
        label_encoder=None
    ):
    """
    Parameters
    ----------
    Batched version of write_pmd_results().
    """
    B = len(utt_ids)  # batch size

    for i in range(B):
        write_pmd_results(
            fp,
            scores_list[i],
            utt_ids[i],
            gt_pmd_lbl_seqs[i],
            pred_pmd_lbl_seqs[i],
            label_encoder
        )