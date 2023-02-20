import numpy as np
import torch
from utils.metric_stats.base_metric_stats import BaseMetricStats


class DomainMetricStats(BaseMetricStats):
    def __init__(self):
        super(DomainMetricStats, self).__init__(metric_fn=batch_seq_d_scoring)
        self.saved_seqs = {}

    def append(self, ids, **kwargs):
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
                self.saved_seqs[key].extend(seqs[key])

    def summarize(self, field=None):
        mean_scores = super(DomainMetricStats, self).summarize()

        eps = 1e-6
        s_domain_ACC = mean_scores['s_domain_ACC']
        s_domain_PRE = mean_scores['s_domain_PRE']
        s_domain_REC = mean_scores['s_domain_REC']
        mean_scores['s_domain_F1'] = (2 * s_domain_PRE * s_domain_REC) / (s_domain_PRE + s_domain_REC + eps)
        t_domain_ACC = mean_scores['t_domain_ACC']
        t_domain_PRE = mean_scores['t_domain_PRE']
        t_domain_REC = mean_scores['t_domain_REC']
        mean_scores['t_domain_F1'] = (2 * t_domain_PRE * t_domain_REC) / (t_domain_PRE + t_domain_REC + eps)


        for key in mean_scores:
            mean_scores[key] = round(mean_scores[key].item(), 2)

        if field is None:
            return mean_scores
        else:
            return mean_scores[field]

    def write_seqs_to_file(self, path, label_encoder=None):
        with open(path, 'w') as f:
            batch_write_pmd_results(fp=f, scores_list=self.scores_list, label_encoder=label_encoder, **self.saved_seqs)


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


def seq_d_scoring(t_domain_pred_seq, t_domain_gt_seq, s_domain_pred_seq, s_domain_gt_seq):  
    # frame level/ sample level scoring
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
    if not isinstance(t_domain_pred_seq, valid_input_types):
        raise TypeError(f'Unsupported input type: {type(t_domain_pred_seq).__name__}')
    if not isinstance(s_domain_pred_seq, valid_input_types):
        raise TypeError(f'Unsupported input type: {type(s_domain_pred_seq).__name__}')

    def get_metrics(pred_seq, gt_seq):
        TP = np.sum(gt_seq * pred_seq)
        FP = np.sum(pred_seq) - TP
        FN = np.sum(gt_seq) - TP
        TN = np.sum((1 - gt_seq) * (1 - pred_seq))
        eps = 1e-6
        ACC = (TP + TN) / (TP + TN + FP + FN + eps) * 100
        PRE = TN / (TN + FN + eps) * 100
        REC = TN / (TN + FP + eps) * 100
        F1 = 2 * PRE * REC / (PRE + REC + eps)
        return ACC, PRE, REC, F1
    
    if isinstance(t_domain_pred_seq, int):
        # when each item has one domain label
        s_domain_ACC, s_domain_PRE, s_domain_REC, s_domain_F1 = get_metrics(s_domain_pred_seq, s_domain_gt_seq)
        t_domain_ACC, t_domain_PRE, t_domain_REC, t_domain_F1 = get_metrics(t_domain_pred_seq, t_domain_gt_seq)
    else:
        s_domain_ACC, s_domain_PRE, s_domain_REC, s_domain_F1 = get_metrics(np.array(s_domain_pred_seq), np.array(s_domain_gt_seq))
        t_domain_ACC, t_domain_PRE, t_domain_REC, t_domain_F1 = get_metrics(np.array(t_domain_pred_seq), np.array(t_domain_gt_seq))

    d_scores = {
        's_domain_ACC': s_domain_ACC,        
        's_domain_PRE': s_domain_PRE,
        's_domain_REC': s_domain_REC,
        's_domain_F1': s_domain_F1,
        't_domain_ACC': t_domain_ACC,
        't_domain_PRE': t_domain_PRE,
        't_domain_REC': t_domain_REC,
        't_domain_F1': t_domain_F1
    }
    return d_scores


def batch_seq_d_scoring(
        s_domain_pred_seqs=None,
        s_domain_gt_seqs=None,
        t_domain_pred_seqs=None,
        t_domain_gt_seqs=None,
        ):
    """
    Compute domain scores for a batch.
    Parameters
    ----------
    s_domain_pred_seqs : list
        List of predicted Domain labels.
    s_domain_gt_seqs : list
        List of ground truth Domain labels.

    Returns
    -------
    batch_d_scores : list
        list of domain metric scores
    """
    # check input
    for x in [s_domain_pred_seqs, s_domain_gt_seqs, t_domain_pred_seqs, t_domain_gt_seqs]:
        if x is not None and not isinstance(x, list):
            raise TypeError(f'Input type must be list, not {type(x).__name__}')

    # compute and save PMD scores for each sample in the batch
    if len(t_domain_pred_seqs) != len(s_domain_pred_seqs):
        raise ValueError(f'Inconsistent batch size: {len(t_domain_pred_seqs)} != {len(s_domain_pred_seqs)}')
    
    d_scores = []
    for i in range(len(t_domain_pred_seqs)):
        seq_d_scores = seq_d_scoring(t_domain_pred_seqs[i], t_domain_gt_seqs[i],
                                        s_domain_pred_seqs[i], s_domain_gt_seqs[i])
        
        # save scores
        d_scores.append(seq_d_scores)

    # save sequences for writing to the file
    seqs_keys = ['s_domain_pred_seqs', 't_domain_pred_seqs']
    seqs_dict = {key: [] for key in seqs_keys}
    for i in range(len(d_scores)):
        seqs_dict['s_domain_pred_seqs'].append(s_domain_pred_seqs)
        seqs_dict['t_domain_pred_seqs'].append(t_domain_pred_seqs)

    return d_scores, seqs_dict


def write_d_results(
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
    # gt_phn_seq : list
    #     List of pronounced phonemes.
    gt_pmd_lbl_seq : list
        Ground truth PMD labels.
    # pred_phn_seq : list
    #     Predicted phonemes.
    pred_pmd_lbl_seq : list
        Predicted PMD labels.
    label_encoder : LabelEncoder
        The label encoder.
    """
    # input check
    if pred_phn_seq is None and pred_md_lbl_seq is None:
        raise ValueError('pred_phn_seq and pred_md_lbl_seq cannot be None at the same time.')
    length = len(gt_phn_seq)
    assert len(gt_pmd_lbl_seq) == length
    assert pred_phn_seq is None or len(pred_phn_seq) == length
    assert pred_md_lbl_seq is None or len(pred_md_lbl_seq) == length

    # handle None input
    if pred_phn_seq is None:
        pred_phn_seq = ['NA'] * length
    if pred_md_lbl_seq is None:
        pred_md_lbl_seq = []
        for cnncl, pred_phn in zip(gt_cnncl_seq, pred_phn_seq):
            if cnncl == pred_phn:
                pred_md_lbl_seq.append(0)
            else:
                pred_md_lbl_seq.append(1)

    # correctness_seq
    correctness_seq = []
    for gt_pmd_lbl, pred_pmd_lbl in zip(gt_pmd_lbl_seq, pred_pmd_lbl_seq):
        if gt_pmd_lbl == pred_pmd_lbl:
            correctness_seq.append('c')
        else:
            correctness_seq.append('x')

    # decoded phonemes
    if label_encoder is not None:
        def decode_seq(seq):
            decoded_seq = []
            for p in seq:
                if p == -1:
                    decoded_seq.append('**')
                else:
                    decoded_seq.append(label_encoder.ind2lab[int(p)])
            return decoded_seq

        gt_phn_seq = decode_seq(gt_phn_seq)
        gt_cnncl_seq = decode_seq(gt_cnncl_seq)
        pred_phn_seq = decode_seq(pred_phn_seq)

    # initialize lines with the first ID line
    lines = [f'ID: {utt_id}\n']

    # template for each line
    line_template = '{:11s}: |' + '|'.join(['{:^4s}'] * length) + '|\n'

    lines.append(line_template.format('pmd_lbl', *[str(x) for x in gt_pmd_lbl_seq]))
    lines.append(line_template.format('pred_pmd_lbl', *[str(x) for x in pred_pmd_lbl_seq]))
    lines.append(line_template.format('correctness', *correctness_seq))

    # scores
    for key, value in scores.items():
        lines.append(f'{key}: {value}\n')

    lines.append('\n')

    # write to file
    fp.writelines(lines)


def batch_write_pmd_results(
        fp,
        scores_list,
        utt_ids,
        gt_pmd_lbl_seqs,
        pred_pmd_lbl_seqs=None,
        label_encoder=None
):
    """
    Parameters
    ----------
    Batched version of write_pmd_results().
    """
    B = len(utt_ids)  # batch size
    assert len(gt_pmd_lbl_seqs) == B

    if pred_pmd_lbl_seqs is None:
        pred_pmd_lbl_seqs = [None] * B

    for i in range(B):
        write_d_results(
            fp,
            scores_list[i],
            utt_ids[i],
            gt_pmd_lbl_seqs[i],
            pred_pmd_lbl_seqs[i],
            label_encoder
        )