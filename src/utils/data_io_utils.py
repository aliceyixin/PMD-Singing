# coding: utf-8
import speechbrain as sb
import speechbrain.dataio.encoder


def get_label_encoder(hparams):
    """
    Get label encoder.
    Parameters
    ----------
    hparams : dict
        Loaded hparams.
    Returns
    -------
    label_encoder : sb.dataio.encoder.CategoricalEncoder
        The label encoder for the dataset.
    """
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    phoneme_set = ['rest', 'breathy', 'neutral', 'pressed']  
    label_encoder.update_from_iterable(phoneme_set, sequence_input=False)
    return label_encoder