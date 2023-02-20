import logging
import warnings
from pathlib import Path
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import speechbrain as sb
from speechbrain.utils.train_logger import FileTrainLogger
from speechbrain.utils.checkpoints import Checkpointer
from utils.metric_stats.loss_metric_stats import LossMetricStats

logger = logging.getLogger(__name__)


class ADDAModel(sb.Brain):
    def __init__(self, label_encoder, **kwargs):
        super(ADDAModel, self).__init__(**kwargs)
        self.label_encoder = label_encoder

    def init_optimizers(self):
        """
        Initialize multiple optimizers or an optimizer.
        """
        if hasattr(self.hparams, 'optimizers'):
            opt_info_dict = self.hparams.optimizers
            if type(opt_info_dict) is list:
                opt_info_dict = {f'optimizer_{i}': opt_info for i, opt_info in enumerate(opt_info_dict)}
        elif hasattr(self.hparams, 'optimizer'):
            opt_info_dict = {'optimizer': self.hparams.optimizer}
        else:
            raise ValueError('No optimizers defined.')

        self.optimizers = {}
        for key, opt_info in opt_info_dict.items():
            if type(opt_info) is dict:
                opt_class = opt_info['opt_class']
                if 'modules' in opt_info:
                    params = []
                    for module_name in opt_info['modules']:
                        params += list(self.modules[module_name].parameters())
                else:
                    params = self.modules.parameters()
                optimizer = opt_class(params)
            else:
                optimizer = opt_info(self.modules.parameters())
            self.optimizers[key] = optimizer

        if self.checkpointer is not None:
            for key, optimizer in self.optimizers.items():
                self.checkpointer.add_recoverable(key, optimizer)

    def fit_batch(self, batch):
        """
        Overwritten for multiple optimizers.
        """
        # Managing automatic mixed precision
        optimizers = [optimizer for _, optimizer in self.optimizers.items()]
        if self.auto_mix_prec:
            for optimizer in optimizers:
                optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            self.scaler.scale(loss).backward()

            for optimizer in optimizers:
                self.scaler.unscale_(optimizer)

            if self.check_gradients(loss):
                for optimizer in optimizers:
                    self.scaler.step(optimizer)

            self.scaler.update()
        else:
            # run this option in else
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            # loss, loss_CLS, loss_DA, loss_discriminator = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
    
            loss.backward()

            if self.check_gradients(loss):
                for optimizer in optimizers:
                    optimizer.step()
            for optimizer in optimizers:
                optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Evaluate one batch, override for different procedure than train.

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """

        out = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(out, batch, stage=stage)
        return loss.detach().cpu()

    def on_fit_start(self):
        super(ADDAModel, self).on_fit_start()
        train_logger_save_file = Path(self.hparams.output_dir) / 'train_log.txt'
        self.train_logger = FileTrainLogger(save_file=train_logger_save_file)
        self.tb_writer = SummaryWriter(log_dir=self.hparams.output_dir)

        logger.info(str(self.modules))
        with open(train_logger_save_file, 'w') as f:
            f.write(str(self.modules) + '\n')

    def on_stage_start(self, stage, epoch=None):
        # initialize metric stats
        self.stats_loggers = {}

        # initialize metric stats for losses
        for loss_key in self.hparams.metric_keys:
            if loss_key.endswith('_loss'):
                stats_key = loss_key + '_stats'
                self.stats_loggers[stats_key] = LossMetricStats(loss_key)

        # debug
        if self.debug and stage == sb.Stage.TRAIN:
            logger.info(f'{torch.cuda.memory_allocated()} {torch.cuda.max_memory_allocated()}')

    def on_stage_end(self, stage, stage_loss, epoch=None):
        stage_name = str(stage).split('.')[1].lower()

        if epoch is None:
            epoch = self.hparams.epoch_counter.current

        # get metrics
        log_metrics = {'loss': round(stage_loss, 3)}

        for metric_key in self.hparams.metric_keys:  # e.g. metric_key = 'PER' or 'md.F1'
            metric_key_list = metric_key.split('.')
            stats_logger_key = f'{metric_key_list[0].lower()}_stats'
            stats = self.stats_loggers.get(stats_logger_key)

            if stats is not None:
                if len(metric_key_list) == 1:  # summarize all metrics
                    summarized_stats = stats.summarize(None)
                    for stats_key, value in summarized_stats.items():
                        complete_metric_key = f'{metric_key}.{stats_key}'
                        log_metrics[complete_metric_key] = round(value, 2)
                else:  # summarize a particular metric
                    summarized_stats = stats.summarize(metric_key_list[1])
                    log_metrics[metric_key] = round(float(summarized_stats), 2)


        if stage == sb.Stage.TRAIN or stage == sb.Stage.VALID:
            # log stats
            # lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            train_logger_stats = {
                'stats_meta': {'stage': stage_name, 'epoch': epoch},
                f'{stage_name}_stats': log_metrics
            }
            self.train_logger.log_stats(**train_logger_stats)

            # TB log
            for metric_key, metric_value in log_metrics.items():
                self.tb_writer.add_scalar(f'{metric_key}/{stage_name}', metric_value, global_step=epoch)

            # save checkpoint after the VALID stage
            if stage == sb.Stage.VALID:
                max_keys = []
                min_keys = []
                if hasattr(self.hparams, 'max_key'):
                    if self.hparams.max_key is not None:
                        max_keys.append(self.hparams.max_key)
                if hasattr(self.hparams, 'min_key'):
                    if self.hparams.min_key is not None:
                        min_keys.append(self.hparams.min_key)
                if len(max_keys) == 0 and len(min_keys) == 0:
                    raise ValueError('no max_key or min_key provided')
                self.checkpointer.save_and_keep_only(
                    meta=log_metrics, max_keys=max_keys, min_keys=min_keys
                )

        if stage == sb.Stage.TEST:
            test_output_dir = Path(self.hparams.output_dir) / 'test_output'

            test_output_dir.mkdir(exist_ok=True)

            # log test metrics
            log_str = ', '.join([f'{k}: {v}' for k, v in log_metrics.items()])
            logger.info(f'Best epoch: {epoch}, {log_str}')
            metric_values = []
            with open(test_output_dir / 'test_metrics.txt', 'w') as f:
                f.write(f'Epoch: {epoch}\n')
                for metric_key, metric_value in log_metrics.items():
                    f.write(f'{metric_key}: {metric_value}\n')
                    metric_values.append(str(metric_value))
                f.write(f'Epoch: {epoch}\t' + '\t'.join(metric_values) + '\n')
                logger.info(f'Test metrics saved to {f.name}')

            # save all stats loggers
            for stats_key, stats_logger in self.stats_loggers.items():
                stats_key = stats_key.replace('_stats', '')
                with open(test_output_dir / f'{stats_key}.txt', 'w') as f:
                    stats_logger.write_stats(f)
                    logger.info(f'{stats_key} stats saved to {f.name}')


    # def compute_and_save_losses(self, losses):  # calculate loss in model.py
    #     loss = 0
    #     for loss_key in losses:
    #         # compute weighted loss
    #         weight_key = loss_key.replace('_loss', '_weight')
    #         weight = getattr(self.hparams, weight_key, 'none')
    #         if weight == 'none':
    #             warnings.warn(f'{weight_key} not found, use 1 as default')
    #             weight = 1
    #         loss += weight * losses[loss_key]

    #         # save loss
    #         loss_stats_logger_key = loss_key + '_stats'
    #         loss_stats_logger = self.stats_loggers.get(loss_stats_logger_key)
    #         if loss_stats_logger is not None:
    #             self.stats_loggers[loss_stats_logger_key].append(losses[loss_key])
    #         else:
    #             # logger.warning(f'loss stats logger {loss_stats_logger_key} not found')
    #             warnings.warn(f'loss stats logger {loss_stats_logger_key} not found')

    #     return loss