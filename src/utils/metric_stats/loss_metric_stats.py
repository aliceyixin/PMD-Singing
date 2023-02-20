import torch


class LossMetricStats:
    def __init__(self, name):
        self.name = name
        self.clear()
        self.total_loss = 0

    def clear(self):
        self.loss_list = []

    def append(self, loss):
        loss = loss.clone().detach().cpu()

        self.loss_list.append(loss)  # save loss

    def summarize(self, field=None):
        if field is not None:
            raise ValueError('field must be None')

        mean_loss = torch.mean(torch.tensor(self.loss_list)).item()

        return {'loss': mean_loss}

    def write_stats(self, f):
        mean_loss = self.summarize()
        f.write(f'{self.name}: {mean_loss}\n')