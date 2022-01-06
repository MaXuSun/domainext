from .basetrainer import SimpleClassTrainer, SimpleSegTrainer
import time
import datetime
from domainext.utils.common import ( MetricMeter, AverageMeter)
from domainext.utils.common.data import ForeverDataIterator

class ClassTrainerX(SimpleClassTrainer):
    """A base trainer using labeled data only."""

    def run_epoch(self):
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        if self.cfg.TRAIN.ITER_TYPE == 'static':
            self.num_batches = len(self.train_loader_x)
        else:
            self.num_batches = self.cfg.TRAIN.ITER_NUM
            
        self.train_loader_x = ForeverDataIterator(self.train_loader_x)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            batch = next(self.train_loader_x)
    
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar('train/' + name, meter.avg, n_iter)
            self.write_scalar('train/lr', self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        domain = batch['domain']

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain


class SegTrainerX(SimpleSegTrainer):
    pass
