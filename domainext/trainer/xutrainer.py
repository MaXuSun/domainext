from .basetrainer import SimpleClassTrainer, SimpleSegTrainer
import time
import datetime
from domainext.utils.common import ( MetricMeter, AverageMeter)
from domainext.utils.common.data import ForeverDataIterator

class TrainerXU(SimpleClassTrainer):
    """A base trainer using both labeled and unlabeled data.

    In the context of domain adaptation, labeled and unlabeled data
    come from source and target domains respectively.

    When it comes to semi-supervised learning, all data comes from the
    same domain.
    """

    def run_epoch(self):
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        if self.cfg.TRAIN.ITER_TYPE == 'static':
            len_train_loader_x = len(self.train_loader_x)
            len_train_loader_u = len(self.train_loader_u)

            if self.cfg.TRAIN.COUNT_ITER == 'train_x':
                self.num_batches = len_train_loader_x
            elif self.cfg.TRAIN.COUNT_ITER == 'train_u':
                self.num_batches = len_train_loader_u
            elif self.cfg.TRAIN.COUNT_ITER == 'smaller_one':
                self.num_batches = min(len_train_loader_x, len_train_loader_u)
            else:
                raise ValueError
        else:
            self.num_batches ==  self.cfg.TRAIN.ITER_NUM

        train_loader_x_iter = ForeverDataIterator(self.train_loader_x)
        train_loader_u_iter = ForeverDataIterator(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            batch_x = next(train_loader_x_iter)
            batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
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

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x['img']
        label_x = batch_x['label']
        input_u = batch_u['img']

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        return input_x, label_x, input_u

class SegTrainerX(SimpleSegTrainer):
    pass
