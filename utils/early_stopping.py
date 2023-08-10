import numpy as np

class EarlyStopping(object):
    def __init__(self,
                 min_delta=0,
                 patience=50,
                 verbose=0,
                 logger=None,
                 baseline=None):

        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.baseline = baseline
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.less  # np.less(x1, x2) == x1 < x2
        self.stop_training = False
        self.best = np.inf
        self.logger = logger
        self.save_checkpoint = False

    def on_train_begin(self):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf
        self.stop_training = False
        self.save_checkpoint = False

    def on_epoch_end(self, epoch, monitor, monitor_name):
        if self.monitor_op(monitor - self.min_delta, self.best):
            self.save_checkpoint = True
            self.best = monitor
            self.wait = 0
        else:
            self.save_checkpoint = False
            self.wait += 1
            if self.verbose > 1:
                if self.logger is not None:
                    self.logger.info(f"EarlyStopping counter: {self.wait} / {self.patience}")
                else:
                    print(f"EarlyStopping counter: {self.wait} / {self.patience}")
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                if self.verbose > 0 and self.verbose < 2:
                    if self.logger is not None:
                        self.logger.info(f'Epoch {self.stopped_epoch}: early stopping, best {monitor_name}: {self.best}')
                    else:
                        print(f'Epoch {self.stopped_epoch}: early stopping, best {monitor_name}: {self.best}')
        return self.save_checkpoint, self.stop_training
