import os

class config():
    __default_dict__ = {
        "batch_size":256,
        "init_lr":0.0005,
        "epochs":500,
        "min_delta":0.00,
        "patience":100,
        "output_dir":os.path.abspath('./outputs'),
        # "config_dir":os.path.abspath('./config'),
        # "log_dir":os.path.abspath('./log'),
        # "checkpoint_dir":os.path.abspath('./checkpoint'),
        "pretrained_model":None
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self.__default_dict__)
        self.__dict__.update(**kwargs)

        # self.__makedirs()

    def update(self, **kwargs):
        self.__dict__.update(**kwargs)

        self.__makedirs()

    def save_config(self, time):
        # self.config_dir = os.path.join(self.config_dir, str(time))
        # self.log_dir = os.path.join(self.log_dir, str(time))
        # self.checkpoint_dir = os.path.join(self.checkpoint_dir, str(time))
        self.output_dir = os.path.join(self.output_dir, str(time))
