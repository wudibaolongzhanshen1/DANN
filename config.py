import os

class config(object):

    __default_dict__ = {
        "pre_model_path":None,
        "checkpoints_dir":os.path.abspath("./checkpoints"),
        "logs_dir":"logs",
        "config_dir":os.path.abspath("./config"),
        "image_input_shape":(28,28,3),
        "image_size":28,
        "init_learning_rate": 1e-3,
        "momentum_rate":0.9,
        "batch_size":256,
        "epoch":500,
        "pixel_mean":[45.652287,45.652287,45.652287],
        "pixel_std":[405.652287,45.652287,45.652287]
    }

    def __init__(self,**kwargs):
        """
        这是参数配置类的初始化函数
        :param kwargs: 参数字典
        """
        # 初始化相关配置参数
        self.__dict__.update(self. __default_dict__)
        # 根据相关传入参数进行参数更新
        self.__dict__.update(kwargs)

        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)

    def set(self,**kwargs):
        """
        这是参数配置的设置函数
        :param kwargs: 参数字典
        :return:
        """
        # 根据相关传入参数进行参数更新
        self.__dict__.update(kwargs)


    def save_config(self,time):
        """
        这是保存参数配置类的函数
        :param time: 时间点字符串
        :return:
        """
        # 更新相关目录
        self.checkpoints_dir = os.path.join(self.checkpoints_dir,time)
        self.config_dir = os.path.join(self.config_dir,time)

        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        config_txt_path = os.path.join(self.config_dir,"config.txt")
        with open(config_txt_path,'a') as f:
            for key,value in self.__dict__.items():
                if key in ["checkpoints_dir","config_dir"]:
                    value = os.path.join(value,time)
                s = str(key)+": "+str(value)+"\n"
                f.write(s)


