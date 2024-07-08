import yaml
import time
from utils.convertor import *

from .utils import obj
import sys, os
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)


def load_class_names(namesfile, trim=True):
    # namesfile = self.DATA.CLASS_NAME_FILE
    all_class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        # print('line: ', line)
        if trim:
            line = line.replace(' ', '')
        all_class_names.append(line)
    return all_class_names


class ConfigParser:
    def __init__(self, args, time_str, test_mode=False):
        self.test_mode = test_mode
        self.time_str = time_str
        self.args = args
        self.exp_dir = args.exp_dir
        self.config_file = args.cfg
        self.load_config()
        self.converter = LabelConverter()
        if args.det != None and self.ATTACKER.TYPE == "TA-C":
            self.DETECTOR.NAME = [args.det]
        elif args.det != None:
            self.DETECTOR.NAME = args.det
        self.target_index = self.get_index_from_label(self.ATTACKER.TARGET_LABEL if args.target == None else args.target)
        if self.ATTACKER.TYPE == "TA-D":
            self.origin_index = self.get_index_from_label(args.origin)
        else:
            self.origin_index = None

    def get_index_from_label(self, target_label):
        if self.DETECTOR.NAME == "faster_rcnn":
            names = self.converter.category91
        elif self.DETECTOR.NAME == "yolov3":
            names = self.converter.category80
        elif self.DETECTOR.NAME == "yolov5":
            names = self.converter.category80
        else:
            return target_label
        target_id = names.index(target_label)
        return target_id

    def get_index_from_model_label(self, model_name, attack_label):
        if model_name == "faster_rcnn":
            names = self.converter.category91
        elif model_name == "yolov3":
            names = self.converter.category80
        elif model_name == "yolov5":
            names = self.converter.category80
        else:
            return int(attack_label)
        attack_index = names.index(attack_label)
        return attack_index

    def load_config(self):
        cfg = yaml.load(open(self.config_file), Loader=yaml.FullLoader)
        for a, b in cfg.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

    def __str__(self):
        pass


def ignore_class(eva_args, cfg):
    # Be careful of the so-called 'attack_list' and 'eva_class' in the evaluate.py
    # For higher reusability of the codes, these variable names may be confusing
    # In this file, the 'attack_list' is loaded from the config file which has been used for training
    # (cuz we don't bother to create a new config file for evaluations)
    # Thus the 'attack_list' refers to the original attacked classes when training the patch
    # while the 'eva_list' denotes the class list to be evaluated, which are to attack in the evaluation
    # (When the eva classes are different from the original attack classes,
    # it is mainly for the partial attack in evaluating unseen-class/cross-class performance)
    eva_args.eva_class_list = cfg.rectify_class_list(eva_args.eva_class, dtype='str')
    # print('Eva(Attack) classes from evaluation: ', cfg.show_class_index(args.eva_class_list))
    # print('Eva classes names from evaluation: ', args.eva_class_list)

    eva_args.ignore_class = list(set(cfg.all_class_names).difference(set(eva_args.eva_class_list)))
    if len(eva_args.ignore_class) == 0: eva_args.ignore_class = None
    return eva_args


def logger_msg(k, v):
    msg = '{:>30} : {:<30}'.format(str(k), str(v))
    print(msg)
    return msg

def logger_banner(banner):
    dot = '------------------------------------------------------------'
    pos = int(len(dot)/2 - len(banner)/2)
    banner = dot[:pos] + banner + dot[pos+len(banner):]
    print(banner)


def logger_cfg(cfg, banner=None):
    if banner is not None:
        logger_banner(banner)
    for k, v in cfg.__dict__.items():
        if isinstance(v, obj):
            logger_cfg(v)
        else:
            logger_msg(k, v)


def logger(cfg, args):
    logger_banner('Training')
    logger_msg('cfg', args.cfg)
    localtime = time.asctime(time.localtime(time.time()))
    logger_msg('time', localtime)
    logger_cfg(cfg.DATA, 'DATA')
    logger_cfg(cfg.DETECTOR, 'DETECTOR')
    logger_cfg(cfg.ATTACKER, 'ATTACKER')
    logger_cfg(cfg.EVAL, 'EVAL')
    logger_banner('END')


def merge_dict_by_key(dict_s, dict_d):
    for k, v in dict_s.items():
        dict_d[k] = [v, dict_d[k]]
    return dict_d


def dict2txt(dict, filename, ljust=16):
    with open(filename, 'a') as f:
        for k, v in dict.items():
            f.write(k.ljust(ljust, ' ') + ':\t' + str(v))
            f.write('\n')

