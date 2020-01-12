import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json


def parse_argumnet():
    with open('./checkpoint/unet_resnet34/' + "params.json", 'r', encoding='utf-8') as json_file:
        config = json.load(json_file)
    # dict to namespace
    config = Namespace(**config)    
