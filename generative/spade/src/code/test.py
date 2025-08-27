"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from tqdm import tqdm

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# test
for i, data_i in tqdm(enumerate(dataloader), total=len(dataloader), desc="Image generation"):
    generated = model(data_i, mode='inference')

    img_path = data_i['path']

    for b in range(generated.shape[0]):
        visualizer.save_images(opt, generated[b], img_path[b])
