"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os


class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        data_list = sorted(self.get_paths(opt), key=lambda x: x['flair'])
        self.data_list = data_list[:opt.max_dataset_size]

        size = len(self.data_list)
        self.dataset_size = size

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths

    def __getitem__(self, index):
        # Label Image
        label_path = self.data_list[index]['seg']
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * (self.opt.label_nc + 1)
        label_tensor[label_tensor == (self.opt.label_nc + 1)] = self.opt.label_nc

        # input image (real images)
        image_path = self.data_list[index]['flair']
        image = Image.open(image_path)

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        input_dict = {
            'label': label_tensor,
            'image': image_tensor,
            'path': self.data_list[index]['path'],
        }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
