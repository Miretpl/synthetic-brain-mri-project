"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import pandas as pd
from os.path import join
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class CustomDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=13)
        parser.set_defaults(contain_dontcare_label=False)

        parser.add_argument('--ids_path', type=str, required=True, help='Path to tsv file with set paths')
        parser.add_argument('--data_dir', type=str, default=None, help='Path to directory where all data is stored')

        return parser

    def get_paths(self, opt):
        if opt.ids_path is None:
            data_dicts = [{
                'flair': f'{opt.data_dir}/01045/03_flair_unhealthy_{idx}.png',
                # This will be used as path for saving image
                'seg': f'{opt.data_dir}/01045/03_seg_unhealthy.png'
            } for idx in range(1000)]
        else:
            df = pd.read_csv(opt.ids_path, sep='\t')

            if opt.data_dir is not None:
                data_dicts = [
                    {
                        'flair': join(opt.data_dir, row['flair']),
                        'seg': join(opt.data_dir, row['seg'])
                    }
                    for index, row in df.iterrows()
                ]
            else:
                data_dicts = [
                    {
                        'flair': row['flair'],
                        'seg': row['seg']
                    }
                    for index, row in df.iterrows()
                ]

        print(f'Found {len(data_dicts)} subjects')
        return data_dicts
