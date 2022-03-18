# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function

import logging
import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from mirror3d.utils.mirror3d_metrics import Mirror3dEval

import errno
import matplotlib.pyplot as plt
from tqdm import tqdm

from custom_dataloader import CustomImgDataset
sys.path.append('./')
from bts import BtsModel

def test(params):
    args.mode = 'test'
    args.output_save_folder = os.path.join(args.coco_val, "BTS_depth")
    dataset = CustomImgDataset(args.input_height, args.input_width, args.coco_val)
    
    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load(args.resume_checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    print('now testing {} files with {}'.format(len(dataset), args.resume_checkpoint_path))
    depth_shift = np.array(args.depth_shift)
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset)):
            image = Variable(sample['image'].cuda())
            focal = Variable(sample['focal'].cuda())
            _, _, _, _, depth_est = model(image, focal)
            pred_depth = depth_est.cpu().numpy().squeeze()
            color_img_path = sample["image_path"][0]
            
            pred_depth_scaled = (np.array(pred_depth) * depth_shift).astype(np.uint16)
            depth_np_save_path = args.output_save_folder + "/" + color_img_path.split("/")[-1]
            cv2.imwrite(depth_np_save_path[:-4] + ".png", pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get Setting :D')

    # Input source config
    parser.add_argument('--refined_depth',             action='store_true',  help='using coco input format or not')
    parser.add_argument('--mesh_depth',             type=bool,  help='using coco input format or not', default=True)
    parser.add_argument('--depth_shift',               type=int,   help='depth shift to meter', default=4000) 

    # Input format config
    parser.add_argument('--coco_val',                  type=str,   help='coco json path', default='')
    parser.add_argument('--coco_train',                type=str,   help='coco json path', default='')
    parser.add_argument('--coco_train_root',           type=str,   help='coco data root', default="")
    parser.add_argument('--coco_val_root',             type=str,   help='coco data root', default="")
    parser.add_argument('--coco_focal_len',            type=str,   help='focal length of input data; correspond to INPUT DEPTH!', default="519")
    parser.add_argument('--input_height',              type=int,   help='input height', default=480) 
    parser.add_argument('--input_width',               type=int,   help='input width',  default=640) 
    
    # Output config
    parser.add_argument('--resume_checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
    parser.add_argument('--output_save_folder',       type=str,   help='output_main_folder only use during inference', default='infer_output')


    parser.add_argument('--model_name',                type=str,   help='model name', default='bts')
    parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                        default='densenet161_bts')
    parser.add_argument('--data_path', type=str, help='path to the data', default="../dataset/nyu_depth_v2/official_splits/test/")
    parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', default="train_test_inputs/nyudepthv2_test_files_with_gt.txt")
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)

    parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
    parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)

    parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=10)
    args = parser.parse_args()

    model_dir = os.path.dirname(args.resume_checkpoint_path)
    sys.path.append(model_dir)
    a = time.time()
    test(args)
    print("TEST TIME:", time.time() - a)
