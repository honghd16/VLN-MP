#!/usr/bin/env python3

''' Script to precompute image features using a Pytorch ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import os
import sys

import MatterSim

import argparse
import numpy as np
import json
import math
import h5py
import copy
from PIL import Image
import time
from progressbar import ProgressBar

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from glob import glob

VIEWPOINT_SIZE = 36 
FEATURE_SIZE = 768


def get_all_objs():
    inpaint_dir = '/mnt/data/haodong/UTSD'
    obj_paths = [x for x in glob(os.path.join(inpaint_dir, '*', '*', '*')) if "_full" in x]
    return obj_paths


def build_feature_extractor(model_name, checkpoint_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = timm.create_model(model_name, pretrained=(checkpoint_file is None)).to(device)
    if checkpoint_file is not None:
        state_dict = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)['state_dict']
        model.load_state_dict(state_dict)
    model.eval()

    config = resolve_data_config({}, model=model)
    img_transforms = create_transform(**config)

    return model, img_transforms, device

def process_features(proc_id, out_queue, objs, args):
    print('start proc_id: %d' % proc_id)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    model, img_transforms, device = build_feature_extractor(args.model_name, args.checkpoint_file)

    for obj in objs:
        scan = obj.split('/')[-3]
        obj_id = obj.split('/')[-2]

        image = Image.open(obj).convert('RGB')
        images = [image]

        images = torch.stack([img_transforms(image).to(device) for image in images], 0)
        fts = model.forward_features(images)
        fts = fts.data.cpu().numpy()

        out_queue.put((scan, obj_id, fts))

    out_queue.put(None)

def build_feature_file(args):
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    objs = get_all_objs()

    num_workers = min(args.num_workers, len(objs))
    num_data_per_worker = len(objs) // num_workers

    out_queue = mp.Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_features,
            args=(proc_id, out_queue, objs[sidx: eidx], args)
        )
        process.start()
        processes.append(process)
    
    num_finished_workers = 0
    num_finished_vps = 0

    progress_bar = ProgressBar(max_value=len(objs))
    progress_bar.start()

    with h5py.File(args.output_file, 'w') as outf:
        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan, obj_id, fts = res
                key = '%s_%s'%(scan, obj_id)
                data = fts
                outf.create_dataset(key, data.shape, dtype='float', compression='gzip')
                outf[key][...] = data
                outf[key].attrs['scanId'] = scan
                outf[key].attrs['objId'] = obj_id

                num_finished_vps += 1
                progress_bar.update(num_finished_vps)

    progress_bar.finish()
    for process in processes:
        process.join()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='vit_base_patch16_224')
    parser.add_argument('--checkpoint_file', default=None)
    parser.add_argument('--out_image_logits', action='store_true', default=False)
    parser.add_argument('--output_file', default="./utsd_obj_full_features.hdf5")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    build_feature_file(args)


