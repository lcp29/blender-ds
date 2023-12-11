# Transform blender-ds dataset to the ordinary blender dataset
# Detailed dataset format in IDR.md

import os, sys, glob
import logging
from absl import app, flags
import shutil
import numpy as np
import json
from PIL import Image
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', '', 'root directory of the source blender-ds dataset')

def main(_):
    # get output directory name
    output_name = os.path.basename(FLAGS.output_dir)
    output_root = os.path.dirname(FLAGS.output_dir)
    result_name = output_name + '_blender'
    result_dir = os.path.join(output_root, result_name)
    if os.path.exists(result_dir):
        logging.warning('Result directory already exists, removing...')
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    # iteratively process all splits
    for split in glob.glob(pathname='transforms_*.json', root_dir=FLAGS.output_dir):
        # get split name
        split_name = split[len('transforms_'):-len('.json')]
        logging.info(f'Processing split {split_name}...')
        # create split directory
        os.mkdir(os.path.join(result_dir, split_name))
        # process all views
        pbar = tqdm(glob.glob(pathname='*', root_dir=os.path.join(FLAGS.output_dir, split_name)), f'Processing split {split_name}...')
        for view in pbar:
            color_exist = os.path.exists(os.path.join(FLAGS.output_dir, split_name, view, 'color.png'))
            normal_exist = os.path.exists(os.path.join(FLAGS.output_dir, split_name, view, 'normal.png'))
            depth_exist = os.path.exists(os.path.join(FLAGS.output_dir, split_name, view, 'depth.png'))
            alpha_exist = os.path.exists(os.path.join(FLAGS.output_dir, split_name, view, 'alpha.png'))
            if color_exist:
                if alpha_exist:
                    # read color and alpha and add alpha channel to color
                    color = np.array(Image.open(os.path.join(FLAGS.output_dir, split_name, view, 'color.png')))
                    if color.shape[2] == 4:
                        color = color[:, :, :3]
                    alpha = np.array(Image.open(os.path.join(FLAGS.output_dir, split_name, view, 'alpha.png')))[:, :, -1:]
                    color = np.concatenate([color, alpha], axis=2)
                    # save color file
                    Image.fromarray(color).save(os.path.join(result_dir, split_name, view + '.png'))
                else:
                    shutil.copy(os.path.join(FLAGS.output_dir, split_name, view, 'color.png'), os.path.join(result_dir, split_name, view + '.png'))
            if normal_exist:
                shutil.copy(os.path.join(FLAGS.output_dir, split_name, view, 'normal.png'), os.path.join(result_dir, split_name, view + '_normal.png'))
            if depth_exist:
                shutil.copy(os.path.join(FLAGS.output_dir, split_name, view, 'depth.png'), os.path.join(result_dir, split_name, view + '_depth.png'))
            if alpha_exist:
                shutil.copy(os.path.join(FLAGS.output_dir, split_name, view, 'alpha.png'), os.path.join(result_dir, split_name, view + '_alpha.png'))
        shutil.copy(os.path.join(FLAGS.output_dir, split), os.path.join(result_dir, split))


if __name__ == '__main__':
    argv = sys.argv
    app.run(main=main, argv=argv)
