# Transform blender-ds dataset to the dataset format used in IDR [https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md]
# Detailed dataset format in IDR.md

import os, sys, glob
import logging
import bpy
from absl import app, flags
import shutil
import numpy as np
import json
from tqdm import tqdm

flags.DEFINE_string('output_dir', '', 'root directory of the source blender-ds dataset')
flags.DEFINE_string('scene_path', '', 'path to the Blender scene')
flags.DEFINE_integer('resx', 800, 'width of output images')
flags.DEFINE_integer('resy', 800, 'height of output images')

FLAGS = flags.FLAGS

def main(_):
    # get output directory name
    output_name = os.path.basename(FLAGS.output_dir)
    output_root = os.path.dirname(FLAGS.output_dir)
    result_name = output_name + '_idr'
    
    # create result directory
    result_dir = os.path.join(output_root, result_name)
    if os.path.exists(result_dir):
        logging.warning('Result directory already exists, removing...')
        shutil.rmtree(result_dir)
    
    os.mkdir(result_dir)
    os.mkdir(os.path.join(result_dir, 'image'))
    os.mkdir(os.path.join(result_dir, 'mask'))

    logging.info('Result directory created.')

    # calculate the bounding sphere of the scene
    # load blender scene
    bpy.ops.wm.open_mainfile(filepath=FLAGS.scene_path)
    # points of all meshes
    points_co_global = []
    # all objects
    objs = bpy.context.scene.objects
    # get all mesh vertices
    for obj in objs:
        if obj.type == 'MESH':
            points_co_global.extend([obj.matrix_world @ vertex.co for vertex in obj.data.vertices])

    # get the bounding box first
    pts = np.array(points_co_global)
    aabb0, aabb1 = np.min(pts, axis=0), np.max(pts, axis=0)
    sphere_center = 0.5 * (aabb0 + aabb1)
    sphere_radius = 0.5 * np.linalg.norm(aabb1 - aabb0) * 1.01  # slightly larger
    scale_mat = np.diag([sphere_radius, sphere_radius, sphere_radius])
    scale_mat = np.concatenate([scale_mat, sphere_center[:, None]], axis=1)
    scale_mat = np.concatenate([scale_mat, np.zeros([1, 4])], axis=0)
    scale_mat[3, 3] = 1.
    scale_mat = scale_mat.astype(np.float32)

    logging.info(f'Scene bounding box: {aabb0}, {aabb1}, bounding sphere c={sphere_center}, r={sphere_radius}')

    # current image index
    image_idx = 0

    npz_file = {}

    # iteratively process all splits
    for split in glob.glob(pathname='transforms_*.json', root_dir=FLAGS.output_dir):
        # get split name
        split_name = split[len('transforms_'):-len('.json')]
        logging.info(f'Processing split {split_name}, starting from index {image_idx}')

        # reading camera information
        split_json = json.load(open(os.path.join(FLAGS.output_dir, split)))

        fovX = split_json['camera_angle_x']
        resx = FLAGS.resx
        resy = FLAGS.resy
        # K matrix (see IDR.md)
        focal = 0.5 * resx / np.tan(0.5 * fovX)
        cx = 0.5 * (resx - 1)
        cy = 0.5 * (resy - 1)
        K = np.array([[focal, 0, cx, 0],
                      [0, focal, cy, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        
        # flip the y and z axes to transform the camera coordinates
        camera_left_right = np.diag([1., -1., -1., 1.]).astype(np.float32)
        
        # process all frames
        pbar = tqdm(split_json['frames'], f'Processing split {split_name}')
        for frame in pbar:
            RtInv = np.array(frame['transform_matrix'], dtype=np.float32)
            Rt = np.linalg.inv(RtInv)
            world_mat = K @ camera_left_right @ Rt
            npz_file[f'world_mat_{image_idx}'] = world_mat
            npz_file[f'scale_mat_{image_idx}'] = scale_mat
            shutil.copy(os.path.join(FLAGS.output_dir, frame['file_path'], 'color.png'), os.path.join(result_dir, 'image', f'{image_idx}.png'))
            shutil.copy(os.path.join(FLAGS.output_dir, frame['file_path'], 'alpha.png'), os.path.join(result_dir, 'mask', f'{image_idx}.png'))
            image_idx += 1
    
    # save npz file
    np.savez(os.path.join(result_dir, 'camera_sphere.npz'), **npz_file)


if __name__ == '__main__':
    argv = sys.argv
    app.run(main=main, argv=argv)

