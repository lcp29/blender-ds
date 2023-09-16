
import sys, os, shutil
import numpy as np
import logging
import bpy
from absl import app, flags
import json
import math
import tqdm

flags.DEFINE_string('scene_path', '', 'path to the Blender scene')
flags.DEFINE_string('cam_dir', '', 'directory containing camera JSON files')
flags.DEFINE_string('output_dir', '', 'root directory of render results')
flags.DEFINE_integer('resx', 800, 'width of output images')
flags.DEFINE_integer('resy', 800, 'height of output images')

FLAGS = flags.FLAGS


def main(_):
    # delete existing output directory
    if os.path.exists(FLAGS.output_dir):
        logging.warning(f'Output directory "{FLAGS.output_dir}" already exists, overwriting.')
        shutil.rmtree(FLAGS.output_dir)
    os.makedirs(FLAGS.output_dir)
    
    # load camera JSONs
    splits = []
    split_json_objs = {}
    for split_json in os.listdir(FLAGS.cam_dir):
        # the JSON file name starts with "transforms_" and has extension .json
        if split_json.startswith('transforms_') and split_json.endswith('.json'):
            # only get the split name part
            split = split_json[len('transforms_'):-len('.json')]
            splits.append(split)
            # read json
            json_path = os.path.join(FLAGS.cam_dir, split_json)
            split_json_objs[split] = json.load(open(json_path, 'r'))
            # copy camera JSONs to the output directory
            shutil.copy(os.path.join(FLAGS.cam_dir, split_json), FLAGS.output_dir)

    logging.info(f'Found split{"s" if len(splits) > 1 else ""} in {FLAGS.cam_dir}: {splits}')

    # load JSON data
    # camera intrinstics for each split
    cam_intrinstics = {}
    # filename and camera to world matrices
    # =================================================================
    # Important: file_path is interpreted as output directory, which 
    #            is different with the original NeRF synthetic dataset.
    #            The output dir tree will be like:
    #            output---train---r_0---color.png
    #                   |       |     |-normal.png
    #                   |       |     |-metadata.json
    #                   |       |     \-...
    #                   |       |-r_1--...
    #                  ...     ...
    # =================================================================
    
    # file directories of each sample in each split
    file_dirs = {}
    # camera extrinstics for each samples in each split
    cam_extrinstics = {}

    for split in splits:
        logging.info(f'Loading split "{split}"...')
        # load camera intrinstics
        cam_fovx = split_json_objs[split]['camera_angle_x']
        cam_intrinstic = {'sensor_width': FLAGS.resx,                       # width
                          'sensor_height': FLAGS.resy,                      # height
                          'sensor_fit': 'AUTO',                             # width or height to which to fit the sensor size 
                          'lens': 0.5 * FLAGS.resx / math.tan(cam_fovx),    # focal
                          'clip_start': 0.1,                                # length in meters
                          'clip_end': 100,
                          'type': 'PERSP'}                                  # perspective camera
        cam_intrinstics[split] = cam_intrinstic
        logging.info(f'Camera intrinstic: {cam_intrinstic}')

        # load output filename and c2w matrices
        file_paths = []
        c2ws = []
        for frame in split_json_objs[split]['frames']:
            file_paths.append(frame['file_path'])
            c2ws.append(np.array(frame['transform_matrix']))
        file_dirs[split] = file_paths
        cam_extrinstics[split] = c2ws
        logging.info(f'{len(file_paths)} sample{"s" if len(file_paths) > 1 else ""}.')

    # load blender scene
    bpy.ops.wm.open_mainfile(filepath=FLAGS.scene_path)

    # render splits
    for split in splits:
        logging.info(f'Rendering split {split}...')
        # find camera in the scene
        camera = None
        for obj in bpy.data.objects:
            if obj.type == 'CAMERA':
                camera = obj
                break
        # if no camera in the scene, create one
        if camera is None:
            bpy.ops.object.camera_add()
            camera = bpy.context.active_object
        # set camera intrinstics
        for k, v in cam_intrinstics[split].items():
            print(k, v)
        logging.info(f'Camera intrinstics set: {cam_intrinstics[split]}')




if __name__ == '__main__':
    argv = sys.argv
    app.run(main=main, argv=argv)
