
import sys, os, shutil
import time
import numpy as np
import logging
import bpy
from absl import app, flags
import json
import math
from tqdm import tqdm

flags.DEFINE_string('scene_path', '', 'path to the Blender scene')
flags.DEFINE_string('cam_dir', '', 'directory containing camera JSON files')
flags.DEFINE_string('output_dir', '', 'root directory of render results')
flags.DEFINE_integer('resx', 800, 'width of output images')
flags.DEFINE_integer('resy', 800, 'height of output images')
flags.DEFINE_bool('film_transparent', False, 'if the background should be transparent, '
                                             'only influences direct background pixels')
# output control
flags.DEFINE_string('rgb_format', 'png', 'output format for rgb image(exr, png, nil)')
flags.DEFINE_bool('rgba', False, 'enable alpha channel for rgb output')
flags.DEFINE_string('alpha_format', 'nil', 'output format for alpha map(exr, png, nil)')
flags.DEFINE_string('depth_format', 'nil', 'output format for depth map(exr, png, nil)')
flags.DEFINE_float('depth_min', 2.0, 'minimum depth used in png format depth map')
flags.DEFINE_float('depth_max', 6.0, 'maximum depth used in png format depth map')
flags.DEFINE_string('normal_format', 'nil', 'output format for normal map(exr, png, nil)')


FLAGS = flags.FLAGS

def set_node_output_format(node, flag):
    assert flag in ['exr', 'png']
    if flag == 'exr':
        node.format.file_format = 'OPEN_EXR'
        node.format.color_depth = '32'
    elif flag == 'png':
        node.format.file_format = 'PNG'
        node.format.color_depth = '16'

def render_splits(splits, cam_intrinstics, cam_extrinstics, file_dirs, outnodes):
    for split in splits:
        logging.info(f'Rendering split "{split}"...')
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
        # set render resolution before setting camera params
        bpy.context.scene.render.resolution_x = FLAGS.resx
        bpy.context.scene.render.resolution_y = FLAGS.resy
        # set camera intrinstics
        for k, v in cam_intrinstics[split].items():
            setattr(camera.data, k, v)
        logging.info(f'Camera intrinstics set: {cam_intrinstics[split]}')

        # render samples
        pbar = tqdm(range(len(file_dirs[split])))

        for idx in pbar:
            pbar.set_description(file_dirs[split][idx])
            # set camera extrinstic
            # blender requires transposed c2w if using numpy matrix as input
            camera.matrix_world = cam_extrinstics[split][idx].T
            # set output file slot
            for outnode in outnodes:
                outnode['node'].base_path = os.path.abspath(FLAGS.output_dir)
                outnode['node'].file_slots[0].path = os.path.join(file_dirs[split][idx], f'{outnode["name"]}')
            bpy.ops.render.render(write_still=True)

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

    # set up Cycles as rendering engine
    bpy.context.scene.render.engine = 'CYCLES'
    if FLAGS.film_transparent:
        bpy.context.scene.render.film_transparent = True
    else:
        bpy.context.scene.render.film_transparent = False

    # use GPU
    cycle_preferences = bpy.context.preferences.addons['cycles'].preferences
    cycle_preferences.refresh_devices()
    device_types = cycle_preferences.get_device_types(bpy.context)
    device_names = [dt[0] for dt in device_types]
    # try to use OPTIX first
    if 'OPTIX' in device_names:
        cycle_preferences.compute_device_type = 'OPTIX'
    elif 'CUDA' in device_names:
        cycle_preferences.compute_device_type = 'CUDA'
    elif 'METAL' in device_names:
        cycle_preferences.compute_device_type = 'METAL'
    
    # enable all non-CPU devices
    logging.info(f'Compute device type: {cycle_preferences.compute_device_type}')
    for device in cycle_preferences.devices:
        if device.type != 'CPU':
            device.use = True
        
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.feature_set = 'SUPPORTED'
    logging.info('Cycles set up as rendering engine')

    # raw output
    bpy.context.scene.view_settings.view_transform = 'Raw'

    # set output composition
    bpy.context.scene.use_nodes = True
    node_tree = bpy.context.scene.node_tree
    # get scene render layer
    render_layers = None
    for node in node_tree.nodes:
        if node.type == 'R_LAYERS':
            render_layers = node
            break
    if render_layers is None:
        render_layers = node_tree.nodes.new("CompositorNodeRLayers")
    # get scene view layer
    view_layer = bpy.context.scene.view_layers[0]
    # file output nodes
    outnodes = []
    # set composition
    # RGB(A)
    if FLAGS.rgb_format != 'nil':
        outnode = node_tree.nodes.new("CompositorNodeOutputFile")
        outnode.label = 'RGB Output'
        outnode.name = 'RGB Output'

        set_node_output_format(outnode, FLAGS.rgb_format)
        outnode.format.color_mode = 'RGBA' if FLAGS.rgba else 'RGB'

        render_layers.outputs['Image'].enabled = True
        node_tree.links.new(render_layers.outputs['Image'], outnode.inputs[0])
        outnodes.append({'node': outnode, 'name': 'color', 'ext': FLAGS.rgb_format})
        logging.info(f'RGB{"A" if FLAGS.rgba else ""} {FLAGS.rgb_format} output enabled.')

    # render splits
    render_splits(splits, cam_intrinstics, cam_extrinstics, file_dirs, outnodes)

    # clear output nodes
    for outnode in outnodes:
        node_tree.nodes.remove(outnode['node'])
    outnodes = []

    # requires mask for other maps
    bpy.context.scene.render.film_transparent = True

    # foreground mask
    # alpha
    if FLAGS.alpha_format != 'nil':
        outnode = node_tree.nodes.new("CompositorNodeOutputFile")
        outnode.label = 'Alpha Output'
        outnode.name = 'Alpha Output'

        set_node_output_format(outnode, FLAGS.alpha_format)
        #outnode.format.color_mode = 'BW' if FLAGS.alpha_format == 'png' else 'RGBA'

        render_layers.outputs['Alpha'].enabled = True
        rgb_compositor = node_tree.nodes.new("CompositorNodeCombineColor")
        node_tree.links.new(render_layers.outputs['Alpha'], rgb_compositor.inputs[0])
        node_tree.links.new(render_layers.outputs['Alpha'], rgb_compositor.inputs[1])
        node_tree.links.new(render_layers.outputs['Alpha'], rgb_compositor.inputs[2])
        node_tree.links.new(render_layers.outputs['Alpha'], rgb_compositor.inputs[3])
        node_tree.links.new(rgb_compositor.outputs[0], outnode.inputs[0])
        outnodes.append({'node': outnode, 'name': 'alpha', 'ext': FLAGS.alpha_format})
        logging.info(f'Alpha {FLAGS.alpha_format} output enabled.')
    
    # depth
    if FLAGS.depth_format != 'nil':
        outnode = node_tree.nodes.new("CompositorNodeOutputFile")
        outnode.label = 'Depth Output'
        outnode.name = 'Depth Output'

        set_node_output_format(outnode, FLAGS.depth_format)

        view_layer.use_pass_z = True
        render_layers.outputs['Depth'].enabled = True
        if FLAGS.depth_format == 'exr':
            # directly connect depth output with file node input with alpha channel
            rgb_compositor = node_tree.nodes.new("CompositorNodeCombineColor")
            node_tree.links.new(render_layers.outputs['Depth'], rgb_compositor.inputs[0])
            node_tree.links.new(render_layers.outputs['Depth'], rgb_compositor.inputs[1])
            node_tree.links.new(render_layers.outputs['Depth'], rgb_compositor.inputs[2])
            node_tree.links.new(render_layers.outputs['Alpha'], rgb_compositor.inputs[3])
            node_tree.links.new(rgb_compositor.outputs[0], outnode.inputs[0])
        elif FLAGS.depth_format == 'png':
            # requires some compression
            depth_map = node_tree.nodes.new(type="CompositorNodeMapRange")
            depth_map.inputs['From Min'].default_value = FLAGS.depth_min
            depth_map.inputs['From Max'].default_value = FLAGS.depth_max
            depth_map.inputs['To Min'].default_value = 0
            depth_map.inputs['To Max'].default_value = 1
            node_tree.links.new(render_layers.outputs['Depth'], depth_map.inputs[0])
            rgb_compositor = node_tree.nodes.new("CompositorNodeCombineColor")
            node_tree.links.new(depth_map.outputs[0], rgb_compositor.inputs[0])
            node_tree.links.new(depth_map.outputs[0], rgb_compositor.inputs[1])
            node_tree.links.new(depth_map.outputs[0], rgb_compositor.inputs[2])
            node_tree.links.new(render_layers.outputs['Alpha'], rgb_compositor.inputs[3])
            node_tree.links.new(rgb_compositor.outputs[0], outnode.inputs[0])

        outnodes.append({'node': outnode, 'name': 'depth', 'ext': FLAGS.depth_format})
        logging.info(f'Depth {FLAGS.depth_format} output enabled.')
    
    # normal
    if FLAGS.normal_format != 'nil':
        outnode = node_tree.nodes.new("CompositorNodeOutputFile")
        outnode.label = 'Normal Output'
        outnode.name = 'Normal Output'

        set_node_output_format(outnode, FLAGS.normal_format)
        outnode.format.color_mode = 'RGBA'

        view_layer.use_pass_normal = True
        render_layers.outputs['Normal'].enabled = True
        if FLAGS.normal_format == 'exr':
            node_tree.links.new(render_layers.outputs['Normal'], outnode.inputs[0])
        else:
            # transform from [-1, 1] to [0, 1] on all channels for png
            rgb_separator = node_tree.nodes.new("CompositorNodeSeparateColor")
            muladds = [node_tree.nodes.new("CompositorNodeMath") for _ in range(3)]
            rgb_combiner = node_tree.nodes.new("CompositorNodeCombineColor")
            # connect normal output to rgb_seperator
            node_tree.links.new(render_layers.outputs['Normal'], rgb_separator.inputs[0])
            # connect muladd units
            for idx, m in enumerate(muladds):
                m.operation = 'MULTIPLY_ADD'    # operation
                m.inputs[1].default_value = 0.5 # scale factor
                m.inputs[2].default_value = 0.5 # bias
                node_tree.links.new(rgb_separator.outputs[idx], m.inputs[0])
                node_tree.links.new(m.outputs[0], rgb_combiner.inputs[idx])
            # add alpha channel to normal map
            node_tree.links.new(render_layers.outputs['Alpha'],rgb_combiner.inputs[3])
            # connect to the final output
            node_tree.links.new(rgb_combiner.outputs[0], outnode.inputs[0])

        outnodes.append({'node': outnode, 'name': 'normal', 'ext': FLAGS.normal_format})
        logging.info(f'Normal {FLAGS.alpha_format} output enabled.')

    # render splits
    render_splits(splits, cam_intrinstics, cam_extrinstics, file_dirs, outnodes)

    # remove all frame indices
    for root, _, files in os.walk(FLAGS.output_dir):
        for file in files:
            if (file.endswith('.png') or file.endswith('.exr')) and ('0001' in file):
                new_file = file.replace('0001', '')
                os.rename(os.path.join(root, file), os.path.join(root, new_file))
    logging.info('Frame index removed.')


if __name__ == '__main__':
    argv = sys.argv
    app.run(main=main, argv=argv)
