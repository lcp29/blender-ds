# Transform blender-ds dataset to the dataset format used in IDR [https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md]
# Detailed dataset format in IDR.md

import os, sys
import bpy
from absl import app, flags

flags.DEFINE_string('output_dir', '', 'root directory of the source blender-ds dataset')
flags.DEFINE_string('scene_path', '', 'path to the Blender scene')

FLAGS = flags.FLAGS

def main(_):
    pass

if __name__ == '__main__':
    argv = sys.argv
    app.run(main=main, argv=argv)

