import data_tools
import os

network_path = os.path.abspath('.') + '/dsb2018_topcoders'
full_image_name = os.path.abspath('.') + '/test_img.jpg'
sample_path = os.path.abspath('.') + '/sample_test'

data_tools.perform_segmentation(full_image_name, sample_path, network_path, force=True)
