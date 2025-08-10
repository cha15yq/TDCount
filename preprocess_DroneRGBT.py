import glob
from PIL import Image
import argparse
import os
import shutil
import json
import numpy as np
import scipy.ndimage as ndimage
import scipy.spatial
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import xml.etree.ElementTree as ET
from scipy.ndimage.filters import gaussian_filter

def process_depth(depth_out):
    predicted_depth = depth_out[0]["predicted_depth"]
    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    depth = depth.detach().cpu().numpy() * 255
    depth = Image.fromarray(depth.astype("uint8"))
    return depth


def parse_args():
    parser = argparse.ArgumentParser(description='Data processing')
    parser.add_argument('--origin-dir', default='DroneRGBT',
                        help='orginal data directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    saved_data_dir = 'RGBT-Drone'

    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")

    if not os.path.exists(saved_data_dir):
        os.mkdir(saved_data_dir)

    for phase in ['Test', 'Train']:
        dir_main = os.path.join(saved_data_dir, phase.lower())
        dir_RGB = os.path.join(saved_data_dir, phase.lower(), 'RGB')
        dir_T = os.path.join(saved_data_dir, phase.lower(), 'T')
        dir_points = os.path.join(saved_data_dir, phase.lower(), 'points')
        dir_depth_rgb = os.path.join(saved_data_dir, phase.lower(), 'depth_rgb')
        dir_depth_t = os.path.join(saved_data_dir, phase.lower(), 'depth_t')
        if phase == 'Train':
            dir_den = os.path.join(saved_data_dir, phase.lower(), 'density_map')
        if not os.path.exists(dir_main):
            os.mkdir(dir_main)
            os.mkdir(dir_RGB)
            os.mkdir(dir_T)
            os.mkdir(dir_points)
            os.mkdir(dir_depth_rgb)
            os.mkdir(dir_depth_t)
            if phase == 'Train':
                os.mkdir(dir_den)

        anno_path_list = glob.glob(os.path.join(args.origin_dir, phase, 'GT_', '*.xml'))

        for anno_path in anno_path_list:
            rgb_path = anno_path.replace('GT_', 'RGB').replace('R.xml', '.jpg')
            t_path = anno_path.replace('GT_', 'Infrared').replace('xml', 'jpg')
            basename = os.path.basename(rgb_path)
            shutil.copy(rgb_path, os.path.join(dir_RGB, basename))
            shutil.copy(t_path, os.path.join(dir_T, basename))

            rgb = Image.open(rgb_path)
            t = Image.open(t_path)
            tree = ET.parse(anno_path)
            root = tree.getroot()

            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)
            gt = np.zeros((height, width))
            x_list = []
            y_list = []

            for person in root.findall('object'):
                x = int(person.find('point').find('x').text)
                y = int(person.find('point').find('y').text)
                x = min(x, width - 1)
                y = min(y, height - 1)
                x_list.append(int(x))
                y_list.append(int(y))
                gt[y, x] = 1
            pts = np.stack([x_list, y_list], axis=1)
            pt_path = os.path.join(dir_points, basename.replace('jpg', 'npy'))
            print('Name:', os.path.basename(anno_path), 'nop:', len(pts))
            np.save(pt_path, pts)
            if phase.lower() == 'train':
                den_map = gaussian_filter(gt, 4)
                den_path = os.path.join(dir_den, basename.replace('jpg', 'npy'))
                np.save(den_path, den_map)

            rgb_inputs = image_processor(images=rgb, return_tensors="pt")
            t_inputs = image_processor(images=t, return_tensors="pt")

            with torch.no_grad():
                outputs_rgb = model(**rgb_inputs)
                outputs_t = model(**t_inputs)

            post_processed_output_rgb = image_processor.post_process_depth_estimation(
                outputs_rgb,
                target_sizes=[(rgb.height, rgb.width)],
            )

            depth_rgb = process_depth(post_processed_output_rgb)
            depth_rgb.save(os.path.join(dir_depth_rgb, basename), quality=100)

            post_processed_output_t = image_processor.post_process_depth_estimation(
                outputs_t,
                target_sizes=[(t.height, t.width)],
            )

            depth_t = process_depth(post_processed_output_t)
            depth_t.save(os.path.join(dir_depth_t, basename), quality=100)




