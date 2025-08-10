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
from scipy.ndimage.filters import gaussian_filter


def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = min((distances[i][1]+distances[i][2]+distances[i][3])*0.1, 15)
        else:
            sigma = np.average(np.array(gt.shape))/2./2.  # case: 1 point
        density += ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('density map count: ', density.sum())
    return density


def process_depth(depth_out):
    predicted_depth = depth_out[0]["predicted_depth"]
    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    depth = depth.detach().cpu().numpy() * 255
    depth = Image.fromarray(depth.astype("uint8"))
    return depth


def parse_args():
    parser = argparse.ArgumentParser(description='Data processing')
    parser.add_argument('--origin-dir', default='RGBT-CC-CVPR2021',
                        help='orginal data directory')
    args = parser.parse_args()
    return args


rotate_clockwise = ['RGBT-CC-CVPR2021/val/1297_GT.json',
                    'RGBT-CC-CVPR2021/val/1902_GT.json',
                    'RGBT-CC-CVPR2021/val/1893_GT.json',
                    'RGBT-CC-CVPR2021/val/2059_GT.json',
                    'RGBT-CC-CVPR2021/test/1213_GT.json',
                    'RGBT-CC-CVPR2021/train/2002_GT.json',
                    'RGBT-CC-CVPR2021/train/1455_GT.json',
                    'RGBT-CC-CVPR2021/train/2024_GT.json',
                    'RGBT-CC-CVPR2021/train/2025_GT.json',
                    'RGBT-CC-CVPR2021/train/2003_GT.json',
                    'RGBT-CC-CVPR2021/train/1436_GT.json',
                    'RGBT-CC-CVPR2021/train/2001_GT.json',
                    'RGBT-CC-CVPR2021/train/1447_GT.json',
                    'RGBT-CC-CVPR2021/train/1439_GT.json',
                    'RGBT-CC-CVPR2021/train/2023_GT.json',
                    'RGBT-CC-CVPR2021/train/2033_GT.json',
                    'RGBT-CC-CVPR2021/train/1453_GT.json',
                    'RGBT-CC-CVPR2021/train/1892_GT.json',
                    'RGBT-CC-CVPR2021/train/2032_GT.json',
                    'RGBT-CC-CVPR2021/train/2022_GT.json',
                    'RGBT-CC-CVPR2021/train/1899_GT.json',
                    'RGBT-CC-CVPR2021/train/2029_GT.json',
                    'RGBT-CC-CVPR2021/train/2030_GT.json',
                    'RGBT-CC-CVPR2021/train/1891_GT.json',
                    'RGBT-CC-CVPR2021/train/2031_GT.json',
                    'RGBT-CC-CVPR2021/train/2021_GT.json',
                    'RGBT-CC-CVPR2021/train/1898_GT.json',
                    'RGBT-CC-CVPR2021/train/2038_GT.json']
rotate_counterclockwise = ['RGBT-CC-CVPR2021/test/2170_GT.json',
                           'RGBT-CC-CVPR2021/test/2171_GT.json',
                           'RGBT-CC-CVPR2021/test/2172_GT.json',
                           'RGBT-CC-CVPR2021/train/1997_GT.json',
                           'RGBT-CC-CVPR2021/train/2046_GT.json',
                           'RGBT-CC-CVPR2021/train/2047_GT.json',
                           'RGBT-CC-CVPR2021/train/1986_GT.json',
                           'RGBT-CC-CVPR2021/train/1996_GT.json',
                           'RGBT-CC-CVPR2021/train/1984_GT.json',
                           'RGBT-CC-CVPR2021/train/1985_GT.json',
                           'RGBT-CC-CVPR2021/train/1995_GT.json',
                           'RGBT-CC-CVPR2021/train/1990_GT.json',
                           'RGBT-CC-CVPR2021/train/2066_GT.json',
                           'RGBT-CC-CVPR2021/train/1981_GT.json',
                           'RGBT-CC-CVPR2021/train/1991_GT.json',
                           'RGBT-CC-CVPR2021/train/1998_GT.json',
                           'RGBT-CC-CVPR2021/train/1983_GT.json',
                           'RGBT-CC-CVPR2021/train/1451_GT.json',
                           'RGBT-CC-CVPR2021/train/1982_GT.json',
                           'RGBT-CC-CVPR2021/train/2065_GT.json']


if __name__ == '__main__':
    args = parse_args()
    saved_data_dir = 'RGBT-CC'

    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")

    if not os.path.exists(saved_data_dir):
        os.mkdir(saved_data_dir)

    for phase in ['train', 'val', 'test']:
        dir_main = os.path.join(saved_data_dir, phase)
        dir_RGB = os.path.join(saved_data_dir, phase, 'RGB')
        dir_T = os.path.join(saved_data_dir, phase, 'T')
        dir_points = os.path.join(saved_data_dir, phase, 'points')
        dir_depth_rgb = os.path.join(saved_data_dir, phase, 'depth_rgb')
        dir_depth_t = os.path.join(saved_data_dir, phase, 'depth_t')
        if phase == 'train':
            dir_den = os.path.join(saved_data_dir, phase, 'density_map')
        if not os.path.exists(dir_main):
            os.mkdir(dir_main)
            os.mkdir(dir_RGB)
            os.mkdir(dir_T)
            os.mkdir(dir_points)
            os.mkdir(dir_depth_rgb)
            os.mkdir(dir_depth_t)
            if phase == 'train':
                os.mkdir(dir_den)

        anno_path_list = glob.glob(os.path.join(args.origin_dir, phase, '*.json'))
        for anno_path in anno_path_list:
            rgb_path = anno_path.replace('GT.json', 'RGB.jpg')
            t_path = anno_path.replace('GT.json', 'T.jpg')
            basename = os.path.basename(rgb_path).replace('_RGB.jpg', '')
            shutil.copy(rgb_path, os.path.join(dir_RGB, basename+'.jpg'))
            shutil.copy(t_path, os.path.join(dir_T, basename+'.jpg'))

            with open(anno_path, "r", encoding="utf-8") as file:
                pts = json.load(file)['points']
            pt_path = os.path.join(dir_points, basename+'.npy')
            np.save(pt_path, np.array(pts))
            print('Name:', os.path.basename(anno_path), 'nop:', len(pts))
            rgb = Image.open(rgb_path)
            t = Image.open(t_path)
            if anno_path in rotate_clockwise:
                rgb = rgb.rotate(270, expand=True)
                rgb.save(os.path.join(dir_RGB, basename+'.jpg'), quality=100)
            elif anno_path in rotate_counterclockwise:
                rgb = rgb.rotate(90, expand=True)
                rgb.save(os.path.join(dir_RGB, basename+'.jpg'), quality=100)
            if phase == 'train':
                gt = np.zeros((rgb.size[1], rgb.size[0]), dtype=np.float32)

                for pt in pts:
                    gt[min(int(pt[1]), rgb.size[1] - 1), min(int(pt[0]), rgb.size[0] - 1)] = 1
                den_map = gaussian_filter(gt, 4) #if adaptive: gaussian_filter_density(gt)
                den_path = os.path.join(dir_den, basename+'.npy')
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
            depth_rgb.save(os.path.join(dir_depth_rgb, basename+'.jpg'), quality=100)

            post_processed_output_t = image_processor.post_process_depth_estimation(
                outputs_t,
                target_sizes=[(t.height, t.width)],
            )

            depth_t = process_depth(post_processed_output_t)
            depth_t.save(os.path.join(dir_depth_t, basename+'.jpg'), quality=100)




