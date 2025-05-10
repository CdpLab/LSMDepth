from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import time
import torch
from torchvision import transforms, datasets
import cv2
import heapq
from PIL import ImageFile

# Avoid loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Import custom modules
import networks
from layers import disp_to_depth


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Simple testing function for Light models.')

    parser.add_argument('--image_path', type=str, help='path to a test image or folder of images', required=True)
    parser.add_argument('--load_weights_folder', type=str, help='path of a pretrained model to use')
    parser.add_argument('--test', action='store_true', help='if set, read images from a .txt file')
    parser.add_argument('--model', type=str, help='name of a pretrained model to use', default="light", choices=["light", "light-s", "light-t", "light-m"])
    parser.add_argument('--ext', type=str, help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda", help='if set, disables CUDA', action='store_true')

    return parser.parse_args()


def load_model(args, device):
    """
    Load the encoder and decoder model weights
    """
    assert args.load_weights_folder is not None, "You must specify the --load_weights_folder parameter"

    print("-> Loading model from ", args.load_weights_folder)
    
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)

    # Extract image size
    feed_height = encoder_dict['height']
    feed_width = encoder_dict['width']

    # Loading encoder
    print("   Loading pretrained encoder")
    encoder = networks.Light(model=args.model, height=feed_height, width=feed_width)
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    encoder.to(device)
    encoder.eval()

    # Loading decoder
    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    depth_model_dict = depth_decoder.state_dict()
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

    depth_decoder.to(device)
    depth_decoder.eval()

    return encoder, depth_decoder, feed_height, feed_width


def get_image_paths(args):
    """
    Get the paths of images to test, based on user input
    """
    if os.path.isfile(args.image_path) and not args.test:
        # Single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isfile(args.image_path) and args.test:
        # Reading images from a .txt file
        gt_path = os.path.join('splits', 'eigen', "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        paths = []

        with open(args.image_path) as f:
            filenames = f.readlines()
            for i in range(len(filenames)):
                filename = filenames[i]
                line = filename.split()
                folder = line[0]
                if len(line) == 3:
                    frame_index = int(line[1])
                    side = line[2]

                f_str = "{:010d}{}".format(frame_index, '.jpg')
                image_path = os.path.join(
                    'kitti_data',
                    folder,
                    "image_0{}/data".format(side_map[side]),
                    f_str)
                paths.append(image_path)

    elif os.path.isdir(args.image_path):
        # Folder with images
        paths = glob.glob(os.path.join(args.image_path, '*.jpg')) + glob.glob(os.path.join(args.image_path, '*.png'))
        output_directory = args.image_path
    else:
        raise Exception(f"Cannot find args.image_path: {args.image_path}")

    print(f"-> Predicting on {len(paths)} test images")
    
    return paths, output_directory


def predict_image(image_path, encoder, depth_decoder, device, feed_width, feed_height, output_directory):
    """
    Perform depth prediction on a single image and save the results
    """
    input_image = pil.open(image_path).convert('RGB')
    original_width, original_height = input_image.size
    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    # Prediction
    input_image = input_image.to(device)
    features = encoder(input_image)
    outputs = depth_decoder(features)

    # Post-process results
    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)
    scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

    output_name = os.path.splitext(os.path.basename(image_path))[0]
    name_dest_npy = os.path.join(output_directory, f"{output_name}_disp.npy")
    np.save(name_dest_npy, scaled_disp.cpu().numpy())

    # Save the colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)

    name_dest_im = os.path.join(output_directory, f"{output_name}_disp.jpeg")
    im.save(name_dest_im)

    print(f"   Processed and saved predictions for {output_name}:")

    return name_dest_im, name_dest_npy


def test_simple(args):
    """
    Main function to predict depth for each image
    """
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Load model
    encoder, depth_decoder, feed_height, feed_width = load_model(args, device)

    # Get image paths
    paths, output_directory = get_image_paths(args)

    total_time = 0  # Total inference time

    # Predicting on each image
    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            if image_path.endswith("_disp.jpg"):
                # Skip disparity images
                continue

            # Start prediction
            start_time = time.time()
            name_dest_im, name_dest_npy = predict_image(image_path, encoder, depth_decoder, device,
                                                       feed_width, feed_height, output_directory)
            end_time = time.time()

            # Calculate inference time for each image
            inference_time = end_time - start_time
            total_time += inference_time

            print(f"   Processed {idx + 1} of {len(paths)} images - saved predictions to:")
            print(f"   - {name_dest_im}")
            print(f"   - {name_dest_npy}")

        # Print average inference time
        average_time = total_time / len(paths)
        fps = len(paths) / total_time
        print(f"-> Average inference time per image: {average_time:.6f} seconds")
        print(f"-> FPS: {fps:.3f} frames per second")
        print(f"{len(paths)},{total_time}")

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
