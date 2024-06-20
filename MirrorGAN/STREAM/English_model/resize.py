import argparse
import os
import cv2

def resize_image(image, size):
    """Resize an image to the given size."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        img = cv2.imread(os.path.join(image_dir, image))
        if img is not None:
            img = resize_image(img, size)
            cv2.imwrite(os.path.join(output_dir, image), img)
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{num_images}] Resized the images and saved into '{output_dir}'.")

def main(args):
    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = (args.image_size, args.image_size)
    resize_images(image_dir, output_dir, image_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='../data/train2017/', help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='../data/resized2017/', help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256, help='size for image after processing')
    args = parser.parse_args()
    main(args)
