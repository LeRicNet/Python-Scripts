from imageio import imread, imsave
from skimage.transform import resize
import argparse
import os


def fetch_flags():
    """worker for args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_dir',
        type="str",
        default="",
        required=True,
        help="Path to Image Directory"
    )
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


def cropimread(crop, xcrop, ycrop, fn):
    """Function to crop center of an image file"""
    img_pre= imread(fn)
    if crop:
        ysize, xsize, chan = img_pre.shape
        xoff = (xsize - xcrop) // 2
        yoff = (ysize - ycrop) // 2
        img= img_pre[yoff:-yoff,xoff:-xoff]
    else:
        img= img_pre
    return resize(img, (299,299))


def read_save_loop(oldf, newf, xcrop, ycrop):
    """crops and saves"""
    im = cropimread(crop=True,
                     xcrop=xcrop,
                     ycrop=ycrop,
                     fn=oldf)
    imsave(uri=newf, im=im)


def get_image_list(img_dir, ext):
    """finds images in img_dir and creates new file names"""
    imgs = [os.path.join(img_dir, img) for img in os.listdir(img_dir)
            if img.endswith('.jpg')]
    new_names = ['{}'.format(img.strip('.jpg') + ext) for img
                 in imgs]
    return imgs, new_names


def loop_manager(crop_sizes, img_dir):
    """loops whole process over each crop size"""
    for crop_size in crop_sizes:
        ext = "_crop{}.jpg".format(str(crop_size))
        oldf, newf = get_image_list(img_dir, ext=ext)
        read_save_loop(
            oldf=oldf,
            newf=newf,
            xcrop=crop_size,
            ycrop=crop_size)


def main():
    FLAGS, _ = fetch_flags()
    crop_sizes = [100, 150, 200]
    loop_manager(crop_sizes=crop_sizes,
                 img_dir=FLAGS.img_dir)


if __name__ == '__main__':
    main()