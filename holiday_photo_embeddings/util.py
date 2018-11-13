from os import makedirs, walk
from os.path import exists, join, splitext, basename
from PIL import Image


def get_square_centre_crop(img):
    """
    This function calculates a centered square crop bounding box
    as big as the image's smaller dimension allows.

    :param img: PIL Image object.
    :return: Tuple with bounding box of the square crop.
    """
    width, height = img.size
    if width > height:
        x_min = int((width - height) / 2)
        x_max = int(width - (width - height) / 2)
        y_min = 0
        y_max = height
    else:
        x_min = 0
        x_max = width
        y_min = int((height - width) / 2)
        y_max = int(height - (height - width) / 2)

    return x_min, y_min, x_max, y_max


def preprocess_image(file_path, dst_path, new_width, new_height):
    """
    Crop an image to square aspect ratio, resize it and save it in specified directory.

    :param file_path: string, absolute path to the image file to process.
    :param dst_path: string, absolute path to which output image is saved to.
    :param new_width: int, the new width for the image
    :param new_height: int, the new height for the image
    """
    img = Image.open(file_path)
    cropped_img = img.crop(get_square_centre_crop(img))

    new_filename = splitext(basename(file_path))[0] + '_' + str(new_width) + '_' + str(new_height)
    new_img = Image.new('RGB', (new_width, new_height))
    new_img.paste(cropped_img.resize((new_width, new_height), resample=Image.LANCZOS))
    new_img.save(join(dst_path, new_filename) + '.jpg')


def generate_resized_images(img_path, dst_path, new_width, new_height):
    """
    Read images from image_path, generate cropped and resized images
    and save them to dst_path.

    :param img_path: string, absolute path to the image folder
    :param dst_path: string, absolute path to the destination folder
    :param new_width: int, the width for the new images
    :param new_height: int, the height for the new images
    """
    if not exists(dst_path):
        makedirs(dst_path)

    file_list = []
    for dirpath, dirnames, filenames in walk(img_path):
        file_list.extend(join(dirpath, f) for f in filenames)

    for i, f in enumerate(file_list):
        if f.lower().endswith('.jpg'):
            preprocess_image(f, dst_path, new_width, new_height)
            print('Processing image {step}/{total}'.format(step=i, total=len(file_list)))


def main():
    img_path = 'E:/slovenia'
    dst_path = 'E:/slovenia/all_days_resized'
    generate_resized_images(img_path, dst_path, 224, 224)


if __name__ == '__main__':
    main()


print('Hello')