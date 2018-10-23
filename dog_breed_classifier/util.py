import glob
import json
import numpy as np
import os
import xml.etree.ElementTree as ET
from PIL import Image


ANNOTATION_PATH = 'C:/Users/JC/Documents/studia/erasmus/dl/lab_1/autonomous/stanford_dogs/annotation/Annotation/'
IMAGE_PATH = 'C:/Users/JC/Documents/studia/erasmus/dl/lab_1/autonomous/stanford_dogs/images/Images/'
BOX_FOLDER = 'boxes_'


def get_annotation_dict(annotation_path):
    """
    get_annotation_dict reads an annotation file, parses the XML
    and returns the labeled dog breed and image file dimensions.

    input:
        annotation_path: string, file path of the annotation file

    output:
        result_dict: dictionary, with filename, breed, width, height,
        and bounding box
    """

    tree = ET.parse(annotation_path)
    root = tree.getroot()

    result_dict = dict()

    result_dict['filename'] = root[1].text

    if root[1].text == "%s":
        result_dict['filename'] = annotation_path.split('\\')[-1]

    result_dict['breed'] = root[5][0].text

    result_dict['width'] = int(root[3][0].text)
    result_dict['height'] = int(root[3][1].text)

    result_dict['xmin'] = int(root[5][4][0].text)
    result_dict['ymin'] = int(root[5][4][1].text)
    result_dict['xmax'] = int(root[5][4][2].text)
    result_dict['ymax'] = int(root[5][4][3].text)

    return result_dict


def get_image_folder_path_name(annotation_dict):
    """
    get_image_folder_path_name returns the folder name which contains the
    image file for the annotation_dict parameter.

    'n02085620_7' => 'n02085620-Chihuahua'

    input:
        annotation_dict: dictionary, contains filename of annotation.

    output:
        string, returns folder name which contains image file
        for this annotation
    """
    filename = annotation_dict['filename']
    folder_name = filename.split('_')[0] + '-' + annotation_dict['breed']
    return folder_name


def get_box_folder_path_name(annotation_dict, new_width, new_height):
    """
    get_box_folder_path_name returns the folder name which contains the
    resized image files for an original image file.

    Given image 'n02085620_7', you can find the resized images at:
        'F:/dogs/images/n02085620-Chihuahua/boxes_64_64/'

    input:
        annotation_dict: dictionary, contains filename

        new_width: int, the new width for the image

        new_height: int, the new height for the image

    output:
        returns a string, the folder path for the resized images
    """
    folder_name = get_image_folder_path_name(annotation_dict)
    box_folder = BOX_FOLDER + str(new_width) + '_' + str(new_height)
    return IMAGE_PATH + folder_name + '/' + box_folder + '/'


def get_image_file_path_name(annotation_dict, new_width, new_height):
    """
    get_image_file_path_name returns the file path and file name for the
    new cropped and resized image.

    Given image 'n02085620_7' resized to 64 x 64,
    it will return:
        'F:/dogs/images/n02085620-Chihuahua/boxes_64_64/n02085620_7_box_64_64.jpg'

    input:
        annotation_dict: dictionary, contains the filename

        new_width: int, the new width for the image

        new_height: int, the new height for the image

    output:
        returns a string, which is the folder path and the file name

    """
    filename = annotation_dict['filename']

    # create the file suffix i.e. '_box_64x64.jpg'
    box_file_ending = '_box_' + str(new_width) + '_' + str(new_height) + '.jpg'

    return get_box_folder_path_name(annotation_dict, new_width, new_height) + filename + box_file_ending


def crop_save_bounded_box(annotation_dict, new_width, new_height, background_color=None, no_background=False):
    """
    crop_save_bounded_box crops the image to the pixels designated
    by its bounded box, resizes the image to the provided
    dimensions (new_width, new_height), and saves the new cropped
    and resized image to disk.  It maintains the aspect ratio
    of the original image by placing the image on a background color,
    if a background color is given.  If background color is None,
    the image is resized to new_width x new_height, scaling without
    maintaining aspect ratio.

    input:
        annotation_dict: dictionary, returned from get_annotation_dict(),
                        contains filename, breed, and other info

        new_width: int, new width in px for resized bounded box

        new_height: int, new height in px for resized bounded box

        background_color: (int, int, int), color for background

        no_background: bool, if true, crops the image to its bounding
                      box and resizes it to fit inside a box of new_width
                      and new_height, but does not place the image on a
                      background color.  the bounding box maintains
                      its aspect ratio, and the saved image to disk
                      is of the dimensions of the bounding box.

    output:
        returns nothing, will throw exception if crop or save fails
    """

    # open the original image (the one which is not resized)
    # for instance, image file n02085620_7 is located at 'F:/dogs/images/n02085620-Chihuahua/n02085620_7.jpg'
    filename = annotation_dict['filename']
    folder_name = get_image_folder_path_name(annotation_dict)
    temp_image = Image.open(IMAGE_PATH + folder_name + '/' + filename + '.jpg')

    # crop the image to the region defined by the bounding box
    cropped_image = temp_image.crop((annotation_dict['xmin'],
                                     annotation_dict['ymin'],
                                     annotation_dict['xmax'],
                                     annotation_dict['ymax']))

    # if a background color is provided, resize the image and maintain aspect ratio
    # otherwise, don't maintain aspect ratio
    if background_color is not None or no_background:

        # keep the aspect ratio of the bounding box
        # if the width is bigger than the height
        #   box_height = (box_height / box_width) * new_width
        # if the height is bigger than the width
        #   box_width = (box_width / box_height) * new_height

        box_width = annotation_dict['xmax'] - annotation_dict['xmin']
        box_height = annotation_dict['ymax'] - annotation_dict['ymin']

        if box_width > box_height:
            box_height = int((box_height * new_width) / box_width)
            box_width = new_width
        else:
            box_width = int((box_width * new_height) / box_height)
            box_height = new_height

        # create an empty background size of the bounding box if no_background is true,
        # that way we won't see a background color
        if no_background:
            background = Image.new('RGB', (box_width, box_height), background_color)
        else:
            # create an empty background size of the new image size
            background = Image.new('RGB', (new_width, new_height), background_color)

        # resize the bounding box while keeping the aspect ratio
        resized_image = cropped_image.resize((box_width, box_height), resample=Image.LANCZOS)

        # paste the bounding box with original aspect ratio onto black background
        # if there is no_background, paste the resize image exactly on the background at
        # (0,0), otherwise, center the bounding box in the background
        if no_background:
            background.paste(resized_image)
        else:
            background.paste(resized_image,
                             (int((new_width - box_width) / 2), int((new_height - box_height) / 2)))

        # save the bounding box on black background to disk
        background.save(get_image_file_path_name(annotation_dict, new_width, new_height))

    else:
        # resize the bounding box but do not maintain the aspect ratio
        # the image may be stretched
        resized_image = cropped_image.resize((new_width, new_height), resample=Image.LANCZOS)

        # save the resized image to disk
        new_image = Image.new('RGB', (new_width, new_height))
        new_image.paste(resized_image)
        new_image.save(get_image_file_path_name(annotation_dict, new_width, new_height), 'jpeg')


def generate_all_resized_images(new_width, new_height, background_color=None, no_background=False):
    """
    generate_all_resized_images reads each annotation file in
    /annotation/* and then creates a resized image based on the
    bounding box for that annotation and saves it to the folder
    /images/breed/boxes_new_width_new_height

    input:
        new_width: int, the width for the output image

        new_height: int, the height for the output image

        background_color: (int, int, int), color for background,
                         not providing a background color will resize the image
                         and not maintain the original aspect ratio of the bounding
                         box

        no_background: bool, crops the bounding box of the image and
                      saves the bounding box to disk with no background color,
                      the saved image will have the dimensions of the bounding box
                      that fits within an image of new_width x new_height

    output:
        returns nothing, but saves images to the corresponding
        breed folders in /images
    """
    count = 0

    # recursively iterate through all the annotation files
    for fname in glob.iglob(ANNOTATION_PATH + '**/*', recursive=True):
        if os.path.isfile(fname):
            annotation_dict = get_annotation_dict(fname)
            box_folder_path = get_box_folder_path_name(annotation_dict, new_width, new_height)

            # create the 'boxes_new_width_new_height' folder if it does not already exist
            if not os.path.exists(box_folder_path):
                os.makedirs(box_folder_path)

            # only write a new image if we haven't come across it yet
            if True or not os.path.exists(get_image_file_path_name(annotation_dict, new_width, new_height)):
                # crop and save the new image file
                crop_save_bounded_box(annotation_dict, new_width, new_height, background_color,
                                   no_background)

            count += 1
            if count % 100 == 0:
                print('Progress: ' + str(count / float(20580) * 100) + ' %')
                print('Just processed ' + get_image_file_path_name(annotation_dict, new_width, new_height))

    print('Images Resized:', count)


def get_resized_image_data(annotation_dict, width, height):
    """
    get_resized_image_data returns a numpy array of the rgb values for the resized image.
    The RGB int values are converted to a float16 (0-255 => 0.0-1.0).
    The shape of the numpy array is (height, width, 3).

    input:
        annotation_dict: dictionary, contains the file name for the original image,
                        used to get the filepath for the resized image

        width: int, the width of the resized image to retrieve

        height: int, the height of the resized image to retrieve

    output:
        returns a numpy array of shape (height, width, 3) representing
        the resized image data
    """
    file_path = get_image_file_path_name(annotation_dict, width, height)
    image = Image.open(file_path)
    image_array = np.array(image)
    # converting to float64 here increases the size per image by a factor of 8 compared
    # to the 1 byte used by 0 - 255 rgb int
    # the RAM usage is insane
    # image_array = image_array / 255.0

    # converting to float 16 cuts the size by 4
    image_array = np.array(image_array / 255.0, dtype=np.float16)
    image.close()
    return image_array


def generate_training_test_lists(training_ratio=0.8):
    """
    generate_training_test_lists outputs three .json files containing
    the training and test splits based on the training ratio,
    along with the one hot encodings for each class (breed). The .json
    files for the training and test splits contain dictionaries,
    where the key is the breed, and the value for each key is a list
    of annotation dictionaries where each annotation dicionary's
    breed is the key.  The .json file for the one hot encodings
    is a dictionary, where the key is a breed, and the value
    is a one hot encoding (a list).

    input:
        training_ratio: float, default is 80% split

    output:
        writes four .json files to disk:
            train_annotations.json,
            test_annotations.json,
            one_hot_encodings.json
    """

    file_list = []
    for filename in glob.iglob(ANNOTATION_PATH + '**/*', recursive=True):
        if os.path.isfile(filename):
            file_list.append(get_annotation_dict(filename))

    # sort the annotations based on breed into a dictionary
    # i.e.: breed_annotations = {'husky': [list of husky annotations]}
    breed_annotations = {}
    for annotation in file_list:
        breed = annotation['breed']
        if breed not in breed_annotations:
            breed_annotations[breed] = [annotation]
        else:
            breed_annotations[breed].append(annotation)

    # dictionaries for training splits, testing splits, and one hot encodings
    train_annotations = {}
    test_annotations = {}
    one_hot_encodings = {}
    nb_breeds = len(breed_annotations.keys())

    for i, breed in enumerate(breed_annotations.keys()):
        # calculate the number of training examples to include from this breed,
        # as each breed has a different number of total annotations
        breed_annotations_len = len(breed_annotations[breed])
        training_size = int(breed_annotations_len * training_ratio)
        train_annotations[breed] = breed_annotations[breed][:training_size]
        test_annotations[breed] = breed_annotations[breed][training_size:]

        # create the one hot encoding for this breed
        # the one hot encoding is a list of 0s, the size being the number of breeds
        # a 1 is inserted at the index of the ith breed for the encodings
        one_hot_encodings[breed] = ([0] * nb_breeds)
        one_hot_encodings[breed][i] = 1

    # write the three dictionaries to file
    with open('train_annotations.json', 'w') as fout:
        json.dump(train_annotations, fout)

    with open('test_annotations.json', 'w') as fout:
        json.dump(test_annotations, fout)

    with open('one_hot_encodings.json', 'w') as fout:
        json.dump(one_hot_encodings, fout)


def one_hot_encoding_to_class(one_hot_encodings):
    """
    Generates a dictionary where the key is the index of the one hot encoding,
    and the value is the class name.

    input:
        one_hot_encoding: dict, a dictionary where the key is the class name,
        and the value is the one hot encoding

    output:
        returns a dictionary where the key is the index of the one hot encoding,
        and the value is the class name
    """
    result = {}
    for class_name, one_hot_encoding in one_hot_encodings.items():
        result[one_hot_encoding.index(1)] = class_name

    return result


def main():
    generate_all_resized_images(64, 64)
    generate_training_test_lists()


if __name__ == "__main__":
    main()