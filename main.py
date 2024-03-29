import os
import cv2 as cv
from pathlib import Path

from Scripts.BG_removal import bg_remove
from Scripts.eyelid_replacement import skin_tone_eye
from Scripts.image_adustments import preprocessing
from Scripts.hairmask import hair_brighten
from Scripts.blendMode_screen import combine_overlay


def main(photo_directory, mask_directory, output_directory):
    photo_directory = Path(photo_directory)
    mask_directory = Path(mask_directory)
    output_directory = output_directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


    print('running main()')
    for i in range(0, 30000):
        file_name = '**/' + str(i).zfill(5) + '_*'

        #list of files in mask
        mask_list = list(mask_directory.glob(file_name))

        full_image_name = '**/' + str(i) + '.jpg'
        original_image = cv.imread(str(Path(list(photo_directory.glob(full_image_name))[0])))
        original_image = cv.resize(original_image, (512, 512), interpolation=cv.INTER_AREA)

        # remove background
        im = bg_remove(mask_list, original_image)

        # eyes
        eyes_file_name = '**/' + str(i).zfill(5) + '*eye.png'
        eyelist = list(mask_directory.glob(eyes_file_name))
        # print(eyelist)

        if len(eyelist) >= 2:
            im = skin_tone_eye(eyelist, im)

        # hair
        hair_file_name = '**/' + str(i).zfill(5) + '*hair.png'
        hairlist = list(mask_directory.glob(hair_file_name))
        im = hair_brighten(hairlist, im)

        #image adjustments
        im = preprocessing(im)

        #overlay
        im=combine_overlay(texture_folder_path="../linen_textures/", base_image=im)

#
        outfile = output_directory + os.path.basename(str(i)) + '.jpg'
        print(outfile)
        status = cv.imwrite(outfile, im)

#TODO: make 5 iterations per image
# randomize hair brightening as effect not idea
# Randomize linen texture overlay i.e. which sections on the texture and different textures


if __name__ == '__main__':
    PHOTO_DIRECTORY = '/home/student/Desktop/shroud_models/Datasets/CelebAMask-HQ/CelebA-HQ-img/'
    MASK_DIRECTORY = '/home/student/Desktop/shroud_models/Datasets/CelebAMask-HQ/CelebAMask-HQ-mask-anno/'
    OUTPUT_SHROUD_DIRECTORY = '/home/student/Desktop/shroud_models/Datasets/CelebAMask-HQ/testing/'

    main(PHOTO_DIRECTORY, MASK_DIRECTORY, OUTPUT_SHROUD_DIRECTORY)
