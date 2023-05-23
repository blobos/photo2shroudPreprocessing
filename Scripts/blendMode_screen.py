import cv2
import random
import numpy as np
from blend_modes import screen

# input: texture image path, preprocessed face

#currently only texture applicable is
#"natural-linen-texture-background-crumpled-fabric-of-gray-color-rough-texture-photo.jpg"
def texture_resize(texture_image_path, face_image):
    texture = cv2.imread(texture_image_path)
    face = cv2.imread(face_image)



    # Resize texture #specific to texture
    # TODO: add height rescale to for different texture iamges
    # print(texture.shape)
    height_scale_factor = face.shape[0]/texture.shape[0]
    new_width = int(height_scale_factor * texture.shape[1])
    texture_resized = cv2.resize(texture, (new_width, face.shape[0]), cv2.INTER_CUBIC)
    # print(texture_resized.shape)

    # Resize texture

    print(texture.shape)
    height_scale_factor = face.shape[0] / texture.shape[0]
    new_width = int(height_scale_factor * texture.shape[1])
    texture_resized = cv2.resize(texture, (new_width, face.shape[0]), cv2.INTER_CUBIC)
    return texture_resized

def randomized_crop(img, crop_width):
  max_width = img.shape[0] - crop_width
  start_width = random.randint(0, max_width)
  cropped_img = img[:, start_width:start_width + crop_width]
  return cropped_img


def combine_overlay(texture_image_path, face):
    #TODO: add linear light to blend mode layer2 texture
    layer1_texture = randomized_crop(texture_resize(texture_image_path, face),512)
    face_RGBA = cv2.cvtColor(face, cv2.COLOR_GRAY2RGBA)
    texture1_RGBA = cv2.cvtColor(layer1_texture, cv2.COLOR_BGR2RGBA)
    textured_image = screen(face_RGBA.astype(float), texture1_RGBA.astype(float), .11)
    return textured_image

#linear light:
#https://www.youtube.com/watch?v=12sURzLXnhs
# if Blend <=0.5[Darker]
#     #Linear burn =
#     # Base + Blend - 1
#     # 1-((1-Base)+(1-Blend))
#     #
#     Result = Base + 2 * Blend -1
#
# else [Blend > 0.5, Linear Dodge]:
#     #Linear Dodge(add)
#     # Base + Blend
#     Result = Base + 2 * (Blend - 0.5)
