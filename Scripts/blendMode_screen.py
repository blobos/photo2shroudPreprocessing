import cv2
import random
import numpy as np
import glob
from blend_modes import screen

# input: texture image path, preprocessed face

#currently only texture applicable is
#"natural-linen-texture-background-crumpled-fabric-of-gray-color-rough-texture-photo.jpg","natural-linen.jpg"
def texture_resize(texture_folder_path, face_image):
    #make list from textures in folder
    texture_image_list = glob.glob(texture_folder_path + '*.jpg')
    texture_image = random.choice(texture_image_list)
    texture = cv2.imread(texture_image)

    #randomly flip vertically (shroud artifacts mostly in vertical lines)
    flip = random.choice([0,1])
    if flip ==1:
        texture = cv2.flip(texture,0)

    # Resize texture #specific to texture
    # TODO: add height rescale to for different texture images
    # print(texture.shape)
    face = cv2.imread(face_image)
    height_scale_factor = face.shape[0]/texture.shape[0]
    new_width = int(height_scale_factor * texture.shape[1])
    texture_resized = cv2.resize(texture, (new_width, face.shape[0]), cv2.INTER_CUBIC)
    # print(texture_resized.shape)
    return texture_resized

def randomized_crop(img, crop_width):
#fixed crop in random area of resized image
  max_width = img.shape[0] - crop_width
  start_width = random.randint(0, max_width)
  cropped_img = img[:, start_width:start_width + crop_width]
  return cropped_img

def linear_light_blend_mode(base_image, texture_overlay_image):
    #for second texture overlay

    # Convert images to float32 for calculations
  base_image = base_image.astype(np.float32)
  texture_overlay_image = texture_overlay_image.astype(np.float32)

  # Normalize images to range [0, 1]
  base_image /= 255.0
  texture_overlay_image /= 255.0

  # Calculate the opacity as a fraction (7% opacity = 0.11)
  opacity = 0.07

  # Apply the linear light blend mode with opacity
  output = base_image + 2 * texture_overlay_image - 1
  output = (1 - opacity) * base_image + opacity * output

  # Scale the output image back to the range [0, 255]
  output *= 255.0
  output = np.clip(output, 0, 255).astype(np.uint8)
  return output
def combine_overlay(texture_folder_path, base_image):
    layer1_texture = randomized_crop(texture_resize(texture_folder_path, base_image), 512)
    face_RGBA = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGBA)
    texture1_RGBA = cv2.cvtColor(layer1_texture, cv2.COLOR_BGR2RGBA)
    textured_image = screen(face_RGBA.astype(float), texture1_RGBA.astype(float), .11)
    layer2_texture = randomized_crop(texture_resize(texture_folder_path, base_image), 512)
    texture2_RGBA = cv2.cvtColor(layer2_texture, cv2.COLOR_BGR2RGBA)
    textured_image = linear_light_blend_mode(textured_image, texture2_RGBA)

    return textured_image

#linear light:
#https://www.youtube.com/watch?v=12sURzLXnhs
# if Blend <=0.5[ Pixel is Darker i.e. after normalization]
#     #Linear burn =
#     # Base + Blend - 1
#     # 1-((1-Base)+(1-Blend))
#     #
#     Result = Base + 2 * Blend -1
#TODO: (problem per pixel basis)
# else [Blend > 0.5, Linear Dodge]:
#     #Linear Dodge(add)
#     # Base + Blend
#     Result = Base + 2 * (Blend - 0.5)
