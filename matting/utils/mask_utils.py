
import PIL.Image
import torch
from matting.utils.image_utils import to_tensor
import numpy as np
import cv2

def composite(foreground,background,alpha_l,device="cpu"):
    """
    Composites foreground with background by following
    https://pymatting.github.io/intro.html#alpha-matting math formula.

    Args:
        device: Processing device
        foreground: Image that will be pasted to background image with following alpha mask.
        background: Background image
        alpha: Alpha Image

    Returns:
        Composited image as PIL.Image instance.
    """

    foreground =  cv2.cvtColor(foreground, cv2.COLOR_RGB2RGBA)

    background = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA)
    alpha_rgba = cv2.cvtColor(alpha_l, cv2.COLOR_RGB2RGBA)
    # alpha_l = alpha.convert("L")
    # alpha_l = alpha[:,:,0]

    fg = to_tensor(foreground).to(device)
    alpha_rgba = to_tensor(alpha_rgba).to(device)
    alpha_l = to_tensor(alpha_l).to(device)
    bg = to_tensor(background).to(device)

    alpha_l = alpha_l / 255
    alpha_rgba = alpha_rgba / 255

    bg = torch.where(torch.logical_not(alpha_rgba >= 1), bg, fg)
    bg[:, :, 0] = alpha_l[:, :] * fg[:, :, 0] + (1 - alpha_l[:, :]) * bg[:, :, 0]
    bg[:, :, 1] = alpha_l[:, :] * fg[:, :, 1] + (1 - alpha_l[:, :]) * bg[:, :, 1]
    bg[:, :, 2] = alpha_l[:, :] * fg[:, :, 2] + (1 - alpha_l[:, :]) * bg[:, :, 2]
    bg[:, :, 3] = alpha_l[:, :] * 255

    del alpha_l, alpha_rgba, fg

    result_bg=bg.cpu().numpy()
    result_bg=cv2.cvtColor(result_bg, cv2.COLOR_RGB2RGBA)

    return result_bg


def apply_mask(image, mask, device="cpu"):
    """
    Applies mask to foreground.

    Args:
        device: Processing device.
        image: Image with background.
        mask: Alpha Channel mask for this image.

    Returns:
        Image without background, where mask was black.
    """
    color = (130,130,130,0) # Green color with alpha value of 255 (fully opaque)

# Create an empty numpy array with the specified size and data type
    background= np.zeros((image.shape[0],image.shape[1], 4), dtype=np.uint8)

# Fill the array with the specified color
    background[:] = color

    # print('image ka size in apply mask----------------------',np.array(image).shape)
    # print('mask ka size in apply mask----------------------',np.array(mask).shape)
    # print('background ka size in apply mask----------------------',np.array(background).shape)

    # background = PIL.Image.new("RGBA", image.size, color=(130, 130, 130, 0))
    return composite(image, background, mask, device=device)


def extract_alpha_channel(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    Extracts alpha channel from the RGBA image.

    Args:
        image: RGBA PIL image

    Returns:
        RGBA alpha channel image
    """
    alpha = image.split()[-1]
    bg = PIL.Image.new("RGBA", image.size, (0, 0, 0, 255))
    bg.paste(alpha, mask=alpha)
    return bg.convert("RGBA")
