# -*- coding: utf-8 -*-
import logging

import matplotlib.pyplot as plt   # for debugging
import numpy as np
import sklearn.metrics.pairwise


log = logging.getLogger(name=__name__)


# in the green channel, these are the colors representing the screen objects
COLORS = {
    'person': 92,
    'flag': 72,
    'end_flag': 50,
    'mogul_1': 214,
    'mogul_2': 192,
    'tree_1': 156,
    'tree_2': 126,
    'border': 0,
    'numbers': 0,
    'snow': 236
}

SKIER_Y = (65, 84)
WINDOW_WIDTH = 160
BORDER_WIDTH = 9
SCREEN_X = (BORDER_WIDTH, WINDOW_WIDTH - BORDER_WIDTH)
SCREEN_WIDTH = SCREEN_X[1] - SCREEN_X[0]
FOG_Y = 203

FLAG_HEIGHT = 15
FLAG_WIDTH = 6
TIP_TO_POLE = 5
POLE_TO_POLE = 32

POSSIBLE_ACTIONS = {'noop': 0, 'left': 1, 'right': 2}


def flag_loc(image, skier_y):
    """
    Get the location of the next flag relative to the top of the skier

    Parameters
    ----------
    image : numpy array (250, 160)
        A green channel only image of the playing screen
    skier_y : int
        Which row the top of the skier is on

    Returns
    -------
    (float, float)
        The x, y midpoint of the next slalom flags
    """
    # find the nearest flags below the top of the skier's head
    flag_slice = image[int(skier_y):FOG_Y, SCREEN_X[0]:SCREEN_X[1]]

    flag_pixels = np.argwhere(flag_slice == COLORS['flag'])

    if flag_pixels.size == 0:
        return end_flag_loc(image, skier_y)

    # Flag pixels are (y, x) pairs. That's hard to think about. Flip
    # the matrix horizontally so that they're (x, y) pairs
    flag_pixels = np.fliplr(flag_pixels)
    x, y = 0, 1

    # what is the left most pixel?
    left_tip_x, left_tip_y = flag_pixels[flag_pixels[:, x].argmin()]

    # is this the nearest flag?
    nearest_y = flag_pixels[:, y].min()

    # this assumes flags are separated vertically by at least a flag's worth
    # of space. That's a safe assumption for this game
    if left_tip_y - FLAG_HEIGHT > nearest_y:
        # throw out any pixels from the further flags
        flag_pixels = flag_pixels[flag_pixels[:, y] < left_tip_y - FLAG_HEIGHT]

        # re-get left most pixel
        left_tip_x, left_tip_y = flag_pixels[flag_pixels[:, x].argmin()]

    # now we can get flag_y, flag_left, and flag_right
    flag_y = left_tip_y + skier_y
    flag_left = left_tip_x + TIP_TO_POLE + SCREEN_X[0]
    flag_mid = flag_left + (POLE_TO_POLE / 2.0) - 3

    return np.array([flag_mid, flag_y], dtype=np.int)


def end_flag_loc(image, skier_y):
    """
    Get the location of the end flag (which is a different color than most)
    relative to the top of the skier

    Parameters
    ----------
    image : numpy array (250, 160)
        A green channel only image of the playing screen
    skier_y : int
        Which row the top of the skier is on

    Returns
    -------
    (float, float)
        The x, y midpoint of the end slalom flags
    """
    # find the nearest flags below the top of the skier's head
    flag_slice = image[int(skier_y):FOG_Y, SCREEN_X[0]:SCREEN_X[1]]

    flag_pixels = np.argwhere(flag_slice == COLORS['end_flag'])

    # Flag pixels are (y, x) pairs. That's hard to think about. Flip
    # the matrix horizontally so that they're (x, y) pairs
    flag_pixels = np.fliplr(flag_pixels)
    x = 0

    # what is the left most pixel?
    left_tip_x, left_tip_y = flag_pixels[flag_pixels[:, x].argmin()]

    # now we can get flag_y, flag_left, and flag_right
    flag_y = left_tip_y + skier_y
    flag_left = left_tip_x + TIP_TO_POLE + SCREEN_X[0]
    flag_mid = flag_left + (POLE_TO_POLE / 2.0) - 6

    return np.array([flag_mid, flag_y], dtype=np.int)


def y_delta_slice(image):
    """
    Find a meaningful row of the image that can be used to track the vertical
    difference between this image and the next.

    Parameters
    ----------
    image : numpy array (250, 160)
        A green channel only image of the playing screen

    Returns
    -------
    numpy array (1, 142)
        One row of the image (minus the border) that contains more than just
        snow

    Raises
    ------
    IndexError
        If for some reason we can't find anything that is not snow (shoud not
        happen)
    """
    subimage = image[SKIER_Y[1] + 1:FOG_Y - 1, SCREEN_X[0]:SCREEN_X[1]]
    height = subimage.shape[0]
    # starting just before the fog and working backwards
    for i in range(1, height):
        if (subimage[-i, :] != COLORS['snow']).sum() > 0:
            return i, subimage[-i, :]

    raise IndexError('No informative slices!')


def delta_y(image, y_delta_slice):
    subimage = image[SKIER_Y[1] + 1:FOG_Y - 1, SCREEN_X[0]:SCREEN_X[1]]
    height = subimage.shape[0]
    y, delta_slice = y_delta_slice

    # starting at -y and working up
    for i in range(height - y):
        if np.all(subimage[-(i + y), :] == delta_slice):
            return i

    raise IndexError('No match! y = {}'.format(y))


def skier_loc(skier_box):
    """
    Get the x, y location of the skier on the screen

    Defined as the middle of the skier horizontally
    and the top of the skier vertically

    Parameters
    ----------
    skier_box : (upper_left_x, upper_left_y, width, height)
        The upper-left x, y coords and
        width, height of the skier's bounding box

    Returns
    -------
    (float, float)
        skier (x, y) location (in pixels)
    """
    ul_x, ul_y, width, height = skier_box
    return ul_x + (width / 2.0), ul_y


def skier_box(img):
    """
    Get the bounding box of the skier

    Parameters
    ----------
    img : numpy array (250, 160)
        A green channel only image of the playing screen

    Returns
    -------
    skier_box : (upper_left_x, upper_left_y, width, height)
        The upper-left x, y coords and
        width, height of the skier's bounding box
    """
    # define the slice of the image that the skier appears in
    skier_slice = img[SKIER_Y[0]:SKIER_Y[1], SCREEN_X[0]:SCREEN_X[1]]

    # find the upper left loc
    skier_pixels = np.argwhere(skier_slice == COLORS['person'])
    ul_y = skier_pixels[:, 0].min()
    ul_x = skier_pixels[:, 1].min()
    width = skier_pixels[:, 1].max() - ul_x
    height = skier_pixels[:, 0].max() - ul_y

    ul_x += SCREEN_X[0]
    ul_y += SKIER_Y[0]

    return (ul_x, ul_y, width, height)


def green(image):
    """
    Gets just the green channel of the image

    Parameters
    ----------
    image : numpy array (250, 160, 3)
        An RGB image of the playing screen

    Returns
    -------
    numpy array (250, 160)
        A green channel only image of the playing screen
    """
    return image[:, :, 1]
