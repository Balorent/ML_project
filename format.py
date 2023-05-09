import numpy as np
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt


def center(image):
    row = np.sum(image, axis=1)
    row = row / np.sum(row)
    mean_row = float(np.sum(row * range(len(row))))

    col = np.sum(image, axis=0)
    col = col / np.sum(col)
    mean_col = float(np.sum(col * range(len(col))))

    return round(mean_row) + 1, round(mean_col) + 1


def center_box(row_center, col_center, image):
    left, top, right, bottom = col_center, row_center, col_center, row_center
    end = False
    begin = False
    found_left, found_top, found_right, found_bottom = False, False, False, False
    while not end:
        if not found_left:
            left -= 10
        if not found_top:
            top -= 10
        if not found_right:
            right += 10
        if not found_bottom:
            bottom += 10

        left_side = np.array(image)[:, left]

        if begin and not left_side.any() and not found_left:
            found_left = True
        top_side = np.array(image)[top, :]
        if begin and not top_side.any() and not found_top:
            found_top = True
        right_side = np.array(image)[:, right]
        if begin and not right_side.any() and not found_right:
            found_right = True
        bottom_side = np.array(image)[bottom, :]
        if begin and not bottom_side.any() and not found_bottom:
            found_bottom = True

        border = np.concatenate((left_side, top_side, right_side, bottom_side))

        if not begin:
            if border.any():
                begin = True
        else:
            if not border.any():
                end = True
    return left, top, right, bottom


def binary_format(image):
    # Transforms the image into a black and white version where the digits / letters are in white (value = 255) and
    # that background is in black (value = 0)

    # Negative of the image
    binary_image = ImageOps.invert(image)

    # remove the border (4 pixels of each side) because my phone creates black pixels in the four corners
    col_nbr, row_nbr = binary_image.size
    binary_image = binary_image.crop((4, 4, col_nbr - 4, row_nbr - 4))
    col_nbr, row_nbr = binary_image.size

    # Remove the background : it is often on the edges that the background is the lighter, then the maximum value of an
    # edge pixel is taken as being the maximal value of background :
    # - If a pixel is darker that that threshold, it will be black
    # - Else, it will be white
    left_side = np.array(binary_image)[:, 0]
    top_side = np.array(binary_image)[0, :]
    right_side = np.array(binary_image)[:, col_nbr - 1]
    bottom_side = np.array(binary_image)[row_nbr - 1, :]
    border = np.concatenate((left_side, top_side, right_side, bottom_side))
    threshold = np.max(border)
    binary_image = np.maximum(binary_image, threshold)
    binary_image = Image.fromarray((binary_image != threshold))

    # return
    return binary_image


def standard_format(image):
    standard_image = image

    # center the digit
    row_center, col_center = center(standard_image)
    center_frame = center_box(row_center, col_center, standard_image)
    standard_image = standard_image.crop(center_frame)
    col_nbr, row_nbr = standard_image.size
    if col_nbr > row_nbr:
        add_col = int(col_nbr / 3)
        new_size = col_nbr + 2 * add_col
        add_row = int((new_size - row_nbr) / 2)

    else:
        add_row = int(row_nbr / 3)
        new_size = row_nbr + 2 * add_row
        add_col = int((new_size - col_nbr) / 2)

    standard_image = ImageOps.expand(standard_image, border=(
        add_col, add_row, add_col, add_row), fill=0)

    # resize to 128x128
    standard_image = standard_image.resize((128, 128))

    # return
    return standard_image


def emnist_format(image):
    # Apply Gaussian blur
    # (a) -> (b)
    emnist_image = image.convert('L')
    emnist_image = emnist_image.filter(ImageFilter.GaussianBlur(radius=1))

    # Remove the margins
    # (b) -> (c)
    row = np.sum(emnist_image, axis=0)
    non_zero_row = np.nonzero(row)
    row_min, row_max = np.min(non_zero_row), np.max(non_zero_row)
    col = np.sum(emnist_image, axis=1)
    non_zero_col = np.nonzero(col)
    col_min, col_max = np.min(non_zero_col), np.max(non_zero_col)
    emnist_image = emnist_image.crop((row_min, col_min, row_max, col_max))

    # Center the digit/letter in a square image
    # (c) -> (d)
    row_center, col_center = center(emnist_image)
    col_count, row_count = emnist_image.size
    while row_count != col_count:
        if row_count < col_count:
            if row_center <= row_count / 2:
                emnist_image = ImageOps.expand(emnist_image, border=(0, 1, 0, 0), fill=0)
                row_center += 1
            else:
                emnist_image = ImageOps.expand(emnist_image, border=(0, 0, 0, 1), fill=0)
            row_count += 1
        else:
            if col_center <= col_count / 2:
                emnist_image = ImageOps.expand(emnist_image, border=(1, 0, 0, 0), fill=0)
                col_center += 1
            else:
                emnist_image = ImageOps.expand(emnist_image, border=(0, 0, 1, 0), fill=0)
            col_count += 1
    emnist_image = ImageOps.expand(emnist_image, border=(2, 2, 2, 2), fill=0)

    # reformat to 28x28 pixels
    # (d) -> (e)
    emnist_image = emnist_image.resize((28, 28))

    # return
    return emnist_image
