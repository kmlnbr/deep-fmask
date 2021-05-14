# This module contains various classes that are used for data augmentation in
# the network training.

import cv2
import numpy as np

np.random.seed(42)


class Rotate90(object):
    """
    Rotates image in steps of 90 degrees.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, inp):
        if np.random.random() > self.prob:
            times = np.random.randint(1, 4)
            inp = [np.rot90(i, times).copy() for i in inp]
        return inp


class Flip(object):
    """
    Flips image in the axis given as input.
    """

    def __init__(self, axis, prob=0.5, ):
        self.prob = prob
        self.axis = axis

    def __call__(self, inp):
        if np.random.random() > self.prob:
            inp = [np.flip(i, self.axis).copy() for i in inp]

        return inp


class VerticalFlip(Flip):
    """
    Flips image in the vertical axis.
    """

    def __init__(self, prob=0.5):
        axis = 1
        super(VerticalFlip, self).__init__(axis, prob)


class HorizontalFlip(Flip):
    """
    Flips image in the horizontal axis
    """

    def __init__(self, prob=0.5):
        axis = 0
        super(HorizontalFlip, self).__init__(axis, prob)


class CutOut(object):
    """
    Places a rectangle of zeros of random shape at a random position. The
    height and width of the rectangle can vary from 4 to 32 in steps of 2.
    """

    def __init__(self, prob=0.5):

        self.prob = prob

    def __call__(self, inp):
        if np.random.random() > self.prob:
            inp_dim = inp[0].shape[0]

            # Randomly select size of rectangle
            w, h = np.random.randint(4, 32, (2,))

            # Randomly select top left coordinate position of rectangle
            pos_x = np.random.randint(0, inp_dim - w)
            pos_y = np.random.randint(0, inp_dim - h)

            for img in inp:
                img[pos_y:pos_y + h, pos_x:pos_x + w, :] = 0

        return inp


class ZoomIn(object):
    """
    Zoom Into the input image from the center. A sub-scene is selected from the
    center of the image. The size of the sub-scene can be between 25% to 95% of
    the original input. This is determined by randomly selecting an element
    from the list named as self.level. The sub-scene is then resized to the
    original image size.
    """

    def __init__(self, prob=0.5):

        self.level = np.arange(0.25, 0.95, 0.05)
        self.prob = prob

    def __call__(self, inp):

        if np.random.random() > self.prob:
            # Randomly select sub-scene size
            new_size = int(np.random.choice(self.level, 1, ) * inp[0].shape[0])

            inp_dim = int(inp[0].shape[0])

            start_coordinate = (inp_dim // 2) - (new_size // 2)
            end_coordinate = start_coordinate + new_size
            inp_aug = []

            # inp is list where the first element (n=0) is the array of satellite
            # image input, and the second element (n=1)is the training label array.
            for n, img in enumerate(inp):
                if n:
                    interpolation = cv2.INTER_NEAREST
                else:
                    interpolation = cv2.INTER_CUBIC

                # Crop the sub-scene from the input image and then resize
                img1 = img[start_coordinate:end_coordinate,
                           start_coordinate:end_coordinate, :]
                img = cv2.resize(img1, (img.shape[0], img.shape[0]), interpolation)

                # Ensure that the labels are 3D arrays (with one channel)
                if n:
                    img = img[..., np.newaxis]

                inp_aug.append(img)
            inp = inp_aug
        return inp


if __name__ == '__main__':
    pass
