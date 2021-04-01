import numpy as np

class Rotate90(object):
    def __init__(self,prob = 0.5):
        """
        Rotates image by multiple of 90 degrees in the axis given
        """
        self.prob = prob

    def __call__(self, inp):
        """
        :param inp: ndarray: Image

        :return: Rotated image
        """
        if np.random.random() > self.prob:
            times = np.random.randint(1, 4)
            inp = [np.rot90(i, times).copy() for i in inp]
        return inp
class Flip(object):

    def __init__(self, axis, prob=0.5, ):
        """
        Flips image in the axis given
        """
        self.prob = prob
        self.axis = axis

    def __call__(self, inp):
        """
        :param inp: ndarray): Image

        :return: Fliped image
        """
        if np.random.random() > self.prob:

            inp = [np.flip(i, self.axis).copy() for i in inp]

        return inp


class VerticalFlip(Flip):
    def __init__(self, prob=0.5):
        axis = 1
        super(VerticalFlip, self).__init__(axis, prob)


class HorizontalFlip(Flip):
    def __init__(self, prob=0.5):
        axis = 0
        super(HorizontalFlip, self).__init__(axis, prob)


if __name__ == '__main__':
    trans = Rotate90()
    l = np.random.random((2, 2,2))
    print(l[:, :, 0])
    print('#' * 28)
    print(l[:, :, 1])
    print('*' * 28)
    print('*' * 28)
    l = trans([l])[0]
    print(l[:, :, 0])
    print('#' * 28)
    print(l[:, :, 1])
