import math
import numpy as np
from skimage.color import rgb2gray
from imageio import imread
import matplotlib.pyplot as plt
GRAYSCALE = 1
RGB = 2
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])
YIQ_RGB_TRANSFORMATION_MATRIX = np.linalg.inv(RGB_YIQ_TRANSFORMATION_MATRIX)


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    # If the input image is grayscale, we won’t call it with representation = 2.
    if representation == GRAYSCALE:
        return rgb2gray(normalize(imread(filename))).astype(np.float64)
    return normalize(imread(filename)).astype(np.float64)


def normalize(im):
    return im / 255


def normalize_256bin(im):
    return im * 255


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    img = read_image(filename, representation)
    if representation == 1:
        plt.imshow(img, cmap=plt.cm.gray)
    if representation == 2:
        plt.imshow(img)
    plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """
    imYIQ = np.zeros_like(imRGB)
    rgbColors = np.array([imRGB[:, :, 0], np.array(imRGB[:, :, 1]), np.array(imRGB[:, :, 2])])
    imYIQ[:, :, 0] = np.dot(RGB_YIQ_TRANSFORMATION_MATRIX[0][0], rgbColors[0]) + np.dot(
        RGB_YIQ_TRANSFORMATION_MATRIX[0][1], rgbColors[1]) + np.dot(RGB_YIQ_TRANSFORMATION_MATRIX[0][2], rgbColors[2])
    imYIQ[:, :, 1] = np.dot(RGB_YIQ_TRANSFORMATION_MATRIX[1][0], rgbColors[0]) + np.dot(
        RGB_YIQ_TRANSFORMATION_MATRIX[1][1], rgbColors[1]) + np.dot(RGB_YIQ_TRANSFORMATION_MATRIX[1][2], rgbColors[2])
    imYIQ[:, :, 2] = np.dot(RGB_YIQ_TRANSFORMATION_MATRIX[2][0], rgbColors[0]) + np.dot(
        RGB_YIQ_TRANSFORMATION_MATRIX[2][1], rgbColors[1]) + np.dot(RGB_YIQ_TRANSFORMATION_MATRIX[2][2], rgbColors[2])
    return imYIQ

def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """
    imRGB = np.zeros_like(imYIQ)
    yiqColors = np.array([imYIQ[:, :, 0], np.array(imYIQ[:, :, 1]), np.array(imYIQ[:, :, 2])])
    imRGB[:, :, 0] = np.dot(YIQ_RGB_TRANSFORMATION_MATRIX[0][0], yiqColors[0]) + np.dot(
        YIQ_RGB_TRANSFORMATION_MATRIX[0][1], yiqColors[1]) + np.dot(YIQ_RGB_TRANSFORMATION_MATRIX[0][2], yiqColors[2])
    imRGB[:, :, 1] = np.dot(YIQ_RGB_TRANSFORMATION_MATRIX[1][0], yiqColors[0]) + np.dot(
        YIQ_RGB_TRANSFORMATION_MATRIX[1][1], yiqColors[1]) + np.dot(YIQ_RGB_TRANSFORMATION_MATRIX[1][2], yiqColors[2])
    imRGB[:, :, 2] = np.dot(YIQ_RGB_TRANSFORMATION_MATRIX[2][0], yiqColors[0]) + np.dot(
        YIQ_RGB_TRANSFORMATION_MATRIX[2][1], yiqColors[1]) + np.dot(YIQ_RGB_TRANSFORMATION_MATRIX[2][2], yiqColors[2])
    return imRGB


def get_image_type(im):
    if len(im.shape) == 3:
        return RGB
    return GRAYSCALE


def round_image(im):
    return im.astype(np.uint8)


def histogram_helper(img):
    orig_hist = np.histogram(round_image(img), 256)[0]
    cum_hist = np.cumsum(orig_hist)
    minimum_value = np.amin(np.nonzero(cum_hist))
    maximum_value = np.amax(np.nonzero(cum_hist))
    tk = np.round(((cum_hist - minimum_value) / (cum_hist[-1] - minimum_value)) * maximum_value)
    im_eq = tk[round_image(img)]
    return normalize(im_eq).astype(np.float64), orig_hist, np.histogram(im_eq, 256)[0]


def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return: [im_eq, hist_orig, hist_eq]
    """
    if get_image_type(im_orig) == RGB:
        yiq_im = rgb2yiq(im_orig.copy())
        y_channel, hist_orig, hist_eq = histogram_helper(normalize_256bin(yiq_im[:, :, 0].copy()))
        yiq_im[:, :, 0] = y_channel
        return yiq2rgb(yiq_im), hist_orig, hist_eq
    return histogram_helper(normalize_256bin(im_orig.copy()))


def quantization(im, error, n_quant, n_iter):
    h = np.histogram(im, 256)[0]
    z = np.concatenate([[-1], [np.argmax(np.cumsum(h) > i * np.sum(h) / n_quant) for i in range(1, n_quant)], [255]]).astype(np.float64)
    q = np.array([])
    for it in range(n_iter):
        q = np.array([(lambda z_it: np.sum(z_it * h[z_it]) / np.sum(h[z_it]))
             (np.arange(math.trunc(z[i]) + 1, math.trunc(z[i + 1]) + 1)) for i in range(n_quant)])
        # calculate error and append it to error list
        error.append(sum((lambda z_it: np.sum(((q[i] - z_it) ** 2) * h[z_it]))(np.arange(math.trunc(z[i]) + 1, math.trunc(z[i + 1]) + 1)) for i in range(n_quant)))
        new_z = np.concatenate([[-1], [math.trunc((q[l - 1] + q[l]) / 2) for l in range(1, n_quant)], [255]]).astype(np.float64)
        if np.array_equal(new_z, z): break
        z = new_z
    return normalize(np.concatenate([np.repeat(q[i], z[i + 1] - z[i]) for i in range(n_quant)])[round_image(im)]).astype(np.float64)


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """
    error = []
    if get_image_type(im_orig) == RGB:
        yiq_im = rgb2yiq(im_orig.copy())
        yiq_im[:, :, 0] = quantization(normalize_256bin(yiq_im[:, :, 0]), error, n_quant, n_iter)
        return [yiq2rgb(yiq_im), error]
    return [quantization(normalize_256bin(im_orig.copy()), error, n_quant, n_iter), error]

x = np.hstack([np.repeat(np.arange(0,50,2),10)[None,:], np.array([255]*6)[None,:]])
grad = np.tile(x,(256,1))
fun = lambda x: quantize(x, 10, 10)[0]
# plt.imshow(grad/255, cmap=plt.cm.gray)
plt.imshow(histogram_equalize(read_image("./images/jerusalem.jpg", 2))[0])
# plt.imshow(histogram_equalize(grad/255)[0])
plt.show()
