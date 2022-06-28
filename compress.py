from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
import scipy.fftpack

# function convert rgb to ycbcr
def rgb2ycbcr(img):
    ycbcr = np.zeros(img.shape)
    ycbcr[:,:,0] = 0.256789062 * img[:,:,0] + 0.426003906 * img[:,:,1] + 0.09790625 * img[:,:,2] + 16
    ycbcr[:,:,1] = 128 - 0.1482 * img[:,:,0] - 0.291 * img[:,:,1] + 0.4392 * img[:,:,2]
    ycbcr[:,:,2] = 128 + 0.4392 * img[:,:,0] - 0.3717 * img[:,:,1] - 0.0714 * img[:,:,2]
    return np.uint8(ycbcr)

# function chroma_subsampling 4:2:0
def chroma_subsampling_420(img):
    # convert rgb to ycbcr
    img = rgb2ycbcr(img)
    print(img.shape)
    show_image(img, 'YCbCr Image')
    img = img[1:, :, :]
    img.shape
    # img = img.copy()
    # Vertically, every second element equals to element above itself
    # Horizontally, every second element equals to element to the left of itself
    img[1::2, :, 1] = img[::2, :, 1]
    img[:, 1::2, 1] = img[:, ::2, 1]
    img[1::2, :, 2] = img[::2, :, 2]
    img[:, 1::2, 2] = img[:, ::2, 2]
    # img = rgb2ycbcr(img)
    print(img.shape)
    show_image(img, 'YCbCr Chroma Image')
    return cosine_blocking(img)

# function cosine blocking 8x8 after chroma subsampling
def cosine_blocking(img):
    # convert rgb to ycbcr
    ycbcr = rgb2ycbcr(img)
    # convert ycbcr to y
    y = ycbcr[:,:,0]
    # convert ycbcr to cb and cr
    cb = ycbcr[:,:,1] - 128
    cr = ycbcr[:,:,2] - 128
    # calculate fft of y
    ffty = scipy.fftpack.fft2(y)
    # calculate fft of cb and cr
    fftcb = scipy.fftpack.fft2(cb)
    fftcr = scipy.fftpack.fft2(cr)
    # calculate fft of ycbcr
    fftycbcr = np.zeros(ycbcr.shape)
    fftycbcr[:,:,0] = ffty
    fftycbcr[:,:,1] = fftcb
    fftycbcr[:,:,2] = fftcr
    # calculate fft of ycbcr
    fftycbcr = scipy.fftpack.fft2(fftycbcr)
    # calculate fft of ycbcr
    fftycbcr = fftycbcr / 8
    # calculate fft of ycbcr
    fftycbcr = scipy.fftpack.ifft2(fftycbcr)
    # calculate fft of ycbcr
    fftycbcr = scipy.fftpack.ifft2(fftycbcr)
    # calculate fft of ycbcr
    fftycbcr = scipy.fftpack.ifft2(fftycbcr)
    # calculate fft of ycbcr
    fftycbcr = scipy.fftpack.ifft2(fftycbcr)
    # calculate fft of ycbcr
    fftycbcr = scipy.fftpack.ifft2(fftycbcr)
    # calculate fft of ycbcr
    fftycbcr = scipy.fftpack.ifft2(fftycbcr)
    # calculate

# function compress jpeg
def compress(img, quality):
    # convert rgb to ycbcr
    ycbcr = rgb2ycbcr(img)
    # convert ycbcr to y
    y = ycbcr[:,:,0]
    # convert ycbcr to cb and cr
    cb = ycbcr[:,:,1] - 128
    cr = ycbcr[:,:,2] - 128
    # calculate fft of y
    ffty = scipy.fftpack.fft2(y)
    # calculate fft of cb and cr
    fftcb = scipy.fftpack.fft2(cb)
    fftcr = scipy.fftpack.fft2(cr)
    # calculate fft of ycbcr
    fftycbcr = np.zeros(ycbcr.shape)
    fftycbcr[:,:,0] = ffty
    fftycbcr[:,:,1] = fftcb
    fftycbcr[:,:,2] = fftcr
    # calculate fft of ycbcr
    fftycbcr = scipy.fftpack.fft2(fftycbcr)
    # calculate fft of ycbcr
    fftycbcr = fftycbcr / quality
    # calculate fft of ycbcr
    fftycbcr = scipy.fftpack.ifft2(fftycbcr)
    # calculate fft of ycbcr
    fftycbcr = scipy.fftpack.ifft2(fftycbcr)
    # calculate fft of ycbcr
    fftycbcr = scipy.fftpack.ifft2(fftycbcr)
    # calculate fft of ycbcr
    fftycbcr = scipy.fftpack.ifft2(fftycbcr)
    # calculate fft of ycbcr
    fftycbcr = scipy.fftpack.ifft2(fftycbcr)
    # calculate fft of ycbcr
    fftycbcr = scipy.fftpack.ifft2(fftycbcr)

# function open jpeg image
def open_jpeg(filename):
    img = Image.open(filename)
    img = np.array(img)
    return img

def show_image(img, name):
    plt.figure(figsize = (10, 5))
    plt.imshow(img)
    plt.title(name)

# main function
if __name__ == '__main__':
    # open jpeg image
    img = open_jpeg('photo1.png')

    print(img.shape)
    show_image(img, "Original image")
    # plt.show()

    chroma_subsampling_420(img)

    plt.show()

# function to add x and y
def add(x, y):
    return x + y
    