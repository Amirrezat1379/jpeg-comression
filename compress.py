from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
import scipy.fftpack

# standard_luminance_quantization_table
QY = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,48,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]]
)

# standard_chrominance_quantization_table
QC = np.array([
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99]]
)

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
    return img

# function cosine blocking 8x8
def cosine_blocking(img):
    img = chroma_subsampling_420(img)
    print(img[img.shape[0]//2, img.shape[1]//2, :])
    width = img.shape[0]
    height = img.shape[1]
    dct = []
    # img.save("imgage_after_subsampling.jpg")
    for i in range(0, width, 8) :
        for j in range(0, height, 8) : 
            block = [[0] * 8] * 8
            for block_x_axis in range(i, i + 8) : 
                for block_y_axis in range(j, j + 8) : 
                    if (block_x_axis >= width) or (block_y_axis >= height) : 
                        data = [0, 0, 0]
                    else : 
                        data = img[block_x_axis, block_y_axis, :]
                    block[block_x_axis - i][block_y_axis - j] = data
            dct.append(scipy.fftpack.dct(scipy.fftpack.dct(block, axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho'))
    return dct

# function to quantize
def quantize(img):
    img = cosine_blocking(img)
    img_quantized = img
    for i in range(len(img)):
        img_quantized[i][:, :, 0] = img[i][:, :, 0] // QY
        img_quantized[i][:, :, 1] = img[i][:, :, 1] // QC
        img_quantized[i][:, :, 2] = img[i][:, :, 2] // QC
    print(img_quantized[106])
    return img_quantized

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

    quantize(img)

    plt.show()

# function to add x and y
def add(x, y):
    return x + y
    