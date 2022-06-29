from operator import le
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import huffman
import pickle
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
    print(img[0:8, 0:8, 1])
    # print(img[img.shape[0]//2, img.shape[1]//2, :])
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
                        data = [16, 128, 128]
                    else : 
                        data = img[block_x_axis, block_y_axis, :]
                        # print(data, "\n\n\n")
                    block[block_x_axis - i][block_y_axis - j] = data
            if i == 0 and j == 0 :
                print(np.float32(np.array(block)[:, :, 1]))
            block = np.array(block)
            block[:, :, 0] = cv2.dct(np.float32(block[:, :, 0]))
            block[:, :, 1] = cv2.dct(np.float32(block[:, :, 1]))
            block[:, :, 2] = cv2.dct(np.float32(block[:, :, 2]))
            if i == 0 and j == 0 :
                # print(np.float32(np.array(block)[:, :, 0]))
                print(np.float32(np.array(block)[:, :, 1]))
                # print(np.float32(np.array(block)[:, :, 2]))
            dct.append(block)
    return dct

# function to quantize
def quantize(img):
    img = cosine_blocking(img)
    img_quantized = img
    for i in range(len(img)):
        img_quantized[i][:, :, 0] = img[i][:, :, 0] // QY
        img_quantized[i][:, :, 1] = img[i][:, :, 1] // QC
        img_quantized[i][:, :, 2] = img[i][:, :, 2] // QC
    # print(img[0], "\n\n\n")
    # print(img_quantized[0])
    return img_quantized

# function to zigzag
def zigzag(img):
    img = quantize(img)
    # print(img[106])
    print(img[0][:, :, 0])
    img_zigzag = img
    img_zigzagy = []
    img_zigzagcb = []
    img_zigzagcr = []
    for i in range(len(img)):
        zigzagy = []
        zigzagcb = []
        zigzagcr = []
        zigzagy.append(img[i][0, 0, 0])
        for j in range(1, 8):
            if j % 2 == 1:
                for k in range(j + 1):
                    zigzagy.append(img[i][k, j - k, 0])
                    zigzagcb.append(img[i][k, j - k, 1])
                    zigzagcr.append(img[i][k, j - k, 2])
            else:
                for k in range(j, -1, -1):
                    zigzagy.append(img[i][k, j - k, 0])
                    zigzagcb.append(img[i][k, j - k, 1])
                    zigzagcr.append(img[i][k, j - k, 2])
        for j in range(1, 8):
            sum = 7 + j
            if j % 2 == 1:
                for k in range(7, j - 1, -1):
                    zigzagy.append(img[i][k, sum - k, 0])
                    zigzagcb.append(img[i][k, sum - k, 1])
                    zigzagcr.append(img[i][k, sum - k, 2])
            else:
                for k in range(j, 8):
                    zigzagy.append(img[i][k, sum - k, 0])
                    zigzagcb.append(img[i][k, sum - k, 1])
                    zigzagcr.append(img[i][k, sum - k, 2])
        img_zigzagy.append(zigzagy)
        img_zigzagcb.append(zigzagcb)
        img_zigzagcr.append(zigzagcr)
    return [img_zigzagy, img_zigzagcb, img_zigzagcr]

# function to run-length coding AC
def run_length_coding(img):
    img_zigzagy, img_zigzagcb, img_zigzagcr = zigzag(img)
    rlcAcy = []
    rlcAcb = []
    rlcAcr = []
    rlcDcy = []
    rlcDcb = []
    rlcDcr = []
    zero_count = 0
    for item in img_zigzagy:
        rlca = []
        for value in item[1:]:        
            if value != 0:            
                rlca.append([zero_count, value])
                if zero_count != 0:
                    zero_count = 0
            else:
                zero_count += 1
        rlca.append([0, 0])
        rlcAcy.append(rlca)
    zero_count = 0
    for item in img_zigzagcb:
        rlcb = []
        for value in item[1:]:        
            if value != 0:            
                rlcb.append([zero_count, value])
                if zero_count != 0:
                    zero_count = 0
            else:
                zero_count += 1
        rlcb.append([0, 0])
        rlcAcb.append(rlcb)
    zero_count = 0
    for item in img_zigzagcr:
        rlcr = []
        for value in item[1:]:        
            if value != 0:            
                rlcr.append([zero_count, value])
                if zero_count != 0:
                    zero_count = 0
            else:
                zero_count += 1
        rlcr.append([0, 0])
        rlcAcr.append(rlcr)
    for item in img_zigzagy:
        rlcDcy.append(item[0])
        last = item[0]
        for value in item[1:]:
            rlcDcy.append(int(value) - int(last))
            last = value
    for item in img_zigzagcb:
        rlcDcb.append(item[0])
        last = item[0]
        for value in item[1:]:
            rlcDcb.append(int(value) - int(last))
            last = value
    for item in img_zigzagcr:
        rlcDcr.append(item[0])
        last = item[0]
        for value in item[1:]:
            rlcDcr.append(int(value) - int(last))
            last = value
    return [[rlcAcy, rlcAcb, rlcAcr], [rlcDcy, rlcDcb, rlcDcr]]
    # print(rlcAc[0])
    # return rlc

# function to huffman coding
def huffman_coding(img):
    rlc = run_length_coding(img)
    rlcA = rlc[0]
    rlcA_cof = {}
    rlcD = rlc[1]
    rlcD_cof = {}
    rlcAcy = rlcA[0]
    rlcAcb = rlcA[1]
    rlcAcr = rlcA[2]
    print(rlcAcy[0])
    rlcDcy = rlcD[0]
    rlcDcy_cof = {}
    rlcDcb = rlcD[1]
    rlcDcb_cof = {}
    rlcDcr = rlcD[2]
    rlcDcr_cof = {}
    for i in range(len(rlcDcy)):
        rlcDcy_cof[rlcDcy[i]] = len(str(bin(rlcDcy[i])[2:]))
        rlcDcy[i] = len(str(bin(rlcDcy[i])[2:]))
    for i in range(len(rlcDcb)):
        rlcDcb_cof[rlcDcb[i]] = len(str(bin(rlcDcb[i])[2:]))
        rlcDcb[i] = len(str(bin(rlcDcb[i])[2:]))
    for i in range(len(rlcDcr)):
        rlcDcr_cof[rlcDcr[i]] = len(str(bin(rlcDcr[i])[2:]))
        rlcDcr[i] = len(str(bin(rlcDcr[i])[2:]))
    rlcD = {}
    rlcD_cof["DCY"] = rlcDcy_cof
    rlcD_cof["DCB"] = rlcDcb_cof
    rlcD_cof["DCR"] = rlcDcr_cof
    DCfile = open("DCCOFF", "ab")
    pickle.dump(rlcD_cof, DCfile)
    DCfile.close()
    rlcDcyhuff = huffman.HuffmanTree(rlcDcy)
    rlcDcbhuff = huffman.HuffmanTree(rlcDcb)
    rlcDcrhuff = huffman.HuffmanTree(rlcDcr)
    rlcD["DCY"] = rlcDcyhuff.main()
    rlcD["DCB"] = rlcDcbhuff.main()
    rlcD["DCR"] = rlcDcrhuff.main()
    DCfile = open("DC", "ab")
    pickle.dump(rlcD, DCfile)
    DCfile.close()
    ry = {}
    for i in range(len(rlcAcy)):
        r = []
        for j in range(len(rlcAcy[i])):
            r.append(rlcAcy[i][j][0])
        rt = huffman.HuffmanTree(r)
        ry[str(i)] = rt.main()
    rb = {}
    for i in range(len(rlcAcb)):
        r = []
        for j in range(len(rlcAcb[i])):
            r.append(rlcAcb[i][j][0])
        rt = huffman.HuffmanTree(r)
        rb[str(i)] = rt.main()
    rr = {}
    for i in range(len(rlcAcr)):
        r = []
        for j in range(len(rlcAcr[i])):
            r.append(rlcAcr[i][j][0])
        rt = huffman.HuffmanTree(r)
        rr[str(i)] = rt.main()
    rlcA = {}
    rlcA["ACY"] = ry
    rlcA["ACB"] = rb
    rlcA["ACR"] = rr
    ACfile = open("AC", "ab")
    pickle.dump(rlcA, ACfile)
    ACfile.close()

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

    huffman_coding(img)

    plt.show()
    