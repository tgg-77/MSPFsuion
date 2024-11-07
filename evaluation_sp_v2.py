import math
import numpy as np
import cv2
import torch
import pandas as pd
from skimage import measure,metrics
from scipy import signal
from MSSSIM import msssim
import math
from PIL import Image
from skimage.measure import shannon_entropy
from pytorch_msssim import ssim, ms_ssim
from sklearn.metrics.cluster import mutual_info_score
from tqdm import trange
# from MEF_MS_SSIM.MS_SSIMc import MS_SSIMc
from scipy.ndimage import convolve
from scipy.ndimage import filters

def QG(fuse, imgs):
    """
    Calculate the fusion quality metric for multiple input images and a fused image.

    Parameters:
    fuse : 2D array
        The fused image.
    imgs : list of 2D arrays
        Input images (five images in this case).

    Returns:
    res : float
        The fusion quality metric.
    """
    # Define Sobel filters
    flt1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    flt2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Calculate gradient and angle for the fused image
    fuseX = convolve(fuse, flt1, mode='reflect')
    fuseY = convolve(fuse, flt2, mode='reflect')
    fuseG = np.sqrt(fuseX ** 2 + fuseY ** 2)
    fuseX = np.where(fuseX == 0, 1e-5, fuseX)
    fuseA = np.arctan(fuseY / fuseX)

    # Initialize total weights and quality sums
    total_weighted_quality = np.zeros_like(fuse, dtype=float)
    total_weights = np.zeros_like(fuse, dtype=float)

    # Parameters for quality metric calculation
    gama1, gama2 = 1, 1
    k1, k2 = -10, -20
    delta1, delta2 = 0.5, 0.75

    # Process each input image
    for img in imgs:
        # Calculate gradients and angles for the input image
        imgX = convolve(img, flt1, mode='reflect')
        imgY = convolve(img, flt2, mode='reflect')
        imgG = np.sqrt(imgX ** 2 + imgY ** 2)

        imgX = np.where(imgX == 0, 1e-5, imgX)
        imgA = np.arctan(imgY / imgX)

        # Edge preservation estimation for current image and fused image
        bimap = imgG > fuseG
        imgG = np.where(imgG == 0, 1e-5, imgG)
        fuseG = np.where(fuseG == 0, 1e-5, fuseG)

        Gaf = bimap * (fuseG / imgG) + (~bimap) * (imgG / fuseG)
        Aaf = np.abs(np.abs(imgA - fuseA) - np.pi / 2) * 2 / np.pi

        # Quality metrics
        Qg_AF = gama1 / (1 + np.exp(k1 * (Gaf - delta1)))
        Qalpha_AF = gama2 / (1 + np.exp(k2 * (Aaf - delta2)))
        Qaf = Qg_AF * Qalpha_AF

        # Weighting matrix (edge strength of current image)
        Wa = imgG ** 1  # L=1 for simplicity

        # Accumulate weighted qualities and weights
        total_weighted_quality += Qaf * Wa
        total_weights += Wa

    # Final fusion quality metric
    res = np.mean(total_weighted_quality / total_weights)
    return res

def spatialF(image):
    image = np.array(image)
    M = image.shape[0]
    N = image.shape[1]

    cf = 0
    rf = 0
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            dx = float(image[i, j - 1]) - float(image[i, j])
            rf += dx ** 2
            dy = float(image[i - 1, j]) - float(image[i, j])
            cf += dy ** 2

    RF = math.sqrt(rf / (M * N))
    CF = math.sqrt(cf / (M * N))
    SF = math.sqrt(RF ** 2 + CF ** 2)

    return round(SF, 3)




#求图像的灰度概率分布
def gray_possiblity(image):
    tmp = np.zeros(256,dtype='float')
    val = 0
    k = 0
    res = 0
    img = np.array(image)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]             #像素值
            val = val.astype(int)
            tmp[val] = float(tmp[val] + 1) #该像素值的次数
            k = k+1              #总次数

    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)   #各个灰度概率
    return tmp





# 求图像的条件熵： H(x|y) = H(x,y) - H(y)
def condition_entropy(image,fuse):
    tmp = np.zeros((256,256),dtype='float')
    res = 0
    k = 0
    image = np.array(image,dtype='int')
    fuse = np.array(fuse,dtype='int')
    rows,cols = image.shape[:2]
    for i in range(len(image)):
        for j in range(len(image[i])):
            tmp[image[i][j]][fuse[i][j]] = float(tmp[image[i][j]][fuse[i][j]]+1)
            k = k+1
    for i in range(256):
        for j in range(256):
            p = tmp[i,j]/k
            if p!=0:
                res = float(res-p*math.log(p,2))
    return res - Entropy(fuse)

#求图像的信息熵 P(a)表示灰度概念 entropy = PA(a)*logPA(a)
def Entropy(image):
    tmp = []
    res = 0
    tmp = gray_possiblity(image)
    for i in range(len(tmp)):
        if(tmp[i]==0):
            res = res
        else:
            res = float(res - tmp[i]*(math.log(tmp[i],2)))
    return res

#交叉熵 求和 -(p(x)logq(x)+(1-p(x))log(1-q(x)))
def cross_entropy(image,aim):
    tmp1 = []
    tmp2 = []
    res = 0
    tmp1 = gray_possiblity(image)
    tmp2 = gray_possiblity(aim)
    for i in range(len(tmp1)):
        if(tmp1[i]!=0)and(tmp2[i]!=0):
            res = tmp1[i]*math.log(1/tmp2[i]) + res
            #res = float(res-(tmp1[i]*math.log2(tmp2[i])+(1-tmp1[i])*math.log2(1-tmp2[i])))
    return res

#求psnr 利用skimage库
def peak_signal_to_noise(true,test):
    return metrics.peak_signal_noise_ratio(true,test,data_range=255)

#求IQM 论文参考文献14
def IQM(image,fused):
    image = np.array(image,float)
    fused = np.array(fused,float)
    N = len(image)*len(image[0])
    k = 0.0
    x_mean = 0.0
    y_mean = 0.0
    for i in range(len(image)):
        for j in range(len(image[i])):
            x_mean = x_mean + image[i][j]
            y_mean = y_mean + fused[i][j]
            k = k+1
    x_mean = float(x_mean/k)
    y_mean = float(y_mean/k)
    for i in range(len(image)):
        for j in range(len(image[i])):
            xy = xy+(image[i][j]-x_mean)*(fused[i][j]-y_mean)
            x = x+(image[i][j]-x_mean)*(image[i][j]-x_mean)
            y = y+(fused[i][j]-y_mean)*(fused[i][j]-y_mean)
    xy = xy/(N-1)
    x = x/(N-1)
    y = y/(N-1)
    Q = 4*xy*x_mean*y_mean/((x*x+y*y)*(x_mean*x_mean+y_mean*y_mean))
    return Q


def QABF(stra, strb, f):
    # model parameters 模型参数
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8

    # Sobel Operator Sobel算子
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
    h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

    # if y is the response to h1 and x is the response to h3;then the intensity is sqrt(x^2+y^2) and  is arctan(y/x);
    # 如果y对应h1，x对应h2，则强度为sqrt(x^2+y^2)，方向为arctan(y/x)

    strA = stra.astype(np.float32)
    strB = strb.astype(np.float32)
    strF = f.astype(np.float32)

    # 数组旋转180度
    def flip180(arr):
        new_arr = arr.reshape(arr.size)
        new_arr = new_arr[::-1]
        new_arr = new_arr.reshape(arr.shape)
        return new_arr

    # 相当于matlab的Conv2
    def convolution(k, data):
        k = flip180(k)
        data = np.pad(data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        n, m = data.shape
        img_new = []
        for i in range(n - 2):
            line = []
            for j in range(m - 2):
                a = data[i:i + 3, j:j + 3]
                line.append(np.sum(np.multiply(k, a)))
            img_new.append(line)
        return np.array(img_new)

    # 用h3对strA做卷积并保留原形状得到SAx，再用h1对strA做卷积并保留原形状得到SAy
    # matlab会对图像进行补0，然后卷积核选择180度
    # gA = sqrt(SAx.^2 + SAy.^2);
    # 定义一个和SAx大小一致的矩阵并填充0定义为aA，并计算aA的值
    def getArray(img):
        SAx = convolution(h3, img)
        SAy = convolution(h1, img)
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        n, m = img.shape
        aA = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if (SAx[i, j] == 0):
                    aA[i, j] = math.pi / 2
                else:
                    aA[i, j] = math.atan(SAy[i, j] / SAx[i, j])
        return gA, aA

    # 对strB和strF进行相同的操作
    gA, aA = getArray(strA)
    gB, aB = getArray(strB)
    gF, aF = getArray(strF)

    # the relative strength and orientation value of GAF,GBF and AAF,ABF;
    def getQabf(aA, gA, aF, gF):
        n, m = aA.shape
        GAF = np.zeros((n, m))
        AAF = np.zeros((n, m))
        QgAF = np.zeros((n, m))
        QaAF = np.zeros((n, m))
        QAF = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if (gA[i, j] > gF[i, j]):
                    GAF[i, j] = gF[i, j] / gA[i, j]
                elif (gA[i, j] == gF[i, j]):
                    GAF[i, j] = gF[i, j]
                else:
                    GAF[i, j] = gA[i, j] / gF[i, j]
                AAF[i, j] = 1 - np.abs(aA[i, j] - aF[i, j]) / (math.pi / 2)

                QgAF[i, j] = Tg / (1 + math.exp(kg * (GAF[i, j] - Dg)))
                QaAF[i, j] = Ta / (1 + math.exp(ka * (AAF[i, j] - Da)))

                QAF[i, j] = QgAF[i, j] * QaAF[i, j]

        return QAF

    QAF = getQabf(aA, gA, aF, gF)
    QBF = getQabf(aB, gB, aF, gF)

    # 计算QABF
    deno = np.sum(gA + gB)
    nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
    output = nume / deno
    return output




#计算图像的Qabf
def matrix_pow(m):
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] = pow(m[i][j],2)
    return m
def matrix_sqrt(m):
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] = math.sqrt(m[i][j])
    return m
def matrix_multi(m1,m2):
    m = np.zeros(m1.shape)
    for i in range(len(m1)):
        for j in range(len(m1[i])):
            m[i][j] = m1[i][j]*m2[i][j]
    return m
def Qabf(pA,pB,pF):
    L = 1
    Tg = 0.9994
    Kg = -15
    Dg = 0.5
    Ta = 0.9879
    Ka = -22
    Da = 0.8
    h3 = [[-1,0,1],
          [-2,0,2],
          [-1,0,1]]
    h3 = np.array(h3)
    h3 = h3.astype(np.float)
    h1 = [[1,2,1],
          [0,0,0],
          [-1,-2,-1]]
    h1 = np.array(h1)
    h1 = h1.astype(np.float)
    h2 = [[0,1,2],
          [-1,0,1],
          [-2,-1,0]]
    h2 = np.array(h2)
    h2 = h2.astype(np.float)
    SAx  = signal.convolve2d(pA,h3,'same')
    SAy = signal.convolve2d(pA,h1,'same')
    gA = matrix_pow(SAx)+matrix_pow(SAy)
    gA = matrix_sqrt(gA)
    M = SAy.shape[0]
    N = SAy.shape[1]
    aA = np.zeros(SAx.shape)
    for i in range(M):
        for j in range(N):
            if SAx[i][j]==0:
                aA[i][j] = math.pi/2
            else:
                aA[i][j] = math.atan(SAy[i][j]/SAx[i][j])

    SBx = signal.convolve2d(pB,h3,'same')
    SBy = signal.convolve2d(pB,h1,'same')
    gB = matrix_pow(SBx) + matrix_pow(SBy)
    gB = matrix_sqrt(gB)
    aB = np.zeros(SBx.shape)
    for i in range(SBx.shape[0]):
        for j in range(SBx.shape[1]):
            if SBx[i][j] == 0:
                aB[i][j] = math.pi/2
            else:

                aB[i][j] = math.atan(SBy[i][j]/SBx[i][j])
    SFx = signal.convolve2d(pF,h3,boundary='symm',mode='same')
    SFy = signal.convolve2d(pF,h1,boundary='symm',mode='same')
    gF = matrix_sqrt((matrix_pow(SFx)+matrix_pow(SFy)))
    M = SFx.shape[0]
    N = SFx.shape[1]
    aF = np.zeros(SFx.shape)
    for i in range(M):
        for j in range(N):
            if SFx[i][j] == 0:
                aF[i][j] = math.pi/2
            else:
                aF[i][j] = math.atan(SFy[i][j]/SFx[i][j])
    #the relative strength and orientation value of GAF,GBF and AAF,ABF
    GAF = np.zeros(SFx.shape)
    AAF = np.zeros(SFx.shape)
    QgAF = np.zeros(SFx.shape)
    QaAF = np.zeros(SFx.shape)
    QAF  = np.zeros(SFx.shape)
    for i in range(M):
        for j in range(N):
            if gA[i][j]>gF[i][j]:
                GAF[i][j] = gF[i][j]/gA[i][j]
            else:
                if gF[i][j] == 0:
                    GAF[i][j] = 0
                else:
                    GAF[i][j] = gA[i][j] / gF[i][j]

            AAF[i][j] = 1- abs(aA[i][j]-aF[i][j])/(math.pi/2)
            QgAF[i][j] = Tg / (1+math.exp(Kg*(GAF[i][j]-Dg)))
            QaAF[i][j] = Ta / (1+math.exp(Ka*(AAF[i][j]-Da)))
            QAF[i][j] = QgAF[i][j]*QaAF[i][j]

    GBF = np.zeros(SFx.shape)
    ABF = np.zeros(SFx.shape)
    QgBF = np.zeros(SFx.shape)
    QaBF = np.zeros(SFx.shape)
    QBF = np.zeros(SFx.shape)
    for i in range(M):
        for j in range(N):
            if gB[i][j] == gF[i][j]:
                GBF[i][j] = gF[i][j]
            else:
                if gF[i][j]==0:
                    GBF[i][j] = 0
                else:
                    GBF[i][j] = gB[i][j]/gF[i][j]
            ABF[i][j] = 1 - abs(aB[i][j]-aF[i][j])/(math.pi/2)
            QgBF[i][j] = Tg / (1 + math.exp(Kg*(GBF[i][j]-Dg)))
            QaBF[i][j] = Ta / (1 + math.exp(Ka*(ABF[i][j]-Da)))
            QBF[i][j] = QgBF[i][j]*QaBF[i][j]
    # compute the QABF
    deno = np.sum(np.sum(gA+gB))
    nume = np.sum(np.sum(matrix_multi(QAF,gA)+matrix_multi(QBF,gB)))
    output = nume/deno
    return  output



def avgGradient(image):
    # image = Image.open(path).convert('L')
    image = image
    width = image.shape[0]
    width = width - 1
    heigt = image.shape[1]
    heigt = heigt - 1
    tmp = 0.0

    for i in range(width):
        for j in range(heigt):
            dx = float(image[i, j + 1]) - float(image[i, j])
            dy = float(image[i + 1, j]) - float(image[i, j])
            ds = math.sqrt((dx * dx + dy * dy) / 2)
            tmp += ds

    imageAG = tmp / (width * heigt)
    return round(imageAG,3)


def ComEntropy(img1, img2):
    width = img1.shape[0]
    hegith = img1.shape[1]
    tmp = np.zeros((256, 256))
    res = 0
    for i in range(width):
        for j in range(hegith):
            val1 = int(img1[i][j])
            val2 = int(img2[i][j])
            tmp[val1][val2] = float(tmp[val1][val2] + 1)
    tmp = tmp / (width * hegith)
    for i in range(256):
        for j in range(256):
            if (tmp[i][j] == 0):
                res = res
            else:
                res = res - tmp[i][j] * (math.log(tmp[i][j] / math.log(2.0)))
    return res

def SD(fusion_image):
    h, w = fusion_image.shape
    m = np.mean(fusion_image)
    s = 0
    for i in range(h):
        for j in range(w):
            s = (fusion_image[i, j] - m)**2 + s
    # sd = (s/(h*w))**0.5
    # sd = (s)**0.5/255
    sd = np.std(fusion_image)
    return sd



def ssim_index(img1, img2):
    """计算SSIM及其映射"""
    ssim_map = metrics.structural_similarity(img1, img2, full=True)[1]
    return metrics.structural_similarity(img1, img2), ssim_map

def QS(img_list, fuse, sw=1):
    # 计算每对图像之间的SSIM
    ssim_values = []
    ssim_maps = []
    sigma_values = []

    # 对每一对图像进行SSIM计算
    for img in img_list:
        ssim_val, ssim_map = ssim_index(fuse, img)
        ssim_values.append(ssim_val)
        ssim_maps.append(ssim_map)
        sigma_values.append(np.var(fuse - img))  # 计算方差作为sigma估计

    # 转换为numpy数组
    ssim_values = np.array(ssim_values)
    ssim_maps = np.array(ssim_maps)
    sigma_values = np.array(sigma_values)

    # 计算加权系数
    buffer = sigma_values.sum(axis=0)
    ramda = sigma_values / buffer

    if sw == 1:
        # 扩展ramda的维度
        Q = ramda[:, np.newaxis, np.newaxis] * ssim_maps + (1 - ramda[:, np.newaxis, np.newaxis]) * ssim_maps
        res = np.mean(Q)

    elif sw == 2:
        # Weighted fusion quality index
        cw = sigma_values / np.sum(sigma_values)
        Q = np.sum(cw[:, np.newaxis, np.newaxis] * (
                    ramda[:, np.newaxis, np.newaxis] * ssim_maps + (1 - ramda[:, np.newaxis, np.newaxis]) * ssim_maps),
                   axis=0)
        res = Q

    elif sw == 3:
        # Edge-dependent quality index
        flt1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        flt2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

        fuseX = cv2.filter2D(fuse, -1, flt1)
        fuseY = cv2.filter2D(fuse, -1, flt2)
        fuseF = np.sqrt(fuseX ** 2 + fuseY ** 2)

        img_gradients = []
        for img in img_list:
            imgX = cv2.filter2D(img, -1, flt1)
            imgY = cv2.filter2D(img, -1, flt2)
            imgF = np.sqrt(imgX ** 2 + imgY ** 2)
            img_gradients.append(imgF)

        img_gradients = np.array(img_gradients)

        # 计算新的SSIM
        ssim_values = []
        for grad_img in img_gradients:
            ssim_val, _ = ssim_index(fuseF, grad_img)
            ssim_values.append(ssim_val)

        ssim_values = np.array(ssim_values)

        # 计算加权平均
        cw = sigma_values / np.sum(sigma_values)
        Qw = np.sum(cw * (ramda * ssim_values))

        alpha = 1
        Qe = Qw ** (1 + alpha)
        res = Qe

    return res

def MI (path_A,path_B,path_C,path_D,path_E,path_F,path_G):
    A = path_A
    B = path_B
    C = path_C
    D = path_D
    E = path_E
    F = path_F
    G = path_G
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)
    E = np.array(E)
    F = np.array(F)
    G = np.array(G)
    mi1 = shannon_entropy(A) + shannon_entropy(G) - ComEntropy(A, G)
    mi2 = shannon_entropy(B) + shannon_entropy(G) - ComEntropy(B, G)
    mi3 = shannon_entropy(C) + shannon_entropy(G) - ComEntropy(C, G)
    mi4 = shannon_entropy(D) + shannon_entropy(G) - ComEntropy(D, G)
    mi5 = shannon_entropy(E) + shannon_entropy(G) - ComEntropy(E, G)
    mi6 = shannon_entropy(F) + shannon_entropy(G) - ComEntropy(F, G)
    mi = ((mi1+mi2+mi3+mi4+mi5)/5+mi6)/2
    return round(mi,3)


def cv2torch(np_img):
    rgb = np.expand_dims(np_img, 0)
    rgb = np.expand_dims(rgb, 0)
    rgb = np.float32(rgb)
    return torch.from_numpy(rgb)


def torch2cv(t_img):
    return np.squeeze(t_img.numpy())


import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def QCB(images, fim):
    """
    Implements Yin Chen's algorithm for image fusion metric.

    Parameters:
        images: List of input images (list of 2D arrays).
        fim: Fused image (2D array).

    Returns:
        res: Metric value.
    """


    # Normalize input images
    if images[0].shape[0] != images[0].shape[1]:
        images = [im[:, :-1] for im in images]
    normalized_images = [normalize1(im) for im in images]

    for im in normalized_images:
        if im.shape != fim.shape:
            fim = normalize1(fim[:, :-1])
    # Constants for the experiment
    f0 = 15.3870
    f1 = 1.3456
    a = 0.7622
    k = 1
    h = 1
    p = 3
    q = 2
    Z = 0.0001
    HH = normalized_images[0].shape[0] // 30
    LL = normalized_images[0].shape[1] // 30

    # Frequency space
    u, v = np.meshgrid(np.fft.fftfreq(normalized_images[0].shape[0]), np.fft.fftfreq(normalized_images[0].shape[1]))
    u = LL * u
    v = HH * v
    r = np.sqrt(u**2 + v**2)

    Sd = np.exp(-(r/f0)**2) - a * np.exp(-(r/f1)**2)

    # Contrast sensitivity filtering for each image
    filtered_images = [ifft2(ifftshift(fftshift(fft2(im)) * Sd)).real for im in normalized_images]
    ffim = ifft2(ifftshift(fftshift(fft2(fim)) * Sd)).real

    # Local contrast computation
    G1 = gaussian2d(normalized_images[0].shape[0], normalized_images[0].shape[1], 2)
    G2 = gaussian2d(normalized_images[0].shape[0], normalized_images[0].shape[1], 4)

    Q_total = np.zeros_like(ffim)

    for fim1 in filtered_images:
        C1 = contrast(G1, G2, fim1)
        C1P = (k * (np.abs(C1)**p)) / (h * (np.abs(C1)**q) + Z)

        Cf = contrast(G1, G2, ffim)
        CfP = (k * (np.abs(Cf)**p)) / (h * (np.abs(Cf)**q) + Z)

        # Contrast preservation calculation
        mask = (C1P < CfP)
        Q1F = (C1P / CfP) * mask + (CfP / C1P) * (1 - mask)

        # Accumulate quality maps
        Q_total += Q1F

    # Average over the number of input images
    Q_avg = Q_total / len(images)

    return np.mean(Q_avg)

def gaussian2d(n1, n2, sigma):
    """
    Creates a 2D Gaussian filter in spatial domain.

    Parameters:
        n1: Height of the filter.
        n2: Width of the filter.
        sigma: Standard deviation of the Gaussian.

    Returns:
        res: 2D Gaussian filter.
    """
    x = np.linspace(-15, 15, n2)
    y = np.linspace(-15, 15, n1)
    X, Y = np.meshgrid(x, y)
    G = np.exp(-(X**2 + Y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return G

def contrast(G1, G2, im):
    """
    Computes the contrast between two filters and an image.

    Parameters:
        G1: First filter.
        G2: Second filter.
        im: Input image.

    Returns:
        res: Contrast value.
    """
    buff = gaussian_filter(im, sigma=1)
    buff1 = gaussian_filter(im, sigma=2)
    return buff / buff1 - 1

def normalize1(im):
    """
    Normalizes the image to the range [0, 1].

    Parameters:
        im: Input image (2D array).

    Returns:
        res: Normalized image.
    """
    return (im - np.min(im)) / (np.max(im) - np.min(im)) if np.max(im) != np.min(im) else im

# Example usage:
# images = [image1, image2, image3, image4, image5]
# res = metric_chen_blum(images, fused_image)





#测试集
if __name__ == '__main__':
    num = 34
    # methed_list = ['GFF', 'GTF', 'MDLatRR', 'MST_SR', 'MSVD']
    # methed_list = ['NSST_PCNN']
    # methed_list = ['densefuse', "ifcnn", "rfn", 'TRP']
    # methed_list = ['densefuse', "rfn", 'TRP']
    # methed_list = ['MST_34_expen', 'ADF', 'GFF_new', 'densefuse', 'DWT_new']
    # methed_list = ['MST_34_expen', 'Swinfusion_new_v2']
    methed_list = ['l2_ssim_expen_1e0', 'l2_ssim_expen_1e1', 'l2_ssim_expen_1e2', 'l2_ssim_expen_1e3', 'sp_34_mean', 'max']
    result = np.zeros((34, 8))
    for methed in methed_list:
        en, ag, ssim_v, mi, sf, sd, ms_ssim, qs, qg, qcb = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        error, error_qg = 0, 0
        for i in range(34):
            fuse = cv2.imread(r'H:\360MoveData\Users\Lenovo\Desktop\fusion\MST_Fusion\sp_34\%s\%d\f.png' % (methed, i), 0)
            s0 = cv2.imread(r'H:\360MoveData\Users\Lenovo\Desktop\fusion\MST_Fusion\sp_34\%s\%d\0.png' % ('densefuse', i), -1)
            s1 = cv2.imread(r'H:\360MoveData\Users\Lenovo\Desktop\fusion\MST_Fusion\sp_34\%s\%d\1.png' % ('densefuse', i), -1)
            s2 = cv2.imread(r'H:\360MoveData\Users\Lenovo\Desktop\fusion\MST_Fusion\sp_34\%s\%d\2.png' % ('densefuse', i), -1)
            s3 = cv2.imread(r'H:\360MoveData\Users\Lenovo\Desktop\fusion\MST_Fusion\sp_34\%s\%d\3.png' % ('densefuse', i), -1)
            s4 = cv2.imread(r'H:\360MoveData\Users\Lenovo\Desktop\fusion\MST_Fusion\sp_34\%s\%d\4.png' % ('densefuse', i), -1)
            d = cv2.imread(r'H:\360MoveData\Users\Lenovo\Desktop\fusion\MST_Fusion\sp_34\%s\%d\d.png' % ('densefuse', i), -1)
            # print(fuse.shape, s0.shape, d.shape, s1.shape, s2.shape, s3.shape, s4.shape)
            sd_ = SD(fuse)
            en_ = Entropy(fuse)
            ag_ = avgGradient(fuse)
            mi_ = MI(s0,s1,s2,s3,s4,d,fuse)
            sf_ = spatialF(fuse)
            ssim_ = ((metrics.structural_similarity(s0, fuse)+metrics.structural_similarity(s1, fuse)+metrics.structural_similarity(s2, fuse)+metrics.structural_similarity(s3, fuse)+metrics.structural_similarity(s4, fuse))/5+metrics.structural_similarity(d, fuse))/2
            # ms_ssim_ = ((msssim(s0, fuse)+msssim(s1, fuse)+msssim(s2, fuse)+msssim(s3, fuse)+msssim(s4, fuse))/5+msssim(d, fuse))/2
            fuse = fuse.astype(np.float64)
            s0 =s0.astype(np.float64)
            s1 =s1.astype(np.float64)
            s2 =s2.astype(np.float64)
            s3 =s3.astype(np.float64)
            s4 =s4.astype(np.float64)
            d =d.astype(np.float64)
            qs_ = (QS([s0,s1,s2,s3,s4],fuse) + QS([d],fuse))/2
            qg_ = (QG(fuse, [s0,s1,s2,s3,s4]) + QG(fuse, [d]))/2
            qcb_ = (QCB([s0,s1,s2,s3,s4],fuse) + QCB([d],fuse))/2

            sd = sd + sd_
            en = en + en_
            ag = ag + ag_
            mi = mi + mi_
            sf = sf + sf_
            ssim_v = ssim_v + ssim_
            qcb = qcb + qcb_

            if not np.isnan(qs_):
                qs = qs + qs_
            else:
                error += 1
            if not np.isnan(qg_):
                qg = qg + qg_
            else:
                error_qg += 1
            # result[i, 0] = en_
            # result[i, 1] = ag_
            # result[i, 2] = ssim_
            # result[i, 3] = mi_
            # result[i, 4] = sf_
            # result[i, 5] = sd_
            # result[i, 6] = ms_ssim_
            result[i, 7] = qs_
            print('number: %d' % i, '*'*5)
            # print('SD:', sd_)
            # print('entropy:', en_)
            # print('AG:', ag_)
            # print('sssim:', ssim_)
            # print('sf:', sf_)
            # print('MI:', mi_)
            # print('msssim:', ms_ssim_)
            # print('QS:', qs_)
            # print('QG:', qg_)
            # print('QCB:', qcb_)

        print(methed)
        print('SD:', sd/num)
        print('entropy:', en/num)
        print('AG:', ag/num)
        print('sssim:', ssim_v/num)
        print('sf:', sf/num)
        print('MI:', mi/num)
        # print('msssim:', ms_ssim/(num - error))
        print('QS:', qs/(num-error))
        print('QCB:', qcb/num)
        print('QG:', qg/(num-error_qg))
        # pd.DataFrame(result).to_csv('%s.csv' % methed_list[0])

