import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import Wrap
import math
import cv2
import sys
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
import itertools
import pyfftw
from scipy.fft import ifft2, ifftshift
from PIL import Image
from typing import List, Tuple, Union
import decimal


# 論文第六頁內有提到參數的相關設置
def zernikePointSpread(coefficients, spectrum=None, **kwargs):
    # Default values
    wavelength = kwargs.get('Wavelength', 555) # 設定要計算的 PSF 波長(是 in focus 的預設波長)
    pupil = kwargs.get('PupilDiameter', 6) # 係數配合的瞳孔直徑(mm)
    pupilSamples = kwargs.get('PupilSamples', 62.89) # 指瞳孔的直徑，和計算 PSF 的準確性有關
    imagesamples = kwargs.get('ImageSamples', 256) # PSF 的 img 大小
    degrees = kwargs.get('Degrees', 0.5) # 也是指 PSF 的 img 大小(以 degree 為單位)
    # 若有指定 degree 的大小，pupilSamples 就要另外計算
    # 若有指定 pupilSamples 的大小， degree 就要另外計算
    apod = kwargs.get('Apodization', False)
    otfq = kwargs.get('OTF', False)
    verbose = kwargs.get('Verbose', False)
    
    # 看 hackmd 的 Trichromatic PSFs 有說明
    if spectrum is not None:
        if spectrum.ndim == 1:
            defocus = [2, 0, ChromaticDefocusZernike(spectrum[0], wavelength, pupil)]
            coefficients = np.append(coefficients,[defocus],axis=0)
            return zernikePointSpread(coefficients, Wavelength=spectrum[0], Degrees=degrees)
    
    if "Degrees" == "Automatic":
        degrees = PSFDegrees(pupilSamples, wavelength, pupil)
    else:
        pupilSamples = PupilSamples(degrees, wavelength, pupil)        
    
    # ZernikeImage 的第三個參數，簡單來說，數值越大，準確率越高
    radius = pupilSamples/2
    ppd = imagesamples / degrees
    # pai 
    if type(apod)==int or type(apod)==float:
        pai = PupilApertureImage(radius, apod * radius / (pupilSamples/2))
    else:
        pai = PupilApertureImage2(radius)
    
    
    # total wavefront aberration image是加權後的Zernike polynomial images總合，
    wai = WaveAberrationImage(coefficients, radius)
    gp = pai * np.exp(((1j *2 * np.pi * 10**3 / wavelength) * wai))

    w=np.size(gp,0)
    
    if imagesamples > pupilSamples:
        pgp = np.pad(gp, (imagesamples-w,0), mode='constant',constant_values=0)
    else:
        pgp = gp

    # InverseFourier
    psf = np.abs((np.fft.fft2(pgp,norm="ortho")))**2    
    psf /= np.sum(psf)
    
    otf = imagesamples * ((np.fft.ifft2(psf,norm='ortho'))) if otfq == True or otfq == "Both" else []
    
    if verbose:
        plot_verbose_output(pai, wai, gp, psf, pupil, degrees)
    
    if spectrum is not None:
        process_spectrum(coefficients, spectrum, kwargs)

    return otf if otfq == True else psf if otfq == False else [psf, otf]

def plot_verbose_output(pai, wai, gp, psf, pupil, degrees):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    axs[0, 0].imshow(pai, extent=(-pupil/2, pupil/2, -pupil/2, pupil/2))
    axs[0, 0].set_title("Pupil aperture")
    
    axs[0, 1].imshow(wai, extent=(-1, 1, -1, 1))
    axs[0, 1].set_title("Wave aberration")
    
    axs[1, 0].imshow(np.real(gp), extent=(-pupil/2, pupil/2, -pupil/2, pupil/2))
    axs[1, 0].set_title("Generalized pupil")
    
    axs[1, 1].imshow(psf, extent=(-degrees/2, degrees/2, -degrees/2, degrees/2))
    axs[1, 1].set_title("PSF")
    
    plt.tight_layout()
    plt.show()
    
# 計算有 a list of wavelength 跟比例的 PSF
def process_spectrum(coefficients, spectrum, kwargs):
    print("get in RGB PSF")
    result = np.zeros((kwargs.get('ImageSamples', 256), kwargs.get('ImageSamples', 256)))
    if kwargs.get('OTF', False) == "Both":
        result = [result, result]
    for wave, intensity in spectrum:
        defocus = [2, 0, ChromaticDefocusZernike(wave, kwargs.get('Wavelength', 555), kwargs.get('PupilDiameter', 6))]
        new_coeffs = coefficients + [defocus]
        temp_result = zernikePointSpread(new_coeffs, Wavelength=wave, **kwargs)
        if isinstance(result, list):
            result[0] += intensity * temp_result[0]
            result[1] += intensity * temp_result[1]
        else:
            result += intensity * temp_result

    return result

def ChromaticDefocusZernike(wavelength, focuswavelength, pupildiameter=6):
    return InverseEquivalentDefocus(ChromaticDefocus(wavelength) - ChromaticDefocus(focuswavelength), pupildiameter)

def ChromaticDefocus(wavelength):
    p = 1.68524
    q = 0.63346
    c = 0.21410
    return p - q/(wavelength*0.001 - c)

def PupilSamples(Degrees, Wavelength, pupildiameter):
    # 在 mathematica 中的 Degree 的功能即為 角度轉弧度(np.deg2rad)
    return np.deg2rad(10**6 * Degrees * pupildiameter) / Wavelength

def WaveAberrationImage(coefficients=None, radius=16):
    # 先看 ZernikeImage
    size = 2 * int(np.ceil(radius))
    total_image = np.zeros((size, size))
    if coefficients is None or len(coefficients) == 0:
        return total_image
    
    for n, m, c in coefficients:
        # total_image 是二維陣列，把算完的ZernikeImage用係數(c)加權後再加到total_image上
        # 就是不同的二維陣列疊加起來就是 total_image
        # 已確定 ZernikeImage 的第一個參數及第二個參數為數字並非陣列
        z=ZernikeImage(n, m, radius)
        
        total_image = np.add(total_image, c * z)
        
    
       
    return total_image

def PSFDegrees(pupilSamples, Wavelength, pupildiameter):
    return pupilSamples * Wavelength * 180 * (10**(-6)) / (pupildiameter * np.pi)

def PupilApertureImage2(radius):
    h = int(np.ceil(radius))
    # 1j 用來表示虛數的 i
    return np.array([[0 if abs(x + 1j*y) > radius else 1 for x in range(-h, h)] for y in range(-h, h)])

def PupilApertureImage(radius, sd):
    h = int(np.ceil(radius))
    # 1j 用來表示虛數的 i
    return np.array([[0 if abs(x + 1j*y) > radius else np.exp(-0.5 * ((abs(x + 1j*y) / sd)**2)) for x in range(-h, h)] for y in range(-h, h)])

def EquivalentDefocus(coefficient, pupildiameter):
    return (16 * np.sqrt(3) * pupildiameter**-2) * coefficient

# 計算模仿 defocus 的 Zernike 參數
def InverseEquivalentDefocus(diopters,pupildiameter):
    return diopters*pupildiameter**2/(16*np.sqrt(3))

def ZernikeImage(n, m, radius=64):
    h = int(np.ceil(radius))
    
    R=PolarList(radius)
    
    r=R[0][:]
    a=R[1][:]

    # 若有 0 出現在 r 中，用一個極小的數字替代他
    r = np.where(r == 0.0, np.finfo(float).eps, r)
    aperture_image = ApertureImage(radius)
    zernike=[]

    zernike=Zernike(n, m, r, a)
    packed_array = zernike.reshape((2 * h, 2 * h))
    
    return aperture_image*packed_array

def PolarList(radius):
    h = int(np.ceil(radius))
    cartesian_points = []
    packed_array=[]
    for i in range(-h, h):
        for j in range(-h, h):
            x = i / radius
            y = j / radius
            # 將笛卡爾坐標系(就是一般數學用的坐標系)轉為極座標
            r, a = CartesianToPolar((x, y))
            cartesian_points.append([r, a])
        packed_array.append(cartesian_points)
        cartesian_points=[]
    
    packed_array=np.array(packed_array)
    packed_array=np.transpose(packed_array, (1, 0, 2)) # Transpose
    packed_array=np.array(list(itertools.chain.from_iterable(packed_array))) #Flatten
    packed_array=np.transpose(packed_array) # Transpose
    
    return packed_array

def CartesianToPolar(point):
    x, y = point
    # r為半徑
    r = np.linalg.norm([x, y])
    # a為角度
    a = np.arctan2(y, x)
    if x==0 and y==0:
        return 0, 0
    return r, a

def Zernike(n, m, r, a):
    # 這裡的計算以 r a 是 array 計算
    cou=1
    if m<0:
        cou=-1.0
    else:
        cou=1.0
        
    n=int(n)
    nz=NZ(n, m)
    rz= np.array(RZ(n, m, r))
    az=np.array(AZ(n, m, a))

    # NZ RZ AZ 沒錯
    # zernike_value 沒錯(格式根長度都是一樣的)
    zernike_value = cou * NZ(n, m) * rz * az
    zernike_value = np.where(r>1, 0, zernike_value)
    
    return zernike_value
    
def ApertureImage(radius):
    h = int(np.ceil(radius))
    image = np.zeros((2 * h, 2 * h))
    for i in range(-h, h):
        for j in range(-h, h):
            if np.linalg.norm([i, j]) > radius:
                image[i + h, j + h] = 0
            else:
                image[i + h, j + h] = 1
    return image

def RZ(n, m, r):
    if m<0:
        m=-m
    if (type(r)==int and r>1) or (n-m)%2:
        return 0
    if m==0 and (type(r)==int and r==0):
        return (-1)**(n/2)
    
    result = []
    for ri in r:
        sum_value = sum(
            (-1)**s * math.factorial(int(n - s)) * ri**(int(n - 2 * s)) / (
                math.factorial(s) * 
                math.factorial(int((n + m) // 2) - s) * 
                math.factorial(int((n - m) // 2) - s)
            ) for s in range(int((n - m) // 2) + 1)
        )
        result.append(sum_value)
    return result

def AZ(n, m, a):
    result=[]
    for ai in a:
        if m < 0:
            result.append(math.sin(m * ai))
        else:
            result.append(math.cos(m * ai))
    return result
    
def NZ(n, m):
    if m == 0:
        return math.sqrt(2 * (n + 1)/2)
    else:
        return math.sqrt(2 * (n + 1))
    
def SpheroCylindricalCoefficients(defocus, astigmatism, angle, pupil):
    num=np.zeros([3])
    num[0]=astigmatism*math.sin(2*angle)*pupil**2/(8*np.sqrt(6))
    num[1]=defocus*pupil**2/(16*np.sqrt(3))
    num[2]=astigmatism*math.cos(2*angle)*pupil**2/(8*np.sqrt(6))
    return num
    
# 畫出PSF
def PSFPlot(psf, Degrees=0.5, Magnification=1, ScaleMark=True, **kwargs):
    dim = psf.shape
    newdim = 2 * np.ceil(np.array(dim) / Magnification / 2).astype(int)
    newmag = dim[0] / newdim[0]
    newpsf = np.fft.fftshift(psf)[:newdim[0], :newdim[1]]
    newDegrees = Degrees / newmag
    
    fig, ax = plt.subplots()
    im = ax.imshow(1 - newpsf / np.max(newpsf), 
                   extent=[-30*newDegrees, 30*newDegrees, -30*newDegrees, 30*newDegrees],
                   cmap='gray')
    
    ax.set_xlabel('arcmin')
    ax.set_ylabel('arcmin')
    
    if ScaleMark:
        mark = 5  # 5 arcmin scale mark
        markx = mark / 2
        marky = -0.95 * newDegrees * 30
        ax.plot([-markx, markx], [marky, marky], 'k-')
    
    plt.tight_layout()
    return fig

# np.set_printoptions(threshold=np.inf)
# zc=np.array([[2,-2,-0.094629],[2,0,0.096927],[2,2,0.30527],[3,-3,0.045947],
#              [3,-1,-0.12144],[3,1,0.026396],[3,3,-0.11346],[4,-4,0.029154],
#              [4,-2,0.030043],[4,0,0.029426],[4,2,0.016292],[4,4,0.063988],
#              [5,-5,0.049899],[5,-3,-0.025238],[5,-1,0.0074414],[5,1,0.0015506],
#              [5,3,-0.0068646],[5,5,0.028846],[6,-6,0.0024519],[6,-4,0.0018495],
#              [6,-2,0.0012222],[6,0,-0.0075453],[6,2,-0.00069273],[6,4,0.00055051],
#              [6,6,-0.014835]    
#              ])

# [5,-5,0.0499],[5,-3,-0.0252],[5,-1,0.00744],[5,1,0.00155],
# [5,3,-0.00686],[5,5,0.0288],[6,-6,0.00245],[6,-4,0.00185],
# [6,-2,0.00122],[6,0,-0.00755],[6,2,-0.000693],[6,4,0.000551],
# [6,6,-0.0148]
# [5,-5,0.049899],[5,-3,-0.025238],[5,-1,0.0074414],[5,1,0.0015506],
#              [5,3,-0.0068646],[5,5,0.028846],[6,-6,0.0024519],[6,-4,0.0018495],
#              [6,-2,0.0012222],[6,0,-0.0075453],[6,2,-0.00069273],[6,4,0.00055051],
#              [6,6,-0.014835]
# [[2,-2,-0.094629],[2,0,0.096927],[2,2,0.30527],[3,-3,0.045947],
#              [3,-1,-0.12144],[3,1,0.026396],[3,3,-0.11346],[4,-4,0.029154],
#              [4,-2,0.030043],[4,0,0.029426],[4,2,0.016292],[4,4,0.063988],
#              [5,-5,0.049899],[5,-3,-0.025238],[5,-1,0.0074414],[5,1,0.0015506],
#              [5,3,-0.0068646],[5,5,0.028846],[6,-6,0.0024519],[6,-4,0.0018495],
#              [6,-2,0.0012222],[6,0,-0.0075453],[6,2,-0.00069273],[6,4,0.00055051],
#              [6,6,-0.014835]    
#              ]

# 文章上方對 E 字做模糊的係數
# zc=np.array([[2, -2, -0.0946], [2, 0, 0.0969], [2, 2, 0.305], [3, -3, 
# 0.0459], [3, -1, -0.121], [3, 1, 0.0264], [3, 3, -0.113], [4, 
# -4, 0.0292], [4, -2, 0.03], [4, 0, 0.0294], [4, 2, 0.0163], 
# [4, 4, 0.064]])

# zc=np.array([[2, -2, -0.094629], [2, 0, 0.096927], [2, 2, 0.30527], [3, -3, 
# 0.045947], [3, -1, -0.12144], [3, 1, 0.026396], [3, 3, -0.11346], [4, 
# -4, 0.029154], [4, -2, 0.030043], [4, 0, 0.029426], [4, 2, 0.016292], 
# [4, 4, 0.063988]])

# TestCoefficients
zc=np.array([[2,-2,-0.0946],[2,0,0.0969],[2,2,0.305],[3,-3,0.0459],
             [3,-1,-0.121],[3,1,0.0264],[3,3,-0.113],[4,-4,0.0292],
             [4,-2,0.03],[4,0,0.0294],[4,2,0.0163],[4,4,0.064],
             [5,-5,0.0499],[5,-3,-0.0252],[5,-1,0.00744],[5,1,0.00155],
             [5,3,-0.00686],[5,5,0.0288],[6,-6,0.00245],[6,-4,0.00185],
             [6,-2,0.00122],[6,0,-0.00755],[6,2,-0.000693],[6,4,0.000551],
             [6,6,-0.0148]])

zc_pupil_3 = [[0, 0, -0.100302], [1, -1, 0.137265], [1, 
  1, -0.0262347], [2, -2, -0.0441613], [2, 0, -0.00524091], [2, 2, 
  0.0637425], [3, -3, 0.0173347], [3, -1, -0.0185977], [3, 1, 
  0.00258734], [3, 3, -0.0110297], [4, -4, 0.00130923], [4, -2, 
  0.00153875], [4, 0, 0.00393156], [4, 2, 0.00121036], [4, 4, 
  0.00384658], [5, -5, 0.00155934], [5, -3, -0.000788688], [5, -1, 
  0.000232544], [5, 1, 0.0000484563], [5, 3, -0.000214519], [5, 5, 
  0.000901438], [6, -6, 0.0000383109], [6, -4, 0.0000288984], [6, -2, 
  0.0000190969], [6, 0, -0.000117895], [6, 2, -0.0000108239], [6, 4, 
  8.60172*10**(-6)], [6, 6, -0.000231797]]

zc_pupil_4 = [[0, 0, -0.0918079], [1, -1, 0.130929], [1, 
  1, -0.0268779], [2, -2, -0.0698653], [2, 0, 0.00924714], [2, 2, 
  0.119573], [3, -3, 0.0339663], [3, -1, -0.0419831], [3, 1, 
  0.00657061], [3, 3, -0.0280821], [4, -4, 0.00455807], [4, -2, 
  0.00514093], [4, 0, 0.0107112], [4, 2, 0.00366791], [4, 4, 
  0.0122822], [5, -5, 0.00657106], [5, -3, -0.00332352], [5, -1, 
  0.000979937], [5, 1, 0.000204194], [5, 3, -0.00090398], [5, 5, 
  0.00379865], [6, -6, 0.000215256], [6, -4, 0.00016237], [6, -2, 
  0.000107299], [6, 0, -0.000662413], [6, 2, -0.0000608158], [6, 4, 
  0.0000483301], [6, 6, -0.00130239]]

# 不同的波長在 RGB 這三種顏色裡面各自所佔的比例(所以總合為 1 )
spectra_R = np.array([[535,0.00986],[555,0.088],[575,0.163],[595,0.19],[615,0.176],[635,0.143],
                    [655,0.106],[675,0.0741],[695,0.0493]])
spectra_G = np.array([[455,0.00477],[475,0.0727],[495,0.175],[515,0.22],[535,0.198],[555,0.143],
                    [575,0.0896],[595,0.0502],[615,0.0258],[635,0.0123],[655,0.00558],[675,0.0024],[695,0.000991]])
spectra_B = np.array([[415,0.000458],[435, 0.0503],[455, 0.157],[475, 0.217],[495,0.204],[515,0.154],
                    [535,0.0994],[555,0.0578],[575,0.031],[595,0.0156],[615,0.00744],[635,0.0034],[655,0.0015],[675,0.000641],[695,0.000266]])

# 畫RGB的波長占比圖
# plt.plot(spectra_R[:,0], spectra_R[:,1], 'r-')  # 紅色
# plt.plot(spectra_G[:,0], spectra_G[:,1], 'g-')  # 綠色
# plt.plot(spectra_B[:,0], spectra_B[:,1], 'b-')  # 藍色
# plt.show()

# 多 wavelength 測試(Polychromatic PSF)
# spectrum = np.array([[455,0.00477],[475,0.0727],[495,0.175],[515,0.22],[535,0.198],[555,0.143],[575,0.0896],
#             [595,0.0502],[615,0.0258],[635,0.0123],[655,0.00558],[675,0.0024]])

# Defocus 測試(近視遠視)
# diopter=1.0 # 設定近視(遠視)度數

########### 計算 RGB 的模擬 #######################

# # 有考慮到色差的
# p = 1.68524
# q = 0.63346
# c = 0.21410
# wavelength=589 # 預設正常是對焦在 555
# # 若要模擬不同近視(遠視)度數，就用此計算出在近視(遠視)為 diopter 度時，眼睛是對焦在哪種波長
# # focus_wavelength=(q*(wavelength*0.001-c)/(diopter*(wavelength*0.001-c)+q) + c )/0.001
# focus_wavelength=555
# # 要顯示不同波長的波長設定
# # wavelength=np.array([555])
# # psf=zernikePointSpread(zc, np.array([spectrum[0][0]]))*spectrum[0][1]

# # degree 設定大一點，才可以包含住比較大的PSF
# degree=0.5
# # 求 RGB 各自的 PSF 總和
# psf_r=zernikePointSpread(zc, spectrum=np.array([spectra_R[0][0]]), Degrees=degree, Wavelength=focus_wavelength)*spectra_R[0][1]
# for wavelength in spectra_R[1:]:
#     psf_r+=zernikePointSpread(zc, spectrum=np.array([wavelength[0]]), Degrees=degree, Wavelength=focus_wavelength)*wavelength[1]
    
# psf_g=zernikePointSpread(zc, np.array([spectra_G[0][0]]),Degrees=degree, Wavelength=focus_wavelength)*spectra_G[0][1]
# for wavelength in spectra_G[1:]:
#     psf_g+=zernikePointSpread(zc, np.array([wavelength[0]]),Degrees=degree, Wavelength=focus_wavelength)*wavelength[1]
    
# psf_b=zernikePointSpread(zc, np.array([spectra_B[0][0]]),Degrees=degree, Wavelength=focus_wavelength)*spectra_B[0][1]
# for wavelength in spectra_B[1:]:
#     psf_b+=zernikePointSpread(zc, np.array([wavelength[0]]),Degrees=degree, Wavelength=focus_wavelength)*wavelength[1]

# psf_img = PSFPlot(psf=psf_r, Degrees=degree)
# plt.show()
# psf_img = PSFPlot(psf=psf_g, Degrees=degree)
# plt.show()
# psf_img = PSFPlot(psf=psf_b, Degrees=degree)
# plt.show()

# # psf_r=psf
# # psf_g=psf
# # psf_b=psf

# # 讀取要處理的圖片
# img_bgr=cv2.imread("C:\\xampp\\htdocs\\Visual-inspection\\PSF\\color_img.png")
# img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

# # 圖片左右翻轉(因為文章中的 Basis 的 Image 有提到，卷積是從圖片的底部開始做的)
# # 不知道為啥是左右翻轉
# img_rgb_flip=cv2.flip(img_rgb, 1)

# # 對圖片做處理
# # 先將 RGB 三通道分開
# R,G,B = cv2.split(img_rgb_flip)
# R = cv2.filter2D(src=R,ddepth=-1,kernel=Wrap.wrap(psf_r))
# G = cv2.filter2D(src=G,ddepth=-1,kernel=Wrap.wrap(psf_g))
# B = cv2.filter2D(src=B,ddepth=-1,kernel=Wrap.wrap(psf_b))
# img_blur = cv2.merge([R,G,B])

# # 把圖片翻回來
# img_blur=cv2.flip(img_blur,1)
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.title("Original")
# plt.imshow(img_rgb)
# plt.subplot(1, 2, 2)
# plt.title("Blurred")
# plt.imshow(img_blur)
# plt.show()

########## 單純計算單一波長的 PSF #######################

# degree 設定大一點，才可以包含住比較大的PSF
degree=0.5

diopter=1.0 # 設定近視-(遠視+)度數

psf=np.zeros([256,256,20])

# 讀取要處理的圖片
img_gray=cv2.imread("C:\\xampp\\htdocs\\Visual-inspection\\PSF\\letter_e.png")

# 圖片左右翻轉(因為文章中的 Basis 的 Image 有提到，卷積是從圖片的底部開始做的)
# 不知道為啥是左右翻轉
img_gray_flip=cv2.flip(img_gray, 1)
psf_num=0
# 在 zernike 參數內加 Defocus，計算近視(遠視)
for i in np.arange(-2,2.25,0.25):
    print(psf_num, i)
    # Defocus 計算近視(遠視)係數
    Defocus=InverseEquivalentDefocus(diopters=i,pupildiameter=4)
    zc_pupil_4_defocus=np.append(zc_pupil_4,[[2,0,Defocus]],axis=0)
    # 計算PSF(模糊 kernel)
    psf[:,:,psf_num]=zernikePointSpread(zc_pupil_4_defocus, Degrees=degree, PupilDiameter=4)
    img_blur=cv2.filter2D(src=img_gray_flip, ddepth=-1, kernel=Wrap.wrap(psf[:,:,psf_num]))
    img_blur=cv2.flip(img_blur,1) # 把圖片翻回來
    img_num=i*100
    s=str(img_num)+'_img.png'
    cv2.imwrite(s, img_blur)  # 存成 png
    psf_num+=1
        

# 輸出 PSF 圖片
# for i in range(9):
#     plt.title(i/2-2)
#     psf_img = PSFPlot(psf=psf[:,:,i], Degrees=degree)
#     plt.show()


plt.figure()
# 對圖片做處理

img_blur_n2= cv2.filter2D(src=img_gray_flip, ddepth=-1, kernel=Wrap.wrap(psf[:,:,0]))
img_blur_n1_5= cv2.filter2D(src=img_gray_flip, ddepth=-1, kernel=Wrap.wrap(psf[:,:,1]))
img_blur_n1= cv2.filter2D(src=img_gray_flip, ddepth=-1, kernel=Wrap.wrap(psf[:,:,2]))
img_blur_n0_5= cv2.filter2D(src=img_gray_flip, ddepth=-1, kernel=Wrap.wrap(psf[:,:,3]))
img_blur_0= cv2.filter2D(src=img_gray_flip, ddepth=-1, kernel=Wrap.wrap(psf[:,:,4]))
img_blur_0_5= cv2.filter2D(src=img_gray_flip, ddepth=-1, kernel=Wrap.wrap(psf[:,:,5]))
img_blur_1= cv2.filter2D(src=img_gray_flip, ddepth=-1, kernel=Wrap.wrap(psf[:,:,6]))
img_blur_1_5= cv2.filter2D(src=img_gray_flip, ddepth=-1, kernel=Wrap.wrap(psf[:,:,7]))
img_blur_2= cv2.filter2D(src=img_gray_flip, ddepth=-1, kernel=Wrap.wrap(psf[:,:,8]))

plt.subplot(2, 5, 1)
plt.title(-2)
img_blur_n2=cv2.flip(img_blur_n2,1) # 把圖片翻回來
plt.imshow(img_blur_n2)

plt.subplot(2, 5, 2)
plt.title(-1.5)
img_blur_n1_5=cv2.flip(img_blur_n1_5,1) # 把圖片翻回來
plt.imshow(img_blur_n1_5)

plt.subplot(2, 5, 3)
plt.title(-1)
img_blur_n1=cv2.flip(img_blur_n1,1) # 把圖片翻回來
plt.imshow(img_blur_n1)

plt.subplot(2, 5, 4)
plt.title(-0.5)
img_blur_n0_5=cv2.flip(img_blur_n0_5,1) # 把圖片翻回來
plt.imshow(img_blur_n0_5)

plt.subplot(2, 5, 5)
plt.title("normal")
img_blur_0=cv2.flip(img_blur_0,1) # 把圖片翻回來
plt.imshow(img_blur_0)

plt.subplot(2, 5, 6)
plt.title(0.5)
img_blur_0_5=cv2.flip(img_blur_0_5,1) # 把圖片翻回來
plt.imshow(img_blur_0_5)

plt.subplot(2, 5, 7)
plt.title(1)
img_blur_1=cv2.flip(img_blur_1,1) # 把圖片翻回來
plt.imshow(img_blur_1)

plt.subplot(2, 5, 8)
plt.title(1.5)
img_blur_1_5=cv2.flip(img_blur_1_5,1) # 把圖片翻回來
plt.imshow(img_blur_1_5)

plt.subplot(2, 5, 9)
plt.title(2)
img_blur_2=cv2.flip(img_blur_2,1) # 把圖片翻回來
plt.imshow(img_blur_2)
plt.show()

########### 計算散光加 defocus #########################
# diopter=0.5 # 設定近視-(遠視+)度數
# degree=0.5 #設定 PSF 大小
# astigmatism=0.2 # 散光度數
# angle=np.pi #散光角度(以弧度為單位 in radians)
# zc = SpheroCylindricalCoefficients(defocus=diopter, astigmatism=astigmatism, angle=angle, pupil=6)  
# zc=[[2,-2,zc[0]],[2,0,zc[1]],[2,2,zc[2]]]
# print(zc)
# psf=zernikePointSpread(coefficients=zc, Degrees=degree, PupilDiameter=6)
# psf_img = PSFPlot(psf=psf, Degrees=degree)
# plt.show()

# # 讀取要處理的圖片
# img_gray=cv2.imread("C:\\xampp\\htdocs\\Visual-inspection\\PSF\\letter_e.png")

# # 圖片左右翻轉(因為文章中的 Basis 的 Image 有提到，卷積是從圖片的底部開始做的)
# # 不知道為啥是左右翻轉
# img_gray_flip=cv2.flip(img_gray, 1)

# plt.figure()
# img_blur= cv2.filter2D(src=img_gray_flip, ddepth=-1, kernel=Wrap.wrap(psf))

# plt.subplot(1, 2, 1)
# plt.title("Blur")
# img_blur=cv2.flip(img_blur,1) # 把圖片翻回來
# plt.imshow(img_blur)

# plt.subplot(1, 2, 2)
# plt.title("Original")
# plt.imshow(img_gray)
# plt.show()