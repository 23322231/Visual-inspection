import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import Wrap
import math
import cv2
import sys

# 論文第六頁內有提到參數的相關設置
def ZernikePointSpread(coefficients, Wavelength=555, PupilDiameter=6, pupilSamples=62.89, ImageSamples=256, **kwargs):
    otfq = kwargs.get("OTF", False)
    apod = kwargs.get("Apodization", False)
    verbose = kwargs.get("Verbose", False)
    Degrees = kwargs.get("Degrees", 0.5)
    print("Wavelength",Wavelength)
    # test用，非正式用
    # PupilSamples=126.2
    print("ImageSamples =",ImageSamples)
    if "Degrees" not in kwargs.keys():
        Degrees = PSFDegrees(pupilSamples, Wavelength, PupilDiameter)
    else:
        print("!!!!!!!!!!!!!!!!!!")
        pupilSamples = PupilSamples(Degrees, Wavelength, PupilDiameter)
    
    # ZernikeImage 的第三個參數，簡單來說，數值越大，準確率越高
    radius = pupilSamples/2
    ppd = ImageSamples / Degrees
    # pai 有問題(6/15 看了感覺應該沒問題)
    print("type(apod) =",type(apod))
    if type(apod)==int or type(apod)==float:
        pai = PupilApertureImage(radius, apod * radius / (pupilSamples/2))
    else:
        pai = PupilApertureImage2(radius)
    

    # total wavefront aberration image是加權後的Zernike polynomial images總合，
    wai = WaveAberrationImage(coefficients, radius)

    gp = pai * np.exp((1j * 2 * np.pi * 10**3 / Wavelength) * wai)
    w=np.size(gp,0)
    
    # 這裡會報錯(ok!)
    if ImageSamples > pupilSamples:
        pgp = np.pad(gp, (ImageSamples-w,0), mode='constant',constant_values=0)
    else:
        pgp = gp
    # mathematica 的輸出
    # img = 256
    # img [1, 1]
    # [256, 256]
    # 從這裡開始看
    psf = np.abs((np.fft.ifft2(pgp,norm='ortho')))**2
    psf /= np.sum(psf)
    # psf = np.abs(pgp)**2
    # psf /= np.sum(psf)
    # psf = np.fft.fftshift(psf)
    
    otf = ImageSamples * ((np.fft.ifft2(psf,norm='ortho'))) if otfq == True or otfq == "Both" else []
    
    if verbose:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(pai, extent=(-PupilDiameter/2, PupilDiameter/2, -PupilDiameter/2, PupilDiameter/2))
        plt.title("Pupil aperture")
        
        plt.subplot(2, 2, 2)
        plt.imshow(wai, extent=(-1, 1, -1, 1))
        plt.title("Wave aberration")
        
        plt.subplot(2, 2, 3)
        plt.imshow(np.real(pgp), extent=(-PupilDiameter/2, PupilDiameter/2, -PupilDiameter/2, PupilDiameter/2))
        plt.title("Generalized pupil")
        
        plt.subplot(2, 2, 4)
        plt.imshow(psf, extent=(-Degrees/2, Degrees/2, -Degrees/2, Degrees/2))
        plt.title("PSF")
        
        plt.show()

    return otf if otfq == True else psf if otfq == False else [psf, otf]
def PupilSamples(Degrees, Wavelength, pupildiameter):
    # 在 mathematica 中的 Degree 的功能即為 角度轉弧度(np.deg2rad)
    return np.deg2rad(10**6 * Degrees * pupildiameter) / Wavelength

def WaveAberrationImage(coefficients=None, radius=16):
    # 先看 ZernikeImage
    size = 2 * int(np.ceil(radius))
    total_image = np.zeros((size, size))
    if coefficients is None or len(coefficients) == 0:
        return total_image
    
    # print("radius = ",radius)
    for n, m, c in coefficients:
        # total_image 是二維陣列，把算完的ZernikeImage用係數(c)加權後再加到total_image上
        # 就是不同的二維陣列疊加起來就是 total_image
        total_image += c * ZernikeImage(n, m, radius)
        # print("total_image =",total_image)
    # np.set_printoptions(threshold=np.inf,linewidth=200)
    # print("total_image =",total_image)
    return total_image

def PSFDegrees(pupilSamples, Wavelength, pupildiameter):
    return pupilSamples * Wavelength * 180 * (10**(-6)) / (pupildiameter * np.pi)

def PupilApertureImage2(radius):
    h = int(np.ceil(radius))
    # print("h =",h)
    # print("Table =", np.array([[0 if abs(x + 1j*y) > radius else 1 for x in range(-h, h)] for y in range(-h, h)]))
    # 1j 用來表示虛數的 i
    return np.array([[0 if abs(x + 1j*y) > radius else 1 for x in range(-h, h)] for y in range(-h, h)])

def PupilApertureImage(radius, sd):
    h = int(np.ceil(radius))
    # 1j 用來表示虛數的 i
    return np.array([[0 if abs(x + 1j*y) > radius else np.exp(-0.5 * ((abs(x + 1j*y) / sd)**2)) for x in range(-h, h)] for y in range(-h, h)])

def EquivalentDefocus(coefficient, pupildiameter):
    return (16 * np.sqrt(3) * pupildiameter**-2) * coefficient

def ZernikeImage(n, m, radius=64):
    h = int(np.ceil(radius))
    # print(type(PolarList(radius)))
    R = PolarList(radius)
    r=R[:,0]
    a=R[:,1]
    # print(a)
    r = np.where(r == 0., np.finfo(float).eps, r)
    # print(r)
    aperture_image = ApertureImage(radius)
    # n, m=2.0,-2.0
    zernike=[]
    # print(np.size(r))
    for i in range(np.size(r)):
        zernike.append(Zernike(n, m, r[i], a[i]))
    zernike=np.array(zernike)
    # print("zernike =",zernike.shape)
    # print("2h =",h*2)
    # print("aperture_image =",aperture_image.shape)
    packed_array = zernike.reshape((2 * h, 2 * h))
    # print("aperture_image*packed_array =",aperture_image*packed_array)
    return aperture_image*packed_array

def PolarList(radius):
    h = int(np.ceil(radius))
    cartesian_points = []
    for i in range(-h, h):
        for j in range(-h, h):
            x = i / radius
            y = j / radius
            r, a = CartesianToPolar((x, y))
            cartesian_points.append([r, a])
            
    # print("cartesian_points =",cartesian_points)
    packed_array = np.array(cartesian_points)
    # packed_array_T = np.transpose(packed_array)
    # packed_array_flatten=packed_array_T.flatten()
    # print("packed_array_flatten =",packed_array_flatten)
    # packed_array_T=np.transpose(packed_array_flatten)
    # print("packed_array =",packed_array)
    # print('packed_array_T.shape =',packed_array_T.shape)
    return packed_array

def Zernike(n, m, r, a):
    # 這裡的計算感覺是以 r a 是一個值的計算，但他們是兩個陣列，所以要改一下計算方式，回傳一陣列
    zernike_value = np.where(r > 1, 0, np.where(m < 0, -1, 1) * NZ(n, m) * RZ(n, m, r) * AZ(n, m, a))
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
    np.set_printoptions(threshold=np.inf,linewidth=200)
    # print(h)
    # print("image =",image)
    return image

def CartesianToPolar(point):
    x, y = point
    r = np.linalg.norm([x, y])
    a = np.arctan2(y, x)
    return r, a

def RZ(n, m, r):
    if m<0:
        m=-m
    # print("r =",r)
    if r>1 or (n-m)%2:
        return 0
    if m==0 and r==0:
        return (-1)**(n/2)
    
    result=[]
    for s in np.arange(0, (n - m) // 2 + 1):            
        result.append(calculate_RZ(n, m, s, r))
    return np.sum(result)

def calculate_RZ(n, m, s, r):
    # print(n - m)
    # 因為計算太長，加一個自己加的 function
    result = (-1) ** s * np.math.factorial(int(n - s)) * r ** (n - 2 * s) / (
        np.math.factorial(int(s)) * np.math.factorial(int((n + m) // 2 - s)) * 
        np.math.factorial(int((n - m) // 2 - s))
    )

    return result

def AZ(n, m, a):
    if m < 0:
        return math.sin(m * a)
    else:
        return math.cos(m * a)
    
def NZ(n, m):
    if m == 0:
        return math.sqrt(2 * (n + 1)/2)
    else:
        return math.sqrt(2 * (n + 1))
    
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


np.set_printoptions(threshold=np.inf)
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
# zc=np.array([[2,-2,-0.0946],[2,0,0.0969],[2,2,0.305],[3,-3,0.0459],
#              [3,-1,-0.121],[3,1,0.0264],[3,3,-0.113],[4,-4,0.0292],
#              [4,-2,0.03],[4,0,0.0294],[4,2,0.0163],[4,4,0.064],
#              ])
# TestCoefficients
zc=np.array([[2,-2,-0.0946],[2,0,0.0969],[2,2,0.305],[3,-3,0.0459],
             [3,-1,-0.121],[3,1,0.0264],[3,3,-0.113],[4,-4,0.0292],
             [4,-2,0.03],[4,0,0.0294],[4,2,0.0163],[4,4,0.064],
             [5,-5,0.0499],[5,-3,-0.0252],[5,-1,0.00744],[5,1,0.00155],
             [5,3,-0.00686],[5,5,0.0288],[6,-6,0.00245],[6,-4,0.00185],
             [6,-2,0.00122],[6,0,-0.00755],[6,2,-0.000693],[6,4,0.000551],
             [6,6,-0.0148]
             ])


psf=ZernikePointSpread(zc)
psf = np.rot90(psf,axes=(1,0))
psf_img = PSFPlot(psf=psf)
plt.show()
# letter=cv2.imread("C:\\xampp\\htdocs\\Visual-inspection\\PSF\\letter_z.png")
# blurredImg=cv2.filter2D(src=letter,ddepth=-1,kernel=Wrap.wrap(psf))
# cv2.imshow('Blurred Img',blurredImg)
# cv2.waitKey(0)

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Original")
# plt.imshow(letter, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.title("Blurred")
# plt.imshow(blurredImg, cmap='gray')
# plt.show()