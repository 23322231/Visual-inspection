import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import Wrap
import math
import cv2
import sys
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

code="""
Options[ZernikePointSpread] = {Verbose -> False, Wavelength -> 555, 
   PupilDiameter -> 6, OTF -> False, PupilSamples -> Automatic , 
   ImageSamples -> 256, Degrees -> 0.5
   , Apodization -> False, 
   DerivedParameters -> {Degrees -> 0, PupilSamples -> 0}};
ZernikePointSpread[coefficients_, opts___Rule] := Module[
  {verbose, wavelength, pupil, otfq, psf, otf, ppd, pai, wai, gp, pgp,
    pupilsamples, imagesamples, radius, degrees, apod},
  {verbose, wavelength, pupil, otfq, pupilsamples, imagesamples, 
    degrees, 
    apod} = {Verbose, Wavelength, PupilDiameter, OTF, PupilSamples, 
      ImageSamples, Degrees, Apodization} /. {opts} /. 
    Options[ZernikePointSpread];
  
  If[degrees === Automatic
   , (degrees = N@PSFDegrees[ pupilsamples, wavelength, pupil];
    SetOptions[ZernikePointSpread, 
     DerivedParameters -> {Degrees -> degrees}])
   , (pupilsamples = N@PupilSamples[degrees, wavelength, pupil];
    SetOptions[ZernikePointSpread, 
     DerivedParameters -> {PupilSamples -> pupilsamples }])];
  
  radius = pupilsamples/2;
  ppd = imagesamples/degrees // N;
  pai = PupilApertureImage @@ 
    If[NumberQ[apod], {radius, apod radius/(pupil/2)}, {radius}];
  
  wai = WaveAberrationImage[coefficients, radius];
  gp = pai Exp[(I 2. Pi 10^3/wavelength) wai];
  pgp = If[imagesamples > pupilsamples, 
    PadLeft[gp, imagesamples {1, 1}, 0], gp];
  psf = Abs[InverseFourier[pgp]]^2;
  psf = psf/Total[psf, 2];
  otf = If[otfq == True || otfq === "Both", 
    imagesamples InverseFourier[psf], {}];
  
  If[verbose, (
    Print @ OpenerView[{"Images", TabView[{
         ShowArray[ pai, XYRanges -> pupil/2 {{-1, 1}, {-1, 1}}, 
          PlotLabel -> "Pupil aperture", ImageSize -> 240]
         , 
         ShowArray[ wai, XYRanges -> {{-1, 1}, {-1, 1}}, 
          PlotLabel -> "Wave aberration", ImageSize -> 240]
         , 
         Labeled[Row[
           Image[#, ImageSize -> 120] & /@ ({Re@#, Im@#} &[gp])], 
          "Generalized pupil", Top]
         , 
         Labeled[Image[Re @ pgp , Real, ImageSize -> 240], 
          "Padded generalized pupil" , Top],
         MTFPlot[otf, Degrees -> degrees, PlotLabel -> "MTF"],
         PSFPlot[psf, Degrees -> degrees, PlotLabel -> "PSF"]
         }]}];
    Print @ OpenerView[{"Statistics", StringJoin @@ ToString /@ {
          "Scale = ", 2 *radius/pupil // N, " pixels/mm."
          , "\npupil samples = ", pupilsamples
          , "\nPSF image dimensions = ", Dimensions[pgp]
          , "\nEquivalent Defocus ", 
          EquivalentDefocus[coefficients, pupil] // N, " diopters"
          , "\nPSF image size = ", degrees // N, " degrees"
          , "\nPSF image resolution = ", ppd , " pixels/deg"
          , "\nPSF image nyquist = ", ppd /2, " pixels/deg"
          , "\nOTF image resolution = ", 1/degrees, " cycles/deg/pixel"
          , "\nOTF image nyquist = ", imagesamples/degrees/2, 
          " cycles/deg"}}];
    )];
  Switch[otfq, True, otf, False, psf, "Both", {psf, otf}]
  ]
  
Options[PSFPlot] = {Degrees -> 0.5, Magnification -> 1, 
   ScaleMark -> True};
   
PSFPlot[psf_, opts___Rule] := Module[
  {scalemark, magnification, degrees, dim, newpsf, newdim, newmag, 
   newdegrees, mark = 5, markx, marky},
  {scalemark, magnification, 
    degrees} = {ScaleMark, Magnification, Degrees} /. {opts} /. 
    Options[PSFPlot];
  dim = Dimensions[psf];
  newdim = 2 Ceiling[ dim/magnification/2];
  newmag = (dim/newdim)[[1]];
  newpsf = Wrap[TakeWrapped[psf, newdim]];
  newdegrees = degrees/newmag;
  markx = mark/2; marky = -.95 newdegrees 60/2;
  Graphics[Raster[1 - newpsf/Max[newpsf]
    , Transpose[(60 newdegrees {{-1, 1}, {-1, 1}}/2)]]
   , FilterRules[{opts}, Options[Graphics]]
   , Frame -> True
   , FrameLabel -> {"arcmin", "arcmin"}
   , Axes -> False
   , FrameTicks -> {{Automatic, None}, {Automatic, None}}
   , Epilog -> 
    If[scalemark, {Line[{{-markx, marky}, {markx, marky}}]}, {}]
   , ZernikeStyles]
  ]

NZ[n_, m_] :=  Sqrt[2 (n + 1)/(1 + If[m == 0, 1, 0])]

AZ[n_, m_, a_] := If[m < 0, Sin[m a], Cos[m a]]

RZ[n_, m_, r_] := 0 /; r > 1
RZ[n_, m_, r_] := RZ[n, -m, r] /; m < 0
RZ[n_, m_, r_] := 0  /; OddQ[n - m]
RZ[n_, 0, 0] := (-1)^(n/2) 
RZ[n_, 0, 0.] := (-1)^(n/2) 
RZ[n_, m_, r_] := 
 Sum[(-1)^s (n - 
      s)! r^(n - 2 s) / (s! ((n + m)/2 - s)! ((n - m)/2 - s)!), {s, 
   0, (n - m)/2}] 

CartesianToPolar[{0, 0}] := {0, 0}
CartesianToPolar[{0, 0.}] := {0, 0}
CartesianToPolar[{0., 0}] := {0, 0}
CartesianToPolar[{0., 0.}] := {0, 0}
CartesianToPolar[{x_, y_}] := {Norm[{x, y}], ArcTan[x, y]}

ApertureImage[radius_] := ApertureImage[radius] = Block[{h},
   h = Ceiling[radius];
   Array[If[Norm[N[{##}]] > radius, 0, 1] &, 2 h {1, 1}, {-h, -h}]]
   
Zernike[n_, m_, r_, a_] := 0 /; r > 1

Zernike[n_, m_, r_, a_] := 
 If[m < 0, -1, 1] NZ[n, m] RZ[n, m, r] AZ[n, m, a]

PolarList[radius_] := PolarList[radius] = Block[{h},
   h = Ceiling[radius];
   ToPackedArray[
    N[Transpose[
      Flatten[Transpose[
        Array[CartesianToPolar[{##}/radius] &, 2 {h, h}, {-h, -h}]], 
       1]]]]]

ZernikeImage[n_, m_, radius_] := ZernikeImage[n, m, radius] = Block[
   {h, r, a},
   h = Ceiling[radius];
   {r, a} = PolarList[radius];
   r = r /. {0. -> $MachineEpsilon};
   ToPackedArray[
    ApertureImage[radius] Partition[ Zernike[n, m, r, a], 2 h]]]
    
EquivalentDefocus[coefficients_List, 
  pupildiameter_] := (16 Sqrt[3] pupildiameter^-2) WavefrontRMS[
   coefficients]
   
EquivalentDefocus[coefficient_, 
  pupildiameter_] := (16 Sqrt[3] pupildiameter^-2) coefficient
  
PupilApertureImage[radius_] := Module[{h},
  h = Ceiling[radius];
  Table[If[Abs[x + I y] > radius, 0, 1], {y, -h, h - 1}, {x, -h, 
    h - 1}]]
    
PupilApertureImage[radius_, sd_] := Module[{h, r},
  h = Ceiling[radius];
  Table[If[(r = Abs[x + I y]) > radius, 0, Exp[-.5 (r/sd)^2]], {y, -h,
     h - 1}, {x, -h, h - 1}]]
     
PSFDegrees[ pupilsamples_, wavelength_, pupildiameter_] := 
 pupilsamples wavelength 180 10^-6/(pupildiameter Pi)
 
PupilSamples[degrees_, wavelength_, pupildiameter_] := 
 10^6 degrees pupildiameter Degree/wavelength
 
 WaveAberrationImage[{}, radius_ : 16] := 
 ConstantArray[0, Ceiling[radius] {2, 2}]
 
 WaveAberrationImage[coefficients_, radius_ : 16] :=
 Total[(#[[3]] ZernikeImage[#[[1]], #[[2]], radius]) & /@ 
   coefficients]
"""

# 文章上方對 E 字做模糊的係數
# zc=np.array([[2,-2,-0.0946],[2,0,0.0969],[2,2,0.305],[3,-3,0.0459],
#              [3,-1,-0.121],[3,1,0.0264],[3,3,-0.113],[4,-4,0.0292],
#              [4,-2,0.03],[4,0,0.0294],[4,2,0.0163],[4,4,0.064]])

# TestCoefficients
zc=np.array([[2,-2,-0.0946],[2,0,0.0969],[2,2,0.305],[3,-3,0.0459],
             [3,-1,-0.121],[3,1,0.0264],[3,3,-0.113],[4,-4,0.0292],
             [4,-2,0.03],[4,0,0.0294],[4,2,0.0163],[4,4,0.064],
             [5,-5,0.0499],[5,-3,-0.0252],[5,-1,0.00744],[5,1,0.00155],
             [5,3,-0.00686],[5,5,0.0288],[6,-6,0.00245],[6,-4,0.00185],
             [6,-2,0.00122],[6,0,-0.00755],[6,2,-0.000693],[6,4,0.000551],
             [6,6,-0.0148]])

# 啟動 Wolfram Engine session
session = WolframLanguageSession("C:\\Users\\user\\Downloads\\Mathematica\\Mathematica\\wolfram.exe")

# 定義 Mathematica 程式碼
mathematica_code = """
  PolarList[radius_] := PolarList[radius] = Block[{h},
h = Ceiling[radius];
ToPackedArray[
    N[Transpose[
    Flatten[Transpose[
        Array[CartesianToPolar[{##}/radius] &, 2 {h, h}, {-h, -h}]], 1]]]]]
CartesianToPolar[{0., 0.}] := {0, 0}
CartesianToPolar[{0., 0}] := {0, 0}
CartesianToPolar[{0, 0.}] := {0, 0}
CartesianToPolar[{0, 0}] := {0, 0}
CartesianToPolar[{x_, y_}] := {Norm[{x, y}], ArcTan[x, y]}
"""

# # 執行 Mathematica 程式碼
# session.evaluate(wlexpr(mathematica_code))
# radius=31.445
# # 測試 PolarList 函數
# R = session.evaluate(wlexpr(f'PolarList[{radius}]'))
# print(R)
# # 關閉 session
# session.terminate()
zc=([[2,-2,-0.0946],[2,0,0.0969],[2,2,0.305]])
print(zc.ndim)


