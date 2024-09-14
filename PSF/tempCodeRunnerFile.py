
# # 計算 PSF
# psf=zernikePointSpread(zc,Wavelength=[400,555,700])

# # 輸出 PSF(要對圖片做卷積的 kernel)
# psf_img = PSFPlot(psf=psf)
# plt.show()