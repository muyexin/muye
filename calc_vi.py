# -*- coding: utf-8 -*-

import numpy as np
from math import sqrt


def rgb2vi(img, vi_type):
    '''
    RGB图像转换为可见光植被指数
    'R', 'G', 'B', 'r', 'g', 'b',
    'G-R','ExG','ExG2','MExG','ExR','ExR2','VDVI','NGBDI','NGRDI','RGRI','GRRI',
    'GBRI','BRRI','RGBVI','ExGR','ExGR2','CIVE','CIVE2','VEG','COM','COM2',
     

    '''
#     VARI,IKAW,'GRVI','MGRVI','GLA'
    
    if 'uint8' in img.dtype.name:
        img = img.astype(np.float32) / 255
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    
    with np.errstate(divide='ignore', invalid='ignore'): # 忽略除以0的情况       
        if vi_type == 'R':
            vi = r
        elif vi_type == 'G':
            vi = g
        elif vi_type == 'B':
            vi = b
        elif vi_type == 'r':
            vi = r / (r+g+b)
        elif vi_type == 'g':
            vi = g / (r+g+b)
        elif vi_type == 'b':
            vi = b / (r+g+b)
        elif vi_type == 'G-R':
            vi = g - r
        elif vi_type == 'ExG':
            vi = 2 * (g / (r+g+b)) - (r / (r+g+b)) - (b / (r+g+b))
        elif vi_type == 'ExG2':
            vi = 2 * g - r - b
        elif vi_type == 'MExG':
            vi = 1.262 * (g / (r+g+b)) - 0.884 * (r / (r+g+b)) - 0.311 * (b / (r+g+b))
        elif vi_type == 'ExR':
            vi = 1.4 * r - g
        elif vi_type == 'ExR2':
            vi = 1.4 * (r / (r+g+b)) - (g / (r+g+b))
        elif vi_type == 'VDVI':
            vi = (2 * g - r - b) / (2 * g + r + b)
        elif vi_type == 'NGBDI':
            vi = (g - b) / (g + b)
        elif vi_type == 'NGRDI':
            vi = (g - r) / (g + r)
        elif vi_type == 'RGRI':
            vi = r / g
        elif vi_type == 'GRRI':
            vi = g / r
        elif vi_type == 'GBRI':
            vi = g / b
        elif vi_type == 'BRRI':
            vi = b / r
        elif vi_type == 'RGBVI':
            vi = (g*g - r*b) / (g*g + r*b)
        elif vi_type == 'ExGR':
            vi = (2 * (g / (r+g+b)) - (r / (r+g+b)) - (b / (r+g+b))) - (1.4 * (r) - (g))
        elif vi_type == 'ExGR2':
            vi = (2 * g - r - b) - (1.4 * r - g)
    #    elif vi_type == 'NHLVI':
    #        hsl = colorspace('HSL<-RGB', img)
    #        vi = (hsl(:,:,1) - hsl(:,:,3)) ./ (hsl(:,:,1) + hsl(:,:,3))
        elif vi_type == 'CIVE':
            vi = 0.441*(r / (r+g+b)) - 0.881*(g / (r+g+b)) + 0.385*(b / (r+g+b)) + 18.78745
        elif vi_type == 'CIVE2':
            vi = 0.441*r - 0.881*g + 0.385*b + 18.78745
        elif vi_type == 'VEG':
            vi = g / (r**0.667 * b**(1 - 0.667))
        elif vi_type == 'COM':
            vi1 = 2 * (g / (r+g+b)) - (r / (r+g+b)) - (b / (r+g+b))
            vi2 = 0.441*(r / (r+g+b)) - 0.881*(g / (r+g+b)) + 0.385*(b / (r+g+b)) + 18.78745
            vi3 = (2 * (g / (r+g+b)) - (r / (r+g+b)) - (b / (r+g+b))) - (1.4 * (r / (r+g+b)) - (g / (r+g+b)))
            vi4 = g / (r**0.667 * b**(1 - 0.667))
            vi = vi1 + vi2 + vi3 + vi4
        elif vi_type == 'COM2':
            vi1 = 2 * (g / (r+g+b)) - (r / (r+g+b)) - (b / (r+g+b))
            vi2 = 0.441*(r / (r+g+b)) - 0.881*(g / (r+g+b)) + 0.385*(b / (r+g+b)) + 18.78745
            vi3 = g / (r**0.667 * b**(1 - 0.667))
            vi = 0.36 * vi1 + 0.47 * vi2 + 0.17 * vi3
#            
#        elif vi_type == 'VARI':
#            vi=(g/(r+g+b)-r/(r+g+b))/(g/(r+g+b)+r/(r+g+b)-b/(r+g+b))   
#            
#        elif vi_type == 'IKAW':
#            vi=(r/(r+g+b)-b/(r+g+b))/(r/(r+g+b)+b/(r+g+b))
#            
#        elif vi_type == 'GRVI':
#            vi=(g-r)/(g+r)
#        elif vi_type == 'MGRVI':
#            vi=((g/(r+g+b))*(g/(r+g+b))-(r/(r+g+b))*(r/(r+g+b)))/((g/(r+g+b))*(g/(r+g+b))+(r/(r+g+b))*(r/(r+g+b)))    
#        elif vi_type == 'GLA':
#            vi=(2*(g/(r+g+b))-r/(r+g+b)-b/(r+g+b))/(2*(g)/(r+g+b)+r/(r+g+b)+b/(r+g+b))    
            
        else:
            raise Exception('Unknown VI')
#            print(type(vi))
        vi[~np.isfinite(vi)] = 0 # -inf inf nan置为0
#        print(type(vi))
    return vi

def rgn2vi(img, vi_type):
    '''
    多光谱RGN图像转换为植被指数
    'NDVI','RVI','NDWI','DVI','PVI','SAVI','EVI','TVI','RDVI','RGRI','GI','NDGI','OSAVI','NRI','CIG','MSR'
    '''
    if 'uint8' in img.dtype.name:
        img = img.astype(np.float32) / 255
    r, g, n = img[:,:,0], img[:,:,1], img[:,:,2]
    
    with np.errstate(divide='ignore', invalid='ignore'): # 忽略除以0的情况
        if vi_type == 'NDVI':
            vi = (n - r) / (n + r)
        elif vi_type == 'RVI':
            vi = n / r
        elif vi_type == 'NDWI':
            vi = (g - n) / (g + n)
        elif vi_type == 'DVI':
            vi = n - r
        elif vi_type == 'PVI':
            a = 10.489
            b = 6.604
            vi = (n - a * r - b) / sqrt(1 + a**2)
        elif vi_type == 'SAVI':
            l = 0.5
            vi = ((1 + l) * (n - r)) / (n + r + l)
        elif vi_type == 'EVI':
            l = 2.5
            vi = (l*(n+r))/(n-r+1)
        elif vi_type == 'TVI':
            vi = 0.5*(120*(n-g))-20*(r-g)  
        elif vi_type == 'RDVI':
            vi = (n-r)/((n+r)**0.5)
        elif vi_type == 'RGRI':
            vi = r/g
        elif vi_type == 'GI':
            vi = g/r  
        elif vi_type == 'NDGI':
            vi = (n-g)/(n+g)
        elif vi_type == 'OSAVI':
            vi = (n-r)/(n+r+0.16)
        elif vi_type == 'NRI':
            vi = (g-r)/(g+r)
        elif vi_type == 'CIG':
            vi = n/r -1
        elif vi_type == 'MSR':
            vi = [(n/r - 1)/  (n/r + 1)]
        else:
            raise Exception('Unknown VI')
        vi[~np.isfinite(vi)] = 0 # -inf inf nan置为0
    return vi

def rgb2vis(img, vi_types):
    '''RGB图像转换为多个可见光植被指数'''
    img = np.array(img)
    vis = []
    for vi_type in vi_types:
        vi = rgb2vi(img, vi_type)
        vis.append(vi)
    vis = np.array(vis)
#    print(vis.shape)
#    print(vis.shape[1])
#    print(vis.shape[2])


    vis = vis.transpose((1,2,0))
    return vis

def rgn2vis(img, vi_types):
    '''多光谱RGN图像转换为多个植被指数'''
    img = np.array(img)
    vis = []
    for vi_type in vi_types:
        vi = rgn2vi(img, vi_type)
        vis.append(vi)
    vis = np.array(vis)
    vis = vis.transpose((1,2,0))
    return vis

if __name__ == '__main__':
    from PIL import Image
    from boxx import show
    
#    path = r'E:\pic\jiansanjiang\contrast\RGB\data\test\img\0a54a1b8-b743-4824-8b5e-8e64893b7d64.jpg'
#    path = r'E:\pic\jiansanjiang\contrast\RGB\img\rgb.jpg'
    path = r'D:\pic\jiansanjiang\contrast\RGN\img\rgn.jpg'
    img = Image.open(path)
    
#    vi_types = ['G-R','ExG','ExG2','MExG','ExR','ExR2','VDVI','NGBDI','NGRDI',
#                'RGRI','GRRI','GBRI','BRRI','RGBVI','ExGR','ExGR2',
#                'CIVE','CIVE2','VEG','COM','COM2']
#    vi_types = ['ExG','ExR','VDVI','NGRDI','RGRI','ExGR']
#    vis = rgb2vis(img, vi_types)
#    show(vis)
    
    vi_types = ['NDVI','RVI','NDWI','DVI','PVI','SAVI']
    vis2 = rgn2vis(img, vi_types)
    show(vis2.transpose((2,0,1)))






