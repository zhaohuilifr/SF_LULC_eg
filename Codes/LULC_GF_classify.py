#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/07/06 08:31:45
@功能: 土地利用/土地覆盖分类
'''

import numpy as np
import os
import glob
# import geopandas
from osgeo import gdal,osr,ogr
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from skimage.util import img_as_float
from skimage import exposure
from skimage.segmentation import felzenszwalb
import joblib
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import gc
import pickle
import scipy.signal as signal
import multiprocessing
import datetime
# import func_glcmfeature
# import func_separability
import cv2

###########
# COLORS=["#000000","#FFFF00","#1CE6FF","#FF34FF","#FF4A46","#008941"]
# COLORS=['tan','slategrey','green','red','blue']
# cmap=mpl.colors.ListedColormap(COLORS)
###########
# 波段选择：band7+5\4\3多光谱

class Raster():
    def __init__(self,array,transform,proj,xsize,ysize):
        self.array=array
        self.transform=transform
        self.proj=proj
        self.xsize=xsize
        self.ysize=ysize


def read_tif_multiband(raster_data_path):
    raster_dataset=gdal.Open(raster_data_path,gdal.GA_ReadOnly)
    geo_transform=raster_dataset.GetGeoTransform()
    proj = raster_dataset.GetProjection()
    bands_data=[]
    for b in range(1,raster_dataset.RasterCount+1):
        band=raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())
    
    bands_data=np.dstack(bands_data)
    rows,cols,n_bands=bands_data.shape
    del raster_dataset,band
    return rows,cols,n_bands,geo_transform,proj,bands_data


def read_tif(raster_data_path):
    raster_dataset=gdal.Open(raster_data_path,gdal.GA_ReadOnly)
    geo_transform=raster_dataset.GetGeoTransform()
    proj = raster_dataset.GetProjection()
    band=raster_dataset.GetRasterBand(1)
    array=band.ReadAsArray()
    rows,cols=array.shape

    del raster_dataset,band
    return rows,cols,geo_transform,proj,array


def read_tif_path(raster_data_path,bands):
    files=glob.glob(raster_data_path+'/LC08*.TIF')
    bands_data=[]
    # print(files)
    for i in range(len(files)):
        filei=files[i]
        bandstr=filei.split('_')[-1][1:-4]
        if bandstr in bands:
            rows,cols,geo_transform,proj,array=read_tif(filei)
            bands_data.append(array)

    bands_data=np.dstack(bands_data)
    rows,cols,n_bands=bands_data.shape
    return rows,cols,n_bands,geo_transform,proj,bands_data

def create_mask_from_vector(vector_path,cols,rows,geo_transform,projection,target_value=1):
    data_source=gdal.OpenEx(vector_path,gdal.OF_VECTOR)
    
    layer=data_source.GetLayer(0)
    driver=gdal.GetDriverByName("MEM")
    target_ds=driver.Create('',cols,rows,1,gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds,[1],layer,burn_values=[target_value])

    return target_ds

def vectors_to_raster(filepaths,rows,cols,geo_transform,projection):
    labeled_pixels=np.zeros((rows,cols))
    for i,path in enumerate(filepaths):
        label=i+1
        print(path.split('/')[-1],':',label)
        ds=create_mask_from_vector(path,cols,rows,geo_transform,projection,target_value=label)
        band=ds.GetRasterBand(1)
        labeled_pixels+=band.ReadAsArray()
        ds=None
    return labeled_pixels

def write_geotiff(fname,data,geo_transform,projection):
    driver=gdal.GetDriverByName("GTiff")
    rows,cols=data.shape
    dataset=driver.Create(fname,cols,rows,1,gdal.GDT_Byte)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band=dataset.GetRasterBand(1)
    band.WriteArray(data)
    dataset=None # 关闭文件

def get_labels(vector_path,rows,cols,geo_transform,proj):
    files=[f for f in os.listdir(vector_path) if f.endswith('.shp')]
    classes=[f.split('.')[0] for f in files]
    shapefiles=[os.path.join(vector_path,f) for f in files]

    labeled_pixels=vectors_to_raster(shapefiles,rows,cols,geo_transform,proj)
    is_train=np.nonzero(labeled_pixels) #返回数组a中非零元素的索引值数组，结果为包含2个array的tuple，第一array为行数，第二array为列数
    ## 统计各样本像元数占总数比例
    tmp=labeled_pixels[is_train] # 返回数组a中所有非零元素
    nums_sum = len(tmp)
    for i,classe in enumerate(shapefiles):
        idx = tmp==i+1
        print(classe+' : %.4f'%(np.sum(idx)/float(nums_sum)*100)+'%')
    #
    del tmp
    return labeled_pixels,is_train,classes

def normalize(data):
    # 数据标准化
    data=data.astype('float') # !!!
    for i in range(data.shape[2]):
        tmp=data[:,:,i]
        tmp=(tmp-np.nanmin(tmp))/(np.nanmax(tmp)-np.nanmin(tmp))
        data[:,:,i]=tmp
    del tmp
    return data

def get_ndvi(nir,red):
    return (nir-red)/(nir+red)

def get_training_data(*args):
    if len(args)==3:
        rows,cols,n_bands,geo_transform,proj,bands_data=read_tif_path(args[1],args[2])
    else:
        rows,cols,n_bands,geo_transform,proj,bands_data=read_tif_multiband(args[1])
    
    bands_data[bands_data==65535]=0
    bands_data[np.isnan(bands_data)]=0
    bands_data[bands_data<0]=0

    bands_data = bands_data[:,:,[0,1,2,3]] # b,g,r,nir 
    
    # 计算NDVI
    # Ndvi=get_ndvi(bands_data[:,:,4],bands_data[:,:,3])
    Ndvi=get_ndvi(bands_data[:,:,3],bands_data[:,:,2])
    # # 计算纹理特征
    # # data=bands_data[:,:,4]
    # data=bands_data[:,:,3]
    # Contrast,Homogenity,Energy,Correlation=func_glcmfeature.get_imgfeature(data)
    # bands_data=np.dstack([bands_data,Ndvi,Contrast,Homogenity,Energy,Correlation])
    bands_data=np.dstack([bands_data,Ndvi])
    # print(bands_data.shape)
    n_bands=bands_data.shape[2]
    # del Contrast,Homogenity,Energy,Correlation
    
    # 标准化
    bands_data=normalize(bands_data)
    #
    labeled_pixels,is_train,classes=get_labels(args[0],rows,cols,geo_transform,proj)
    training_labels=labeled_pixels[is_train] # 返回数组a中所有非零元素
    training_samples=bands_data[is_train]
    return rows,cols,n_bands,geo_transform,proj,bands_data,training_labels,training_samples,classes

def train_data(training_samples, training_labels, is_train_split):
    # 分类器01-随机森林
    classifier=RandomForestClassifier(n_estimators=300,n_jobs=-1)
    # 分类器02-SVM
    # classifier=svm.SVC(C=1.0, kernel='rbf', gamma=2)
    print('model training...')
    if is_train_split:
    # 训练数据
        # 对训练集划分,默认90%训练,10%验证
        x_train, x_test, y_train, y_test = train_test_split(training_samples, training_labels, train_size=0.9,test_size=0.1)
        # 训练模型
        classifier.fit(x_train,y_train)
        # 保存分类器
        print('model saving...')
        joblib.dump(classifier,"./Classifiermodel.m")
        return classifier,x_test,y_test
    else:
        classifier.fit(training_samples,training_labels)
        # 保存分类器
        print('model saving...')
        joblib.dump(classifier,"./Classifiermodel.m")
        return classifier


def valid_data(*args): 
    print('model validdtion...')
    if len(args)==4:            # classifier,x_test,y_test,classes
        # 无测试集精度评价
        print('model accuracy assessment:')
        verification_labels=args[2]
        predicted_labels=args[0].predict(args[1])
        accuracy_asses(predicted_labels,verification_labels,args[3])
    else:                       # validation_data_path,bands_data,rows,cols,geo_transform,proj,classifier
        # 有测试精度评价
        verification_pixels,is_verfication,classes=get_labels(args[0],args[2],args[3],args[4],args[5])
        verification_labels=verification_pixels[is_verfication]
        bands_data=args[1]
        validation_samples=bands_data[is_verfication]
        predicted_labels=args[6].predict(validation_samples)
        accuracy_asses(predicted_labels,verification_labels,classes)
        del bands_data

def accuracy_asses(predicted_labels,verification_labels,classes):
    print("Confussion matrix:\n%s" %
      metrics.confusion_matrix(verification_labels, predicted_labels))
    target_names = ['Class %s' % s for s in classes]
    print("Classification report:\n%s" %
        metrics.classification_report(verification_labels, predicted_labels,
                                        target_names=target_names))
    print("Classification accuracy: %f" %
        metrics.accuracy_score(verification_labels, predicted_labels))

def classify(classifier,rows,cols,n_bands,geo_transform,proj,bands_data):
    n_samples=rows*cols
    flat_pixels=bands_data.reshape((n_samples,n_bands))

    result=[]
    print('classifying...')
    for i in np.arange(0,flat_pixels.shape[0],100000):
        if i<flat_pixels.shape[0]-100000:
            tmp=flat_pixels[i:i+100000,:]
            res=classifier.predict(tmp)
        else:
            tmp=flat_pixels[i:,:]
            res=classifier.predict(tmp)
        result.append(res)
        del tmp,res
        # print(i,'-',i+100000,' Done.')
    result=np.hstack(result)
    classfication=result.reshape((rows,cols))
    return classfication

def img_morphology(img,kernelsize):
    kernel = np.ones((kernelsize, kernelsize), np.uint8)
    closed = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
    return closed

def imgSegmentation(bands_data,scale,geo_transform,proj,savename='segmentation.tif'):
    # 分割
    print('Segmentation Starting...')
    img = img_as_float(bands_data[:, :, 0:3])
    img = exposure.rescale_intensity(img)
    segments_fz = felzenszwalb(img, scale=scale, sigma=0.001, min_size=200)
    print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
    # 保存分割结果
    driver=gdal.GetDriverByName("GTiff")
    rows,cols=segments_fz.shape
    dataset=driver.Create('./results/'+savename,cols,rows,1,gdal.GDT_Float32)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(proj)
    band=dataset.GetRasterBand(1)
    band.WriteArray(segments_fz)
    dataset=None # 关闭文件
    print("Segmentation Finished!")

    return segments_fz

def classPostProcess(classfication,array_s,geo_transform,proj,saveraster,savevector):
    # 开运算
    array_c = img_morphology(classfication,3)
    # 对分类结果应用分割结果
    print('Applying segmentation on classification result...')
    idx = array_c ==0
    res_seg_classify = np.zeros_like(array_c)
    nums = np.unique(array_s)
    # for i,num in enumerate(nums):
    #     idx_num = array_s == num
    #     array_c_in_idx = array_c[idx_num]
    #     counts = np.bincount(array_c_in_idx)
    #     res_seg_classify[idx_num] = np.argmax(counts)
    #     print(i,num,np.argmax(counts))
    for i,num in enumerate(nums):
        idx_num = array_s == num
        array_c_in_idx = array_c[idx_num]
        # 'float64'转'int64',似乎np.bincount()需要输入是int64
        array_c_in_idx = array_c_in_idx.astype('int64') 
        counts = np.bincount(array_c_in_idx)
        res_seg_classify[idx_num] = np.argmax(counts)

        # frequency = counts/float(len(array_c_in_idx))
        # if np.nanmax(frequency)>0.4:
        #     res_seg_classify[idx_num] = np.argmax(counts)
        # else:
        #     res_seg_classify[idx_num] = array_c[idx_num]
        # print(i,np.argmax(counts),np.nanmax(frequency))
    # 保存栅格分类结果
    write_geotiff(saveraster,res_seg_classify,geo_transform,proj)
    # 保存矢量分类结果
    raster=array2raster(res_seg_classify,geo_transform,proj, \
        res_seg_classify.shape[1],res_seg_classify.shape[0])
    raster2vector_esrishp(raster,savevector)
    print('Classification Finished!')

def array2raster(data,transform,proj,xsize,ysize):
    driver = gdal.GetDriverByName('MEM')
    output = driver.Create('',xsize, ysize, 1, gdal.GDT_Byte)  #gdal.GDT_Float32
    output.SetGeoTransform(transform)
    output.SetProjection(proj)
    output.GetRasterBand(1).WriteArray(data)
    return output

def raster2vector_esrishp(raster,savefile):
    outband=raster.GetRasterBand(1)
    proj = raster.GetProjection()

    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(savefile):
        drv.DeleteDataSource(savefile)

    dst_ds = drv.CreateDataSource(savefile)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    dst_layer = dst_ds.CreateLayer(savefile, srs )
    # 添加属性列
    newField = ogr.FieldDefn('Class', ogr.OFTInteger)
    dst_layer.CreateField(newField)

    gdal.Polygonize(outband, None, dst_layer,0, [], callback=None )

    dst_ds.Destroy()
    raster= None
    outband=None

    # 删除不需要的属性值的shp,这里把0值全部去掉
    ioShapefile = ogr.Open(savefile,update = 1)
    lyr = ioShapefile.GetLayerByIndex(0)
    lyr.ResetReading()
    for i in lyr:
        lyr.SetFeature(i)
        if i.GetField('Class') == 0: 
            lyr.DeleteFeature(i.GetFID())
    ioShapefile.Destroy()

    return savefile

def addfieeld(shpfile,classes):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shpfile, 1)
    layer = dataSource.GetLayer()
    # 添加属性列
    newField = ogr.FieldDefn('classname', ogr.OFTString)
    layer.CreateField(newField)
    # 
    for i in range(len(classes)):
        layer.SetAttributeFilter("Class = "+str(i+1))
        for feature in layer:
            feature.SetField('classname',classes[i])
            layer.SetFeature(feature)
    dataSource.Destroy()



if __name__=="__main__":
    ###########Lst-8###########
    os.chdir(r'/Database')
    raster_data_path = "./rawdata/Extract_tiff21.tif"
    os.makedirs('./results/features',exist_ok=True)
    output_fname_raster = "./results/classification.tif"
    output_fname_vector = "./results/classification.shp"
    train_data_path = "./traindata"

    starttime=datetime.datetime.now()
    # inputbands=['1','2','3','4']
    # rows,cols,n_bands,geo_transform,proj,bands_data, \
    #     training_labels,training_samples,classes=get_training_data(train_data_path,raster_data_path,inputbands)
    rows,cols,n_bands,geo_transform,proj,bands_data, \
    training_labels,training_samples,classes=get_training_data(train_data_path,raster_data_path)

    # # 分割
    _,_,_,_,array_s = read_tif('./results/segmentation.tif')
    # array_s= imgSegmentation(bands_data,300,geo_transform,proj)

    # 训练分类器
    classifier=train_data(training_samples, training_labels,False)
    # classifier,x_test,y_test=train_data(training_samples, training_labels,True)
    
    # # # 验证分类精度
    # # 测试数据
    # valid_data(classifier,x_test,y_test,classes)
    # # # 验证数据
    # # valid_data(validation_data_path,bands_data,rows,cols,geo_transform,proj,classifier)

    # # # 分类器本地加载
    # classifier=joblib.load("./Classifiermodel.m")
    # 土地利用分类
    bands_data[np.isnan(bands_data)]=0
    classfication=classify(classifier,rows,cols,n_bands,geo_transform,proj,bands_data)

    # # 分类结果后处理与保存
    classPostProcess(classfication,array_s,geo_transform,proj,output_fname_raster,output_fname_vector)
    ## 添加类别属性列
    addfieeld(output_fname_vector,classes)

    endtime=datetime.datetime.now()
    spendtime=(endtime-starttime).total_seconds()
    print(spendtime,' s used in all',', with '+str(rows)+'*'+str(cols)+' image size')