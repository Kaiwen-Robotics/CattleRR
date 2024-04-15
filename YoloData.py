import os
import numpy as np
from YoloDetection import *
from YoloSegmentation import *
import warnings
warnings.filterwarnings("ignore")
        
def getExtensionFilePaths(dirPath,extension):
    filePaths=[]
    for file in os.listdir(dirPath):
        if extension in file:
            filePath=os.path.join(dirPath,file)
            filePaths.append(filePath)
    return filePaths

if __name__=='__main__':
    index=1
    device='0'
    timeInterval=1
    outputDir=r'Data'
    SegmentWeights=r'best.pt'
    DetectWeights=r'best2.pt'
    source=getExtensionFilePaths('Video','mp4')[0]
    conditionDict={}
    conditionDict['index']=index
    detectGenerater=getDetectGenerater(DetectWeights,source,conf_thres=0.65,device=device,conditionDict=conditionDict)
    segmentGenerater=getSegmentGenerater(SegmentWeights,source,imgsz=1920,device=device,conditionDict=conditionDict)
    startTime=time.time()
    while True:
        start=time.time()
        try:
            print('yolo processing index :',index)
            conditionDict['index']=index
            if index==1 or time.time()-startTime>=timeInterval:
                img,mask,xyxys,labels=next(segmentGenerater)
                img,xyxys2,labels2=next(detectGenerater)
                np.savez(os.path.join(outputDir,str(index)+'.npz'),img,mask,xyxys,labels,xyxys2,labels2)
                index+=timeInterval*30
        except:
            break
        print('time : ',time.time()-start)