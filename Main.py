import os
import cv2
import time
import numpy as np
from Visual import *
from VideoProcessing import *
from CowBreathTracker import *
import warnings
warnings.filterwarnings("ignore")

def getExtensionFilePaths(dirPath,extension):
    filePaths=[]
    for file in os.listdir(dirPath):
        if extension in file:
            filePath=os.path.join(dirPath,file)
            filePaths.append(filePath)
    return filePaths

class Main:
    def __init__(self,outputDir,inputDir,videoPath,mode=3,rgb=None):
        self.outputDir=outputDir
        self.inputDir=inputDir
        self.vr=VideoProcessing(videoPath,readCount=200)
        self.vw=VideoProcessing(os.path.join(outputDir,'output.avi'),30)
        self.cvws={}
        self.cbt=CowBreathTracker(interval=20,mode=mode,rgb=rgb)
        self.startTime=None
        self.index=1
        self.updateInfos()

    def updateInfos(self):
        while True:
            try:
                filePath=os.path.join(self.inputDir,str(self.index)+'.npz')
                if os.path.exists(filePath):
                    data=np.load(filePath)
                    img,mask,xyxys,labels,xyxys2,labels2 = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3'],data['arr_4'],data['arr_5']
                    self.cbt.updateInfos(img,mask,xyxys,xyxys2,labels,labels2)
                    break
                if self.index!=1:
                    break
            except:
                pass
        
    def run(self):
        start=time.time()
        while True:
            curTime=time.time()
            try:
                print('processing index :',self.index)
                img,grayGpu=self.vr.readThread()
                if img is None:
                    break
                self.cbt.track(img,grayGpu)
                self.index+=1
                self.updateInfos()
                for info in self.cbt.cowInfos:
                    if info[8]==False:
                        continue
                    info[1].astype(np.int32)
                    if info[0] not in self.cvws:
                        self.cvws[info[0]]=VideoProcessing(os.path.join(outputDir,str(info[0])+'.avi'),30)
                    roiImg=self.cbt.getRoiImg(img,info)
                    rate=self.cbt.getRate(info)
                    meanRate=self.cbt.getMeanRate(info)
                    if rate is not None:
                        cv2.putText(roiImg, '{:.1f}'.format(rate), (roiImg.shape[1]//2-120,roiImg.shape[0]//2-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                        cv2.putText(roiImg, '{:.1f}'.format(meanRate), (roiImg.shape[1]//2+10,roiImg.shape[0]//2-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                    self.cvws[info[0]].write(roiImg)
                for info in self.cbt.cowInfos:
                    if info[8]==False:
                        continue
                    xyxy=self.cbt.getBoundingRect(info)
                    cv2.rectangle(img,xyxy[0:2],xyxy[2:4],[0,255,0],1)
                    point=info[1].astype(np.int32)
                    meanRate=self.cbt.getMeanRate(info)
                    if meanRate is not None:
                        print(point,'{:.0f}'.format(meanRate*60)+'bpm')
                        cv2.putText(img,'{:.0f}'.format(meanRate*60)+'bpm', (point[0]-60,point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                self.vw.write(img)
            except:
                self.vw.close()
                for key in list(self.cvws.keys()):
                    self.cvws[key].close()
                break
            nowTime=time.time()
            meanFPS=(self.index-1)/(nowTime-start) if nowTime-start>0 else 0
            print('TIME : ',nowTime-curTime,'MEAN FPS : ',meanFPS)
                
outputDir=r'Output'
inputDir=r'Data'
videoPath=getExtensionFilePaths('Video','mp4')[0]
main=Main(outputDir,inputDir,videoPath)
main.run()