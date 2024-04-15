import cv2
import numpy as np
import cupy as cp

sparse_flow = cv2.cuda_SparsePyrLKOpticalFlow.create()
sparse_flow.setMaxLevel(2)
sparse_flow.setNumIters(10)
sparse_flow.setWinSize((15,15))

def morphologyEx(binary,size,mtype):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    binary = cv2.morphologyEx(binary, mtype, kernel)
    return binary
    
def getDistancesMatrix(coords1,coords2):
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    distances = np.sqrt(np.sum((coords1[:, np.newaxis] - coords2) ** 2, axis=2))
    return distances

def getMeshGridPoints(shape,interval):
    x=np.arange(0,shape[1],interval)
    y=np.arange(0,shape[0],interval)
    mx,my=np.meshgrid(x,y)
    points=np.stack([mx,my],axis=2)
    points=np.reshape(points,(np.prod(points.shape[0:2]),2))
    return points

def getDistances(points1,points2):
    points1=np.array(points1)
    points2=np.array(points2)
    distances=np.linalg.norm(points1-points2,axis=1)
    return distances
    
def getLeftRightRangeMinIndexs(data,radius):
    indexs=cp.arange(data.shape[0])
    leftMin=data.copy()
    rightMin=data.copy()
    leftIndexs=indexs.copy()
    rightIndexs=indexs.copy()
    leftEnable=cp.ones(data.shape[0])
    rightEnable=cp.ones(data.shape[0])
    for i in range(radius):
        size=(i+1)*2+1
        curRadius=i+1
        curLeftIndexs=indexs-curRadius
        curRightIndexs=indexs+curRadius
        curLeftIndexs[curLeftIndexs<0]=0
        curRightIndexs[curRightIndexs>=data.shape[0]]=data.shape[0]-1
        leftEnable[data[curLeftIndexs]>data]=0
        rightEnable[data[curRightIndexs]>data]=0
        leftCondition=(data[curLeftIndexs]<leftMin) & (leftEnable>0)
        rightCondition=(data[curRightIndexs]<rightMin) & (rightEnable>0)
        leftMin[leftCondition]=data[curLeftIndexs[leftCondition]]
        leftIndexs[leftCondition]=curLeftIndexs[leftCondition]
        rightMin[rightCondition]=data[curRightIndexs[rightCondition]]
        rightIndexs[rightCondition]=curRightIndexs[rightCondition]
    return leftIndexs,rightIndexs
    
def getFFTSpectrum(datas,frameCount):
    fft_y = cp.fft.fft(datas,axis=0)
    freqs = cp.fft.fftfreq(len(datas), 1/frameCount)
    freqs = cp.fft.fftshift(freqs)
    fft_y = cp.fft.fftshift(fft_y)
    abs_fft_y=cp.abs(fft_y)
    return freqs,abs_fft_y

def getFilterFFTSpectrum(datas,frameCount,lowFreq,highFreq):
    freqs,abs_fft_y=getFFTSpectrum(datas,frameCount)
    freqs_2=freqs[(freqs>=lowFreq) & (freqs<=highFreq)]
    abs_fft_y_2=abs_fft_y[(freqs>=lowFreq) & (freqs<=highFreq)]
    return freqs_2,abs_fft_y_2
    
def getCowBreathRate(flowSinClips,frameCount=30):
    freqs,abs_fft_y=getFilterFFTSpectrum(flowSinClips,frameCount,0.3,2.4)
    abs_fft_y=cp.sum(abs_fft_y,axis=1)
    rate=freqs[cp.argmax(abs_fft_y)]
    return rate
    
def getRGBCowBreathRate(flowSinClips,frameCount=30):
    freqs,abs_fft_y=getFilterFFTSpectrum(flowSinClips,frameCount,0.5,2.4)
    abs_fft_y=cp.sum(abs_fft_y,axis=1)
    leftIndexs,rightIndexs=getLeftRightRangeMinIndexs(abs_fft_y,int(0.5//cp.diff(freqs)[0]//2))
    weights=[]
    for i in range(len(leftIndexs)):
        weights.append(cp.prod(abs_fft_y[i]/cp.array([abs_fft_y[leftIndexs[i]],abs_fft_y[rightIndexs[i]]])))
    weights=cp.array(weights)
    rate=freqs[cp.argmax(weights)]
    return rate

def getFilterXyxys(xyxys,minSize):
    filterXyxys=[]
    for i in range(len(xyxys)):
        x0,y0,x1,y1=xyxys[i]
        if np.min([x1-x0,y1-y0])<minSize:
            continue
        filterXyxys.append([x0,y0,x1,y1])
    return filterXyxys
    
def getUVSin(uvs):
    radius=cp.arctan2(uvs[...,1],uvs[...,0])
    sins=cp.sin(radius)
    return sins

class CowBreathTracker:

    def __init__(self,interval=5,fftLength=120,mode=3,rgb=None):
        self.backLength=None
        self.trackPoints=None
        self.cowInfos=None
        self.fftLength=fftLength if rgb is None else 416
        self.interval=interval if rgb is None else 1
        self.mode=mode
        self.rgb=rgb
        self.label=0
        self.gray_gpus=[]
        self.timePointsDict={}
            
    def track(self,img,grayGpu):
        self.gray_gpus.append(grayGpu)
        if len(self.gray_gpus)>=2:
            points_cpu=np.reshape(self.trackPoints,(1, -1, 2)).astype(np.float32)
            points_gpu=cv2.cuda_GpuMat(points_cpu)
            nextPoints_gpu=cv2.cuda_GpuMat(points_cpu)
            sparse_flow.calc(self.gray_gpus[0], self.gray_gpus[1], points_gpu,nextPoints_gpu)
            points=nextPoints_gpu.download()[0]
            self.gray_gpus.pop(0)
            self.updateData(img,points)
            self.updateBreathRate()
            self.trackPoints = points.copy()
    
    def getBackGroundPoints(self,gray,mask):
        points = cv2.goodFeaturesToTrack(gray, maxCorners=10000, qualityLevel=0.01, minDistance=10)[:,0,:]
        points = points.astype(np.int32)
        points = points[mask[points[:,1],points[:,0]]==0]
        return points
    
    def getCowBreathPoints(self,gray,mask,xyxys2,labels2):
        points=getMeshGridPoints(gray.shape,self.interval)
        if self.rgb is None:
            grad=morphologyEx(gray,3,cv2.MORPH_GRADIENT)
            points=points[grad[points[:,1],points[:,0]]>0]
        if self.mode>=2:
            points=points[mask[points[:,1],points[:,0]]>0]
        if self.mode==3:
            marks = np.zeros(len(points))
            for i in range(len(xyxys2)):
                if labels2[i]=='belly_up':
                    x0,y0,x1,y1=xyxys2[i]
                    marks[(points[:,0]>=x0) & (points[:,0]<=x1) & (points[:,1]>=y0) & (points[:,1]<=y1)]=1
            points=points[marks>0]
        return points
        
    def trackWithUpdate(self,img,mask,xyxys,xyxys2,labels,labels2):
        self.track(img)
        self.updateInfos(img,mask,xyxys,xyxys2,labels,labels2)
       
    def updateInfos(self,img,mask,xyxys,xyxys2,labels,labels2):
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        curCenters=[]
        if self.cowInfos is not None:
            curCenters=[cowInfo[1] for cowInfo in self.cowInfos]
        allCowBackPoints=None
        allCowBreathPoints=None
        backPoints=self.getBackGroundPoints(gray,mask)
        breathPoints=self.getCowBreathPoints(gray,mask,xyxys2,labels2)
        cowInfos=[]
        for i in range(len(xyxys)):
            if labels[i]=='standing':
                continue
            x0,y0,x1,y1=xyxys[i]
            radius=int(np.min([x1-x0,y1-y0])/2)
            point=np.array([(x0+x1)/2,(y0+y1)/2],dtype=np.float32)
            cowInfo=None
            if len(curCenters)>0:
                infoDistacnes=getDistances(point,curCenters)
                minIndex=np.argmin(infoDistacnes)
                if infoDistacnes[minIndex]<self.cowInfos[minIndex][2]/2:
                    cowInfo=self.cowInfos[minIndex]
            if cowInfo is None:
                if np.min([x1-x0,y1-y0])<100:
                    continue
                backDistances=getDistances(point,backPoints)
                breathDistances=getDistances(point,breathPoints)
                cowBackPoints=backPoints[backDistances<300]
                cowBreathPoints=breathPoints[breathDistances<radius]
            else:
                number=mask[int(point[1]),int(point[0])]
                curCowBreathPoints=self.getBreathPoints(self.trackPoints,cowInfo).astype(np.int32)
                curCowBreathPoints=curCowBreathPoints[(curCowBreathPoints[:,0]<img.shape[1]) & (curCowBreathPoints[:,1]<img.shape[0])]
                numbers=mask[curCowBreathPoints[:,1],curCowBreathPoints[:,0]]
                if np.sum(numbers==number)/len(numbers)<0.3:
                    continue
                cowBackPoints=self.getBackPoints(self.trackPoints,cowInfo)
                if self.rgb is None:
                    cowBreathPoints=self.getBreathPoints(self.trackPoints,cowInfo)
            if self.rgb is None and len(cowBreathPoints)==0:
                continue
            allCowBackPointsLength=len(allCowBackPoints) if allCowBackPoints is not None else 0
            backIndexs=np.arange(len(cowBackPoints))+allCowBackPointsLength
            allCowBackPoints=np.concatenate([allCowBackPoints,cowBackPoints]) if allCowBackPoints is not None else cowBackPoints
            if self.rgb is None:
                allCowBreathPointsLength=len(allCowBreathPoints) if allCowBreathPoints is not None else 0
                breathIndexs=np.arange(len(cowBreathPoints))+allCowBreathPointsLength
                allCowBreathPoints=np.concatenate([allCowBreathPoints,cowBreathPoints]) if allCowBreathPoints is not None else cowBreathPoints
            else:
                breathIndexs=[]
            if cowInfo is None:
                cowInfo=[self.label,point,radius,backIndexs,breathIndexs,np.array([0,0],dtype=np.float32),cowBreathPoints-point,[],True]
                self.label+=1
            else:
                cowInfo[3]=backIndexs
                cowInfo[4]=breathIndexs
            cowInfos.append(cowInfo)
        if self.rgb is None:
            self.trackPoints=np.concatenate([allCowBackPoints,allCowBreathPoints])
        else:
            self.trackPoints=allCowBackPoints
        self.backLength=len(allCowBackPoints)
        self.cowInfos=cowInfos
        
    def updateData(self,img,points):
        labels=[]
        for i in range(len(self.cowInfos)):
            lastBackPoints=self.getBackPoints(self.trackPoints,self.cowInfos[i])
            curBackPoints=self.getBackPoints(points,self.cowInfos[i])
            offsetVector=np.mean(curBackPoints-lastBackPoints,axis=0)
            self.cowInfos[i][1]+=offsetVector
            self.cowInfos[i][5]+=offsetVector
            label=self.cowInfos[i][0]
            labels.append(label)
            if self.rgb is None:
                breathPoints=self.getBreathPoints(points,self.cowInfos[i])
                timeData=breathPoints-self.cowInfos[i][5]
            else:
                breathPoints=(self.cowInfos[i][1]+self.cowInfos[i][6]).astype(np.int32)
                if self.rgb==True:
                    timeData=np.sum(img[breathPoints[:,1],breathPoints[:,0]].astype(np.uint16),axis=-1)
                else:
                    timeData=img[breathPoints[:,1],breathPoints[:,0],2]
            if label in self.timePointsDict:
                self.timePointsDict[label].append(timeData)
                if len(self.timePointsDict[label])>=self.fftLength+2:
                    self.timePointsDict[label].pop(0)
                    timePoints=np.stack(self.timePointsDict[label])
                    timePointsDisp=np.linalg.norm(timePoints[1:] - timePoints[:-1], axis=-1)
                    meanTimePointDisp=np.mean(timePointsDisp,axis=1)
                    stdMeanTimePointDisp=np.std(meanTimePointDisp)
                    if stdMeanTimePointDisp>0.2:
                        self.cowInfos[i][8]=False
                    else:
                        self.cowInfos[i][8]=True
            else:
                self.timePointsDict[label]=[timeData]
        keys=list(self.timePointsDict.keys())
        deleteLabels=np.setdiff1d(keys,labels)
        for label in deleteLabels:
            self.timePointsDict.pop(label)
            
    def updateBreathRate(self):
        labels=np.array([info[0] for info in self.cowInfos])
        timeKeys=np.array(list(self.timePointsDict.keys()))
        indexs=np.argmax(timeKeys[:,np.newaxis]==labels,axis=1)
        for i in range(len(timeKeys)):
            key=timeKeys[i]
            if len(self.timePointsDict[key])==self.fftLength+1:
                if self.rgb is None:
                    timeVectors=cp.diff(self.timePointsDict[key],axis=0)
                    fftData=getUVSin(timeVectors)
                    rate=getCowBreathRate(fftData)
                else:
                    fftData=cp.array(self.timePointsDict[key])[1:,:]
                    rate=getRGBCowBreathRate(fftData)
                self.cowInfos[indexs[i]][7].append(rate)
            
    def getMeanRate(self,cowInfo):
        if len(cowInfo[7])>0:
            return cp.mean(cp.array(cowInfo[7]))
        else:
            return None
        
    def getRate(self,cowInfo):
        if len(cowInfo[7])>0:
            return cowInfo[7][-1]
        else:
            return None
            
    def getBackPoints(self,points,cowInfo):
        return points[cowInfo[3]]

    def getBreathPoints(self,points,cowInfo):
        return points[self.backLength:][cowInfo[4]]
        
    def getRoiImg(self,img,cowInfo):
        point=cowInfo[1]
        radius=cowInfo[2]
        point=point.astype(np.int32)
        roiImg=np.zeros((2*radius,2*radius,3),dtype=np.uint8)
        strX=point[0]-radius if point[0]-radius>=0 else 0
        strY=point[1]-radius if point[1]-radius>=0 else 0
        offsetX=0 if point[0]-radius>=0 else np.abs(point[0]-radius)
        offsetY=0 if point[1]-radius>=0 else np.abs(point[1]-radius)
        rimg=img[strY:point[1]+radius,strX:point[0]+radius].copy()
        roiImg[offsetY:offsetY+rimg.shape[0],offsetX:offsetX+rimg.shape[1]]=rimg
        return roiImg
        
    def getBoundingRect(self,cowInfo):
        point=cowInfo[1]
        radius=cowInfo[2]
        point=point.astype(np.int32)
        x0=point[0]-radius
        x1=point[0]+radius
        y0=point[1]-radius
        y1=point[1]+radius
        xyxy=[x0,y0,x1,y1]
        return xyxy