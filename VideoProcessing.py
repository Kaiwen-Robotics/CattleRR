import cv2
import threading
import numpy as np
import time
lock = threading.Lock()

class VideoProcessing:
    
    def __init__(self,filePath,fps=None,size=None,readCount=None):
        self.cap=None
        self.fps=fps
        self.filePath=filePath
        self.images=[]
        self.grayGpus=[]
        self.videoWriter=None
        if fps is None:
            self.cap = cv2.VideoCapture(filePath)
        if readCount is not None:
            self.startThreadRead(readCount)
            
    def startThreadRead(self,readCount):
        def loopRead():
            while True:
                if len(self.images) < readCount:
                    frame=self.read()
                    if frame is not None:
                        gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
                        grayGpu = cv2.cuda_GpuMat(gray.astype(np.float32))
                    else:
                        grayGpu = None
                    self.images.append(frame)
                    self.grayGpus.append(grayGpu)
                    if frame is None:
                        break
                else:
                    time.sleep(0.0001)
        self.thread = threading.Thread(target=loopRead)
        self.thread.start()
        
    def readThread(self):
        with lock:
            while self.thread.is_alive() or len(self.images)>0:
                if len(self.images)>0:
                    frame=self.images.pop(0)
                    grayGpu=self.grayGpus.pop(0)
                    break
        return frame,grayGpu
            
    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            frame=None
        else:
            frame=frame[:,:,[2,1,0]]
            frame=np.ascontiguousarray(frame)
        return frame
    
    def write(self,img):
        if self.videoWriter is None:
            self.videoWriter=self.getVideoWriter(self.filePath,self.fps,[img.shape[1],img.shape[0]])
        self.videoWriter.write(img[:,:,[2,1,0]])
            
    def setFrame(self,frameIndex):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
            
    def getVideoWriter(self,filePath,fps,size):
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        videoWriter = cv2.VideoWriter(filePath,fourcc,fps,size)
        return videoWriter
    
    def close(self):
        if self.cap is not None:
            self.cap.release()
        else:
            self.videoWriter.release()

if __name__=='__main__':
    vp=VideoProcessing(r'C:\Users\Alan\Desktop\CODE9 (3)\CODE9 (2)\Output\5.avi')
    vp2=VideoProcessing(r'd:/output.avi',30,[294,294])
    while True:
        img=vp.read()
        if img is not None:
            vp2.write(img)
        else:
            vp2.close()
            break