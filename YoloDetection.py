import numpy as np
from yolov5.detect import *

def getDetectGenerater(weights,source,data=False,imgsz=(640, 640),conf_thres=0.25,iou_thres=0.45,max_det=1000,device='',classes=None,
                                       agnostic_nms=False,augment=False,visualize=False,half=False,dnn=False,vid_stride=1,conditionDict=None):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    for path, im, im0s, vid_cap, s in dataset:
        if conditionDict is not None:
            vid_cap.set(cv2.CAP_PROP_POS_FRAMES, conditionDict['index']-0)
            path, im, im0s, vid_cap, s = next(dataset)
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        
        # Inference
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)
        
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
                xyxys=[]
                labels=[]
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xyxy=[int(v) for v in xyxy]
                    xyxys.append(xyxy)
                    labels.append(names[int(cls)])
                    
                yield im0,xyxys,labels
                
class WriteVideo:
    def __init__(self,outputPath,videoPath=None,fps=30):
        self.out=None
        self.fps=fps
        self.outputPath=outputPath
        if videoPath is not None:
            cap = cv2.VideoCapture(videoPath)
            self.fps = cap.get(cv2.CAP_PROP_FPS)
    
    def write(self,frame):
        if self.out is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(self.outputPath, fourcc, self.fps, (width, height))
        self.out.write(frame)
        
    def close(self):
        self.out.release()

if __name__ == '__main__':
    weights=r'C:\Users\Alan\Desktop\best2.pt'
    source=r'D:\BaiduNetdiskDownload\sp\herd1_0727_0902_01.mp4'
    generater=getDetectGenerater(weights,source,conf_thres=0.65)
    wv=WriteVideo(r'd:/1.avi')
    while True:
        try:
            img,xyxys,labels=next(generater)
            for i in range(len(xyxys)):
                x1,y1,x2,y2=xyxys[i]
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
                #cv2.putText(img, labels[i], ((x1+x2)//2-50,(y1+y2)//2-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0,0,255), 2)
            cv2.imshow('Frame',img)
            cv2.waitKey(1)
            wv.write(img)
        except StopIteration:
            wv.close()
            break