import numpy as np
from yolov5.segment.predict import *
import time

def getSegmentGenerater(weights,source,device='',imgsz=(640, 640),conf_thres=0.25,iou_thres=0.45,
           max_det=1000,agnostic_nms=False,classes=None,vid_stride=1,augment=False,visualize=False,conditionDict=None):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=False, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz=imgsz, s=stride)
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
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
            pred, proto = model(im, augment=augment, visualize=visualize)[:2]
            
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
        
        mask=None
        for i, det in enumerate(pred):  # per image
            im0=im0s.copy()
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                #mask, _ = torch.max(masks, dim=0)
                mask=masks[0,:,:]
                for k in range(1,masks.shape[0]):
                    mask+=masks[k,:,:]*(k+1)
                mask=mask.cpu().numpy().astype(np.uint8)
        xyxys=[]
        labels=[]
        for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
            xyxy=[int(v) for v in xyxy]
            xyxys.append(xyxy)
            labels.append(names[int(cls)])
            
        yield im0,mask,xyxys,labels

if __name__=='__main__':
    weights=r'C:\Users\Alan\Desktop\DSK2\best.pt'
    source=r'D:\BaiduNetdiskDownload\sp\herd1_0727_0902_01.mp4'
    generater=getDetectGenerater(weights,source)
    index=0
    while True:
        try:
            img,masks,xyxys,labels=next(generater)
            print(np.unique(masks))
            np.save('d:/1.npy',masks.astype(np.uint8))
            #np.savez(os.path.join(r'D:\Output',str(index)+'.npz'),img,masks,xyxys,labels)
            #img[masks[0,:,:]>0]=[0,255,0]
            #cv2.imshow('frame',img)
            #key=cv2.waitKey(1)
            #print(labels[0])
            #if key==ord('q'):
            #    cv2.destroyAllWindows()
            #    raise StopIteration
            index+=1
        except StopIteration:
            break