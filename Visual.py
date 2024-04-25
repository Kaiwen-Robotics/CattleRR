import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

class PyplotImage:
    
    def __init__(self,size=None,axisOff=True):
        fig, self.plt = plt.subplots()
        if size is not None:
            fig.set_size_inches(size[0]/fig.dpi, size[1]/fig.dpi)
        self.canvas=fig.canvas
        self.axisOff=axisOff
        
    def numpy(self):
        if self.axisOff==True:
            self.plt.set_axis_off()
            self.plt.set_position([0, 0, 1, 1])
            self.plt.set_frame_on(False)
        self.canvas.draw()
        data = self.canvas.tostring_rgb()
        width, height = self.canvas.get_width_height()
        img = np.fromstring(data, dtype=np.uint8)
        img = img.reshape((height, width, 3))
        return img
    
def drawQuiver(img,points,uvs):
    uvs=np.array(uvs)
    points=np.array(points)
    nextPoints=points+uvs
    border=10
    points=points[(nextPoints[:,0]>=border) & (nextPoints[:,0]<img.shape[1]-border) & (nextPoints[:,1]>=border) & (nextPoints[:,1]<img.shape[0]-border)]
    uvs=uvs[(nextPoints[:,0]>=border) & (nextPoints[:,0]<img.shape[1]-border) & (nextPoints[:,1]>=border) & (nextPoints[:,1]<img.shape[0]-border)]
    plti=PyplotImage(img.shape[0:2][::-1])
    img=img[::-1,:]
    plti.plt.imshow(img,extent=[0, img.shape[1], 0, img.shape[0]])
    plti.plt.quiver(points[:,0],points[:,1],uvs[:,0],uvs[:,1],color='g')
    img=plti.numpy()[::-1,:]
    return img