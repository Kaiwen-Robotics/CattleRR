import os

def clearDir(dirPath):
    for file in os.listdir(dirPath):
        filePath=os.path.join(dirPath,file)
        os.remove(filePath)
        
clearDir('Data')
clearDir('Output')