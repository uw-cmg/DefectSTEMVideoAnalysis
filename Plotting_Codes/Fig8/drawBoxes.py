
import cv2
from os import walk, getcwd
from PIL import Image
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
#from skimage import data
linewidth = 5
yume=10
# Mag=1340/416
# MagY=952/416
with open("./wholeVideo/positionforeachDefect/87.txt") as filehandle:
    images= filehandle.readline()
    imagelist= images.split("),")
    images=[]
    for image in imagelist:
        images.append(image.replace("(", "").replace("[","").strip().split(","))
    listToOpen=list()
    for item in images:
        listToOpen.append((item[0], (item[1], item[2])))
    for item in listToOpen:
        im = mpimg.imread("./frame_crop/Frame"+item[0]+".jpg")
        #resized_image = cv2.resize(im, (1340, 952))
        fig, ax = plt.subplots(figsize=(50, 50))
        Xmin= float(item[1][0])-yume
        Xmax= float(item[1][0])+yume
        Ymin= float(item[1][1])-yume
        Ymax= float(item[1][1])+yume
        width=2*yume
        height=2*yume
        rect = patches.Rectangle((Xmin, Ymin), width, height, linewidth=linewidth, edgecolor='deepskyblue', facecolor = 'none')
        ax.add_patch(rect)
        ax.imshow(im)
        print(item[0])
        if(int(item[0])%50==0):
            fig.savefig("./trajectoryImagesPlotted/"+item[0]+".jpg",format='jpeg',dpi=300, bbox_inches="tight",pad_inches=0.2)
        plt.close()