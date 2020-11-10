import os
#import skimage
import cv2
from os import walk, getcwd
from PIL import Image
import PIL
import math
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import statistics
import numpy as np

for i in range(1176):
    im = cv2.imread("./frame3/frame"+str(i+1)+".jpg")
    try:
        os.mkdir("./crop_set1_final/"+str(i+1))
        os.mkdir("./crop_set2_final/"+str(i+1))
    except OSError:
        print("Creation of the directory is failed")

    
    with open("./result5/"+str(i+1)+".txt") as f:
        inputData = f.readline().split()
        l = 0
        k = 10
        list_1 =[]
        for line in inputData:

            splLine= line.split(",")
            Xmin= int(splLine[1])
            Xmax= int(splLine [3])
            Ymin= int(splLine[2])
            Ymax= int(splLine[0])

            width = Xmax-Xmin
            height= Ymax-Ymin

            a = Ymax - k
            b = Ymin + k
            c = Xmax + k
            d = Xmin - k
            if(a<0):
                a = 0
            if(b>1728):
                b = 1728
            if(c>=2412):
                c = 2412
            if(d<=0):
                d = 0

            #save enlarged bounding box information
            with open ("./enlarged_bbx/"+str(i+1)+".txt", 'a+') as output_file_forPR:
                output_file_forPR.write(str(a)+","+ str(d)+","+ str(b)+","+str(c)+"\n")

            # crop_im1 = im[Ymax:Ymin, Xmin:Xmax]
            crop_im2 = im[a:b, d:c]

            # diameter1 = max(width*(1344/2412)/4.8248, -height*(962/1728)/4.8248)
            # list_1.append(diameter1)       

            #cv2.imshow("cropped", crop_im2)       
            cv2.imwrite('./crop_set2_final/'+str(i+1)+'/'+str(l)+'.jpg', crop_im2)

            l = l + 1

with open("./size_watershed_final/allframe.txt",'w+') as out_Results:
    out_Results.write("frame"+","+"longaxis"+","+"x"+","+"y"+"\n")

for i in range(1176):

    try:
        os.mkdir("./convert_set2_final/"+str(i+1))
        os.mkdir("./fitEllipse_final/"+str(i+1))
    except OSError:
        print("Creation of the directory is failed")

    #convert to binary plot
    path1 = './crop_set2_final/'+str(i+1)
    for filename in os.listdir(path1):
        img = cv2.imread(str(path1)+ '/'+ filename)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        # Finding sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,1,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
        # Finding unknown area
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown ==255] = 0
        markers = cv2.watershed(img, markers)
        img[markers == -1] = [0, 255, 0]
        
        #cv2.imshow('p',sure_fg)
        cv2.imwrite('./convert_set2_final/'+str(i+1)+'/'+filename, sure_fg)
        cv2.waitKey(0)

    #fit ellipse and draw ellipse to every frame
    list_2 = []
    list_3 = []
    list_4 = []
    list_5 = []
    path2 = './convert_set2_final/'+str(i+1)
    frame_im = cv2.imread("./frame3/frame"+str(i+1)+".jpg")

    with open("./enlarged_bbx/"+str(i+1)+".txt") as f:
        inputData = f.readlines()
        for line in inputData:
            splLine= line.split(",")
            list_3.append([splLine[0],splLine[1]])#enlargerd bbx upper left coordinate

    for filename in os.listdir(path2):
        img1 = cv2.imread(str(path2)+ '/'+ filename)
        img2 = cv2.imread(str(path1)+ '/'+ filename)
        imgray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,180,255,cv2.THRESH_BINARY)
        contours,hierarchy= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)       
        cnt = contours[0]

        if(len(contours[0]) >= 5):
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(img2,ellipse,(0,255,0),2) 
            cv2.imwrite('./fitEllipse_final/'+str(i+1)+'/'+filename, img2)
            newEll = (ellipse[0][0]+float(list_3[int(filename[:len(filename)-4])][1]),ellipse[0][1]+float(list_3[int(filename[:len(filename)-4])][0])),ellipse[1],ellipse[2]
            cv2.ellipse(frame_im,newEll,(0,255,0),2)
            
        else:
            cv2.imwrite('./fitEllipse_final/'+str(i+1)+'/'+filename, img2)

        a,b,c=cv2.minAreaRect(cnt)
        list_2.append(max(b[0],b[1])*(1344/2412)/4.8248) # long axis
        list_4.append(a[0]+float(list_3[int(filename[:len(filename)-4])][1])) # center of x
        list_5.append(a[1]+float(list_3[int(filename[:len(filename)-4])][0])) # center of y

    # if(i == 105 or i == 462 or i == 818 or i == 1174):
    # with open("./size_watershed_final/frame"+ str(i+1) +".txt",'w+') as out_Results:
    #     for w in range(len(list_2)):
    #         out_Results.write(str(list_2[w])+"\n")

    with open("./size_watershed_final/allframe.txt",'a+') as out_Results:
        for w in range(len(list_2)):
            out_Results.write(str(i)+","+str(list_2[w])+","+str(list_4[w])+","+str(list_5[w])+"\n")


    cv2.imwrite('./frameplot_final/frame'+str(i+1)+'.jpg',frame_im)





