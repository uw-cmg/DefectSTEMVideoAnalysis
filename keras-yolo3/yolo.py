#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import math
import json

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.utils import multi_gpu_model
from collections import defaultdict
import pandas as pd 
gpu_num=1
imageName=""

class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/PaulIV.h5' # model path or trained weights path
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.score = 0.1
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()
        self.sumFrameDefectPosition=list()
        self.sumFrameDefectID= list()
        self.nextDefectID=0
        self.generatedList=list()
        self.frameNum=0
        self.distance_theshold = 50
        self.allframe=list()
        self.disappear=list()
        self.firstAppear=list()
        self.size=0  # pretend that we have a such thing after implement the watershed
        self.sizeChange=list()
        self.density=list()
        self.storeForLAST=list()
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        # current global lists
        current_newID=list()
        current_ID=list()
        current_Position=list()
        formeanfree=list()
        # tmpID = 0

        start = timer()
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 600
        
        if(self.frameNum==0):
            with open ("/home/guanzhao/Documents/NextGenCode/Precision&Recall/PaulIV/"+imageName[:-4]+".txt", 'a+') as output_file_forPR:
                output_file_forPR.write(imageName+" ")
            print("Frame: "+str(self.frameNum))
            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                #########################################################################
                size= np.sqrt((top-bottom)**2+(right-left)**2)
                ########################################################################
                centerX= left+(right-left)/2
                centerY= bottom+(top-bottom)/2
                        
                # print(label, (left, top), (right, bottom))
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # collecting defects information
                # tmpID = i + 1
                current_Position.append((self.nextDefectID, centerX, centerY,size,self.frameNum))
                current_ID.append(self.nextDefectID)
                current_newID.append(self.nextDefectID)
                self.firstAppear.append((self.nextDefectID, 0))
                #self.meanFreeDist.append(self.nextDefectID, 0)
                self.nextDefectID += 1

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                # draw.rectangle(
                #     [tuple(text_origin), tuple(text_origin + label_size)],
                #     fill=self.colors[c])
                
                #draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                with open ("/home/guanzhao/Documents/NextGenCode/Precision&Recall/PaulIV/"+imageName[:-4]+".txt", 'a+') as output_file_forPR:
                    output_file_forPR.write(str(top)+","+ str(left)+","+ str(bottom)+","+str(right)+","+ str(0)+" ")
                del draw

        if(self.frameNum==1):
            with open ("/home/guanzhao/Documents/NextGenCode/Precision&Recall/PaulIV/"+imageName[:-4]+".txt", 'a+') as output_file_forPR:
                output_file_forPR.write(imageName+" ")
            print("Frame: "+str(self.frameNum))
            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                centerX= left+(right-left)/2
                centerY= bottom+(top-bottom)/2
                ############################################################
                size= np.sqrt((top-bottom)**2+(right-left)**2)
                ############################################################
                        
                #print(label, (left, top), (right, bottom))
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])


                FlagFound = False
                # compare and update
                for item in self.sumFrameDefectPosition[0]:
                    if (centerX-item[1])**2+(centerY-item[2])**2 < self.distance_theshold:
                        current_Position.append((item[0],centerX,centerY,size,self.frameNum))
                        current_ID.append(item[0])
                        FlagFound = True
                        break 
                        # this a old defect ID which has been detected, so break loop

                # if this is a new defect
                if not FlagFound:
                    current_Position.append((self.nextDefectID,centerX,centerY,size,self.frameNum))
                    current_ID.append(self.nextDefectID)
                    current_newID.append(self.nextDefectID)
                    self.firstAppear.append((self.nextDefectID, 1))
                    self.nextDefectID += 1

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                # draw.rectangle(
                #     [tuple(text_origin), tuple(text_origin + label_size)],
                #     fill=self.colors[c])
                
                #draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                with open ("/home/guanzhao/Documents/NextGenCode/resultsForPR.txt", 'a+') as output_file_forPR:
                    output_file_forPR.write(str(top)+","+ str(left)+","+ str(bottom)+","+str(right)+","+ str(0)+" ")
                del draw

        if(self.frameNum==2):
            print("Frame: "+str(self.frameNum))
            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                centerX= left+(right-left)/2
                centerY= bottom+(top-bottom)/2
                ############################################################
                size= np.sqrt((top-bottom)**2+(right-left)**2)
                ############################################################
                        
                #print(label, (left, top), (right, bottom))
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])


                FlagFound = False
                # compare and update with frame 2
                for item in self.sumFrameDefectPosition[1]:
                    if (centerX-item[1])**2+(centerY-item[2])**2 < self.distance_theshold:
                        current_Position.append((item[0],centerX,centerY,size,self.frameNum))
                        current_ID.append(item[0])
                        FlagFound = True
                        break 
                        # this a old defect ID which has been detected, so break loop

                # check whether this is detected in frame 1
                if FlagFound is not True:
                    for item in self.sumFrameDefectPosition[0]:
                        if item[0] not in current_ID: # this is an old defects ID that is not tracked in frame 0
                            if (centerX-item[1])**2+(centerY-item[2])**2 < self.distance_theshold:
                                current_Position.append((item[0],centerX,centerY, size,self.frameNum))
                                current_ID.append(item[0])
                                FlagFound = True
                                break 

                # if this is a new defect
                if not FlagFound:
                    current_Position.append((self.nextDefectID,centerX,centerY, size, self.frameNum))
                    current_ID.append(self.nextDefectID)
                    current_newID.append(self.nextDefectID)
                    self.firstAppear.append((self.nextDefectID, 2))
                    self.nextDefectID += 1

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                # draw.rectangle(
                #     [tuple(text_origin), tuple(text_origin + label_size)],
                #     fill=self.colors[c])
                
                #draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                with open ("/home/guanzhao/Documents/NextGenCode/resultsForPR.txt", 'a+') as output_file_forPR:
                    output_file_forPR.write(str(top)+","+ str(left)+","+ str(bottom)+","+str(right)+","+ str(0)+" ")
                del draw
        
        if(self.frameNum==3):
            print("Frame: "+str(self.frameNum))
            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                centerX= left+(right-left)/2
                centerY= bottom+(top-bottom)/2
                ############################################################
                size= np.sqrt((top-bottom)**2+(right-left)**2)
                ############################################################
                        
                #print(label, (left, top), (right, bottom))
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])


                FlagFound = False
                # compare and update with frame 2
                for item in self.sumFrameDefectPosition[1]:
                    if (centerX-item[1])**2+(centerY-item[2])**2 < self.distance_theshold:
                        current_Position.append((item[0],centerX,centerY,size,self.frameNum))
                        current_ID.append(item[0])
                        FlagFound = True
                        break 
                        # this a old defect ID which has been detected, so break loop
                # compare and update with frame 3
                if FlagFound is not True:
                    for item in self.sumFrameDefectPosition[2]:
                        if (centerX-item[1])**2+(centerY-item[2])**2 < self.distance_theshold:
                            current_Position.append((item[0],centerX,centerY,size,self.frameNum))
                            current_ID.append(item[0])
                            FlagFound = True
                            break 

                # check whether this is detected in frame 1
                if FlagFound is not True:
                    for item in self.sumFrameDefectPosition[0]:
                        if item[0] not in current_ID: # this is an old defects ID that is not tracked in frame 0
                            if (centerX-item[1])**2+(centerY-item[2])**2 < self.distance_theshold:
                                current_Position.append((item[0],centerX,centerY, size,self.frameNum))
                                current_ID.append(item[0])
                                FlagFound = True
                                break 

                # if this is a new defect
                if not FlagFound:
                    current_Position.append((self.nextDefectID,centerX,centerY, size, self.frameNum))
                    current_ID.append(self.nextDefectID)
                    current_newID.append(self.nextDefectID)
                    self.firstAppear.append((self.nextDefectID, 2))
                    self.nextDefectID += 1

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                # draw.rectangle(
                #     [tuple(text_origin), tuple(text_origin + label_size)],
                #     fill=self.colors[c])
                
                #draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                with open ("/home/guanzhao/Documents/NextGenCode/resultsForPR.txt", 'a+') as output_file_forPR:
                    output_file_forPR.write(str(top)+","+ str(left)+","+ str(bottom)+","+str(right)+","+ str(0)+" ")
                del draw

        if(self.frameNum>3 and self.frameNum<50):
            print("Frame: "+str(self.frameNum))
            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                centerX= left+(right-left)/2
                centerY= bottom+(top-bottom)/2
                ############################################################
                size= np.sqrt((top-bottom)**2+(right-left)**2)
                ############################################################
                        
                #print(label, (left, top), (right, bottom))
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])
                
                FlagFound = False
                j=0
                gaiBreakLe= False
                while(j<self.frameNum):
                    j+=1
                    if FlagFound is not True:
                        for item in self.sumFrameDefectPosition[self.frameNum - j]:
                            if (centerX-item[1])**2+(centerY-item[2])**2 < self.distance_theshold:
                                current_Position.append((item[0],centerX,centerY, size, self.frameNum))
                                current_ID.append(item[0])
                                FlagFound = True
                                gaiBreakLe=True
                                break 
                    if gaiBreakLe is True:
                        break
                if not FlagFound:
                    current_Position.append((self.nextDefectID,centerX,centerY, size, self.frameNum))
                    current_ID.append(self.nextDefectID)
                    current_newID.append(self.nextDefectID)
                    self.firstAppear.append((self.nextDefectID,self.frameNum))
                    self.nextDefectID += 1

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                # draw.rectangle(
                #     [tuple(text_origin), tuple(text_origin + label_size)],
                #     fill=self.colors[c])
                
                #draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                with open ("/home/guanzhao/Documents/NextGenCode/resultsForPR.txt", 'a+') as output_file_forPR:
                    output_file_forPR.write(str(top)+","+ str(left)+","+ str(bottom)+","+str(right)+","+ str(0)+" ")
                del draw    


        if(self.frameNum >= 50):
            print("Frame: "+str(self.frameNum))
            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                centerX= left+(right-left)/2
                centerY= bottom+(top-bottom)/2
                ############################################################
                size= np.sqrt((top-bottom)**2+(right-left)**2)
                ############################################################
                        
                #print(label, (left, top), (right, bottom))
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])


                FlagFound = False
                j=0
                gaiBreakLe= False
                while(j<50):
                    j+=1
                    if FlagFound is not True:
                        for item in self.sumFrameDefectPosition[self.frameNum - j]:  
                            if item[0] not in current_ID: # this is an old defects ID that is not tracked in frame 0
                                if (centerX-item[1])**2+(centerY-item[2])**2 < self.distance_theshold:
                                    current_Position.append((item[0],centerX,centerY, size, self.frameNum))
                                    current_ID.append(item[0])
                                    FlagFound = True
                                    gaiBreakLe=True
                                    break
                    if gaiBreakLe is True:
                        break
                if not FlagFound:
                    current_Position.append((self.nextDefectID,centerX,centerY, size, self.frameNum))
                    current_ID.append(self.nextDefectID)
                    current_newID.append(self.nextDefectID)
                    self.firstAppear.append((self.nextDefectID,self.frameNum))
                    self.nextDefectID += 1

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                # draw.rectangle(
                #     [tuple(text_origin), tuple(text_origin + label_size)],
                #     fill=self.colors[c])
                
                #draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw    
        
        # update global information
        end = timer()
        self.sumFrameDefectPosition.append(current_Position)
        self.sumFrameDefectID.append(current_ID)
        self.generatedList.append((self.frameNum, current_newID))
        
        if(len(self.sumFrameDefectID)>3):
            for IDE in self.sumFrameDefectID[len(self.sumFrameDefectID)-4]:            #this line find if the defect is found in the last 3 frame but not in current
                if IDE in self.sumFrameDefectID[len(self.sumFrameDefectID)-3]:
                    if IDE in self.sumFrameDefectID[len(self.sumFrameDefectID)-2]:
                        if IDE not in self.sumFrameDefectID[len(self.sumFrameDefectID)-1]:
                            self.disappear.append((IDE, len(self.sumFrameDefectID)-1))
        # with open("reusltForFrame.txt", "r+") as f:
        #    f.write(str(self.sumFrameDefectPosition))
        #   f.write("\n"+str(self.frameNum)+"\n")
        # frame 0 is a list update for nextDefectsID
        #if (self.frameNum == 0):
        #    self.nextDefectID = tmpID + 1
        #    print(self.nextDefectID)
        self.allframe.append(self.frameNum)
        self.frameNum=self.frameNum+1
        self.density.append((len(out_boxes)/1.3124*1E18, self.frameNum))

        # if(self.frameNum>2):
        #     for item in self.sumFrameDefectPosition[len(self.sumFrameDefectPosition)-1]:
        #         for item2 in self.sumFrameDefectPosition[len(self.sumFrameDefectPosition)-2]:
        #             if(item[0]==item2[0]):
        #                 dist=math.sqrt((item[1]-item2[1])**2+(item[2]-item2[2])**2)
        #                 if(item[0] not in [i[0] for i in formeanfree]):
        #                     formeanfree.append((item[0], dist))
        #     self.storeForMFD.append((self.frameNum, sum(formeanfree)/len(formeanfree)))

        print(end - start)

        return image

    def close_session(self):
        if not os.path.exists('./wholeVideo/distanceForEachDefect/'):
            os.makedirs('./wholeVideo/distanceForEachDefect/')
        oneIDsizeChange=list()
        firstMechanism=list() # this is the disappear mechanism of growth
        alist=list()
        framesForPlot=list()
        storeForPosition=list()
        firstMechanismPos=list()
        forclearance=list()
        temp=list()
        list1=list()
        list2=list()
        sumUp=0
        loacalStoreForMFD={}

        ################################ This is for mean free distance######################
        for i in range(len(self.sumFrameDefectPosition)-2):
            for item in self.sumFrameDefectPosition[i]:
                for item2 in self.sumFrameDefectPosition[i+1]:
                    if(item[0]==item2[0]):
                        
                        if(item[0] in loacalStoreForMFD.keys()):
                            loacalStoreForMFD[item[0]].append((i, math.sqrt((item[1]-item2[1])**2+(item[2]-item2[2])**2)))
                        else:
                            loacalStoreForMFD[item[0]]=[(i, math.sqrt((item[1]-item2[1])**2+(item[2]-item2[2])**2))]
        for key, values in loacalStoreForMFD.items():
            for item in values:
                sumUp+=item[1]
            mean=sumUp/len(values)
            self.storeForMFD.append((key,mean))
            sumUp=0

        for item in self.storeForMFD:
            list1.append(item[0])
            list2.append(item[1])
        df = pd.DataFrame( {'DefectID': list1,'mean distance': list2})
        df.to_csv("./wholeVideo/mean.csv",header = True)

        ######################################################################################

        for item in self.disappear: #disappear has (id, frameNum)
            for everyitme in self.firstAppear:
                if item[0]== everyitme[0]:
                    firstframeAppear= everyitme[1]
                    lastframeAppear=item[1]
                    framesForPlot.append((item[0], firstframeAppear, lastframeAppear))
                    for i in range(firstframeAppear, lastframeAppear): # i is the frame 
                        for defects in self.sumFrameDefectPosition[i]:
                            if defects[0]== item[0]:
                                oneIDsizeChange.append((defects[3],defects[4]))
                                storeForPosition.append((i,defects[1],defects[2],defects[4]))
                    self.sizeChange.append((item[0], oneIDsizeChange))
                    self.storeForLAST.append((item[0], storeForPosition))
                    oneIDsizeChange=[]
                    storeForPosition=[]
        #max(data, key=lambda item: item[1]
        for item in self.sizeChange: #sizeChange: (id,((size, frameNum), (size,frameNum), (size, frameNum)....)) 
            maxSize=max(item[1], key=lambda item: item[0])
            minSize=min(item[1], key=lambda item: item[0])

            if((maxSize[0]-minSize[0])/minSize[0])> 1.2:
                firstMechanism.append(item)

        for XXX in self.storeForLAST:
            for YYY in firstMechanism:
                if(XXX[0]==YYY[0]):
                    firstMechanismPos.append(XXX)

        # num Lists to store total defects
        for clearance in firstMechanism:
            if clearance not in forclearance:
                forclearance.append(clearance)
        firstMechanism=forclearance
        forclearance=[]

        for clearance in firstMechanismPos:
            if clearance not in forclearance:
                forclearance.append(clearance)
        firstMechanismPos=forclearance
        forclearance=[]
        for key, values in loacalStoreForMFD.items():
            with open("./wholeVideo/distanceForEachDefect/"+str(key)+".txt", 'w+') as filehandle:
                filehandle.write(str(values))

        with open("./wholeVideo/sizeChange.txt", "w+") as filehandle:
            filehandle.write("Total: "+str(len(self.sizeChange))+" loops disappeared")
            for item in self.sizeChange:
                filehandle.write(str(item[0]))
                filehandle.write("\n")
                filehandle.write(str(item[1]))
                filehandle.write("\n")
                
        print("Total: "+str(len(firstMechanism))+" firstMechanism")
        for item in firstMechanism:
            with open("./wholeVideo/firstMechanism/"+str(item[0])+".txt", "w+") as filehandle:
                filehandle.write(str(item[1]))
                filehandle.write("\n")
        
        with open('./wholeVideo/density.txt', 'w+') as filehandle:
            for items in self.density:
                filehandle.write(str(items[0])+","+str(items[1]))
                filehandle.write('\n')
        
        with open('./wholeVideo/firstMechanismPos.txt','w+') as filehandle:
            for item in firstMechanismPos:
               filehandle.write(str(item[0]))
               filehandle.write("\n")
               filehandle.write(str(item[1]))
               filehandle.write("\n")

        num_total = list()
        for item in self.sumFrameDefectID:
            num_total.append(len(item))
        # num_list of new defects
        # first reorgnize the new generated defects
        generatedDict = dict()
        for (frame_i, idList) in self.generatedList:
            generatedDict[frame_i] = len(idList)
            # print("dicttionary:  "+str(generatedDict))
        # then populate all the lists
        num_generated = list()
        for i in range(0,len(self.sumFrameDefectID)):
            if i in generatedDict.keys():
                num_generated.append(generatedDict[i])
            else:
                num_generated.append(0)
        # check correctedness
        assert(len(num_generated) == len(num_total))
        # calculate number of annihilation
        num_annihilation = list()
        # first frame shuold be 0
        num_annihilation.append(0)
        for i in range(1,len(num_total)):
             num_annihilation.append(num_total[i-1] - (num_total[i] - num_generated[i]))
        # store files
        #with open("reusltForFrame.txt", "r+") as f:
        #    f.write(str(self.sumFrameDefectPosition))
        #    f.write("\n"+str(self.frameNum)+"\n")
        with open('./wholeVideo/sumFrameDefectPosition.txt', 'w+') as filehandle:  
            filehandle.write( str(self.sumFrameDefectPosition))

        with open('./wholeVideo/sumFrameID.txt', 'w+') as filehandle:  
            filehandle.write( str(self.sumFrameDefectID))

        with open('./wholeVideo/sumNewDefectsID.txt', 'w+') as filehandle:  
            filehandle.write( str(self.generatedList))

        with open('./wholeVideo/pltGenerateAndAnni.txt', 'w+') as filehandle:
            for i in range(len(self.allframe)):
                filehandle.write("\n"+ str(i)+"," + str(len(self.sumFrameDefectID[i]))+","+str(len(self.generatedList[i][1]))+","+str(num_annihilation[i]))
        with open('./wholeVideo/disappearID.txt', 'w+') as filehandle:
            filehandle.write(str(self.disappear))
        # store all the number information into a csv
        df = pd.DataFrame( {'TotalNum': num_total,'GeneratedNum': num_generated,'AnnihilationNum': num_annihilation})
        df.to_csv("./wholeVideo/NumberStats.csv",header = True)
        # store all results into the local
        print('------------ Rainbow -------------')
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    # video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_FourCC = cv2.VideoWriter_fourcc(*'XVID')
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        if not return_value:
            break
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        #cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    # with open("reusltForFrame.txt", "r+") as f:
    #     f.write(self.sumFrameDefectPosition)

    yolo.close_session()


def detect_img(yolo):
    global imageName
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            imageName=img
            r_image.save('detect'+img, "JPEG")
    yolo.close_session()



if __name__ == '__main__':
    detect_img(YOLO())
