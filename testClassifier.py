from __future__ import print_function

import numpy as np
import argparse
import glob
import cv2
import os 
import sys
import random
from sklearn.svm import SVC
import pickle

sys.path.append('/Users/developer/guru/utility')

#from utility import basics
#from fileOp.imgReader import ImageReader
from fileOp.conf import Conf
from annotation.pascal_voc import pacasl_voc_reader
from feature.HOG import HOG
from fileOp.h5_dataset import h5_dump_dataset

classInfo = []

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--conf", required=True, help="json file configuration")
	ap.add_argument("-t", "--test", required=True, help="path of the test image")
	ap.add_argument("-n", "--hard", default=False, action='store_true', required=False, help="write hard negative")
	
	args = vars(ap.parse_args())

	conf = Conf(args['conf'])
	classInfo = []
	if (conf['class'] != None):
		for name in open(conf['class']).read().split("\n"):
			classInfo.append(name)
	
	orientation = conf['orientations']
	pixels_per_cell = conf['pixels_per_cell']
	cells_per_block = conf['cells_per_block']
	transform_sqrt = True if conf['transform_sqrt']==1 else False
	normalize = str(conf['normalize'])

	hog = HOG(orientation, pixels_per_cell, cells_per_block, transform_sqrt, normalize)

	model = pickle.loads(open(conf["classifier_path"], "rb").read())

	fileList = []
	for f in os.listdir(args['test']):
		if f.endswith(".xml") == False:
			continue
		fileList.append(args['test']+'/'+f)

	negativeFeatureList = []
	negativeLabels = []
	error_predict_cnt = 0
	total_predict_cnt = 0
	e1 = cv2.getTickCount()
	for xmlName in fileList:
		voc = pacasl_voc_reader(xmlName)

		imgName = xmlName.replace('.xml', '.png')
		img = cv2.imread(imgName)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		objectList = voc.getObjectList()
		print('classify {}'.format(imgName))
		for (className, (xmin, ymin, xmax, ymax)) in objectList:
			roi = img[ymin:ymax+1, xmin:xmax+1]
			(feature, _) = hog.describe(roi)

			predictIdx = model.predict([feature])
			realIdx = classInfo.index(className) if className != '__distract' else -1
			#print('predict = {}, real = {}'.format(idx, realIdx))
			if realIdx != predictIdx:
				print('		predict error: {} - {}, real {} predict {}'.format(imgName, (xmin, ymin, xmax, ymax), realIdx, predictIdx))
				error_predict_cnt = error_predict_cnt + 1
				if realIdx == -1:
					negativeFeatureList.append(feature)
					negativeLabels.append(-1)
					print('			write hard negative')
			total_predict_cnt = total_predict_cnt + 1

	e2 = cv2.getTickCount()
	time = (e2 - e1)/ cv2.getTickFrequency()
	time = time * 1000
	print('===========================')
	print('total {} predicts, {} errors '.format(total_predict_cnt, error_predict_cnt))
	print('each predict takes {}'.format(time/total_predict_cnt))
	if (args['hard'] == True):
		print('write {} hard negative feature'.format(len(negativeLabels)))
		h5_dump_dataset(negativeFeatureList, 
					negativeLabels, 
					'./output/icon_featur.hdf5', 
					'hard_negative', 
					'a')

main()
