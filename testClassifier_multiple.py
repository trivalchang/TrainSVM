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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/utility')

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
	ap.add_argument("-l", "--label", required=True, help="class label index to classify")
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

	classIdx = classInfo.index(args['label'])
	model = pickle.loads(open(conf['classifier'+str(classIdx)+'_path'], "rb").read())

	fileList = []
	for f in os.listdir(args['test']):
		if f.endswith(".xml") == False:
			continue
		fileList.append(args['test']+'/'+f)

	negativeFeatureList = []
	negativeLabels = []
	error_predict_cnt = 0
	total_predict_cnt = 0
	for xmlName in fileList:
		voc = pacasl_voc_reader(xmlName)

		imgName = xmlName.replace('.xml', '.png')
		img = cv2.imread(imgName)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		objectList = voc.getObjectList()
		#print('classify {}'.format(imgName))
		for (className, (xmin, ymin, xmax, ymax)) in objectList:
			roi = img[ymin:ymax+1, xmin:xmax+1]
			(feature, _) = hog.describe(roi)

			predictIdx = model.predict([feature])

			realIdx = 0 if className != '__distract' and classIdx == classInfo.index(className) else -1
			
			#print('predict = {}, real = {}'.format(predictIdx, realIdx))
			if realIdx != predictIdx:
				#print('		predict error: {} - {}, real {} predict {}'.format(imgName, (xmin, ymin, xmax, ymax), realIdx, predictIdx))
				error_predict_cnt = error_predict_cnt + 1
				if realIdx == -1:
					negativeFeatureList.append(feature)
					negativeLabels.append(-1)
			total_predict_cnt = total_predict_cnt + 1
	
	print('===========================')
	print('Running classifier for {} total {} errors {}'.format(args['label'], total_predict_cnt, error_predict_cnt))
	if (args['hard'] == True):
		h5_dump_dataset(negativeFeatureList, 
						negativeLabels, 
						conf['classifier'+str(classIdx)+'_hard'], 
						'hard_negative', 
						'w')
main()
