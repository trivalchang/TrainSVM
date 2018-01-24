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

sys.path.insert(0, '/Users/developer/guru/')

from utility import basics
from fileOp.imgReader import ImageReader
from fileOp.conf import Conf
from annotation.pascal_voc import pacasl_voc_reader
from feature.HOG import HOG

classInfo = []

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--conf", required=True,  help="json file configuration")
	ap.add_argument("-p", "--positive", required=False, help="path of the positive dataset")
	ap.add_argument("-n", "--negative", required=False, help="path of the negative dataset")

	args = vars(ap.parse_args())

	conf = Conf(args['conf'])
	classInfo = []
	if (conf['class'] != None):
		for name in open(conf['class']).read().split("\n"):
			print('{}'.format(name))
			classInfo.append(name)
			#print('{}'.name)
	
	orientation = conf['orientations']
	pixels_per_cell = conf['pixels_per_cell']
	cells_per_block = conf['cells_per_block']
	transform_sqrt = True if conf['transform_sqrt']==1 else False
	normalize = str(conf['normalize'])

	hog = HOG(orientation, pixels_per_cell, cells_per_block, transform_sqrt, normalize)
	labels = []
	featureList = []

	fileName = []
	if (args['positive'] != None):
		for f in os.listdir(args['positive']):
			fileName.append(args['positive']+'/'+f)
	if (args['negative'] != None):
		for f in os.listdir(args['negative']):
			fileName.append(args['negative']+'/'+f)

	for fpath in fileName:
		print('file {}'.format(fpath))
		if fpath.endswith(".xml") == False:
			continue
		voc = pacasl_voc_reader(fpath)
		#imgName, _, _, _ = voc.imageInfo()
		imgName = fpath.replace('.xml', '.png')
		objectList = voc.getObjectList()
		img = cv2.imread(imgName)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		for (className, (xmin, ymin, xmax, ymax)) in objectList:
			roi = img[ymin:ymax+1, xmin:xmax+1]
			if (className == '__distract'):
				labels.append(-1)
			else:	
				labels.append(classInfo.index(className))
			#print('roi shape {}'.format(roi.shape))
			(feature, _) = hog.describe(roi)
			featureList.append(feature)
			#print('		{}'.format(feature))
		#break

	model = SVC(kernel="linear", C=conf["C"], probability=True, random_state=42)
	model.fit(featureList, labels)

	f = open(conf["classifier_path"], "wb")
	f.write(pickle.dumps(model))
	f.close()
main()
