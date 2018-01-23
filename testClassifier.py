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
	ap.add_argument("-c", "--conf", required=True, help="json file configuration")
	ap.add_argument("-t", "--test", required=True, help="path of the test image")

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

	xmlName = args['test']
	voc = pacasl_voc_reader(xmlName)

	imgName = xmlName.replace('.xml', '.png')
	img = cv2.imread(imgName)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	objectList = voc.getObjectList()
	for (className, (xmin, ymin, xmax, ymax)) in objectList:
		roi = img[ymin:ymax+1, xmin:xmax+1]
		(feature, _) = hog.describe(roi)

		idx = model.predict([feature])
		realIdx = classInfo.index(className) if className != '__distract' else -1
		print('predict = {}, real = {}'.format(idx, realIdx))

main()
