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
from fileOp.h5_dataset import h5_dump_dataset, h5_load_dataset

classInfo = []

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--conf", required=True,  help="json file configuration")
	ap.add_argument("-p", "--path", required=True, help="path of the dataset")
	ap.add_argument("-a", "--append", default=False, action='store_true', required=False, help="name of the dataset")

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

	fileList = []
	if (args['path'] != None):	
		for f in os.listdir(args['path']):
			if f.endswith(".xml") == False:
				continue
			fileList.append(args['path']+'/'+f)

	cnt = 0
	for fpath in fileList:
		print('file {}'.format(fpath))
		if fpath.endswith(".xml") == False:
			continue
		voc = pacasl_voc_reader(fpath)
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
			(feature, _) = hog.describe(roi)
			featureList.append(feature)
			cnt = cnt + 1
		if (cnt >= conf['sample_cnt']):
			break

	print('write {} feature'.format(len(labels)))
	h5_dump_dataset(featureList, 
						labels, 
						conf['feature_file'], 
						conf['dataset_feature_name'], 
						'a' if args['append']==True else 'w')

main()
