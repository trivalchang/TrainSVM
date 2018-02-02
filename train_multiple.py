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
from fileOp.h5_dataset import h5_load_dataset

classInfo = []

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--conf", required=True,  help="json file configuration")
	ap.add_argument("-p", "--path", required=True, help="path of the dataset")
	ap.add_argument("-d", "--dataset", required=True, help="name of the dataset")

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
	(featureList, labels) = h5_load_dataset(args['path'], args['dataset'])
	(hard_featureList, hard_labels) = h5_load_dataset(args['path'], 'hard_negative')

	for classIdx in range(0, len(classInfo)):
		idx = np.where(np.asarray(labels)==classIdx)
		labelByClass = np.take(labels, idx)
		featureByClass = np.take(featureList, idx)
		model = SVC(kernel="linear", C=conf["C"], probability=True, random_state=42)
		model.fit(featureList, labels)

		f = open(conf['classifier'+str(classIfx)+'_path'], "wb")
		f.write(pickle.dumps(model))
		f.close()

main()
