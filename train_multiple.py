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

from fileOp.conf import Conf
from annotation.pascal_voc import pacasl_voc_reader
from feature.HOG import HOG
from fileOp.h5_dataset import h5_load_dataset

classInfo = []

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--conf", required=True,  help="json file configuration")

	args = vars(ap.parse_args())

	conf = Conf(args['conf'])
	classInfo = []
	if (conf['class'] != None):
		for name in open(conf['class']).read().split("\n"):
			if (name == ''):
				continue
			print('{}'.format(name))
			classInfo.append(name)
			#print('{}'.name)
	
	(featureList, labels) = h5_load_dataset(conf['feature_file'], conf['dataset_feature_name'])
	if conf['dataset_hard_negative']!="":
		(hard_featureList, hard_labels) = h5_load_dataset(args['dataset_hard_negative'], 'hard_negative')

	if conf['multiple_classifier']==1:
		for classIdx in range(0, len(classInfo)):
			labelsToFit = np.asarray(labels)

			positive_cnt = conf["positive_sample"]
			negative_cnt = conf["negative_sample"]
			PWhere = np.where(labelsToFit == classIdx)[0]
			NWhere = np.where(labelsToFit != classIdx)[0]
			if positive_cnt < len(PWhere):
				PWhere = np.random.choice(PWhere, positive_cnt)
			if negative_cnt < len(NWhere):
				NWhere = np.random.choice(NWhere, negative_cnt)
			
			PLabels = np.zeros(len(PWhere))
			NLabels = np.full(len(NWhere), -1)
			labelsToFit = np.hstack([PLabels, NLabels])
			PFeatureList = np.take(featureList, PWhere, 0)
			NFeatureList = np.take(featureList, NWhere, 0)
			featureToFit = np.vstack([PFeatureList, NFeatureList])
			
			hardFeatureList = []
			if conf['dataset_hard_negative']!="":
				hardFile = conf['classifier'+str(classIdx)+'_hard']
				(hardFeatureList, hardLabels) = h5_load_dataset(hardFile, 'hard_negative')
				labelsToFit = np.hstack([labelsToFit, hardLabels])
				featureToFit = np.vstack([featureToFit, hardFeatureList])
			
			print('training class {} by {} positive {} negative {}'.format(classInfo[classIdx], len(PFeatureList), len(NFeatureList), len(hardFeatureList)))
			
			model = SVC(kernel="linear", C=conf["C"], probability=True, random_state=42)
			model.fit(featureToFit, labelsToFit)

			f = open(conf['classifier'+str(classIdx)+'_path'], "wb")
			f.write(pickle.dumps(model))
			f.close()
	else:
		(featureList, labels) = h5_load_dataset(conf['feature_file'], conf['dataset_feature_name'])
		print('total {} normal features'.format(len(labels)))
		print('write to {}'.format(conf["classifier_path"]))
		model = SVC(kernel="linear", C=conf["C"], probability=True, random_state=42)
		model.fit(featureList, labels)

		f = open(conf["classifier_path"], "wb")
		f.write(pickle.dumps(model))
		f.close()

main()
