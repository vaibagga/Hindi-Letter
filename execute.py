# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 23:14:48 2018

@author: lenovo
"""

import os
import numpy as np
from keras.models import load_model
import h5py
import cv2

characters  = ['character_10_yna',
 'character_11_taamatar',
 'character_12_thaa',
 'character_13_daa',
 'character_14_dhaa',
 'character_15_adna',
 'character_16_tabala',
 'character_17_tha',
 'character_18_da',
 'character_19_dha',
 'character_1_ka',
 'character_20_na',
 'character_21_pa',
 'character_22_pha',
 'character_23_ba',
 'character_24_bha',
 'character_25_ma',
 'character_26_yaw',
 'character_27_ra',
 'character_28_la',
 'character_29_waw',
 'character_2_kha',
 'character_30_motosaw',
 'character_31_petchiryakha',
 'character_32_patalosaw',
 'character_33_ha',
 'character_34_chhya',
 'character_35_tra',
 'character_36_gya',
 'character_3_ga',
 'character_4_gha',
 'character_5_kna',
 'character_6_cha',
 'character_7_chha',
 'character_8_ja',
 'character_9_jha',
 'digit_0',
 'digit_1',
 'digit_2',
 'digit_3',
 'digit_4',
 'digit_5',
 'digit_6',
 'digit_7',
 'digit_8',
 'digit_9']

characters.sort()

os.chdir('C://Users//lenovo//Desktop//Docs//Hindi Letters//DevanagariHandwrittenCharacterDataset')
model = load_model('model.h5')

file_path = 'C://Users//lenovo//Desktop//Docs//Hindi Letters//DevanagariHandwrittenCharacterDataset//4201.png'
img = cv2.imread(file_path, 0)
img = cv2.resize(img, (28, 28))
img = np.reshape(img, (-1, 28, 28, 1))
img.shape
model.summary()
pred = model.predict(img)