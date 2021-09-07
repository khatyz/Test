import cv2
from os import listdir
from typing import List
from mahotas import features
import numpy as np
import pandas as pd

# define haralick function
def haralick2(file, foldername, imageClass):
    features_haralick = features.haralick(file)
    features_haralick = [var_each_feat for var_sub_array in features_haralick for var_each_feat in var_sub_array]
    final_list = np.append(foldername, features_haralick)
    final_list = np.append(final_list, imageClass)
    return final_list


image_path = 'DatasetTexture/'
image_dir : List[str] = listdir(image_path)
ListOfList_features = list()
iteration = 1
for folder in image_dir:
    # Pick all images of each subfolder
    for image in listdir(image_path + folder):
        path = image_path + folder + '/' + image
        # img = cv2.imread(path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.imread(path, 0) # Returns a grayscale image (path, 0)
        haralick_vector = haralick2(img, folder, iteration)
        #print('%s'% haralick_vector)
        # Add each feature vector to the list of all features
        ListOfList_features.append(haralick_vector)
        print('Lenght of final list is %d: '% len(ListOfList_features))
    print('Folder %s successfully extracted (Class: %d). ' % (folder, iteration))
    iteration  += 1
print('Features Extraction done!')
print('Start generating the dataframe from the final list ->')
df = pd.DataFrame.from_records(data=ListOfList_features)
print('Dataframe generated!')
print('Start generating the csv file from the Dataframe ->')
df.to_csv('MyCSV/feat_haralick3.csv', sep=',', index=False)
print('CSV generated!')
print('Process finished!')