import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import *
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import glob
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
import pickle


def color_hist(img, hist_bins=32, color_range=(0, 256)):
    ch1 = np.histogram(img[:, :, 0], hist_bins, range=color_range)
    ch2 = np.histogram(img[:, :, 1], hist_bins, range=color_range)

    ch3 = np.histogram(img[:, :, 2], hist_bins, range=color_range)
    hist_features = np.hstack((ch1[0], ch2[0], ch3[0]))
    return hist_features



def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size)
    return features.ravel()


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features



def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, hist_bins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)



def extract_features(imgs, cspace='RGB', orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_size=(16, 16), hist_bins=32,
                     spatial_feet=True, hist_feet=True):
    # Create a list to append feature vectors to
    #创建一个列表来追加特征向量
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_feature = single_img_features(image, color_space=cspace, spatial_size=spatial_size,
                                          hist_bins=hist_bins, orient=orient,
                                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                          hog_channel=hog_channel,
                                          spatial_feat=spatial_feet, hist_feat=hist_feet, hog_feat=True)
        features.append(img_feature)

    # Return list of feature vectors
    return features


def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    print(data_dict)
    # Read in a test image, either car or notcar
    # Define a key "image_shape" and store the test image shape 3-tuple
    test_img = cv2.imread(car_list[0])
    data_dict["image_shape"] = test_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = test_img.dtype
    # Return data_dict
    return data_dict



car_path=[]
notcar_path=[]
car_dir=os.listdir('../../all/vehicles')
notcar_dir=os.listdir('../../all/non-vehicles')
cars=[]
notcars=[]
for path in car_dir:
    car_path.append('D:/gitcode/linux-udacity/all/vehicles'+'/'+path)
for path in notcar_dir:
    notcar_path.append('D:/gitcode/linux-udacity/all/non-vehicles'+'/'+path)

for path in car_path:
    if os.path.isdir(path)==True:
        infile=glob.glob(path+'/'+'*.png')
        for img in infile:
            cars.append(img)

print(car_path)
for path in notcar_path:
    if os.path.isdir(path)==True:
        infile=glob.glob(path+'/'+'*.png')
        for img in infile:
            notcars.append(img)

print(notcar_path)

data_info = data_look(cars, notcars)

print('There are',
      data_info["n_cars"], ' cars and',
      data_info["n_notcars"], ' non-cars')
print('of size: ', data_info["image_shape"], ' and data type:',
      data_info["data_type"])
# Just for fun choose random car / not-car indices and plot example images
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(car_image)
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(notcar_image)
plt.title('Example Not-car Image')
plt.show()


# test_imgs=glob.glob('../test_image/*.jpg')
# image=mpimg.imread('../test_image/test2.jpg')
# windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[350, None],
#                     xy_window=(128, 128), xy_overlap=(0.5, 0.5))
# window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
# plt.imshow(window_img)
# plt.show()




car_features=extract_features(cars, cspace='YCrCb', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',spatial_size=(16,16),hist_bins=32,
                             spatial_feet=True, hist_feet=True)
noncar_features=extract_features(notcars, cspace='YCrCb', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',spatial_size=(16,16),
                                hist_bins=32, spatial_feet=True, hist_feet=True)

X = np.vstack((car_features, noncar_features)).astype(np.float64)
scaler=StandardScaler()
X=scaler.fit_transform(X)
y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))
print(X.shape,y.shape)


X, y=shuffle(X, y)
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)
svc=LinearSVC(C=0.001, verbose=True, random_state=0)
svc.fit(X_train,y_train)
test_acc=svc.score(X_test, y_test)
#测试精度为
print('The test accuracy is %.4f'%(test_acc))



orient = 9
spatial_size = 16, 16
hist_bins = 32
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
xy_windows = [(96, 96), (128, 128)]
image_windows = []



#保存模型
model_combine = 'svc_model.p'
try:
    with open(model_combine,'wb') as pfile:
        pickle.dump(
        {
            'svc':svc,
            "scaler":scaler,
            'spatial_size':spatial_size,
            'hist_bins':hist_bins,
            'orient':orient,
            'pix_per_cell':pix_per_cell,
            'cell_per_block':cell_per_block,
            'hog_channel':hog_channel,
            'xy_windows':xy_windows
        },
            pfile,pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to',model_combine,':',e)
    raise





