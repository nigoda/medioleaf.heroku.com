import pandas as pd
import numpy as np
import PIL
import os
import os.path
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
# %matplotlib inline
from skimage.io import imread, imshow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from os import listdir
from PIL import Image
from tqdm import tqdm
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.io import imshow, imread
from skimage.morphology import closing
from skimage.measure import label, regionprops, regionprops_table
from matplotlib import pyplot as plt
from skimage.io import imshow, imread
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR,'static')
filecsv = STATIC_DIR + "/myapp/csv/datapp3.csv"
df = pd.read_csv(filecsv)

newimage2 = STATIC_DIR + "/myapp/images/newimage2/"

image_path_list = os.listdir(newimage2)
df4=pd.DataFrame()
for i in range(len(image_path_list)):
    filename = image_path_list[i]
    if filename.endswith('.jpg'):
        try:
            img = Image.open(newimage2+filename) # open the image file
            img.verify()
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename)
            os.remove(newimage2+filename)
            continue
        image = imread(newimage2+filename, as_gray=True)
        #imshow(image)
        a=image.shape
        features = np.reshape(image, a)
        x=features[0:1,0:17]
        df5= pd.DataFrame(x)
        df4=df4.append(df5,ignore_index=True)

df4.insert(0,"id",[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102],True)
image.shape
re=pd.merge(df,df4)
re.shape

X = re.drop(['id','species','medicinal'], axis=1)
y = re['medicinal']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 99)
#X_train.shape, X_test.shape, y_train.shape, y_test.shape

list1=[]
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
accuracy1=metrics.accuracy_score(y_test,y_pred)
list1.append(accuracy1)


model=LogisticRegression()
model.fit(X_train,y_train)
pred=model.predict(X_test)
#metrics.confusion_matrix(y_test,pred)
accuracy3=metrics.accuracy_score(y_test,y_pred)
list1.append(accuracy3)




knn = KNeighborsClassifier(n_neighbors=15)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)                                  knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy2=metrics.accuracy_score(y_test, y_pred)
list1.append(accuracy2)


algo=['RF','Logistic Reg', 'KNN']
#import seaborn as sns

#
# fig = plt.figure(figsize = (10, 5))
#
# # creating the bar plot
# plt.bar(algo, list1, color ='maroon',
#         width = 0.4)
#
# plt.xlabel("algo used")
# plt.ylabel("accuracy")
# plt.title("comparison")
# plt.show()


row=re.iloc[-(len(image_path_list)):]
name=row.pop('species')
row.pop('medicinal')
test_id=row.pop('id')
row.head(len(image_path_list))




x_test = row.values

scaler = StandardScaler()

x_test = scaler.fit_transform(x_test)

y_pred = clf.predict_proba(x_test)
#y_pred



test_pred = clf.predict(row)

output = pd.DataFrame({'Specie Name':name,'Medicinal or Not':test_pred
                    })
output.to_csv('submission.csv', index=False)
output = output.reset_index(drop=True)


def test(img_path,ph,temperature):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target = STATIC_DIR + "/myapp/images/" #jejj
    image_path_list = [img_path]
    df = pd.DataFrame()
    for i in range(len(image_path_list)):
        filename = image_path_list[i]
        if filename.endswith('.jpg'):
            try:
                img = Image.open(target+filename) # open the image file
                img.verify()
            except (IOError, SyntaxError) as e:
                print('Bad file:', filename)
                os.remove(target+filename)
                continue
            image = rgb2gray(imread(target+filename))
            binary = image < threshold_otsu(image)
            #imshow(binary)
            #plt.figure()
            #imshow(image)
            #plt.figure()
            binary = closing(binary)
            label_img = label(binary)

            table = pd.DataFrame(regionprops_table(label_img, image,
                              ['convex_area', 'area', 'eccentricity',
                               'extent', 'inertia_tensor',
                               'major_axis_length', 'minor_axis_length',
                               'perimeter', 'solidity', 'image',
                               'orientation', 'moments_central',
                               'moments_hu', 'euler_number',
                               'equivalent_diameter',
                               'mean_intensity', 'bbox']))
            table['perimeter_area_ratio'] = table['perimeter']/table['area']
            real_images = []
            std = []
            mean = []
            percent25 = []
            percent75 = []
            for prop in regionprops(label_img):
                min_row, min_col, max_row, max_col = prop.bbox
                img = image[min_row:max_row,min_col:max_col]
                real_images += [img]
                mean += [np.mean(img)]
                std += [np.std(img)]
                percent25 += [np.percentile(img, 25)]
                percent75 += [np.percentile(img, 75)]
            table['real_images'] = real_images
            table['mean_intensity'] = mean
            table['std_intensity'] = std
            table['25th Percentile'] = mean
            table['75th Percentile'] = std
            table['iqr'] = table['75th Percentile'] - table['25th Percentile']
            table['label'] = filename[5]
            df = pd.concat([df, table], axis=0)


    arr = np.array([[1,ph,temperature]])
    df = pd.DataFrame(arr, columns=['id','ph','temperature(F)'])
    image_path_list = ["test.jpg"]


    df4=pd.DataFrame()
    for i in range(len(image_path_list)):
        filename = image_path_list[i]
        if filename.endswith('.jpg'):
            try: # open the image file
                img = Image.open(target+filename)
                img.verify()
            except (IOError, SyntaxError) as e:
                print('Bad file:', filename)
                os.remove(target+filename)
                continue
            image = imread(target+filename, as_gray=True)
            #imshow(image)
            a=image.shape
            features = np.reshape(image, a)
            x=features[0:1,0:17]
    #         print(x)
            df5= pd.DataFrame(x)
            df4=df4.append(df5,ignore_index=True)
    #         print(df4)

    df4.insert(0,"id",[1],True)
    re=pd.merge(df,df4)

    row=re.iloc[-(len(image_path_list)):]
    row.head(len(image_path_list))
    test_id=row.pop('id')
    x_test = row.values

    scaler = StandardScaler()

    x_test = scaler.fit_transform(x_test)

    y_pred = clf.predict_proba(x_test)
    #y_pred



    test_pred = clf.predict(row)
    return test_pred
