import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Model,Sequential, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.applications import DenseNet121
print("hello")
disease_types = ['Apple Apple scab','Apple Black rot','Apple Cedar apple rust','Apple healthy','Cherry (including sour) healthy','Cherry (including sour) Powdery mildew','Corn (maize) Cercospora leaf spot Gray leaf spot','Corn (maize) Common rust','Corn (maize) healthy','Grape Black rot','Grape Esca (Black Measles)','Grape healthy','Orange Haunglongbing (Citrus greening)','Peach Bacterial spot','Peach healthy','Pepper bell Bacterial spot','Pepper bell healthy','Potato Early blight','Potato healthy','Potato Late blight','Squash Powdery mildew','Strawberry healthy','Strawberry Leaf scorch','Tomato Target Spot','Tomato Tomato mosaic virus','Tomato Tomato YellowLeaf Curl Virus','Tomato Bacterial spot','Tomato Early blight','Tomato healthy','Tomato Late blight','Tomato Leaf Mold','Tomato Septoria leaf spot','Tomato Spider mites Two spotted spider mite']
data_dir = "D://PlantVillage/"
train_dir = os.path.join(data_dir)
test_dir = os.path.join(data_dir,'test')

train_data = []
for defects_id, sp in enumerate(disease_types):
    for file in os.listdir(os.path.join(train_dir, sp)):
        train_data.append(['{}/{}'.format(sp, file), defects_id, sp])
        
train = pd.DataFrame(train_data, columns=['File', 'DiseaseID','Disease Type'])
train.head()

SEED = 42
train = train.sample(frac=1, random_state=SEED) 
train.index = np.arange(len(train)) # Reset indices
train.head()
# Plot a histogram
plt.hist(train['DiseaseID'])
plt.title('Frequency Histogram of Species')
plt.figure(figsize=(12, 12))
plt.show()
IMAGE_SIZE = 64

def read_image(filepath):
    return cv2.imread(os.path.join(data_dir, filepath)) # Loading a color image is the default flag
# Resize image to target size
def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)

X_train = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
for i, file in tqdm(enumerate(train['File'].values)):
    image = read_image(file)
    if image is not None:
        X_train[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))
# Normalize the data
X_Train = X_train / 255.
Y_train = train['DiseaseID'].values
Y_train = to_categorical(Y_train, num_classes=34)
print(Y_train)


BATCH_SIZE = 64
X_Train = X_Train[:50000]
Y_train = Y_train[:50000]
# Split the train and validation sets 
X_train, X_val, Y_train, Y_val = train_test_split(X_Train, Y_train, test_size=0.2, random_state=SEED)

print(len(Y_train))
print(len(X_train))
print(len(X_val))
print(len(Y_val))

EPOCHS =25
SIZE=64
N_ch=3

def build_densenet():
    densenet = DenseNet121(weights='imagenet', include_top=False)

    input = Input(shape=(SIZE, SIZE, N_ch))
    x = Conv2D(3, (3, 3), padding='same')(input)
    
    x = densenet(x)
    
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # multi output
    output = Dense(34,activation = 'softmax', name='root')(x)
 

    # model
    model = Model(input,output)
    
    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    return model

model = build_densenet()
annealer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
checkpoint = ModelCheckpoint('model.h6', verbose=1, save_best_only=True)
# Generates batches of image data with data augmentation
datagen = ImageDataGenerator(rotation_range=360, # Degree range for random rotations
                        width_shift_range=0.2, # Range for random horizontal shifts
                        height_shift_range=0.2, # Range for random vertical shifts
                        zoom_range=0.2, # Range for random zoom
                        horizontal_flip=True, # Randomly flip inputs horizontally
                        vertical_flip=True) # Randomly flip inputs vertically
print(model)
datagen.fit(X_train)
# Fits the model on batches with real-time data augmentation
hist = model.fit(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
               steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
               epochs=EPOCHS,
               verbose=2,
               callbacks=[annealer, checkpoint],
               validation_data=(X_val, Y_val))

#model = load_model('http://localhost:8888/tree/Downloads/model.h2/assets')
final_loss, final_accuracy = model.evaluate(X_val, Y_val)
print('Final Loss: {}, Final Accuracy: {}'.format(final_loss, final_accuracy))

Y_pred = model.predict(X_val)

Y_pred = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_val, axis=1)

cm = confusion_matrix(Y_true, Y_pred)
plt.figure(figsize=(12, 12))
ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, xticklabels=disease_types, yticklabels=disease_types)
ax.set_ylabel('Actual', fontsize=40)
ax.set_xlabel('Predicted', fontsize=40)

# accuracy plot 
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# loss plot
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

from skimage import io
from keras.preprocessing import image
#path='imbalanced/Scratch/Scratch_400.jpg'
img = image.load_img('D://PlantVillage//Grape Esca (Black Measles)/0c0064e9-f2f4-4264-95a2-58e6e795d1a7___FAM_B.Msls 1328.jpg', grayscale=False, target_size=(64, 64))
show_img=image.load_img('D://PlantVillage//Grape Esca (Black Measles)/0c0064e9-f2f4-4264-95a2-58e6e795d1a7___FAM_B.Msls 1328.jpg', grayscale=False, target_size=(200, 200))
disease_class = ['Apple Apple scab','Apple Black rot','Apple Cedar apple rust','Apple healthy','Cherry (including sour) healthy','Cherry (including sour) Powdery mildew','Corn (maize) Cercospora leaf spot Gray leaf spot','Corn (maize) Common rust','Corn (maize) healthy','Grape Black rot','Grape Esca (Black Measles)','Grape healthy','Orange Haunglongbing (Citrus greening)','Peach Bacterial spot','Peach healthy','Pepper bell Bacterial spot','Pepper bell healthy','Potato Early blight','Potato healthy','Potato Late blight','Squash Powdery mildew','Strawberry healthy','Strawberry Leaf scorch','Tomato Target Spot','Tomato Tomato mosaic virus','Tomato Tomato YellowLeaf Curl Virus','Tomato Bacterial spot','Tomato Early blight','Tomato healthy','Tomato Late blight','Tomato Leaf Mold','Tomato Septoria leaf spot','Tomato Spider mites Two spotted spider mite']
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = np.array(x, 'float32')
x /= 255

custom = model.predict(x)
print(custom[0])



#x = x.reshape([64, 64]);

#plt.gray()
plt.imshow(img)
plt.show()

a=custom
a= a.astype(float)
print('a',a)    
ind=np.argmax(a)
print('ind',ind)
        
print('Prediction:',disease_class[ind])