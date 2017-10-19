import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from skimage.transform import warp, resize

#########################################
### Exploring Data and Data Smoothing ###
#########################################

samples = []
with open('./data/driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        samples.append(line)

print('Number of samples(only center images): ', len(samples))

angles = []
for sample in samples:
    angle = float(sample[3])
    angles.append(angle)

#plt.hist(angles, bins=41, rwidth=0.8)
#plt.title('Data distribution for angle')

angles = np.array(angles)
indices = np.where(angles == 0)
tmp = np.ones(len(angles), dtype=bool)
tmp[indices] = False
samples = np.array(samples)
samples_tmp = samples[tmp]
samples = np.concatenate((samples_tmp, samples[indices][:1000]), axis=0)

print('Number of samples(after smoothing): ', len(samples))

samples = shuffle(samples)

angles = []
for sample in samples:
    angle = float(sample[3])
    angles.append(angle)

#plt.hist(angles, bins=41, rwidth=0.8)
#plt.title('Data distribution for angle after smoothing')


'''
############################################
### Data Preprocessing and Visualization ###
############################################

samples = shuffle(samples)
images = []
angles = []
for i in range(3):
    for j in [1,0,2]:
        filename = samples[i][j].split('/')[-1]
        current_path = './data/IMG/' + filename
        image = mpimg.imread(current_path)
        angle = float(samples[i][3])
        if j == 1:
            angle = angle + 0.2 # left
        elif j == 2:
            angle = angle - 0.2 # right
        images.append(image)
        angles.append(angle)

# original images (left, center, right)
fig, axs = plt.subplots(3, 3, figsize=(20, 10))
axs = axs.ravel()

for i in range(0,9,3):
    axs[i].axis('off')
    axs[i].set_title('left, angle: {:.5f}'.format(angles[i]))
    axs[i].imshow(images[i])
    axs[i+1].axis('off')
    axs[i+1].set_title('center, angle: {:.5f}'.format(angles[i+1]))
    axs[i+1].imshow(images[i+1])
    axs[i+2].axis('off')
    axs[i+2].set_title('right, angle: {:.5f}'.format(angles[i+2]))
    axs[i+2].imshow(images[i+2])

# cropped and resized images
fig, axs = plt.subplots(3, 3, figsize=(20, 7))
axs = axs.ravel()

for i in range(0,9,3):
    axs[i].axis('off')
    axs[i].set_title('left')
    axs[i].imshow(resize(images[i][70:-20], (66,200)))
    axs[i+1].axis('off')
    axs[i+1].set_title('center')
    axs[i+1].imshow(resize(images[i+1][70:-20], (66,200)))
    axs[i+2].axis('off')
    axs[i+2].set_title('right')
    axs[i+2].imshow(resize(images[i+2][70:-20], (66,200)))

# flipped images
fig, axs = plt.subplots(3, 2, figsize=(15, 7))
axs = axs.ravel()

for i in range(0,6,2):
    image = resize(images[i][70:-20], (66,200))
    image_flipped, angle_flipped = image[:, ::-1, :], -angles[i]

    axs[i].axis('off')
    axs[i].set_title('original, angle: {:.5f}'.format(angles[i]))
    axs[i].imshow(image)
    axs[i+1].axis('off')
    axs[i+1].set_title('flipped, angle: {:.5f}'.format(angle_flipped))
    axs[i+1].imshow(image_flipped)

# shifted images
def shift(image, angle):
    dx = 40 * (np.random.rand() - 0.5)
    dy = 20 * (np.random.rand() - 0.5)
    trans_matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    image = warp(image, trans_matrix)
    angle += -dx * 0.005
    return image, angle

fig, axs = plt.subplots(3, 2, figsize=(15, 7))
axs = axs.ravel()

for i in range(0,6,2):
    image = resize(images[i][70:-20], (66,200))
    image_shift, angle_shift = shift(image, angles[i])

    axs[i].axis('off')
    axs[i].set_title('original, angle: {:.5f}'.format(angles[i]))
    axs[i].imshow(image)
    axs[i+1].axis('off')
    axs[i+1].set_title('shift, angle: {:.5f}'.format(angle_shift))
    axs[i+1].imshow(image_shift)
'''


#######################################
### Model Architecture and Training ###
#######################################

def flip(image, angle):
    if np.random.randint(0,2) == 1:
        image = image[:, ::-1, :]
        angle = -1 * angle
    return image, angle

def shift(image, angle):
    dx = 40 * (np.random.rand() - 0.5)
    dy = 20 * (np.random.rand() - 0.5)
    trans_matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    image = warp(image, trans_matrix)
    angle += -dx * 0.005
    return image, angle

def generator(samples, batch_size=32, train=False):
    num_samples = len(samples)
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                index = np.random.randint(0,3)
                name = './data/IMG/'+batch_sample[index].split('/')[-1]
                image = mpimg.imread(name)
                angle = float(batch_sample[3])
                if index == 1:
                    angle = angle + 0.2 # left
                elif index == 2:
                    angle = angle - 0.2 # right
                image = resize(image[70:-20], (66,200))
                if train == True:
                    image, angle = flip(image, angle)
                    image, angle = shift(image, angle)
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

model = Sequential()
model.add(Lambda(lambda x: x-0.5, input_shape=(66,200,3)))
model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-4))

#model = load_model('model.h5')
#model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-5))

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32, train=True)
validation_generator = generator(validation_samples, batch_size=32, train=False)

early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
checkpoint = ModelCheckpoint('model-{epoch:02d}.h5', monitor='val_loss', save_best_only=True, mode='auto')

hist = model.fit_generator(train_generator, steps_per_epoch=(20000/32),
                           validation_data=validation_generator, validation_steps=(4480/32),
                           epochs=50, callbacks=[early_stop, checkpoint])

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
