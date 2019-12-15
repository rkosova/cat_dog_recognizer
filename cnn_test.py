from keras.models import load_model 
from keras.preprocessing import image
import numpy as np
import sys
#img_size = Image.open(r'cat1.png').size

f_image = sys.argv[1]
s_image = sys.argv[2]

classifier = load_model("models/dog_cat_classifier.h5")
classifier.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

img = image.load_img(f_image, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

dog = image.load_img(s_image, target_size=(64, 64))
y = image.img_to_array(dog)
y = np.expand_dims(y, axis=0)

images = np.vstack([x, y])

pred = classifier.predict(images)

for i in range(len(pred)):
    if pred[i] == 0:
        print("\n \n \n")
        print("The first image is a cat!")
    else:
        print("The second image is a dog!")

