import os

clr = lambda: os.system('cls')
clr()

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import random as rnd

clr()


q = 150 # kvalita (max. 200)
aug_per_image = 3 # počet augmentovaných obrázků na obrázek
min_res_factor = 1 # minimální zvětšení/zmenšení obrázku (1 = 100% z kvality)



target_size = (4*q, 3*q)
x = 4*q
y = 3*q
max_res_factor = 200/q
a = max_res_factor-min_res_factor

datagen = ImageDataGenerator(
    rotation_range=40,
    zoom_range=0.2
)

for root, _, files in os.walk("C:/Users/badam/Documents/RozpoznavaniTextuDataset"):
    for file in files:
        if not file.endswith((".jpg", ".jpeg", ".png")):
            continue

        r = rnd.random()*a+min_res_factor
        img = image.img_to_array(image.load_img(os.path.join(root, file), target_size=(int(x*r), int(y*r))))
        img = img.reshape((1,) + img.shape)

        i = 0
        for batch in datagen.flow(img, batch_size=1, save_to_dir="C:/Users/badam/Documents/RozpoznavaniTextuAugmented", save_prefix="aug", save_format="jpeg"):
            i += 1
            if i >= aug_per_image:
                break

print("Augmentace dokončena.")
