import os

clr = lambda: os.system('cls')
clr()

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import random as rnd

clr()

res_scale = 0.75 # základní rozlišení obrázku (0.75 = 75% z originálního rozlišení)
aug_per_image = 30 # počet augmentovaných obrázků na obrázek
min_res_factor = 1 # minimální zvětšení/zmenšení obrázku (1 = 100% ze základního rozlišení)


res_width = 1200    
res_height = 1600


x = res_scale*res_width
y = res_scale*res_height
a = 1/res_scale-min_res_factor

datagen = ImageDataGenerator(
    rotation_range=40,
    zoom_range=0.2
)

for root, _, files in os.walk("C:/Users/badam/Documents/RozpoznavaniTextuDataset"):
    for file in files:
        if not file.endswith((".jpg", ".jpeg", ".png")):
            continue

        for i in range(aug_per_image):
            r = rnd.random()*a+min_res_factor
            img = image.img_to_array(image.load_img(os.path.join(root, file), target_size=(int(y*r), int(x*r))))
            img = img.reshape((1,) + img.shape)
            datagen.flow(img, batch_size=1, save_to_dir="C:/Users/badam/Documents/RozpoznavaniTextuAugmented", save_prefix="aug", save_format="jpeg").next()

print("Augmentace dokončena.")
