import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model, Sequential
#from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras.models import Sequential
articletype_ = ['Backpacks', 'Belts', 'Briefs', 'Casual Shoes', 'Flats', 'Flip Flops',
                'Formal Shoes', 'Handbags', 'Heels', 'Jeans', 'Kurtas',
                'Perfume and Body Mist', 'Sandals', 'Shirts', 'Shorts', 'Sports Shoes',
                'Sunglasses', 'Tops', 'Trousers', 'Tshirts', 'Wallets', 'Watches']
gender_mastercategory_ = ['BoysApparel', 'GirlsApparel', 'MenAccessories', 'MenApparel',
                          'MenFootwear', 'MenPersonal Care', 'UnisexAccessories',
                          'UnisexFootwear', 'WomenAccessories', 'WomenApparel', 'WomenFootwear',
                          'WomenPersonal Care']
subcategory_ = ['Bags', 'Belts', 'Bottomwear', 'Eyewear', 'Flip Flops', 'Fragrance',
                'Innerwear', 'Jewellery', 'Sandal', 'Shoes', 'Topwear', 'Wallets',
                'Watches']

model_article_type = keras.models.load_model(
    'vgg19_transfer_articletype (1).h5')
model_gender_mastercategory = keras.models.load_model(
    'vgg19_transfer_gendermastercat (1).h5')
model_subcategory = keras.models.load_model('vgg19_transfer_subcat1.h5')


# print(gender_mastercategory_df.groupby(['gender_masterCategory']).count().iloc[:,0].index[model_gender_mastercategory.predict(image_batch[0]).argmax()])
def yellowbacks(image):
   # Convert single image to a batch.
    article_type = model_article_type.predict(image)
    gender_mastercategory = model_gender_mastercategory.predict(image)
    subcategory = model_subcategory.predict(image)

    return articletype_[article_type.argmax()], gender_mastercategory_[gender_mastercategory.argmax()], subcategory_[subcategory.argmax()]


def result(image_batch):
    result_list = []
    database1 = pd.DataFrame(
        columns=('img_array', 'article_type', 'genmcat', 'subcat'))
    for i in image_batch:
        i.shape = (1, i.shape[0], i.shape[1], i.shape[2])
        a, b, c = yellowbacks(i)
        database1.loc[len(database1)] = [i, a, b, c]
        result_list.append([a, b, c])

    return result_list
    #print(a, b, c)
    # save database1 in db to convert in json

# database1.to_json('database1.json')
