from flask import Flask, render_template, request, json
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import io
import detectron as det
import classification as clsf
import uuid


class_category_indices = {'Accessories': 0, 'Apparel Set': 1,
                          'Bags': 2, 'Bath and Body': 3, 'Beauty Accessories': 4,
                          'Belts': 5, 'Bottomwear': 6, 'Cufflinks': 7,
                          'Dress': 8, 'Eyes': 9, 'Eyewear': 10,
                          'Flip Flops': 11, 'Fragrance': 12, 'Free Gifts': 13,
                          'Gloves': 14, 'Hair': 15, 'Headwear': 16, 'Home Furnishing': 17,
                          'Innerwear': 18, 'Jewellery': 19, 'Lips': 20, 'Loungewear and Nightwear': 21,
                          'Makeup': 22, 'Mufflers': 23, 'Nails': 24, 'Perfumes': 25, 'Sandal': 26,
                          'Saree': 27, 'Scarves': 28, 'Shoe Accessories': 29, 'Shoes': 30, 'Skin': 31,
                          'Skin Care': 32, 'Socks': 33, 'Sports Accessories': 34, 'Sports Equipment': 35,
                          'Stoles': 36, 'Ties': 37, 'Topwear': 38, 'Umbrellas': 39, 'Vouchers': 40, 'Wallets': 41,
                          'Watches': 42, 'Water Bottle': 43, 'Wristbands': 44}
class_article_type = {'Accessory Gift Set': 0, 'Baby Dolls': 1, 'Backpacks': 2, 'Bangle': 3, 'Basketballs': 4, 'Bath Robe': 5, 'Beauty Accessory': 6, 'Belts': 7, 'Blazers': 8, 'Body Lotion': 9, 'Body Wash and Scrub': 10, 'Booties': 11, 'Boxers': 12, 'Bra': 13, 'Bracelet': 14, 'Briefs': 15, 'Camisoles': 16, 'Capris': 17, 'Caps': 18, 'Casual Shoes': 19, 'Churidar': 20, 'Clothing Set': 21, 'Clutches': 22, 'Compact': 23, 'Concealer': 24, 'Cufflinks': 25, 'Cushion Covers': 26, 'Deodorant': 27, 'Dresses': 28, 'Duffel Bag': 29, 'Dupatta': 30, 'Earrings': 31, 'Eye Cream': 32, 'Eyeshadow': 33, 'Face Moisturisers': 34, 'Face Scrub and Exfoliator': 35, 'Face Serum and Gel': 36, 'Face Wash and Cleanser': 37, 'Flats': 38, 'Flip Flops': 39, 'Footballs': 40, 'Formal Shoes': 41, 'Foundation and Primer': 42, 'Fragrance Gift Set': 43, 'Free Gifts': 44, 'Gloves': 45, 'Hair Accessory': 46, 'Hair Colour': 47, 'Handbags': 48, 'Hat': 49, 'Headband': 50, 'Heels': 51, 'Highlighter and Blush': 52, 'Innerwear Vests': 53, 'Ipad': 54, 'Jackets': 55, 'Jeans': 56, 'Jeggings': 57, 'Jewellery Set': 58, 'Jumpsuit': 59, 'Kajal and Eyeliner': 60, 'Key chain': 61, 'Kurta Sets': 62, 'Kurtas': 63, 'Kurtis': 64, 'Laptop Bag': 65, 'Leggings': 66, 'Lehenga Choli': 67, 'Lip Care': 68, 'Lip Gloss': 69, 'Lip Liner': 70,
                      'Lip Plumper': 71, 'Lipstick': 72, 'Lounge Pants': 73, 'Lounge Shorts': 74, 'Lounge Tshirts': 75, 'Makeup Remover': 76, 'Mascara': 77, 'Mask and Peel': 78, 'Mens Grooming Kit': 79, 'Messenger Bag': 80, 'Mobile Pouch': 81, 'Mufflers': 82, 'Nail Essentials': 83, 'Nail Polish': 84, 'Necklace and Chains': 85, 'Nehru Jackets': 86, 'Night suits': 87, 'Nightdress': 88, 'Patiala': 89, 'Pendant': 90, 'Perfume and Body Mist': 91, 'Rain Jacket': 92, 'Rain Trousers': 93, 'Ring': 94, 'Robe': 95, 'Rompers': 96, 'Rucksacks': 97, 'Salwar': 98, 'Salwar and Dupatta': 99, 'Sandals': 100, 'Sarees': 101, 'Scarves': 102, 'Shapewear': 103, 'Shirts': 104, 'Shoe Accessories': 105, 'Shoe Laces': 106, 'Shorts': 107, 'Shrug': 108, 'Skirts': 109, 'Socks': 110, 'Sports Sandals': 111, 'Sports Shoes': 112, 'Stockings': 113, 'Stoles': 114, 'Sunglasses': 115, 'Sunscreen': 116, 'Suspenders': 117, 'Sweaters': 118, 'Sweatshirts': 119, 'Swimwear': 120, 'Tablet Sleeve': 121, 'Ties': 122, 'Ties and Cufflinks': 123, 'Tights': 124, 'Toner': 125, 'Tops': 126, 'Track Pants': 127, 'Tracksuits': 128, 'Travel Accessory': 129, 'Trolley Bag': 130, 'Trousers': 131, 'Trunk': 132, 'Tshirts': 133, 'Tunics': 134, 'Umbrellas': 135, 'Waist Pouch': 136, 'Waistcoat': 137, 'Wallets': 138, 'Watches': 139, 'Water Bottle': 140, 'Wristbands': 141}
class_gender_type = {'Boys': 0, 'Girls': 1, 'Men': 2, 'Unisex': 3, 'Women': 4}
model = {}
app = Flask(__name__)


@app.route("/upload")
def upload():
    return render_template('image-upload.html')


@app.route("/")
def create_data():
    request_data = request.get_json()
    # images = request_data['imagelist']  # array ["url"]
    images = ['https://cdn.luxe.digital/media/2019/09/12084906/casual-dress-code-men-street-style-luxe-digital-1.jpg']
    #product_name = request_data['product_name']
    #product_id = str(uuid.uuid1())
    #product_ecom_url = request_data['product_url']
    # other infos to be saved
    # ...
    imagedetailist = []
    #imageclassifierdetailslist = []
    # for i in range(0, len(images)):
    # print(images[i])
    image_batch = det.crop_images(
        'https://cdn.luxe.digital/media/2019/09/12084906/casual-dress-code-men-street-style-luxe-digital-1.jpg')
    # print(len(image_batch))
    # imagedetailist.append(image_batch)
    # for i in range(0, len(imagedetailist)):
    imageclassifierdetailslist = clsf.result(image_batch)
    # print(imageclassifierdetailslist)
    return json.dumps({
        "success": True,
        # "image_details": imagedetailist,
        "classified_info": imageclassifierdetailslist
    })


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    #image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


@app.route("/upload_image", methods=["GET", "POST"])
def upload_image():
    image = request.files['filename'].read()
    image = Image.open(io.BytesIO(image))
    # print("model",model)
    # preprocess the image and prepare it for classification
    image = prepare_image(image, target=(96, 96))
    #byte_im = f.read()
    # img = keras.preprocessing.image.load_img(myfile, target_size=(96, 96)
    # )
    # img_array = keras.preprocessing.image.img_to_array(img)
    # img_array = tf.expand_dims(img_array, 0) # Create a batch
    # subcategory
    #reconstructed_model_subcat = keras.models.load_model("fashion_pre_trained_subCategory_3.h5")
    predictions_subcat = model["reconstructed_model_subcat"].predict(image)
    score_subcat = tf.nn.softmax(predictions_subcat[0])
    labels_subcat = dict((v, k) for k, v in class_category_indices.items())
    pred_subcat = labels_subcat[np.argmax(score_subcat)]
    # article_type
    #reconstructed_model_articleType = keras.models.load_model("fashion_pre_trained_articleType.h5")
    predictions_articleType = model["reconstructed_model_articleType"].predict(
        image)
    score_articleType = tf.nn.softmax(predictions_articleType[0])
    labels_articleType = dict((v, k) for k, v in class_article_type.items())
    pred_articleType = labels_articleType[np.argmax(score_articleType)]
    # gender type
    #reconstructed_model_gender = keras.models.load_model("fashion_pre_trained_gender2.h5")
    predictions_gender = model["reconstructed_model_gender"].predict(image)
    score_gender = tf.nn.softmax(predictions_gender[0])
    labels_gender = dict((v, k) for k, v in class_gender_type.items())
    pred_gender = labels_gender[np.argmax(score_gender)]
    f = "This image most likely belongs to {} with a {:.2f} percent confidence."
    data = [
        {"attribute": "subcategory", "value": pred_subcat,
            "message": f.format(pred_subcat, 100 * np.max(score_subcat))},
        {"attribute": "articletype", "value": pred_articleType, "message": f.format(
            pred_articleType, 100 * np.max(score_articleType))},
        {"attribute": "gender", "value": pred_gender, "message": f.format(pred_gender, 100 * np.max(score_gender))}]
    return json.dumps(data)


def load_model():
    model["reconstructed_model_subcat"] = keras.models.load_model(
        "fashion_pre_trained_subCategory_3.h5")
    # model["reconstructed_model_articleType"]=keras.models.load_model("fashion_pre_trained_articleType.h5")
    #model["reconstructed_model_gender"] = keras.models.load_model("fashion_pre_trained_gender2.h5")


if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0')
