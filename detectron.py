import uuid
import requests
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from PIL import Image
import numpy as np
import pandas as pd
import detectron2
#import classification as clsc
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
lst_name = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear',
            'vest', 'sling', 'shorts', 'bottom_wear', 'skirt', 'short_sleeved_dress',
            'long_sleeved_dress', 'vest_dress', 'sling_dress']
file_path = 'config.yaml'
model_path = 'model_final.pth'
# json_file_path='xyz_path'
cfg = get_cfg()
cfg.merge_from_file(file_path)
cfg.MODEL.WEIGHTS = model_path
cfg.DATASETS.TEST = ("deepfashion_val", )
# set the testing threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55
predictor = DefaultPredictor(cfg)
detectron_database = pd.DataFrame(columns=('img_array', 'class'))


def crop_images(url):

    image = Image.open(requests.get(url, stream=True).raw)
    im = np.asarray(image)
    outputs = predictor(im)
    boxes = {}
    for coordinates in outputs["instances"].to("cpu").pred_boxes:
        coordinates_array = []
        for k in coordinates:
            coordinates_array.append(int(k))

        boxes[uuid.uuid4().hex[:].upper()] = coordinates_array

    img_batch = []
    for k, v in boxes.items():
        crop_img = im[v[1]:v[3], v[0]:v[2], :]
        img_batch.append(crop_img)

    image_details = []

    for i in range(0, len(boxes)):
        image_details.append(
            lst_name[outputs['instances'][i].pred_classes.item()])

    for i in range(len(img_batch)):
        detectron_database.loc[len(detectron_database)] = [
            img_batch[i], image_details[i]]
    # convert detectron_database in json :  save in db ---
    return img_batch


#img_btch, image_details = crop_images(
#   'https://cdn.luxe.digital/media/2019/09/12084906/casual-dress-code-men-street-style-luxe-digital-1.jpg')

#print(image_details)
#clsc.result(img_btch)
# detectron_database.to_json('detectron_database.json')
