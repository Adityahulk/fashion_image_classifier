# fashion_image_classifier
https://yellowbacks.com/

"Basically It took 2 long months to do"
"Reseach and development based Deep Learning pipeline with end to end Deployement for a Fashion Tech startup yellowbacks.com "


A whole deep learning pipeline from very scratch that operates on computer vision technologies like:-
- 1. Image classification using VGG19 on fashion classifying 44000 image dataset
- 2. Multiple object detection using Facebook AI's Detextron2 on DeepFashion2 dataset
- 3. Fashion style recognition using mask rcnn
- 4. Flask API deployement of all the services into a single API.

At the startup Yellowbacks, I in my team, experimented many machine learning and deep learning models through out
many datasets and after research and development and deployement for two months,Here is my work:-
- First for fashion image classification I used pretrained vgg19 with fine tuning on the dataset on fashion image (https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)
, I made model based on reasearch paper of NUS singapore for fashion image classification https://arxiv.org/pdf/2005.08170.pdf
- After three model were developed for gender, master category,article type and sub Category classification after training on google colab, Three model weights were saved for later use
- Still due to multiple clothes and article in an image there was still problem in robustness, So I decided to go for object detection.
- For object detection , I used most advanced DeepFashion2 dataset and Facebook AI's Detectron 2 model on pytorch to train and use the model and along with its configuation on config yaml file for later use.
- Again to make more robust work, I used another mask-rcnn model[main issue here was the main mask-rcnn was not working so I had to moadify the main code and use my own mask rcnn version] for styling detection ,since due to memory and ram constraint ,I used kaggle kernel for training this time
- At last all detection work was done using trained model in detection.py file and classification work in classification.py file and using both files in flask code for api deployement.
- Flask main file is written flexible so that developer can use this whole pipeline to make automatic url fashion article detection and classification system, automated databse filling system or making recommenders system.
- This API typically takes image url input and saves it's fashion style, type of clothes, gender,category and other detected information into your database automatically.

Alternatives we tries but failed:-
- Advance models like resnet,xception,inception didn't worked here and yield very poor result in real world data
- Modanet and other pretrained fashion object detection dataset didn't worked
- mask rcnn main code was not working well, I had to modify the main mask rcnn structure and then use my own version for it
- flask api deployment was failing on macbook so deployed it on a ubuntu based server

My own mask-rcnn repo is present so you can use it in your own deployement
