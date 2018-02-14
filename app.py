import os
import cv2
import numpy as np
from flask import Flask, render_template, request,jsonify
from keras.models import load_model
from keras.preprocessing import image 
from keras.applications.resnet50 import ResNet50, preprocess_input


dog_names = ['/dogImages/train/001.Affenpinscher',
 '/dogImages/train/002.Afghan_hound',
 '/dogImages/train/003.Airedale_terrier',
 '/dogImages/train/004.Akita',
 '/dogImages/train/005.Alaskan_malamute',
 '/dogImages/train/006.American_eskimo_dog',
 '/dogImages/train/007.American_foxhound',
 '/dogImages/train/008.American_staffordshire_terrier',
 '/dogImages/train/009.American_water_spaniel',
 '/dogImages/train/010.Anatolian_shepherd_dog',
 '/dogImages/train/011.Australian_cattle_dog',
 '/dogImages/train/012.Australian_shepherd',
 '/dogImages/train/013.Australian_terrier',
 '/dogImages/train/014.Basenji',
 '/dogImages/train/015.Basset_hound',
 '/dogImages/train/016.Beagle',
 '/dogImages/train/017.Bearded_collie',
 '/dogImages/train/018.Beauceron',
 '/dogImages/train/019.Bedlington_terrier',
 '/dogImages/train/020.Belgian_malinois',
 '/dogImages/train/021.Belgian_sheepdog',
 '/dogImages/train/022.Belgian_tervuren',
 '/dogImages/train/023.Bernese_mountain_dog',
 '/dogImages/train/024.Bichon_frise',
 '/dogImages/train/025.Black_and_tan_coonhound',
 '/dogImages/train/026.Black_russian_terrier',
 '/dogImages/train/027.Bloodhound',
 '/dogImages/train/028.Bluetick_coonhound',
 '/dogImages/train/029.Border_collie',
 '/dogImages/train/030.Border_terrier',
 '/dogImages/train/031.Borzoi',
 '/dogImages/train/032.Boston_terrier',
 '/dogImages/train/033.Bouvier_des_flandres',
 '/dogImages/train/034.Boxer',
 '/dogImages/train/035.Boykin_spaniel',
 '/dogImages/train/036.Briard',
 '/dogImages/train/037.Brittany',
 '/dogImages/train/038.Brussels_griffon',
 '/dogImages/train/039.Bull_terrier',
 '/dogImages/train/040.Bulldog',
 '/dogImages/train/041.Bullmastiff',
 '/dogImages/train/042.Cairn_terrier',
 '/dogImages/train/043.Canaan_dog',
 '/dogImages/train/044.Cane_corso',
 '/dogImages/train/045.Cardigan_welsh_corgi',
 '/dogImages/train/046.Cavalier_king_charles_spaniel',
 '/dogImages/train/047.Chesapeake_bay_retriever',
 '/dogImages/train/048.Chihuahua',
 '/dogImages/train/049.Chinese_crested',
 '/dogImages/train/050.Chinese_shar-pei',
 '/dogImages/train/051.Chow_chow',
 '/dogImages/train/052.Clumber_spaniel',
 '/dogImages/train/053.Cocker_spaniel',
 '/dogImages/train/054.Collie',
 '/dogImages/train/055.Curly-coated_retriever',
 '/dogImages/train/056.Dachshund',
 '/dogImages/train/057.Dalmatian',
 '/dogImages/train/058.Dandie_dinmont_terrier',
 '/dogImages/train/059.Doberman_pinscher',
 '/dogImages/train/060.Dogue_de_bordeaux',
 '/dogImages/train/061.English_cocker_spaniel',
 '/dogImages/train/062.English_setter',
 '/dogImages/train/063.English_springer_spaniel',
 '/dogImages/train/064.English_toy_spaniel',
 '/dogImages/train/065.Entlebucher_mountain_dog',
 '/dogImages/train/066.Field_spaniel',
 '/dogImages/train/067.Finnish_spitz',
 '/dogImages/train/068.Flat-coated_retriever',
 '/dogImages/train/069.French_bulldog',
 '/dogImages/train/070.German_pinscher',
 '/dogImages/train/071.German_shepherd_dog',
 '/dogImages/train/072.German_shorthaired_pointer',
 '/dogImages/train/073.German_wirehaired_pointer',
 '/dogImages/train/074.Giant_schnauzer',
 '/dogImages/train/075.Glen_of_imaal_terrier',
 '/dogImages/train/076.Golden_retriever',
 '/dogImages/train/077.Gordon_setter',
 '/dogImages/train/078.Great_dane',
 '/dogImages/train/079.Great_pyrenees',
 '/dogImages/train/080.Greater_swiss_mountain_dog',
 '/dogImages/train/081.Greyhound',
 '/dogImages/train/082.Havanese',
 '/dogImages/train/083.Ibizan_hound',
 '/dogImages/train/084.Icelandic_sheepdog',
 '/dogImages/train/085.Irish_red_and_white_setter',
 '/dogImages/train/086.Irish_setter',
 '/dogImages/train/087.Irish_terrier',
 '/dogImages/train/088.Irish_water_spaniel',
 '/dogImages/train/089.Irish_wolfhound',
 '/dogImages/train/090.Italian_greyhound',
 '/dogImages/train/091.Japanese_chin',
 '/dogImages/train/092.Keeshond',
 '/dogImages/train/093.Kerry_blue_terrier',
 '/dogImages/train/094.Komondor',
 '/dogImages/train/095.Kuvasz',
 '/dogImages/train/096.Labrador_retriever',
 '/dogImages/train/097.Lakeland_terrier',
 '/dogImages/train/098.Leonberger',
 '/dogImages/train/099.Lhasa_apso',
 '/dogImages/train/100.Lowchen',
 '/dogImages/train/101.Maltese',
 '/dogImages/train/102.Manchester_terrier',
 '/dogImages/train/103.Mastiff',
 '/dogImages/train/104.Miniature_schnauzer',
 '/dogImages/train/105.Neapolitan_mastiff',
 '/dogImages/train/106.Newfoundland',
 '/dogImages/train/107.Norfolk_terrier',
 '/dogImages/train/108.Norwegian_buhund',
 '/dogImages/train/109.Norwegian_elkhound',
 '/dogImages/train/110.Norwegian_lundehund',
 '/dogImages/train/111.Norwich_terrier',
 '/dogImages/train/112.Nova_scotia_duck_tolling_retriever',
 '/dogImages/train/113.Old_english_sheepdog',
 '/dogImages/train/114.Otterhound',
 '/dogImages/train/115.Papillon',
 '/dogImages/train/116.Parson_russell_terrier',
 '/dogImages/train/117.Pekingese',
 '/dogImages/train/118.Pembroke_welsh_corgi',
 '/dogImages/train/119.Petit_basset_griffon_vendeen',
 '/dogImages/train/120.Pharaoh_hound',
 '/dogImages/train/121.Plott',
 '/dogImages/train/122.Pointer',
 '/dogImages/train/123.Pomeranian',
 '/dogImages/train/124.Poodle',
 '/dogImages/train/125.Portuguese_water_dog',
 '/dogImages/train/126.Saint_bernard',
 '/dogImages/train/127.Silky_terrier',
 '/dogImages/train/128.Smooth_fox_terrier',
 '/dogImages/train/129.Tibetan_mastiff',
 '/dogImages/train/130.Welsh_springer_spaniel',
 '/dogImages/train/131.Wirehaired_pointing_griffon',
 '/dogImages/train/132.Xoloitzcuintli',
 '/dogImages/train/133.Yorkshire_terrier']

ResNet50_model = ResNet50(weights='imagenet')

def ResNet50_predict_labels(img_path):
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def face_detector(img_path):
    img = cv2.imread(img_path, 0)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(img)
    return len(faces) > 0


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def extract_Resnet50(tensor):
	return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def resnet50_predict_breed(img):
    
    #loading the saved model
    res_model = load_model("./res_model.h5")
	# extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img))
    # obtain predicted vector
    predicted_vector = res_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)][21:]
    #return predicted_vector

def tell_dog_breed(img):
    # read the image
    #image = cv2.imread(img, 0)
    # convert BGR image to RGB for plotting
    #color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # display the image along with bounding box
    
    # predict the breed
    prediction = resnet50_predict_breed(img)
    
    # algorithm
    if dog_detector(img):
        description =  'This is a doggie: %s' % (prediction)
    elif face_detector(img):
        description = 'This is a human doggiee hahah...They look like a %s' % (prediction)
    else:
    	description = 'I have no clue my man!!'
    return description

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
	image = request.files['image']
	filename = os.path.join(app.config['UPLOAD_FOLDER'], "image.jpeg")
	image.save(filename)

	#image = cv2.imread('filename',0)
	description = tell_dog_breed(filename)
	
	return jsonify(description=description)
	return render_template('index.html')