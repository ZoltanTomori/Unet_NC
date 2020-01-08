import model 
import data 
import matplotlib.pyplot as plt
import sys
from keras.callbacks import ModelCheckpoint
from tensorflow.python.client import device_lib
#from tensorflow.keras.callbacks import ModelCheckpoint

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'


# PREMENNE
from_user = sys.argv
if (len(from_user) > 1):
    label_dir = from_user[1] # label, label_area
    steps = int(from_user[2])
    epochs = int(from_user[3])
    mode = from_user[4] # create, load
    test = from_user[5] # yes, no
    num_of_test_imgs = int(from_user[6])
else:
    label_dir = 'label'
    steps = 20
    epochs = 40
    mode = 'create'
    test = 'yes'
    num_of_test_imgs = 4
train_dir = '../data/train'
test_dir = '../data/test'

print('Sample folder: ' + label_dir)
print('Batch size: ' + str(steps))
print('Number of epochs: ' + str(epochs))
print('Mode: ' + mode)
print('Run testing: ' + test)

print(device_lib.list_local_devices())
import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


# GENEROVAT DATA
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='reflect')
myGene = data.trainGenerator(2,train_dir,'image',label_dir,data_gen_args,save_to_dir = None)


# VYTVORIT U-NET a) nanovo
def create_unet(myGene):
    mdl = model.unet()
    model_checkpoint = ModelCheckpoint('../model/unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
    history = mdl.fit_generator(myGene,steps_per_epoch=steps,epochs=epochs,callbacks=[model_checkpoint])
    print(history.history['loss'])
    # ulozit historiu
    #with open('../model/history.json', 'w') as f:
        #json.dump(history.history, f)
    data.saveHist('../model/history.json', history)
    return mdl, history.history
# b) zo suboru
def load_unet():
    mdl = model.unet('../model/unet_membrane.hdf5')
    # nacitat historiu
    #history_dict = json.load(open('../model/history.json', 'r'))
    history_dict = data.loadHist('../model/history.json')
    return mdl, history_dict
if (mode == 'create'):
    myModel, history = create_unet(myGene) # a)
else:
    myModel, history = load_unet() # b)


# ULOZIT MODEL
def save_model(mdl):
    model_json = mdl.to_json() # struktura
    with open("../model/modelStructure.json", "w") as json_file:
        json_file.write(model_json)
    mdl.save_weights("../model/modelWeights.h5") # vahy
    mdl.save("../model/ourModel.h5")
    print("Model saved to disk")
save_model(myModel)


# VIZUALIZOVAT VYVOJ PRESNOSTI A CHYBY
def visualize_accuracy(history_dict):
    #print(history_dict.keys())

    plt.plot(history_dict['acc'])
    plt.title('Presnosť modelu')
    plt.ylabel('presnosť')
    plt.xlabel('epocha')
    plt.legend(['trénovanie'], loc='upper left')
    plt.savefig('../model/accuracy.png')

    plt.plot(history_dict['loss'])
    plt.title('Chyba modelu')
    plt.ylabel('chyba')
    plt.xlabel('epocha')
    plt.legend(['trénovanie'], loc='upper left')
    plt.savefig('../model/loss.png')
visualize_accuracy(history)

# TESTOVANIE (da sa spustit aj zvlast pomocou use_model.py)
def testing(test_dir):
    testGene = data.testGenerator(test_dir)
    results = myModel.predict_generator(testGene, num_of_test_imgs, verbose=1)
    data.saveResult(test_dir,results)
if (test == 'yes'):
    testing(test_dir)