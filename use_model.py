import model 
import data
import sys

# PREMENNE
test_dir = "../data/test"
generated_dir = "../data/test_generated"
if (len(sys.argv) > 1):
    num_of_test = int(sys.argv[1]) # zistit pocet od pouzivatela
else:
    num_of_test = 4

# NACITAT MODEL
# load json and create model
json_file = open('../model/modelStructure.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
myModel = model.model_from_json(loaded_model_json)
# load weights into new model
myModel.load_weights("../model/modelWeights.h5")
print("Model loaded from disk") 

# evaluate loaded model on test data
myModel.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# SPUSTIT NA TESTOVACICH VZORKACH
testGene = data.testGenerator(test_dir) #generated_dir)
results = myModel.predict_generator(testGene,num_of_test,verbose=1)
data.saveResult(test_dir,results)