import torch
from train import FeedForward, download_datasets
from torch.utils.data import DataLoader
'''
1. load model
2. load data (validation)
3. inference
'''

feed_fwd_net = FeedForward()
statesDict = torch.load("feed_fwd_net.pth")
feed_fwd_net.load_state_dict(statesDict)
# print(feed_fwd_net.load_state_dict(statesDict))

train_data, validation_data = download_datasets()
# print(validation_data.shape)
# print(validation_data[0][0].shape) # torch of my image

input, target = validation_data[0][0], validation_data[0][1]

class_mapping = ["0",'1','2','3','4','5','6','7','8','9']

def predict(model, input, label, class_mapping):
    model.eval() # a switch to get model into evaluation mode. (opposite of model.predict)
    # dropout or batch normalization is switched off 
    with torch.no_grad():
        prediction = model(input)
        # the model outputs 1 x 10 tensor 
        predicted_label_index = class_mapping[prediction[0].argmax(0)]
        print("predicted_label_index",predicted_label_index)
        print("label",label)
        
validation_DataLoader = DataLoader(validation_data, batch_size=128)
# print(validation_DataLoader)
predict(feed_fwd_net, validation_data[1][0], validation_data[1][1], class_mapping)