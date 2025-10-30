import torch
from train import FeedForward, download_datasets

'''
1. load model
2. load data (validation)
3. inference
'''

feed_fwd_net = FeedForward()
statesDict = torch.load("feed_fwd_net.pth")
print(feed_fwd_net.load_state_dict(statesDict))

