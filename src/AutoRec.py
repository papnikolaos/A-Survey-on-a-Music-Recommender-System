import torch.optim as optim
import torch
import torch.nn as nn
from PrepareData import GetData
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from recmetrics import rmse, mse,recommender_precision,recommender_recall

def get_data(num_of_users,num_of_items):
	train_file = open("pipeline/train.txt")
	test_file = open("pipeline/test.txt")
	mat = np.zeros((num_of_users,num_of_items))
	mat_mask = np.zeros((num_of_users,num_of_items))
	for line in train_file.readlines():
		user, item, rating = map(int,line.split("\t"))
		mat[user,item] = rating
		mat_mask[user,item] = 1
	for line in test_file.readlines():
		user, item, rating = map(int,line.split("\t"))
		mat[user,item] = rating
		mat_mask[user,item] = 1
	
	return mat,mat_mask


class Autorec(nn.Module):
    def __init__(self, num_users,num_items,hidden_units,lambda_value):
        super(Autorec, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_units = hidden_units
        self.lambda_value = lambda_value

        self.encoder = nn.Sequential(
            nn.Linear(self.num_items, self.hidden_units),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_units, self.num_items),
        )


    def forward(self,torch_input):

        encoder = self.encoder(torch_input)
        decoder = self.decoder(encoder)

        return decoder

    def loss(self,decoder,input,optimizer,mask_input):
        cost = 0
        temp2 = 0

        cost += (( decoder - input) * mask_input).pow(2).sum()
        rmse = cost

        for i in optimizer.param_groups:
            for j in i['params']:
                if j.data.dim() == 2:
                    temp2 += torch.t(j.data).pow(2).sum()

        cost += temp2 * self.lambda_value * 0.5
        return cost,rmse

def train(epoch,loader,rec,optimizer,train_mask):
    RMSE = 0
    cost_all = 0
    for _, (batch_x, batch_mask_x, _) in enumerate(loader):
        batch_x = batch_x.type(torch.FloatTensor)
        batch_mask_x = batch_mask_x.type(torch.FloatTensor)
        decoder = rec(batch_x)
        loss, rmse = rec.loss(decoder=decoder, input=batch_x, optimizer=optimizer, mask_input=batch_mask_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cost_all += loss
        RMSE += rmse
    RMSE = np.sqrt(RMSE.detach().cpu().numpy() / (train_mask == 1).sum())
    print('epoch ', epoch,  ' train RMSE : ', RMSE)

def test(num_users,num_items,rec,test_r,test_mask):
    test_r_tensor = torch.from_numpy(test_r).type(torch.FloatTensor)
    test_mask_r_tensor = torch.from_numpy(test_mask).type(torch.FloatTensor)
    decoder = rec(test_r_tensor)
    y_preds = (decoder*test_mask_r_tensor).detach().numpy()
    y_true = (test_r_tensor*test_mask_r_tensor).detach().numpy()
    RMSE = rmse(y_preds,y_true)
    
    y_preds = np.ceil(y_preds)
    users, items, rating = [],[],[]
    for user in (range(y_true.shape[0])):
        for item in range(num_items):
            if y_preds[user,item] != 0:
                users.append(user)
                items.append(item)
                rating.append(y_preds[user,item])
    df = pd.DataFrame()
    df["userID"] = users
    df["itemID"] = items
    df["rating"] = rating
    pr = recommender_precision(y_preds,y_true)
    re = recommender_recall(y_preds,y_true)
    return (RMSE,pr,re)

def make_recommendations_autorec(user,p,recommendation_num):
    test_r_tensor = torch.from_numpy(p[0][user]).type(torch.FloatTensor)
    test_mask_r_tensor = torch.from_numpy(p[2][user]).type(torch.FloatTensor)
    decoder = p[1](test_r_tensor)
    recoms = (decoder*test_mask_r_tensor).detach().numpy()
    df = pd.DataFrame()
    df["items"] = np.nonzero(recoms)[0]
    df["ratings"] = recoms[np.where(recoms!=0)]
    df = df.sort_values("ratings",ascending=False)
    recommendations_str = "\n" + "*"*10 + " TOP %d RECOMMENDATIONS FOR USER %s "%(recommendation_num,str(user)) + "*"*10 + "\n"
    i = 1
    for _,item in df.iterrows():
        recommendations_str += (str(i) + ") Song: " + str(int(item['items'])) + ", Predicted rating: " +  str(item['ratings'])) + "\n"
        i += 1
        if(i > recommendation_num):
            break
    return recommendations_str
