import os
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np
import pickle
import pandas as pd

class GetData:
    def __init__(self,train_file_path,test_file_path,number_of_users,number_of_items) -> None:
        self.train_file = open(train_file_path,"r")
        self.test_file = open(test_file_path,"r")
        self.number_of_users = number_of_users
        self.num_of_items = number_of_items
        self.__is_empty = False
        try:
            os.mkdir("pipeline")
        except:
            pass
    def is_empty(self):
        return self.__is_empty
    def __init_pipelines(self):
        self.train_pipe = open("pipeline/train.txt","w")
        self.test_pipe = open("pipeline/test.txt","w")
    
    def get_raw_data(self):
        self.__init_pipelines()
        num_of_users = -1
        prev = ""
        while True:
            train_line = self.train_file.readline()
            if train_line == "":
                self.__is_empty = True
                break
            train_line_splitted = train_line.split("\t")
            user = train_line_splitted[0]
            if prev != user:
                prev = user
                num_of_users+=1
                if num_of_users  == self.number_of_users:
                    break
            self.train_pipe.write(train_line)
        self.train_pipe.close()
        num_of_users = -1
        prev = ""
        while True:
            test_line = self.test_file.readline()
            test_line_splitted = test_line.split("\t")
            user = test_line_splitted[0]
            if prev != user:
                prev = user
                num_of_users+=1
                if num_of_users  == self.number_of_users:
                    break
            self.test_pipe.write(test_line)
        self.test_pipe.close()

    def get_top_n_occ(self):
        train = pd.read_csv("pipeline/train.txt",delimiter="\t",engine="python",header=None)
        train.columns = ["user_id","item_id","rating"]
        songs_counter = Counter(train.item_id)
        self.songs_counter_sorted = sorted(songs_counter.items(), key=lambda x:x[1],reverse=True)[:self.num_of_items]

    def create_filtered_dataset(self):
        self.get_raw_data()
        self.get_top_n_occ()

        encoder = LabelEncoder()
        
        ids = [item[0] for item in self.songs_counter_sorted]

        encoder.fit(np.array(ids))
        pickle.dump(encoder,open('ids_label_encoder.pkl', 'wb'))


        train = pd.read_csv("pipeline/train.txt",delimiter="\t",engine="python",header=None)
        train.columns = ["user_id","item_id","rating"]
        train = train[train["item_id"].isin(ids)]
        train["item_id"] = encoder.transform(train["item_id"].to_numpy())

        test = pd.read_csv("pipeline/test.txt",delimiter="\t",engine="python",header=None)
        test.columns = ["user_id","item_id","rating"]
        test = test[test["item_id"].isin(ids)]
        test["item_id"] = encoder.transform(test["item_id"].to_numpy())

        train.to_csv('pipeline/train.txt', header=None, index=None, sep='\t', mode='w')
        test.to_csv('pipeline/test.txt', header=None, index=None, sep='\t', mode='w')


