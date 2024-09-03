import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence


def chunkify(big_list, chunk_size):
    chunks = [big_list[x:x + chunk_size] for x in range(0, len(big_list), chunk_size)]
    return chunks
def designate_batches(soccernet_dataset_path,data_type, batch_size=32):
    images_x=os.listdir(soccernet_dataset_path+data_type)
    images_y=os.listdir(soccernet_dataset_path+"cnn"+data_type)
    dataset=[{"x":soccernet_dataset_path+data_type+"/"+x,"y":soccernet_dataset_path+"cnn"+data_type+"/"+y} for x,y in zip(images_x,images_y)]
    chunked_dataset=chunkify(dataset,batch_size)
    return  chunked_dataset



class DataGenerator(Sequence):
    def __init__(self, data_batch, batch_size=32, image_size=(960, 560)):
        self.data_batch = data_batch
        self.batch_size = batch_size
        self.image_size = image_size
        self.length = len(data_batch)



    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        X_batch = []
        y_batch = []
        for i in range(len(self.data_batch[idx])):
            x_path = self.data_batch[idx][i]["x"]
            y_path= self.data_batch[idx][i]["y"]

            x = cv2.imread(x_path, cv2.COLOR_BGR2RGB)
            x = cv2.resize(x, self.image_size)
            x = x / 255.0

            y = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
            y = cv2.resize(y, self.image_size)
            y = y / 255.0

            X_batch.append(x)
            y_batch.append(y)

        X_batch_array = np.array(X_batch)
        y_batch_array = np.array(y_batch)

        X_batch_array = X_batch_array.astype(np.float32)
        y_batch_array = y_batch_array.astype(np.float32)

        return X_batch_array, y_batch_array