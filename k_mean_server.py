from flask import Flask, request
from mnist import MNIST
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import random
import json


def np_encoder(obj):
    if isinstance(obj, np.generic):
        return obj.item()


class KmeanMnist:

    def __init__(self):
        self.mndata = MNIST('mnist')
        self.preprocess_data()


    def preprocess_data(self):
        # import data
        self.x_train, x_labels = self.mndata.load_training()
        # reshape and normalize to be in range [0,1]
        self.X = np.array(self.x_train).reshape(len(self.x_train), -1)
        self.X = self.X.astype(float) / 255.
        self.data_size = len(self.X)

    def train_model(self, k, num_samples_to_use):
        self.kmeans = MiniBatchKMeans(n_clusters=k)
        self.kmeans.fit(self.X[0:num_samples_to_use])

    def get_random_images(self, image_num):
        random_indexes = random.sample(range(0, len(self.kmeans.labels_)), image_num)
        result = []  # [[[image], label], [[image], label]...]
        for index in random_indexes:
            result.append([self.x_train[index], self.kmeans.labels_[index]])
        return result


k_mean_mnist = KmeanMnist()
app = Flask(__name__)


@app.route('/')
def train_k_means():
    k = int(request.args.get('k', 0))
    num_samples_to_use = int(request.args.get('num_samples_to_use', 0))
    num_samples_per_clusters = int(request.args.get('num_samples_per_clusters', 0))
    response = {}
    if k <= 0:
        response['status'] = 'failed'
        response['reason'] = 'Please give a valid k size'
        return response
    if num_samples_to_use <= 0 or num_samples_to_use > k_mean_mnist.data_size:
        response['status'] = 'failed'
        response['reason'] = 'Please give a valid num_samples_to_use, should be between 1 and {}'.format(k_mean_mnist.data_size)
        return response
    if num_samples_per_clusters <= 0 or k + k * num_samples_per_clusters > k_mean_mnist.data_size:
        response['status'] = 'failed'
        response['reason'] = 'Please ask for valid num_samples_per_clusters'
        return response
    k_mean_mnist.train_model(k, num_samples_to_use)
    result = k_mean_mnist.get_random_images(k + k * num_samples_per_clusters)
    response['status'] = 'success'
    response['result'] = result
    return json.dumps(response, default=np_encoder)