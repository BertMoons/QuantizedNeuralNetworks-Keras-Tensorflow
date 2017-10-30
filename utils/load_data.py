# Copyright 2017 Bert Moons

# This file is part of QNN.

# QNN is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# QNN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# The code for QNN is based on BinaryNet: https://github.com/MatthieuCourbariaux/BinaryNet

# You should have received a copy of the GNU General Public License
# along with QNN.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.datasets.mnist import MNIST


def load_dataset(dataset):
    if (dataset == "CIFAR-10"):

        print('Loading CIFAR-10 dataset...')

        train_set_size = 45000
        train_set = CIFAR10(which_set="train", start=0, stop=train_set_size)
        valid_set = CIFAR10(which_set="train", start=train_set_size, stop=50000)
        test_set = CIFAR10(which_set="test")

        train_set.X = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., train_set.X), 1.), (-1, 3, 32, 32)),(0,2,3,1))
        valid_set.X = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., valid_set.X), 1.), (-1, 3, 32, 32)),(0,2,3,1))
        test_set.X = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., test_set.X), 1.), (-1, 3, 32, 32)),(0,2,3,1))
        # flatten targets
        train_set.y = np.hstack(train_set.y)
        valid_set.y = np.hstack(valid_set.y)
        test_set.y = np.hstack(test_set.y)

        # Onehot the targets
        train_set.y = np.float32(np.eye(10)[train_set.y])
        valid_set.y = np.float32(np.eye(10)[valid_set.y])
        test_set.y = np.float32(np.eye(10)[test_set.y])

        # for hinge loss
        train_set.y = 2 * train_set.y - 1.
        valid_set.y = 2 * valid_set.y - 1.
        test_set.y = 2 * test_set.y - 1.
        # enlarge train data set by mirrroring
        x_train_flip = train_set.X[:, :, ::-1, :]
        y_train_flip = train_set.y
        train_set.X = np.concatenate((train_set.X, x_train_flip), axis=0)
        train_set.y = np.concatenate((train_set.y, y_train_flip), axis=0)

    elif (dataset == "MNIST"):

        print('Loading MNIST dataset...')

        train_set_size = 50000
        train_set = MNIST(which_set="train", start=0, stop=train_set_size)
        valid_set = MNIST(which_set="train", start=train_set_size, stop=60000)
        test_set = MNIST(which_set="test")

        train_set.X = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., train_set.X), 1.), (-1, 1, 28, 28)),(0,2,3,1))
        valid_set.X = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., valid_set.X), 1.), (-1, 1,  28, 28)),(0,2,3,1))
        test_set.X = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., test_set.X), 1.), (-1, 1,  28, 28)),(0,2,3,1))
        # flatten targets
        train_set.y = np.hstack(train_set.y)
        valid_set.y = np.hstack(valid_set.y)
        test_set.y = np.hstack(test_set.y)

        # Onehot the targets
        train_set.y = np.float32(np.eye(10)[train_set.y])
        valid_set.y = np.float32(np.eye(10)[valid_set.y])
        test_set.y = np.float32(np.eye(10)[test_set.y])

        # for hinge loss
        train_set.y = 2 * train_set.y - 1.
        valid_set.y = 2 * valid_set.y - 1.
        test_set.y = 2 * test_set.y - 1.
        # enlarge train data set by mirrroring
        x_train_flip = train_set.X[:, :, ::-1, :]
        y_train_flip = train_set.y
        train_set.X = np.concatenate((train_set.X, x_train_flip), axis=0)
        train_set.y = np.concatenate((train_set.y, y_train_flip), axis=0)




    else:
        print("wrong dataset given")

    return train_set, valid_set, test_set
