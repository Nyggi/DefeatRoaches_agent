import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, ZeroPadding2D
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from collections import namedtuple
from Config import *
import keras
from numpy import unravel_index
import json

np.set_printoptions(threshold=np.nan)

Coords = namedtuple('Coords', ['x', 'y', 'z'])

class DQNAgent:
    def __init__(self):
        self.action_size = ACTIONS
        self.memory = deque(maxlen=MEMORY)
        self.gamma = DISCOUNT_RATE
        self.epsilon = EXPLORATION_RATE
        self.epsilon_min = EXPLORATION_RATE_MIN
        self.epsilon_decay = EXPLORATION_RATE_DECAY
        self.learning_rate = LEARNING_RATE
        self.learning_rate_decay = LEARNING_RATE_DECAY
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        error = prediction - target
        return K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1)

    def _square_loss(self, target, prediction):
        error = target - prediction
        return abs(error)

    def _build_model(self):
        input_shape = (SCREEN_SIZE, SCREEN_SIZE, INPUT_LAYERS)

        model = Sequential()
        model.add(ZeroPadding2D(padding=(2, 2), input_shape=input_shape))
        model.add(Conv2D(16, (5, 5)))
        model.add(Activation('relu'))

        model.add(ZeroPadding2D(padding=(1, 1)))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))

        #model.add(Dropout(0.5))

        model.add(Conv2D(2, (1, 1)))
        #model.add(Activation('relu'))

        model.compile(optimizer=RMSprop(lr=LEARNING_RATE), loss=self._huber_loss, metrics=['accuracy'])
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon: #Explore
            rn = random.randrange(self.action_size)
            coords = unravel_index(rn, (SCREEN_SIZE, SCREEN_SIZE, 2))

            return Coords(coords[0], coords[1], coords[2])
        else:
            act_values = self.model.predict(state)

            coords = unravel_index(act_values[0].argmax(), (SCREEN_SIZE, SCREEN_SIZE, 2))

            if np.random.rand() < MUTATE_COORDS:
                dx = np.random.randint(-4, 5)
                targetx = int(max(0, min(84 - 1, coords[0] + dx)))

                dy = np.random.randint(-4, 5)
                targety = int(max(0, min(84 - 1, coords[1] + dy)))
                coords = (targetx, targety, coords[2])

            return Coords(coords[0], coords[1], coords[2]) #returns coordinates

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action.x][action.y] = reward
            else:
                a = self.model.predict(next_state)
                coords = unravel_index(a[0].argmax(), (SCREEN_SIZE, SCREEN_SIZE, 2))
                coords = Coords(coords[0], coords[1], coords[2])
                t = self.target_model.predict(next_state)[0]
                target[0][action.x][action.y] = reward + self.gamma * t[coords.x][coords.y][coords.z]
            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.learning_rate *= self.learning_rate_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
