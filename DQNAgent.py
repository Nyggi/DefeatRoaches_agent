import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, ZeroPadding2D
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from collections import namedtuple
import keras
from numpy import unravel_index
import json

np.set_printoptions(threshold=np.nan)

Coords = namedtuple('Coords', ['x', 'y', 'z'])

class DQNAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.action_size = cfg.ACTIONS
        self.memory = deque(maxlen=cfg.MEMORY)
        self.gamma = cfg.DISCOUNT_RATE
        self.epsilon = cfg.EXPLORATION_RATE
        self.epsilon_min = cfg.EXPLORATION_RATE_MIN
        self.epsilon_decay = cfg.EXPLORATION_RATE_DECAY
        self.learning_rate = cfg.LEARNING_RATE
        self.learning_rate_decay = cfg.LEARNING_RATE_DECAY
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
        input_shape = (self.cfg.SCREEN_SIZE, self.cfg.SCREEN_SIZE, self.cfg.INPUT_LAYERS)

        model = Sequential()
        model.add(ZeroPadding2D(padding=(2, 2), input_shape=input_shape))
        model.add(Conv2D(16, (5, 5), activation='relu', use_bias=False))

        model.add(ZeroPadding2D(padding=(1, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu', use_bias=False))

        model.add(Conv2D(2, (1, 1), use_bias=False))

        model.compile(optimizer=RMSprop(lr=self.cfg.LEARNING_RATE), loss=self._huber_loss, metrics=['accuracy'])

        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon: #Explore
            rn = random.randrange(self.action_size)

            coords = unravel_index(rn, (self.cfg.SCREEN_SIZE, self.cfg.SCREEN_SIZE, 2))


            return Coords(coords[0], coords[1], coords[2])
        else:
            act_values = self.model.predict(state)

            coords = unravel_index(act_values[0].argmax(), (self.cfg.SCREEN_SIZE, self.cfg.SCREEN_SIZE, 2))


            if np.random.rand() < self.cfg.MUTATE_COORDS:
                dx = np.random.randint(-4, 5)
                targetx = int(max(0, min(84 - 1, coords[0] + dx)))

                dy = np.random.randint(-4, 5)
                targety = int(max(0, min(84 - 1, coords[1] + dy)))
                coords = (targetx, targety, coords[2])

            return Coords(coords[0], coords[1], coords[2]) #returns coordinates

    def replay(self, memory):

        for state, action, reward, next_state, done in memory:
            target = self.model.predict(state)
            if done:
                target[0][action.x][action.y][action.z] = reward
            else:
                next = self.model.predict(next_state)[0]
                coords = unravel_index(next.argmax(), (self.cfg.SCREEN_SIZE, self.cfg.SCREEN_SIZE, 2))
                coords_n = Coords(coords[0], coords[1], coords[2])

                coords = unravel_index(target[0].argmax(), (self.cfg.SCREEN_SIZE, self.cfg.SCREEN_SIZE, 2))
                coords_f = Coords(coords[0], coords[1], coords[2])

                target[0][action.x][action.y][action.z] = reward + self.gamma * (next[coords_n.x][coords_n.y][coords_n.z] - target[0][coords_f.x][coords_f.y][coords_f.z])
            self.model.fit(state, target, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
