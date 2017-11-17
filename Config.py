from random import uniform, randrange


class BaseConfig():
    def __init__(self):
        # Pysc2 env variables
        self.RENDER = False
        self.SAVE_REPLAY = True
        self.MAP = "DefeatRoaches"

        self.MAX_AGENT_STEPS = 240
        self.STEP_MUL = 8

        self.MAX_EPISODES = 2000
        self.BATCH_SIZE = 32
        self.SCREEN_SIZE = 84
        self.INPUT_LAYERS = 3
        self.MINIMAP_SIZE = 0

        self.MEMORY = 20000
        self.DISCOUNT_RATE = 0.95
        self.EXPLORATION_RATE = 1.0
        self.EXPLORATION_RATE_MIN = 0.01
        self.EXPLORATION_RATE_DECAY = 0.995
        self.LEARNING_RATE = 0.00005
        self.LEARNING_RATE_DECAY = 1
        self.MUTATE_COORDS = 0.2

        self.ACTIONS = self.SCREEN_SIZE ** 2 * 2

    def dump(self):
        result = {}
        for sk in vars(self):
            result[sk] = self.__getattribute__(sk)
        return result

class MultiConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.MAX_EPISODES = 5000

        self.MEMORY = [5000, 40000]
        self.DISCOUNT_RATE = [0.90, 0.995, 2]
        self.EXPLORATION_RATE = 1
        self.EXPLORATION_RATE_MIN = [0.002, 0.02, 3]
        self.EXPLORATION_RATE_DECAY = [0.995, 0.9, 3]
        self.LEARNING_RATE = [0.00001, 0.001, 5]
        self.LEARNING_RATE_DECAY = 1

        self.MUTATE_COORDS = [0.05, 0.3, 2]

        for lvk in vars(self):
            lv = self.__getattribute__(lvk)
            if isinstance(lv, list):
                if len(lv) == 3 and isinstance(lv[2], int) and lv[2] >= 0:
                    # we assume last value is precision limit
                    prec = lv[2]
                else:
                    prec = 2

                if isinstance(lv[0], int) and isinstance(lv[1], int):
                    lv = randrange(lv[0], lv[1] + 1)
                elif isinstance(lv[0], float) or isinstance(lv[1], float):
                    lv = round(uniform(lv[0], lv[1]), prec)

                self.__setattr__(lvk, lv)


class SingleConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        self.MAX_AGENT_STEPS = 240
        self.STEP_MUL = 8

        self.MAX_EPISODES = 2000
        self.BATCH_SIZE = 32
        self.SCREEN_SIZE = 84
        self.INPUT_LAYERS = 3
        self.MINIMAP_SIZE = 0

        self.MEMORY = 20000
        self.DISCOUNT_RATE = 0.99
        self.EXPLORATION_RATE = 1.0
        self.EXPLORATION_RATE_MIN = 0.01
        self.EXPLORATION_RATE_DECAY = 0.995
        self.LEARNING_RATE = 0.00005
        self.LEARNING_RATE_DECAY = 1
        self.MUTATE_COORDS = 0.2
