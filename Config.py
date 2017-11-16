from random import uniform

class Config():
    def __init__(self, firstonly=False):
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

        self.MEMORY = [20000, 40000]
        self.DISCOUNT_RATE = [0.95, 0.99]
        self.EXPLORATION_RATE = [1.0, 1.1]
        self.EXPLORATION_RATE_MIN = [0.01, 0.05]
        self.EXPLORATION_RATE_DECAY = [0.995, 0.9]
        self.LEARNING_RATE = [0.00005, 0.0001]
        self.LEARNING_RATE_DECAY = [1, 1.9]

        self.MUTATE_COORDS = [0.2, 0.3]

        self.ACTIONS = self.SCREEN_SIZE ** 2 * 2

        for lvk in vars(self):
            lv = self.__getattribute__(lvk)
            if isinstance(lv, list):
                if len(lv) == 3 and isinstance(lv[2], int) and lv[2] >= 0:
                    # we assume last value is precision limit
                    prec = lv[2]
                else:
                    prec = 2

                if firstonly:
                    lv = lv[0]
                else:
                    lv = round(uniform(lv[0], lv[1]), prec)

                self.__setattr__(lvk, lv)


