from DQNAgent import DQNAgent
from Config import *
import sys
import gflags as flags
from pysc2.env import sc2_env
from datetime import datetime
from pysc2.lib import features
from pysc2.lib import actions
import numpy


FLAGS = flags.FLAGS

try:
  argv = FLAGS(sys.argv)
except flags.FlagsError as e:
  sys.stderr.write("FATAL Flags parsing error: %s\n" % e)
  sys.stderr.write("Pass --help to see help on flags.\n")
  sys.exit(1)


#Starcraft II stuff
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

def mainrun():
    file = open("data/" + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), "w")

    with sc2_env.SC2Env(
            map_name=MAP,
            step_mul=STEP_MUL,
            screen_size_px=(SCREEN_SIZE, SCREEN_SIZE),
            minimap_size_px=(MINIMAP_SIZE, MINIMAP_SIZE),
            visualize=RENDER) as env:

        dqnAgent = DQNAgent()

        for episode in range(MAX_EPISODES): # Game iterations
            env.reset()
            obs = env.step(actions=[actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])[0]

            for step in range(MAX_AGENT_STEPS):
                player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
                state = Descritize(player_relative)

                action = dqnAgent.act(state)

                target = [action.y, action.x]

                obs = env.step(actions=[actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])])[0]
                player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
                next_state = Descritize(player_relative)

                if step == MAX_AGENT_STEPS:
                    done = True
                else:
                    done = False

                dqnAgent.remember(state, action, obs.reward, next_state, done)

                if len(dqnAgent.memory) > BATCH_SIZE:
                    dqnAgent.update_target_model()
                    dqnAgent.replay(BATCH_SIZE)

            final_score = int(obs.observation["score_cumulative"][0])

            print("Episode: {}/{}, score: {} e: {:.2}".format(episode, MAX_EPISODES, final_score, dqnAgent.epsilon))
            file.write("Episode: {}/{}, score: {} e: {:.2} \n".format(episode, MAX_EPISODES, final_score, dqnAgent.epsilon))


        if SAVE_REPLAY:
            env.save_replay("DefeatRoaches")

def Descritize(feature_layer):
    layers = []
    layer = numpy.zeros([SCREEN_SIZE, SCREEN_SIZE, INPUT_LAYERS], dtype=numpy.int8)
    for j in [0, 1]:
        indy, indx = (feature_layer == j * 3 + 1).nonzero()
        layer[indy, indx, j] = 1
    layers.append(layer)
    return numpy.array(layers)

mainrun()