from DQNAgent import DQNAgent
from Config import *
import sys
import gflags as flags
from pysc2.env import sc2_env, environment
from datetime import datetime
from pysc2.lib import features, actions
import numpy as np
import math


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
_UNIT_HP = features.SCREEN_FEATURES.unit_hit_points.index
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

            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            unit_hp = obs.observation["screen"][_UNIT_HP]

            state = GatherObservations([(player_relative, True), (unit_hp, False)])

            for step in range(MAX_AGENT_STEPS):
                a_actions = obs.observation['available_actions']
                if _ATTACK_SCREEN in a_actions and _MOVE_SCREEN in a_actions:

                    action = dqnAgent.act(state)

                    target = [action.y, action.x]

                    if action.z == 0:
                        obs = env.step(actions=[actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])])[0]
                    else:
                        obs = env.step(actions=[actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])])[0]

                    player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
                    unit_hp = obs.observation["screen"][_UNIT_HP]

                    next_state = GatherObservations([(player_relative, True), (unit_hp, False)])

                    if obs.step_type == environment.StepType.LAST:
                        done = True
                    else:
                        done = False

                    dqnAgent.remember(state, action, obs.reward, next_state, done)

                    state = next_state

                    if done:
                        break
                else:
                    obs = env.step(actions=[actions.FunctionCall(_NO_OP, [])])[0]

            if len(dqnAgent.memory) > BATCH_SIZE:
                dqnAgent.update_target_model()
                dqnAgent.replay(BATCH_SIZE)


            final_score = int(obs.observation["score_cumulative"][0])

            print("Episode: {}/{}, score: {}".format(episode, MAX_EPISODES, final_score))
            file.write("Episode: {}/{}, score: {}\n".format(episode, MAX_EPISODES, final_score))


        if SAVE_REPLAY:
            env.save_replay("DefeatRoaches")

def Descritize(feature_layer):
    layer = np.zeros([SCREEN_SIZE, SCREEN_SIZE, 2], dtype=np.int8)
    for j in [0, 1]:
        indx, indy = (feature_layer == j * 3 + 1).nonzero()
        layer[indx, indy, j] = 1
    return layer

def GatherObservations(feature_layers):
    output = np.zeros([SCREEN_SIZE, SCREEN_SIZE, 0], dtype=np.float32)

    for layer in feature_layers:
        if layer[1]:
            l = Descritize(layer[0])
        else:
            l = np.zeros([SCREEN_SIZE, SCREEN_SIZE, 1], dtype=np.float32)

            ind_x, ind_y = layer[0].nonzero()

            l[ind_x, ind_y, 0] = np.log2(layer[0][ind_x, ind_y])

        output = np.dstack((output, l))

        o = [output]
    return np.array(o)

mainrun()