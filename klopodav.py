from collections import defaultdict
import random
import datetime
import os

import numpy
import torch
from .abstract_game import AbstractGame
# ○● ⊗x

class MuZeroConfig:
    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game



        ### Game
        self.observation_shape = (3, 10, 10)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = [i for i in range(201)]  # Fixed list of all possible actions. You should only edit the length
        self.players = [i for i in range(2)]  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_actors = 4  # Number of simultaneous threads self-playing to feed the replay buffer
        self.max_moves = 250  # Maximum number of moves if game is not finished before
        self.num_simulations = 30  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping temperature to 0 (ie playing according to the max)

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size

        # Residual Network
        self.downsample = False  # Downsample observations before representation network (See paper appendix Network Architecture)
        self.blocks = 2  # Number of blocks in the ResNet
        self.channels = 8  # Number of channels in the ResNet
        self.reduced_channels = 8  # Number of channels before heads of dynamic and prediction networks
        self.resnet_fc_reward_layers = [5]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [5]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [5]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network

        ### Training
        self.results_path = os.path.join(os.path.dirname(__file__), "../results", os.path.basename(__file__)[:-3],
                                         datetime.datetime.now().strftime(
                                             "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.training_steps = 100000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for sef-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.training_device = "cuda" if torch.cuda.is_available() else "cpu"  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.01  # Initial learning rate
        self.lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        ### Replay Buffer
        self.window_size = 1000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
        self.td_steps = 50  # Number of steps in the future to take into account for calculating the target value
        self.use_last_model_value = False  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)

        # Prioritized Replay (See paper appendix Training)
        self.PER = True  # Select in priority the elements in the replay buffer which are unexpected for the network
        self.use_max_priority = True  # Use the n-step TD error as initial priority. Better for large replay buffer
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
        self.PER_beta = 1.0

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired self played games per training step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.
        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Klop(10)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def close(self):
        """
        Properly close the game.
        """
        pass

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        valid = False
        while not valid:
            valid, action = self.env.human_input_to_action()
        return action

    def action_to_string(self, action):
        """
        Convert an action number to a string representing the action.
        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        return self.env.action_to_human_input(action)

    def random_agent(self):
        return self.env.random_agent()


class Klop:
    def __init__(self, sze):
        self.sze = sze
        self.board = numpy.zeros((self.sze, self.sze)).astype(int)
        self.p1con = []
        self.p2con = []
        self.p1cluster = [[]]
        self.p2cluster = [[]]
        self.state = ''
        self.positions = ''
        self.player = 0
        self.playerTurn = 1
        self.reward = 0
        self.moves = {}
        self.backmoves = {}

        for i, j in enumerate([str(m)+str(s)+str(l) for l in range(self.sze) for s in range(self.sze) for m in range(2)]):
            self.moves[i] = j
            self.backmoves[j] = i
        self.moves[i+1] = 'f'

    def common_data(self, l):
        out = []
        while len(l) > 0:
            first, *rest = l
            lf = -1
            while len(first) > lf:
                lf = len(first)

                rest2 = []
                for r in rest:
                    if any(i in first for i in r):
                        first += r
                    else:
                        rest2.append(r)
                rest = rest2

            out.append(first)
            l = rest
        return out

    def get_dots(self, x, y):
        dot = []
        for m in (-1, 0, 1):
            for n in (-1, 0, 1):
                if not m == n == 0 and self.sze > (x + m) >= 0 and self.sze > (y + n) >= 0:
                    dot.append([x + m, y + n])
        return dot

    def step(self, action):
        done = False
        action = self.moves[int(action)]
        if action == 'f':
            if self.player == 0 and len(self.p2con) != 0:
                self.player = 1
                self.reward = len(self.p2con) + 1
            else:
                self.player = 0
                self.reward = len(self.p1con) + 1

            done = True
            print()
            print(str(self.player) + ' | WON')
            self.render()

            return self.get_observation(), self.reward, done

        x = int(action[1]) if self.player == 0 else self.sze - 1 - int(action[1])
        y = int(action[2]) if self.player == 0 else self.sze - 1 - int(action[2])
        state = int(action[0]) + 1

        self.state = state
        tmp = 0

        if self.player == 0:
            if state == 1:
                self.p1con.append([x, y])
            elif state == 2:
                for i, m in enumerate(self.p1cluster):
                    for n in self.get_dots(x, y):
                        if n in m:
                            self.p1cluster[i].append([x, y])
                            tmp += 1
                            break
                if not tmp:
                    self.p1cluster.append([[x, y]])
                elif tmp > 1:
                    self.p1cluster = list(self.common_data(self.p1cluster))
                self.p2con.remove([x, y])
            self.board[x][y] = state
            if self.playerTurn == 3:
                self.player = 1
            self.playerTurn += 1

        elif self.player == 1:
            if state == 1:
                self.p2con.append([x, y])
            elif state == 2:
                for i, m in enumerate(self.p2cluster):
                    for n in self.get_dots(x, y):
                        if n in m:
                            self.p2cluster[i].append([x, y])
                            tmp += 1
                            break
                if not tmp:
                    self.p2cluster.append([[x, y]])
                elif tmp > 1:
                    self.p2cluster = list(self.common_data(self.p2cluster))
                self.p1con.remove([x, y])
            self.board[x][y] = -1 * state

            if self.playerTurn == 6:
                self.playerTurn = 1
                self.player = 0
            else:
                self.playerTurn += 1

        return self.get_observation(), self.reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """

        return self.player


    def legal_actions(self):
        if self.player == 0:
            state = 1
            opstate = -1
            kstate = 2
            player = self.p1con
            cluster = self.p1cluster
        else:
            state = -1
            opstate = 1
            kstate = -2
            player = self.p2con
            cluster = self.p2cluster

        checked = []
        valid_pos = []
        start_pos = [self.sze - 1, 0][::state]
        if (not player) and (self.board[start_pos[0]][start_pos[1]] == 0):
            if self.player == 0:
                return [self.backmoves[str(0) + str(start_pos[0]) + str(start_pos[1])]]
            else:
                return [self.backmoves[str(0) + str(self.sze - 1 - start_pos[0]) + str(self.sze - 1 - start_pos[1])]]

        for p in player:
            x = p[0]
            y = p[1]
            for m in self.get_dots(x, y):
                d = self.board[m[0]][m[1]]
                if d == 0:
                    valid_pos.append(str(state * state - 1) + str(m[0]) + str(m[1]))
                elif d == opstate:
                    valid_pos.append(str(kstate * state - 1) + str(m[0]) + str(m[1]))
                elif d == kstate:
                    for c in cluster:
                        if (c not in checked) and (m in c):
                            for dot in c:
                                for j in self.get_dots(int(dot[0]), int(dot[1])):
                                    if self.board[j[0]][j[1]] == 0:
                                        valid_pos.append(str(state * state - 1) + str(j[0]) + str(j[1]))
                                    elif self.board[int(j[0])][int(j[1])] == opstate:
                                        valid_pos.append(str(kstate * state - 1) + str(j[0]) + str(j[1]))
                            checked.append(c)
        if not valid_pos:
            return [len(self.moves)-1]
        if self.player == 0:
            return [self.backmoves[i] for n, i in enumerate(valid_pos) if i not in valid_pos[:n]]
        else:
            return [self.backmoves[i[0]+str(self.sze - int(i[1]) - 1)+str(self.sze - int(i[2]) - 1)]
                    for n, i in enumerate(valid_pos) if i not in valid_pos[:n]]

    def get_observation(self):
        board_player1 = numpy.array([[float(abs(i)) if (i == 1 or i == 2) else 0.0 for i in j] for j in self.board])
        board_player2 = numpy.array([[float(abs(i)) if (i == -1 or i == -2) else 0.0 for i in j[::-1]] for j in self.board[::-1]])
        board_to_play = numpy.full((self.sze, self.sze), self.player).astype(int)
        return numpy.array([board_player1, board_player2, board_to_play])

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        self.board = numpy.zeros((self.sze, self.sze)).astype(int)
        self.p1con = []
        self.p2con = []
        self.p1cluster = [[]]
        self.p2cluster = [[]]
        self.state = ''
        self.positions = ''
        self.player = 0
        self.playerTurn = 1
        return self.get_observation()


    def render(self):
        print(' ', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        for m, n in enumerate(self.board):
            a = ' '.join([str(i) for i in n]).replace('-1', '○')
            a = a.replace('-2', '⦻')
            a = a.replace('1', 'x')
            a = a.replace('2', '●')
            a = a.replace('0', '.')

            print(m, a)
        print()

    def random_agent(self):
        a = self.get_valid_pos()
        move = random.choice(a)
        return move

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return str(action_number)

