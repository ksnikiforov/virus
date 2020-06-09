from collections import defaultdict
import random
import datetime
import baseconvert
import os

import numpy
import torch
from .abstract_game import AbstractGame
# ○● ⊗x

class MuZeroConfig:
    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game



        ### Game
        self.observation_shape = (3, 6, 6)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = [i for i in range(37)]  # Fixed list of all possible actions. You should only edit the length
        self.players = [i for i in range(2)]  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class
                 ### Self-Play
        self.num_actors = 1  # Number of simultaneous threads self-playing to feed the replay buffer
        self.max_moves = 72  # Maximum number of moves if game is not finished before
        self.num_simulations = 100  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = 6  # Number of moves before dropping temperature to 0 (ie playing according to the max)

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size

        # Residual Network
        self.downsample = False  # Downsample observations before representation network (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(__file__), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.training_steps = 1000000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.training_device = "cuda" if torch.cuda.is_available() else "cpu"  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.01  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.window_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 72  # Number of game moves to keep for every batch element
        self.td_steps = 72 # Number of steps in the future to take into account for calculating the target value
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)

        # Prioritized Replay (See paper appendix Training)
        self.PER = True  # Select in priority the elements in the replay buffer which are unexpected for the network
        self.use_max_priority = False  # If False, use the n-step TD error as initial priority. Better for large replay buffer
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
        self.env = Klop(6)

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
        a = input('Your move:')
        x, y = (5 - int(a[0])), (5-int(a[1]))
        return baseconvert.base(int(str(x)+str(y)), 6, 10, string=True)

    def action_to_string(self, action):
        """
        Convert an action number to a string representing the action.
        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        return baseconvert.base(action, 10, 6, string=True)

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
        self.moveCount = 0

    # Перебирает данные данные из массивов, соединяет массивы в один если данные пересекаются.
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

    # Возвращает точки вокруг координаты
    def get_dots(self, x, y):
        dot = []
        for m in (-1, 0, 1):
            for n in (-1, 0, 1):
                if not m == n == 0 and self.sze > (x + m) >= 0 and self.sze > (y + n) >= 0:
                    dot.append([x + m, y + n])
        return dot

    # Функция, котрая обрабатывает ходы
    def step(self, action):
        done = False
        if action == 36: # В случае поражения одной из сторон
            if self.player == 0 and len(self.p2con) != 0:
                self.player = 1
            else:
                self.player = 0

            self.reward = ((self.sze**2)/self.moveCount)**2
            done = True
            print()
            print(str(self.player) + ' | WON')
            self.render()

            return self.get_observation(), self.reward, done

        action = baseconvert.base(int(action), 10, 6, string=True).zfill(2)

        # Переворачиваю координаты в зависимотси вот игрока. (что бы бот всегда играл как бы с одной стороны)
        x = int(action[0]) if self.player == 0 else self.sze - 1 - int(action[0])
        y = int(action[1]) if self.player == 0 else self.sze - 1 - int(action[1])
        state = 1 if self.board[x][y] == 0 else 2
        self.state = state
        tmp = 0

        if self.player == 0:
            if state == 1:  # state 1 - Поставить новую клетку
                self.p1con.append([x, y])
            elif state == 2:  # state 2 - Убрать чужую клетку
                for i, m in enumerate(self.p1cluster):
                    for n in self.get_dots(x, y):
                        if n in m:  # находим в каком кластере клетка и добавляем туда.
                            self.p1cluster[i].append([x, y])
                            tmp += 1
                            break
                if not tmp:  # Если клетка не в кластере, создаем новый
                    self.p1cluster.append([[x, y]])
                elif tmp > 1:  # Если клетка в нескольких кластерах - объединяем их
                    self.p1cluster = list(self.common_data(self.p1cluster))
                self.p2con.remove([x, y])

            self.board[x][y] = state

            # Это нужно тк каждый игрок делает по 3 хода
            if self.playerTurn == 2:
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

            # для уменьшения числа возможных ходов, у каждый игрок вводит просто коодинату без состояния,
            # но для другого игрока используются отрисцательные значения.
            self.board[x][y] = -1 * state

            if self.playerTurn == 4:
                self.playerTurn = 1
                self.player = 0
            else:
                self.playerTurn += 1
        self.moveCount += 1
        return self.get_observation(), self.reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """

        return self.player

    # Возвращает массив легальных ходов. Поэтому ходы не проверяются в step()
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
        # если игрок еще не сделал не одного хода - возвращаем стартовую
        if (not player) and (self.board[start_pos[0]][start_pos[1]] == 0):
            return [30]
        # Находим валидные ходы
        for p in player:
            x = p[0]
            y = p[1]
            for m in self.get_dots(x, y):
                d = self.board[m[0]][m[1]]
                if d == 0 or d == opstate:
                    valid_pos.append(int(str(m[0]) + str(m[1])))
                elif d == kstate:
                    for c in cluster:
                        if (c not in checked) and (m in c):
                            for dot in c:
                                for j in self.get_dots(int(dot[0]), int(dot[1])):
                                    if self.board[j[0]][j[1]] in [0, opstate]:
                                        valid_pos.append(int(str(j[0]) + str(j[1])))
                            checked.append(c)  # Для того что бы не проверять один кластер несколько раз
        if not valid_pos:
            return [36]  # Возвращаем ход "Сдаться" если нет доступных ходов
        else:
            valid_pos = list(set(valid_pos))

        if self.player == 0:
            # print(self.player)
            # print('valid', valid_pos)
            # print('p1con:', self.p1con)
            # print('p2con:', self.p2con)
            # print('p1cluster:', self.p1cluster)
            # print('p2cluster:', self.p2cluster)
            # self.render()
            return [int(baseconvert.base(i, 6, 10, string=True)) for i in valid_pos]
        else:
            return [int(baseconvert.base(int(str(self.sze - int(str(i).zfill(2)[0]) - 1) +
                    str(self.sze - int(str(i).zfill(2)[1]) - 1)), 6, 10, string=True))
                    for i in valid_pos]  # Переворачиваем ходы для 2го игрока

    def get_observation(self):
        board_player1 = numpy.array([[float(abs(i)) if (i == 1 or i == 2) else 0.0 for i in j] for j in self.board])
        board_player2 = numpy.array([[float(abs(i)) if (i == 1 or i == -2) else 0.0 for i in j[::-1]] for j in self.board[::-1]])
        board_to_play = numpy.full((self.sze, self.sze), self.player).astype(int)
        # Возвращаем стол для 1го игрока и обратный стол для 2го
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
        self.moveCount = 0
        return self.get_observation()

    # Графический вывод
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
