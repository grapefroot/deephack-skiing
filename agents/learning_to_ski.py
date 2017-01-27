# -*- coding: utf-8 -*-
import collections
import logging

import numpy as np
import sklearn.linear_model
import sklearn.preprocessing

import agents.agent
from agents import features


log = logging.getLogger(name=__name__)


class Learning2Ski(agents.agent.Agent):
    """
    Fitted Q-Iteration + Experience Replay + Pseudo Rewards + Hand-engineered
    features + Q-function approximation
    """
    def __init__(self,
                 learning=True,
                 n_history=5,
                 discount=0.99,
                 iteration_size=75,
                 batch_size=3000):
        super(Learning2Ski, self).__init__()

        self.discount = discount

        # use this to track how long it's been since we created fake costs
        # and trained
        self.fake_episode_length = 0

        # use this to track how far into the episode we are because we
        # don't want history features to cross real episodes
        self.real_episode_length = 0

        self.n_fake_episodes = 0.0

        self.n_trainings = 0

        # how far back in history to go when creating features
        self.n_history = n_history

        self.iteration_size = iteration_size
        self.batch_size = batch_size
        log.debug('Batch size: {}'.format(self.batch_size))
        self.sars = collections.deque(maxlen=batch_size)

        self.chosen_actions = []

        # metadata that is useful for calculating features, etc., but
        # are not good as features themselves
        self.states = []

        # are we training the network?
        self.learning = learning

        feature_size = len(self.get_features(
            collections.defaultdict(float), 0
        ).ravel())

        self.ridge = sklearn.linear_model.SGDRegressor(warm_start=True)
        self.ridge.t_ = None
        self.ridge.coef_ = np.random.rand(feature_size) - 0.5
        self.ridge.intercept_ = np.random.rand(1) - 0.5

        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(self.get_features(
            collections.defaultdict(float), 0
        ))

        # keep track of the score states we've seen
        self.seen_scores = []

    def game_score_state(self, image):
        score = image[30:39, 65:82]

        # have we seen this score before?
        for i, seen_score in enumerate(self.seen_scores):
            # if we've seen it
            if np.all(score == seen_score):
                # return its index
                return i

        # if not, keep track of it and return
        self.seen_scores.append(score)
        return len(self.seen_scores) - 1

    def make_metadata(self, image, centiseconds):
        m = {}
        centiseconds = float(centiseconds)
        m['centiseconds'] = centiseconds
        m['game_score_state'] = self.game_score_state(image)

        # get the skier location
        try:
            skier = np.array(
                features.skier_loc(features.skier_box(image))
            )
        except:
            # Skier is probably behind a tree
            if len(self.states) > 0:
                skier = self.states[-1]['skier']
            else:
                skier = np.array([
                    features.WINDOW_WIDTH / 2.0,
                    (
                        features.SKIER_Y[1] - features.SKIER_Y[0] / 2.0
                    ) + features.SKIER_Y[0]
                ])
        m['skier'] = skier

        # get the nearest flag location
        flag = features.flag_loc(image, skier[1])

        m['flag'] = flag

        m['y_distance_to_flag'] = flag[1] - skier[1]
        m['x_distance_to_flag'] = np.abs(skier[0] - flag[0])
        m['distance_to_flag'] = np.linalg.norm(flag - skier)

        # use this slice to track the speed in y
        # looks like (y, array)
        m['y_delta_slice'] = features.y_delta_slice(image)

        # there is a set of things that we are interested in knowing about
        # the difference between this state and last
        if self.real_episode_length == 0:
            # pixel diffs init to 0
            m['delta_x'] = 0
            m['delta_y'] = 0
            m['score_changed'] = False

            # if we can't get that info, just return
            return m

        m['score_changed'] = (
            m['game_score_state'] != self.states[-1]['game_score_state']
        )

        # get the pixel difference in x between this frame and the last
        m['delta_x'] = skier[0] - self.states[-1]['skier'][0]

        # get the pixel difference in y between this frame and the last
        m['delta_y'] = features.delta_y(
            image, self.states[-1]['y_delta_slice']
        )
        # and then there is a set of things we're interested in knowing about
        # the difference between this state and each previous state up to
        # n_history
        delta_x, delta_y = m['delta_x'], m['delta_y']

        cum_centiseconds = centiseconds

        for i in range(1, self.n_history):
            if len(self.states) < i:
                break

            # speed in pixels per centisecond
            m['speed_x_{}'.format(i)] = delta_x / cum_centiseconds

            # speed in pixels per centisecond
            m['speed_y_{}'.format(i)] = delta_y / cum_centiseconds

            # am i moving downward?
            right = np.array([1, 0])
            moving = np.array(
                [m['speed_x_{}'.format(i)], m['speed_y_{}'.format(i)]]
            )
            norm_moving = np.linalg.norm(moving)
            cos = np.dot(right, moving) / norm_moving

            # if nan (or, surprisingly, 0 -- to handle sliding down the wall)
            # TODO: fix this
            if np.isnan(cos) or cos == 0:
                cos = self.states[-1].get('cos_movement_{}'.format(i), 0)

            m['cos_movement_{}'.format(i)] = cos

            m['delta_x_{}'.format(i)] = delta_x
            m['delta_y_{}'.format(i)] = delta_y

            delta_x += self.states[-i]['delta_x']
            delta_y += self.states[-i]['delta_y']
            cum_centiseconds += self.states[-i]['centiseconds']

        return m

    def col_map(self, val, threshold):
        if val < -threshold:
            return 0
        elif -threshold <= val <= threshold:
            return 1
        elif val > threshold:
            return 2
        else:
            raise ValueError('val {}, threshold {}'.format(val, threshold))

    def get_features(self, meta, action):
        val_scale = {
            'right': [-2, -2, 2],
            'left': [2, -2, -2],
            'noop_ld': [-1, 0, 1],
            'noop_d': [0, 2, 0],
            'noop_rd': [1, 0, -1],
            'noop_lr': [-3, -3, -3],
            'noop_none': [0, 2, 0],
        }

        # transform the action with the `f()` function form the paper
        # get the action
        left = action == agents.agent.POSSIBLE_ACTIONS['left']
        noop = action == agents.agent.POSSIBLE_ACTIONS['noop']
        right = action == agents.agent.POSSIBLE_ACTIONS['right']

        # define noop type
        speed_x = meta.get('speed_x_2', 0.0)
        speed_y = meta.get('speed_y_2', 0.0)

        noop_ld = noop and speed_x < 0 and speed_y > 0
        noop_d = noop and speed_x == 0 and speed_y > 0
        noop_rd = noop and speed_x > 0 and speed_y > 0
        noop_lr = noop and np.abs(speed_x) > 0 and speed_y == 0
        noop_none = noop and speed_x == 0 and speed_y == 0

        if left:
            state = 'left'
        elif right:
            state = 'right'
        elif noop_ld:
            state = 'noop_ld'
        elif noop_d:
            state = 'noop_d'
        elif noop_rd:
            state = 'noop_rd'
        elif noop_lr:
            state = 'noop_lr'
        elif noop_none:
            state = 'noop_none'
        else:
            raise ValueError('Action: {}, Speed X: {}, Speed Y: {}'.format(
                action, speed_x, speed_y
            ))

        # now, calculate features with `g()` and `f()`
        f = []

        f.append(speed_x)
        f.append(speed_x**2)
        f.append(speed_y)
        f.append(speed_y**2)

        skier = meta.get('skier', np.array([0, 0]))
        flag = meta.get('flag', np.array([0, 0]))
        # x distance to flag
        val = (skier[0] - flag[0])

        half_slalom = features.POLE_TO_POLE / 2.0  # 16
        threshold = half_slalom

        col = self.col_map(val, threshold)
        scaled = val
        f_val = val_scale[state][col] * (np.abs(scaled) + 0.0001)
        f.append(f_val)

        # x distance from mid (edge)
        mid = features.WINDOW_WIDTH / 2.0
        val = (skier[0] - mid)
        threshold = 60
        col = self.col_map(val, threshold)
        scaled = val  # / mid
        f_val = val_scale[state][col] * (np.abs(scaled) + 0.0001)
        f.append(f_val)

        # encourage the skier to keep pointing itself downhill
        val = meta.get('cos_movement_2', 0.0)
        threshold = 0.44
        col = self.col_map(val, threshold)
        scaled = val  # / mid
        f_val = val_scale[state][col] * (np.abs(scaled) + 0.0001)
        f.append(f_val)

        return np.array(f, dtype=np.float).reshape(1, -1)

    def getQ(self, sa_features):
        X = self.scaler.transform(sa_features)
        return self.ridge.predict(X).ravel()[0]

    @staticmethod
    def softmax(x):
        x = np.asarray(x)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        x = np.exp(x - x.max(axis=1, keepdims=True))
        return x / x.sum(axis=1, keepdims=True)

    def sample_action(self, qs):
        soft_qs = (1.0 - self.softmax(qs).ravel()) / 2.0

        try:
            return np.random.choice(self.actions(), p=soft_qs)
        except:
            raise ValueError('qs {}, soft qs {}'.format(qs, soft_qs))

    def make_fake_costs(self, states):
        zeros = np.zeros(len(states), dtype=np.float)

        time = sum(meta['centiseconds'] for meta in states)
        distance = sum(meta.get('delta_y', 0.0) for meta in states)
        sloth = zeros + (time / (distance + 0.0001))
        slalom = np.array([
            -500.0 * meta.get('score_changed', 0) for meta in states
        ])

        cost = np.sum([
            zeros,
            sloth,
            slalom,
            # off_track
        ], axis=0)

        return cost


    def update_Q(self, inputs, targets):
        """
        Train the Q approximator with the batch of inputs and targets from
        the current Q approximator
        """
        X = np.vstack(inputs)
        y = np.asarray(targets)

        # shuffle
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        X = X[indices, :]
        y = y[indices]

        # scale the features
        X = self.scaler.transform(X)

        self.ridge.partial_fit(X, y)

