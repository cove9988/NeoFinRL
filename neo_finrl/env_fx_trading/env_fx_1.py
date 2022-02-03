import math
import datetime
import random
import json
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from neo_finrl.env_fx_trading.util.plot_chart import TradingChart
from neo_finrl.env_fx_trading.util.log_render import render_to_file
from neo_finrl.env_fx_trading.util.read_config import EnvConfig
from neo_finrl.env_fx_trading.util.action_enum import ActionEnum, form_action

class TradingSellEnv(gym.Env):
    """forex/future/option trading gym environment
    
    For env parameters, please refer to configure file and readme.md

    1. Three action space discrete (-1 Sell, 1 Buy, 0 Nothing).
    1.1 Open position until transtion log empty. which means if Short position, then action Buy to close it before another Sell.
    2. Multiple trading pairs (EURUSD, GBPUSD...) under same time frame
    3. Timeframe from 1 min to daily as long as use candlestick bar (Open, High, Low, Close)
    4. Use StopLose, ProfitTaken to realize rewards. each pair can configure it own SL and PT in configure file
    5. Configure over night cash penalty and each pair's transaction fee and overnight position holding penalty
    6. Split dataset into daily, weekly or monthly..., with fixed time steps, at end of len(df). The business
        logic will force to Close all positions at last Close price (game over).
    7. Must have df column name: [(time_col),(asset_col), Open,Close,High,Low,day] (case sensitive)
    8. Addition indicators can add during the data process. 78 available TA indicator from Finta
    9. Customized observation list handled in json config file. 
    13. render mode:
        human -- display each steps realized reward on console
        file -- create a transaction log
        graph -- create transaction in graph (under development)

    14. Reward, we want to incentivize profit that is sustained over long periods of time. 
        At each step, we will set the reward to the account balance multiplied by 
        some fraction of the number of time steps so far.The purpose of this is to delay 
        rewarding the agent too fast in the early stages and allow it to explore 
        sufficiently before optimizing a single strategy too deeply. 
        It will also reward agents that maintain a higher balance for longer, 
        rather than those who rapidly gain money using unsustainable strategies.
    15. Observation_space contains all of the input variables we want our agent 
        to consider before making, or not making a trade. We want our agent to “see” 
        the forex data points (Open price, High, Low, Close, time serial, TA) in the game window, 
        as well a couple other data points like its account balance, current positions, 
        and current profit.The intuition here is that for each time step, we want our agent 
        to consider the price action leading up to the current price, as well as their 
        own portfolio’s status in order to make an informed decision for the next action.
    16. reward is forex trading unit Point, it can be configure for each trading pair
    17. observation and reward normalization are done by SB3 wrapper 
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    18. this is on policy learning as it is SARSA, best training argtheism is PPO
    19. add timelimit wrapper as we train on a fixed time frame (weekly, monthly, daily)
    """
    metadata = {'render.modes': ['graph', 'human', 'file']}
    env_id = "TradingGym-v1"
    
    def __init__(self, df, env_config_file='./neo_finrl/env_fx_trading/config/TradingSellEnv_gdbusd.json') -> None:
        super(TradingSellEnv, self).__init__()
        self.cf = EnvConfig(env_config_file)
        self.observation_list = self.cf.env_parameters("observation_list")

        self.balance_initial = self.cf.env_parameters("balance")
        self.over_night_cash_penalty = self.cf.env_parameters("over_night_cash_penalty")
        self.asset_col = self.cf.env_parameters("asset_col")
        self.time_col = self.cf.env_parameters("time_col")
        self.random_start = self.cf.env_parameters("random_start")
        self.do_nothing = self.cf.env_parameters("do_nothing")
        self.log_filename = self.cf.env_parameters("log_filename") + \
            datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.csv'
        self.is_training = self.cf.env_parameters("is_training")    
        self.description = self.cf.env_parameters("description")

        self.df = df
        self.df["_time"] = df[self.time_col]
        self.df["_day"] = df["weekday"]
        self.assets = df[self.asset_col].unique()
        self.dt_datetime = df[self.time_col].sort_values().unique()
        self.df = self.df.set_index(self.time_col)
        self.visualization = False
    

        # --- reset value ---
        self.equity_list = [0] * len(self.assets)
        self.balance = self.balance_initial
        self.total_equity = self.balance + sum(self.equity_list)
        self.ticket_id = 0
        self.transaction_live = []
        self.transaction_history = []
        self.transaction_limit_order = []
        self.current_draw_downs = [0.0] * len(self.assets)
        self.max_draw_downs = [0.0] * len(self.assets)
        self.max_draw_down_pct = sum(self.max_draw_downs) / self.balance * 100
        self.current_step = 0
        self.episode = -1
        self.current_holding = [0] * len(self.assets)
        self.tranaction_open_this_step = []
        self.tranaction_close_this_step = []
        self.current_day = 0
        self.done_information = ''
        self.log_header = True
        # --- end reset ---
        self.cached_data = [
            self.get_observation_vector(_dt) for _dt in self.dt_datetime
        ]
        self.cached_time_serial = ((self.df[["_time", "_day"]].sort_values("_time"))\
            .drop_duplicates()).values.tolist()
        
        self.reward_range = (-np.inf, np.inf)
        self.action_space = spaces.Box(low=-1,
                                       high=1, shape=(len(self.assets),))
        # first two 3 = balance,current_holding, max_draw_down_pct
        _space = 3 + len(self.assets) \
                 + len(self.assets) * len(self.observation_list)
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(_space, ))
        print(
            f'initial done:\n'
            f'observation_list:{self.observation_list}\n '
            f'assets:{self.assets}\n '
            f'time serial: {min(self.dt_datetime)} -> {max(self.dt_datetime)} length: {len(self.dt_datetime)}'
        )
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _take_action(self, actions, done):
        rewards = [0] * len(self.assets)
        self.tranaction_open_this_step = []
        self.tranaction_close_this_step = []
        # need use multiply assets
        for i, _action in enumerate(actions):
            self._o = self.get_observation(self.current_step, i, "Open")
            self._h = self.get_observation(self.current_step, i, "High")
            self._l = self.get_observation(self.current_step, i, "Low")
            self._c = self.get_observation(self.current_step, i, "Close")
            self._t = self.get_observation(self.current_step, i, "_time")
            self._day = self.get_observation(self.current_step, i,"_day")

            if _action == ActionEnum.HOLD or self.do_nothing > 0:
                rewards[i] += self.do_nothing
            
            elif self.current_holding[i] == 0:
                if _action in (ActionEnum.BUY, ActionEnum.SELL) and not done :
                    self.ticket_id += 1
                    transaction = {
                        "Ticket": self.ticket_id,
                        "Symbol": self.assets[i],
                        "ActionTime": self._t,
                        "Type": _action,
                        "Lot": 1,
                        "ActionPrice": self._c,
                        "MaxDD": 0,
                        "Swap": 0.0,
                        "CloseTime": "",
                        "ClosePrice": 0.0,
                        "Point": 0,
                        "Reward": -self.cf.symbol(self.assets[i],"transaction_fee"),
                        "DateDuration": self._day,
                        "Status": 0,
                        "ActionStep":self.current_step,
                        "CloseStep":-1,  
                    }
                    self.current_holding[i] += 1
                    self.tranaction_open_this_step.append(transaction)
                    self.balance -= self.cf.symbol(self.assets[i],"transaction_fee")
                    self.transaction_live.append(transaction)
            elif self.current_holding[i] > 0:
                rewards[i] = self._calculate_reward(i, done, _action)

        return sum(rewards)

    def _calculate_reward(self, i, done, _action):
        _total_reward = 0
        _max_draw_down = 0
        for tr in self.transaction_live:
            if tr["Symbol"] == self.assets[i]:
                _point = self.cf.symbol(self.assets[i],"point")
                #cash discount overnight
                if self._day > tr["DateDuration"]:
                    tr["DateDuration"] = self._day
                    tr["Reward"] -= self.cf.symbol(self.assets[i],"over_night_penalty")

                if tr["Type"] == ActionEnum.BUY:  #buy
                    #stop loss trigger
                    if done or _action == ActionEnum.SELL:
                        p = (self._c - tr["ActionPrice"]) * _point
                        self._manage_tranaction(tr, p, self._c, status=2)
                        _total_reward += p
                        self.current_holding[i] -= 1
                    else:  # still open
                        self.current_draw_downs[i] = int(
                            (self._l - tr["ActionPrice"]) * _point)
                        # _total_reward += self.current_draw_downs[i]
                        _max_draw_down += self.current_draw_downs[i]
                        if self.current_draw_downs[i] < 0:
                            if tr["MaxDD"] > self.current_draw_downs[i]:
                                tr["MaxDD"] = self.current_draw_downs[i]

                elif tr["Type"] == ActionEnum.SELL:  # sell
                    if done or _action == ActionEnum.BUY:
                        p = (tr["ActionPrice"] - self._c) * _point
                        self._manage_tranaction(tr, p, self._c, status=2)
                        _total_reward += p
                    else:
                        self.current_draw_downs[i] = int(
                            (tr["ActionPrice"] - self._h) * _point)
                        # _total_reward += self.current_draw_downs[i]
                        _max_draw_down += self.current_draw_downs[i]
                        if self.current_draw_downs[i] < 0:
                            if tr["MaxDD"] > self.current_draw_downs[i]:
                                tr["MaxDD"] = self.current_draw_downs[i]

                if _max_draw_down > self.max_draw_downs[i]:
                    self.max_draw_downs[i] = _max_draw_down

        return _total_reward
             
    def _manage_tranaction(self, tr, _p, close_price, status=1):
        self.transaction_live.remove(tr)
        tr["ClosePrice"] = close_price
        tr["Point"] = int(_p)
        tr["Reward"] = int(tr["Reward"] + _p)
        tr["Status"] = status
        tr["CloseTime"] = self._t
        self.balance += int(tr["Reward"])
        self.total_equity -= int(abs(tr["Reward"]))
        self.tranaction_close_this_step.append(tr)
        self.transaction_history.append(tr)

    def step(self, actions):
        # Execute one time step within the environment
        self.current_step += 1
        done = (self.balance <= 0
                or self.current_step == len(self.dt_datetime) - 1)
        if done:
            self.done_information += f'Episode: {self.episode} Balance: {self.balance} Step: {self.current_step}\n'
            self.visualization = True
        reward = self._take_action(actions, done)
        if self._day > self.current_day:
            self.current_day = self._day
            self.balance -= self.over_night_cash_penalty
        if self.balance != 0:
            self.max_draw_down_pct = abs(
                sum(self.max_draw_downs) / self.balance * 100)
        
        # no action anymore
        # print(f"step:{self.current_step}, action:{actions}, reward :{reward}, balance: {self.balance} {self.max_draw_down_pct}")
        obs = ([self.balance, self.max_draw_down_pct] +
               self.current_holding +
               self.current_draw_downs +
               self.get_observation(self.current_step))
        return np.array(obs).astype(np.float32), reward, done, {
            "Close": self.tranaction_close_this_step
        }

    def get_observation(self, _step, _iter=0, col=None):
        if (col is None):
            return self.cached_data[_step]
        else:
            if col == '_time':
                return self.cached_time_serial[_step][0]
            elif col == '_day':
                return self.cached_time_serial[_step][1]

            col_pos = -1
            for i, _symbol in enumerate(self.observation_list):
                if _symbol == col:
                    col_pos = i
                    break
            assert col_pos >= 0
            return self.cached_data[_step][_iter * len(self.observation_list) +
                                           col_pos]

    def get_observation_vector(self, _dt, cols=None):
        cols = self.observation_list
        v = []
        for a in self.assets:
            subset = self.df.query(
                f'{self.asset_col} == "{a}" & {self.time_col} == "{_dt}"')
            assert not subset.empty
            v += subset.loc[_dt, cols].tolist()
        assert len(v) == len(self.assets) * len(cols)
        return v

    def reset(self):
        # Reset the state of the environment to an initial state
        self.seed()

        if self.random_start:
            self.current_step = random.choice(
                range(int(len(self.dt_datetime) * 0.5)))
        else:
            self.current_step = 0

        self.equity_list = [0] * len(self.assets)
        self.balance = self.balance_initial
        self.total_equity = self.balance + sum(self.equity_list)
        self.ticket_id = 0
        self.transaction_live = []
        self.transaction_history = []
        self.transaction_limit_order = []
        self.current_draw_downs = [0.0] * len(self.assets)
        self.max_draw_downs = [0.0] * len(self.assets)
        self.max_draw_down_pct = sum(self.max_draw_downs) / self.balance * 100
        self.episode = -1
        self.current_holding = [0] * len(self.assets)
        self.tranaction_open_this_step = []
        self.tranaction_close_this_step = []
        self.current_day = 0
        self.done_information = ''
        self.log_header = True
        self.visualization = False

        _space = (
            [self.balance, self.max_draw_down_pct] +
            [0] * len(self.assets) +
            [0] * len(self.assets) +
            self.get_observation(self.current_step))
        return np.array(_space).astype(np.float32)

    def render(self, mode='human', title=None, **kwargs):
        # Render the environment to the screen
        if mode in ('human', 'file'):
            printout = False
            if mode == 'human':
                printout = True
            pm = {
                "log_header": self.log_header,
                "log_filename": self.log_filename,
                "printout": printout,
                "balance": self.balance,
                "balance_initial": self.balance_initial,
                "tranaction_close_this_step": self.tranaction_close_this_step,
                "done_information": self.done_information
            }
            render_to_file(**pm)
            if self.log_header: self.log_header = False
        elif mode == 'graph' and self.visualization:
            print('plotting...')
            p = TradingChart(self.df, self.transaction_history)
            p.plot()

    def close(self):
        pass
    
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs