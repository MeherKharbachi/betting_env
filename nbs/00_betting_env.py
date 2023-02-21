#!/usr/bin/env python
# coding: utf-8

# # OpenAI Gym Env
# 
# > Create a custom GYM environment to simulate trading strategy.

# In[ ]:


# | default_exp betting_env


# # Import Librairies

# In[ ]:


# | export

import pandas as pd
import warnings
import gym
import numpy as np
import numexpr
import json
import os
import sys
import torch
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)
from betting_env.asian_handicap_pnl import *
from betting_env.datastructure.odds import MarketOdds
from betting_env.config.mongo import mongo_init
from pymatchpred.datastructure.lineup import TeamSheet
from pymatchpred.datastructure.features import GameFeatures
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.dataset import TransitionMiniBatch
from inspect import signature
from typing_extensions import Protocol
from infi.traceback import pretty_traceback_and_exit_decorator
from pandas.core.common import SettingWithCopyWarning
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    Type,
    Tuple,
)

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# # Observations Output

# We provide here a simple class that stores our RL observations. The first format is a numerical numpy array that holds numerical game information and the second format is a user-friendly output to show game information in each observation.

# In[ ]:


# | export

class Observation:
    def __init__(
        self,
        game_id: int,  # game Id
        lineups: np.ndarray,  # lineups
        lineups_ids: np.ndarray,  # lineups opta Ids
        teams_names: pd.core.series.Series,  # team names
        teams_ids: np.ndarray,  # teams opta Ids
        betting_market: np.ndarray,  # odds
        ah_line: float,  # Asian handicap line
        observation_shape: set # observation shape
    ):
        self.game_id = game_id
        self.lineups = lineups
        self.lineups_ids = lineups_ids
        self.teams_names = teams_names
        self.teams_ids = teams_ids
        self.betting_market = betting_market
        self.ah_line = ah_line
        self.shape = observation_shape

    def __call__(self):
        """numerical output"""
        self.numerical_observation = np.concatenate(
            (
                np.array([self.game_id]).reshape(1, -1),  # opta gameId
                np.array([self.teams_ids]),  # teams Opta Ids
                np.array([self.lineups_ids[0]]),  # home lineup (players opta Id)
                np.array([self.lineups_ids[1]]),  # away lineup (players opta Id)
                self.betting_market,  # odds (1x2 and AH)
            ),
            axis=1,
        ).reshape(self.shape)

        self.dtype = self.numerical_observation.dtype

        return self
    
    def reshape(self,new_shape):
        self.numerical_observation = self.numerical_observation.reshape(new_shape)
        return self

    def astype(self, data_type):
        self.numerical_observation = self.numerical_observation.astype(data_type)
        return self

    def observation_pretty_output(self):
        """User-friendly output"""
        self.observation = {
            "gameId": [self.game_id],
            "homeTeam": [self.teams_names[0]],
            "awayTeam": [self.teams_names[1]],
            "homeLineup": self.lineups[0],
            "awayLineup": self.lineups[1],
            "odds1": self.betting_market[:, 0:3][0][0],
            "oddsX": self.betting_market[:, 0:3][0][1],
            "odds2": self.betting_market[:, 0:3][0][2],
            "ahLine": [self.ah_line],
            "oddsAhHome": self.betting_market[:, 3:][0][0],
            "oddsAhAway": self.betting_market[:, 3:][0][1],
        }

        return pd.DataFrame(self.observation, index=[0])


# # Betting Environment

# Reinforcement Learning is a branch of machine learning (ML) that focuses on the complex and all-encompassing issue of training a system to behave appropriately in a given situation. Only the value of the reward and observations made about the environment are used to drive learning. The generality of this model allows it to be used in a wide range of real-world contexts, from gaming to the improvement of sophisticated industrial procedures.
# 
# In this perspective, the environment and the agent are two crucial elements of RL. The environment is the Agent's world where it exists and the Agent can engage in interactions with this environment by taking certain actions which cannot change the environment's laws or dynamics.
# 
# The goal of this work is to develop a Deep Reinforcement Learning environment that simulates a betting strategy. The theory underlying this environment is quite straightforward: placing a bet entails selecting a potential outcome, deciding on a stake size, multiplying it by the winning odds, and then deducting the initial wager and any losses.
# 
# Here, the agent can choose a discrete action space with the following options for actions: 
# - choose a small, medium, or big wager size. And,
# - Wagering on the home, draw, or away (1X2 lines), or on the home or away Asian line.
# 
# It should be noted that the agent can only choose one action from the 15 preceding suggestions.
# 
# 
# 
# In addition, our RL betting environment is a subclass of an OpenAI Gym environment, with an observation space equal to (gameId, home team lineup, away team lineup, betting line(1X2, Asian handicap) and selected odds) and an action space equal to the options available to the agent (the wager size and the chosen outcome). 
# 
# A simple action in the environment consists of getting the current observation and placing a bet. The reward (the investment return), which can be positive or negative, is then calculated and deducting the total amount of the wager.
# 
# The line that the agent should select will determine the determined amount. In other words, if we bet 1X2 on the line, we can say that the profit can be expressed as follows:
# 
#     - profit = (bet * invested_amount * results * odds) - (bet * invested_amount)
#              = reward - expense
# 
#     with : 
#     * bet = the chosen outcome or side (Home win, Draw, Away win)
#     * invested_amount = bet size
#     * results = postgame outcome
#     * odds = 1X2 odds
# 
# 
# If the agent selects the Asian handicap, the profit will depend on the outcome of the game's goal-difference and the chosen line (Half Integer Line, Integer Line, Quarter Integer Line). The agent can win the full bet or just the half of it, lose the full bet or just the half of it, or return its stake.
# 
# For instance, the profit could be expressed as follows if the agent had won the wager:
# 
#     - profit = invested_amount * (odds_ah -1)
# 
# If the agent wins the half of the bet :
# 
#     - profit = invested_amount * ((odds_ah - 1) * 0.5))
# 
#     with : 
#     * invested_amount = bet size
#     * odds_ah = Asian Handicap odds
# 
# 
# When there are no more games to play or the user's bank balance is exhausted, an episode will be concluded. 

# In[ ]:


# | export


class BettingEnv(gym.Env):
    """Base class for sports betting environments.

    Creates an OpenAI Gym environment that supports betting a (small / medium / large) amount
    on a single outcome for a single game.

    Parameters
    ----------
    observation_space : gym.spaces.Box
        The observation space for the environment.
        The observation space shape is (1, N) where N is the number of possible
        outcomes for the game + len(gameId, 2 lineups, ah line) .

    action_space : gym.spaces.Discrete
        The action space for the environment.
        The action space is a set of choices that the agent can do.

    balance : float
        The current balance of the environment.

    starting_bank : int, default=100
        The starting bank / balance for the environment.
    """

    metadata = {"render_modes": ["human"]}
    
    # actions
    ACTIONS_LIST = [
        [0, 0, 0, 0, 0],  # no bets
        [0.05, 0, 0, 0, 0],  # betting on home team (1x2)
        [0.4, 0, 0, 0, 0],  # betting on home team (1x2)
        [0.7, 0, 0, 0, 0],  # betting on home team (1x2)
        [0, 0, 0.05, 0, 0],  # betting on away team (1x2)
        [0, 0, 0.4, 0, 0],  # betting on away team (1x2)
        [0, 0, 0.7, 0, 0],  # betting on away team (1x2)
        [0, 0.05, 0, 0, 0],  # betting on draw (1x2)
        [0, 0.4, 0, 0, 0],  # betting on draw (1x2)
        [0, 0.7, 0, 0, 0],  # betting on draw (1x2)
        [0, 0, 0, 0.05, 0],  # betting on home (Asian Handicap)
        [0, 0, 0, 0.4, 0],  # betting on home (Asian Handicap)
        [0, 0, 0, 0.7, 0],  # betting on home (Asian Handicap)
        [0, 0, 0, 0, 0.05],  # betting on away (Asian Handicap)
        [0, 0, 0, 0, 0.4],  # betting on away (Asian Handicap)
        [0, 0, 0, 0, 0.7],  # betting on away (Asian Handicap)
    ]

    def __init__(
        self,
        game_odds,
        odds_column_names=[
            "preGameOdds1",
            "preGameOdds2",
            "preGameOddsX",
            "preGameAhHome",
            "preGameAhAway",
        ],
        starting_bank=100,
    ):
        """Initializes a new environment

        Parameters
        ----------
        game_odds: pandas dataframe
            A list of games, with their betting odds.
        odds_column_names: list of str
            A list of column names with length == number of odds.
        bet_size: list
            3 possible bets : small, medium and large
        starting_bank: int
            bank account

        """

        super().__init__()
        # games df
        self._game = game_odds.copy()
        # sort data by date
        if "gameDate" in self._game.columns:
            self._game["gameDate"] = pd.to_datetime(self._game["gameDate"])
            self._game = self._game.sort_values(by="gameDate")
        # odds (1X2 and Asian handicap) values
        self._odds = self._game[odds_column_names].values
        # results
        self._results = self._game["result"].values
        # ah lines
        self._lines = self._game["lineId"].values
        # game goal-difference
        self._gd = self._game["postGameGd"].values
        # teams names
        self._teams_names = self._game[["homeTeamName", "awayTeamName"]]
        # teams opta id
        self._teams_ids = self._game[["homeTeamOptaId", "awayTeamOptaId"]].values

        # teams lineups (names and positions)
        self._lineups = self._game[["homeTeamLineup", "awayTeamLineup"]].values
        # teams lineups (opta ids)
        self._lineups_ids = self._game[
            ["homeTeamLineupIds", "awayTeamLineupIds"]
        ].values
        # games ids
        self._game_ids = self._game["optaGameId"].values
        # observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self._odds.shape[1] + 25,
            ),  # 25 = 22(players Ids) + 2(home and away team ids) + 1(gameId)
            dtype=np.float64,
        )
        # actions space
        self.action_space = gym.spaces.Discrete(
            len(BettingEnv.ACTIONS_LIST)
        )  # betting action
        # env balance
        self.balance = self.starting_bank = starting_bank
        # current step (game)
        self.current_step = 0
        # bet size for each outcome
        self.bet_size_matrix = None

    def _get_current_index(self):
        return self.current_step % self._odds.shape[0]

    def get_odds(self):
        """Returns the odds for the current step.

        Returns
        -------
        odds : numpy.ndarray of shape (1, n_odds)
            The odds for the current step.
        """
        return pd.DataFrame([self._odds[self.current_step]]).values

    def get_bet(self, action):
        """Returns the betting matrix for the action provided.

        Parameters
        ----------
        action : int
            An action provided by the agent.

        Returns
        -------
        bet : array of shape (1, n_odds)
            The betting matrix, where each outcome specified in the action
            has a value of 1 and 0 otherwise.
        """
        return BettingEnv.ACTIONS_LIST[action]

    @pretty_traceback_and_exit_decorator
    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of episode is reached,
        you are responsible for calling reset() to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Parameters
        ----------
        action : int
            An action provided by the agent.

        Returns
        -------
        observation : dataframe
            The agent's observation of the current environment
        reward : float
            The amount of reward returned after previous action
        done : bool
            Whether the episode has ended, in which case further step() calls will return undefined results
        info : dict
            Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        """
        # init observation
        observation = np.ones(shape=self.observation_space.shape)
        # reward
        reward = 0
        # finish
        done = False
        # episode info
        info = self.create_info(action)
        
        if self.balance < 1:  # no more money
            done = True
        else:
            # bet action
            bet = self.get_bet(action)
            # game result
            results = self.get_results()
            if self.legal_bet(bet):  # making sure agent has enough money for the bet
                # current odds
                odds = self.get_odds()
                # reward (positive or negative)
                reward = self.get_reward(bet, odds, results)
                # update balance
                self.balance += reward
                info.update(legal_bet=True)
            else:
                reward = -(bet * self.bet_size_matrix).sum()
            # update info
            info.update(results=results.argmax())
            info.update(reward=reward)
            # increment step
            self.current_step += 1
            # check if we are finished
            if self.finish():
                done = True
            else:
                observation = self.get_observation()
        
        # update flag
        info.update(done=done)
        # return
        return observation, reward, done, info

    def get_observation(self):
        """return the observation of the current step.

        Returns
        -------
        obs : numpy.ndarray of shape (1, n_odds + 22)
            The observation of the current step.
        """
        # current game
        index = self._get_current_index()
        # current game id
        game_id = self._game_ids[index]
        # current game lineups
        lineups = self._lineups[index]
        lineups_ids = self._lineups_ids[index]
        # teams
        teams_names = self._teams_names.iloc[index]
        teams_ids = self._teams_ids[index]
        # 1X2 and AH odds
        betting_market = self.get_odds()
        # chosen line (AH line)
        ah_line = self._lines[index]

        # observation
        observation = Observation(
            game_id,
            lineups,
            lineups_ids,
            teams_names,
            teams_ids,
            betting_market,
            ah_line,
            self.observation_space.shape
        )
        observation = observation()#.reshape(self.observation_space.shape)

        return observation

    def get_reward(self, bet, odds, results):
        """Calculates the reward

        Parameters
        ----------
        bet : array of shape (1, n_odds)
        odds: dataframe of shape (1, n_odds)
            A games with its betting odds.
        results : array of shape (1, n_odds)

        Returns
        -------
        reward : float
            The amount of reward returned after previous action
        """
        # agent choice
        bet_index = np.argmax(np.array(bet))
        # bet size
        bet_size_matrix = self.bet_size_matrix
        # balance
        balance = self.balance
        # if the action is a AH bet
        if bet_index in [3, 4]:
            # game goal_difference
            obs_gd = (
                self._gd[self.current_step]
                if bet_index == 3
                else -self._gd[self.current_step]
            )
            # ah line
            ah_line = float(
                self._lines[self.current_step]
                if bet_index == 3
                else -self._lines[self.current_step]
            )
            # ah side odds
            ah_odds = (
                odds[:, 3:4][0].item() if bet_index == 3 else odds[:, 4:][0].item()
            )
            # calculate profit
            profit = AsianHandicap.pnl(obs_gd, ah_line, ah_odds)
            profit = (
                0 if profit is None else numexpr.evaluate("sum(bet * balance * profit)")
            )
        else:  # case 1X2
            reward = numexpr.evaluate("sum(bet * balance * results * odds)")
            expense = numexpr.evaluate("sum(bet * balance)")
            profit = reward - expense

        return profit

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns
        -------
        observation : dataframe
            the initial observation.
        """
        self.balance = self.starting_bank
        self.current_step = 0
        return self.get_observation()

    def render(self, mode="human"):
        """Outputs the current balance and the current step.

        Returns
        -------
        msg : str
            A string with the current balance,
            the current step and the current game info.
        """
        index = self._get_current_index()
        teams = self._teams_names.iloc[index]
        game_id = self._game_ids[index]
        teams = teams.itertuples() if isinstance(teams, pd.DataFrame) else [teams]
        teams_str = ", ".join(
            [
                "Home Team: {} VS Away Team: {}".format(
                    row.homeTeamName, row.awayTeamName
                )
                for row in teams
            ]
        )

        print("Current balance at step {}: {}".format(self.current_step, self.balance))
        print("Current game id : {}".format(game_id))
        print(teams_str)

    def finish(self):
        """Checks if the episode has reached an end.

        The episode has reached an end if there are no more games to bet.

        Returns
        -------
        finish : bool
            True if the current_step is equal to n_games, False otherwise
        """
        return self.current_step == self._odds.shape[0]  # no more games left to bet

    def get_results(self):
        """Returns the results matrix for the current step.

        Returns
        -------
        result : array of shape (1, n_odds)
            The result matrix, where the index of the outcome that happened
            value is 1 and the rest of the indexes values are 0.
        """
        result = np.zeros(shape=(1, self._odds.shape[1]))
        result[
            np.arange(result.shape[0], dtype=np.int32),
            np.array([self._results[self.current_step]], dtype=np.int32),
        ] = 1

        return result

    def legal_bet(self, bet):
        """Checks if the bet is legal.

        Checks that the bet does not exceed the current balance.

        Parameters
        ----------
        bet : array of shape (1, n_odds)
            The bet to check.

        Returns
        -------
        legal : bool
            True if the bet is legal, False otherwise.
        """
        bet_size = sum([b * self.balance for b in bet])
        return bet_size <= self.balance

    def create_info(self, action):
        """Creates the info dictionary for the given action.

        The info dictionary holds the following information:
            * the current step
            * game odds of the current step
            * bet action of the current step
            * bet size of the current step
            * the balance at the start of the current step
            * reward of the current step
            * game result of the current step
            * state of the current step
        Parameters
        ----------
        action : int
            An action provided by the agent.

        Returns
        -------
        info : dict
            The info dictionary.
        """
        return {
            "current_step": self.current_step,
            "odds": self.get_odds(),
            "bet_action": env.ACTIONS_LIST[action],
            "balance": self.balance,
            "reward": 0,
            "legal_bet": False,
            "results": None,
            "done": False,
        }


# # Prepare Input

# Load games data from a csv file

# In[ ]:


# | include: false

# load data
raw_odds_data = pd.read_csv("Hosts_edges.csv").head(5)
# extract specific fields
odds_dataframe = raw_odds_data[
    [
        "gameId",
        "gameDate",
        "homeTeamId",
        "homeTeamName",
        "awayTeamId",
        "awayTeamName",
        "preGame_odds1",
        "preGame_oddsX",
        "preGame_odds2",
    ]
].sort_values(by="gameDate")
# change columns names
odds_dataframe.rename(
    columns={
        "preGame_odds1": "preGameOdds1",
        "preGame_oddsX": "preGameOddsX",
        "preGame_odds2": "preGameOdds2",
    },
    inplace=True,
)


# Add new features

# In[ ]:


# | include: false

# initialise connections
mongo_init("prod_atlas")

# add opta game Id
odds_dataframe["optaGameId"] = odds_dataframe.apply(
    lambda row: GameFeatures.get_game(row["gameId"]).game_opta_id,
    axis="columns",
)
# add home team and away team opta Ids
odds_dataframe["homeTeamOptaId"] = odds_dataframe.apply(
    lambda row: GameFeatures.get_game(row["gameId"]).ht_opta_id,
    axis="columns",
)

odds_dataframe["awayTeamOptaId"] = odds_dataframe.apply(
    lambda row: GameFeatures.get_game(row["gameId"]).at_opta_id,
    axis="columns",
)


# add asian handicap
odds_dataframe["preGameAhHome"] = odds_dataframe.apply(
    lambda row: MarketOdds.get_latest(row["gameId"], "asian")["odds1"][0],
    axis="columns",
    result_type="expand",
)
odds_dataframe["preGameAhAway"] = odds_dataframe.apply(
    lambda row: MarketOdds.get_latest(row["gameId"], "asian")["odds2"][0],
    axis="columns",
    result_type="expand",
)
odds_dataframe["lineId"] = odds_dataframe.apply(
    lambda row: MarketOdds.get_latest(row["gameId"], "asian")["line_id"][0],
    axis="columns",
    result_type="expand",
)
# add home team lineup
odds_dataframe["homeTeamLineup"] = odds_dataframe.apply(
    lambda row: json.dumps(
        {
            player.name: player.position
            for player in TeamSheet.get_latest(
                ra_team_id=row["homeTeamId"], date=row["gameDate"]
            ).starting
        }
    ),
    axis="columns",
    result_type="expand",
)
odds_dataframe["homeTeamLineupIds"] = odds_dataframe.apply(
    lambda row: list(
        player.opta_id
        for player in TeamSheet.get_latest(
            ra_team_id=row["homeTeamId"], date=row["gameDate"]
        ).starting
    ),
    axis="columns",
)

# add away team lineup
odds_dataframe["awayTeamLineup"] = odds_dataframe.apply(
    lambda row: json.dumps(
        {
            player.name: player.position
            for player in TeamSheet.get_latest(
                ra_team_id=row["awayTeamId"], date=row["gameDate"]
            ).starting
        }
    ),
    axis="columns",
    result_type="expand",
)

odds_dataframe["awayTeamLineupIds"] = odds_dataframe.apply(
    lambda row: list(
        player.opta_id
        for player in TeamSheet.get_latest(
            ra_team_id=row["awayTeamId"], date=row["gameDate"]
        ).starting
    ),
    axis="columns",
)

# map results {homewin -> 0 , draw -> 1, awaywin -> 2}
odds_dataframe["result"] = raw_odds_data["postGame_tgt_outcome"].map(
    {1.0: 0.0, 0.0: 2.0, 0.5: 1.0}
)
# gd results
odds_dataframe["postGameGd"] = raw_odds_data["postGame_tgt_gd"]


# In[ ]:


# | include: false
odds_dataframe.head()


# # Agent - Env

# Here, we'll set up our betting environment and let the computer program play and make decisions at random.

# In[ ]:


# | include: false
env = BettingEnv(odds_dataframe.reset_index())
max_steps_limit = odds_dataframe.shape[0]


# In[ ]:


# | include: false
env.reset()
for _ in range(0, max_steps_limit):
    print(env.render())
    print("\n Info: \n")
    obs, reward, done, info = env.step(env.action_space.sample())
    print(info)
    print("\n Observation: \n")
    print(obs)
    print("----------------------------------------------------")
    if done:
        break


# ## Test environment with D3rlpy

# In[ ]:


# | export

from d3rlpy import torch_utility
from d3rlpy.torch_utility import _WithDeviceAndScalerProtocol, TorchMiniBatch


# ### Utility function
# In order to apply the monkey-patching concept, we have to set a function that reload all affected packages and modules. To do so, we need to delete them from memory cache.

# In[ ]:


#| export

def uncache(exclude):
    """
    Remove package modules from cache except excluded ones.
    On next import they will be reloaded.
    
    Args:
        exclude (iter<str>): Sequence of module paths.
    """
    
    pkgs = []
    for mod in exclude:
        pkg = mod.split('.', 1)[0]
        pkgs.append(pkg)

    to_uncache = []
    for mod in sys.modules:
        if mod in exclude:
            continue

        if mod in pkgs:
            to_uncache.append(mod)
            continue

        for pkg in pkgs:
            if mod.startswith(pkg + '.'):
                to_uncache.append(mod)
                break

    for mod in to_uncache:
        del sys.modules[mod]


# We will override this function from the d3rlpy package and adapt it to our use case.
# Here we will add the Observation output instance in the condition part.

# In[ ]:


# | export


def torch_api(
    scaler_targets: Optional[List[str]] = None,
    action_scaler_targets: Optional[List[str]] = None,
    reward_scaler_targets: Optional[List[str]] = None,
) -> Callable[..., np.ndarray]:
    def _torch_api(f: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        # get argument names
        sig = signature(f)
        arg_keys = list(sig.parameters.keys())[1:]
        
        def wrapper(
            self: _WithDeviceAndScalerProtocol, *args: Any, **kwargs: Any
        ) -> np.ndarray:
            tensors: List[Union[torch.Tensor, TorchMiniBatch]] = []
            # convert all args to torch.Tensor
            for i, val in enumerate(args):
                tensor: Union[torch.Tensor, TorchMiniBatch]
                if isinstance(val, torch.Tensor):
                    tensor = val
                elif isinstance(val, list):
                    tensor = default_collate(val)
                    tensor = tensor.to(self.device)
                elif isinstance(val, np.ndarray) or isinstance(val, Observation):
                    if val.dtype == np.uint8:
                        dtype = torch.uint8
                    else:
                        dtype = torch.float32
                    tensor = torch.tensor(
                        data=val.numerical_observation,
                        dtype=dtype,
                        device=self.device,
                    )
                elif val is None:
                    tensor = None
                elif isinstance(val, TransitionMiniBatch):
                    tensor = TorchMiniBatch(
                        val,
                        self.device,
                        scaler=self.scaler,
                        action_scaler=self.action_scaler,
                        reward_scaler=self.reward_scaler,
                    )
                else:
                    tensor = torch.tensor(
                        data=val,
                        dtype=torch.float32,
                        device=self.device,
                    )

                if isinstance(tensor, torch.Tensor) or isinstance(val, Observation):
                    # preprocess
                    if self.scaler and scaler_targets:
                        if arg_keys[i] in scaler_targets:
                            tensor = self.scaler.transform(tensor)

                    # preprocess action
                    if self.action_scaler and action_scaler_targets:
                        if arg_keys[i] in action_scaler_targets:
                            tensor = self.action_scaler.transform(tensor)

                    # preprocessing reward
                    if self.reward_scaler and reward_scaler_targets:
                        if arg_keys[i] in reward_scaler_targets:
                            tensor = self.reward_scaler.transform(tensor)

                    # make sure if the tensor is float32 type
                    if tensor is not None and tensor.dtype != torch.float32:
                        tensor = tensor.float()

                tensors.append(tensor)
            return f(self, *tensors, **kwargs)

        return wrapper

    return _torch_api


torch_utility.torch_api = torch_api
uncache(["d3rlpy.torch_utility"])


# Now we can monkey-patch the original function with our new implementation
# 

# In[ ]:


# | export

from d3rlpy.algos import DQN
from torch.optim import Adam
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.preprocessing.scalers import Scaler, register_scaler


# In this CustomScaler, we can do all the transformations that we want to apply on our observations.

# In[ ]:


# | export


class CustomScaler(Scaler):
    def __init__(self):
        pass

    def fit_with_env(self, env: gym.Env):
        pass

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # add numerical data to observations
        observations = x
        return observations

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {}


# In[ ]:


# | include: false
register_scaler(CustomScaler)


# D3rlpy provides not only offline training, but also online training utilities. Here the buffer will try different experiences to collect a decent dataset.

# In[ ]:


# | include: false
buffer = ReplayBuffer(maxlen= 1000000, env= env)


# The majority of the time, the epsilon-greedy strategy chooses the action with the highest estimated reward. Exploration and exploitation should coexist in harmony. Exploration gives us the freedom to experiment with new ideas, often at contradiction with what we have already learnt.

# In[ ]:


# | include: false

# create the epsilon-greedy explorer
explorer = LinearDecayEpsilonGreedy(start_epsilon=1.0,
                                    end_epsilon=0.1,
                                    duration=100000)


# Optimizer to update weights and reduce losses for the Neural Network

# In[ ]:


# | include: false

# modify weight decay
optim_factory = OptimizerFactory(Adam, weight_decay=1e-4)


# We will Implement the DQN Algo

# In[ ]:


# | include: false

custom_scaler = CustomScaler()

dqn = DQN(
    batch_size=32,
    learning_rate=2.5e-4,
    target_update_interval=100,
    optim_factory=optim_factory,
    scaler=custom_scaler,
)

dqn.build_with_env(env)


# Launch training

# In[ ]:


# | include: false

dqn.fit_online(
    env,
    buffer,
    explorer,
    n_steps=100000,  # train for 100K steps
    n_steps_per_epoch=100,  # evaluation is performed every 100 steps
    update_start_step=100,  # parameter update starts after 100 steps
    eval_epsilon=0.3,
    save_metrics=True,
    # tensorboard_dir= 'runs'
)

