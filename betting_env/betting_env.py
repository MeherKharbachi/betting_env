# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/Environment/03_betting_env.ipynb.

# %% auto 0
__all__ = ['SMALL_BET', 'MEDIUM_BET', 'LARGE_BET', 'ACTIONS_LIST', 'Observation', 'BettingEnv']

# %% ../nbs/Environment/03_betting_env.ipynb 4
import pandas as pd
import gym
import numpy as np
import numexpr
import json
import requests
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
from .config.mongo import mongo_init
from .datastructure.fixtures import *
from .utils.asian_pnl import ah_pnl

# %% ../nbs/Environment/03_betting_env.ipynb 7
class Observation:
    def __init__(
        self,
        game_id: int,  # Game Id.
        lineups: np.ndarray,  # Lineups.
        lineups_ids: np.ndarray,  # Lineups opta Ids.
        teams_names: pd.core.series.Series,  # Team names.
        teams_ids: np.ndarray,  # Teams opta Ids.
        betting_market: np.ndarray,  # Odds.
        ah_line: float,  # Asian handicap line.
        observation_shape: tuple,  # Observation shape.
    ):
        self.game_id = game_id
        self.lineups = lineups
        self.lineups_ids = lineups_ids
        self.teams_names = teams_names
        self.teams_ids = teams_ids
        self.betting_market = betting_market
        self.ah_line = ah_line
        self.shape = observation_shape

    def __call__(self) -> "Observation":
        "numerical output"
        self.numerical_observation = np.concatenate(
            (
                np.array([self.game_id]).reshape(1, -1),  # Opta gameId.
                np.array([self.teams_ids]),  # Teams Opta Ids.
                np.array([self.lineups_ids[0]]),  # Home lineup (players opta Id).
                np.array([self.lineups_ids[1]]),  # Away lineup (players opta Id).
                self.betting_market,  # Odds (1x2 and AH).
            ),
            axis=1,
        ).reshape(self.shape)

        self.dtype = self.numerical_observation.dtype
        return self

    def reshape(
        self,
        new_shape: tuple,  # new shape
    ) -> "Observation":
        "reshape observation"
        self.numerical_observation = self.numerical_observation.reshape(new_shape)
        return self

    def astype(
        self,
        data_type: str,  # data type
    ) -> "Observation":
        "cast observation type"
        self.numerical_observation = self.numerical_observation.astype(data_type)
        return self

    def observation_pretty_output(self) -> pd.DataFrame:
        "User-friendly output"
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

# %% ../nbs/Environment/03_betting_env.ipynb 10
# Bet size(small, medium, large)
SMALL_BET, MEDIUM_BET, LARGE_BET = 0.05, 0.2, 0.7

# Actions
ACTIONS_LIST = [
    [0, 0, 0, 0, 0],  # No bets.
    [SMALL_BET, 0, 0, 0, 0],  # Betting on home team (1x2).
    [MEDIUM_BET, 0, 0, 0, 0],  # Betting on home team (1x2).
    [LARGE_BET, 0, 0, 0, 0],  # Betting on home team (1x2).
    [0, 0, SMALL_BET, 0, 0],  # Betting on away team (1x2).
    [0, 0, MEDIUM_BET, 0, 0],  # Betting on away team (1x2).
    [0, 0, LARGE_BET, 0, 0],  # Betting on away team (1x2).
    [0, SMALL_BET, 0, 0, 0],  # Betting on draw (1x2).
    [0, MEDIUM_BET, 0, 0, 0],  # Betting on draw (1x2).
    [0, LARGE_BET, 0, 0, 0],  # Betting on draw (1x2).
    [0, 0, 0, SMALL_BET, 0],  # Betting on home (Asian Handicap).
    [0, 0, 0, MEDIUM_BET, 0],  # Betting on home (Asian Handicap).
    [0, 0, 0, LARGE_BET, 0],  # Betting on home (Asian Handicap).
    [0, 0, 0, 0, SMALL_BET],  # Betting on away (Asian Handicap).
    [0, 0, 0, 0, MEDIUM_BET],  # Betting on away (Asian Handicap).
    [0, 0, 0, 0, LARGE_BET],  # Betting on away (Asian Handicap).
]

# %% ../nbs/Environment/03_betting_env.ipynb 11
class BettingEnv(gym.Env):
    """OpenAI Gym class for football betting environments."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        game_odds: pd.DataFrame,  # Games with their betting odds.
        odds_column_names: list = [
            "preGameOdds1",
            "preGameOdds2",
            "preGameOddsX",
            "preGameAhHome",
            "preGameAhAway",
        ],  # Betting odds column names.
        starting_bank: float = 100.0,  # Starting bank account.
    ) -> None:
        "Initializes a new environment."

        super().__init__()
        # Games df.
        self._game = game_odds.copy()
        # Sort data by date.
        if "gameDate" in self._game.columns:
            self._game["gameDate"] = pd.to_datetime(self._game["gameDate"])
            self._game = self._game.sort_values(by="gameDate").reset_index()
        # Odds (1X2 and Asian handicap) values.
        self._odds = self._game[odds_column_names].values
        # Results.
        self._results = self._game["result"].values
        # Ah lines.
        self._lines = self._game["lineId"].values
        # Game goal-difference.
        self._gd = self._game["postGameGd"].values
        # Teams names.
        self._teams_names = self._game[["homeTeamName", "awayTeamName"]]
        # Teams opta id.
        self._teams_ids = self._game[["homeTeamOptaId", "awayTeamOptaId"]].values
        # Teams lineups (names and positions).
        self._lineups = self._game[["homeTeamLineup", "awayTeamLineup"]].values
        # Teams lineups (opta ids).
        self._lineups_ids = self._game[
            ["homeTeamLineupIds", "awayTeamLineupIds"]
        ].values
        # Games ids.
        self._game_ids = self._game["optaGameId"].values
        # Env balance.
        self.balance = self.starting_bank = starting_bank
        # Current step (game).
        self.current_step = 0
        # Bet size for each outcome.
        self.bet_size_matrix = None
        # cummulative reward
        self.cummulative_profit = [0]
        # cummulative balance
        self.cummulative_balance = [self.balance]
        # plotly fig
        self.fig = go.Figure()
        self.fig.update_layout(
            title="Cumulative Profit and Balance over time",
            xaxis_title="Step",
            yaxis_title="Cumulative Profit and Balance",
        )

        # Gym action space.
        self.action_space = gym.spaces.Discrete(len(ACTIONS_LIST))  # Betting action

        # Gym observation space.
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self._odds.shape[1] + 25,
            ),  # 25 = 22(players Ids) + 2(home and away team ids) + 1(gameId).
            dtype=np.float64,
        )

    def _get_current_index(self) -> None:
        "Returns the current index of the current game."
        return self.current_step % self._odds.shape[0]

    def get_odds(self) -> np.ndarray:
        "Returns odds for the current step"
        return pd.DataFrame([self._odds[self.current_step]]).values

    def get_bet(
        self,
        action: int,  # The chosen action by the agent.
    ) -> list:
        "Returns the betting matrix for the provided action."
        return ACTIONS_LIST[action]

    def step(
        self,
        action: int,  # The chosen action by the agent.
    ) -> tuple:
        "Run one timestep of the environment's dynamics. It accepts an action and returns a tuple (observation, reward, done, info)"
        # Init observation.
        observation = np.ones(shape=self.observation_space.shape)
        # Reward.
        reward = 0
        # finish
        done = False
        # Episode info.
        info = self.create_info(action)

        # If no more money.
        if self.balance < 1:
            done = True
        else:
            # Bet action.
            bet = self.get_bet(action)
            # Game result.
            results = self.get_results()
            # Making sure agent has enough money for the bet.
            if self.legal_bet(bet):
                # Current odds.
                odds = self.get_odds()
                # Reward (positive or negative).
                reward = self.get_reward(bet, odds, results)
                # Update balance.
                self.balance += reward
                info.update(legal_bet=True)
            else:
                reward = -(bet * self.bet_size_matrix).sum()
            # Update info.
            info.update(results=results.argmax())
            info.update(reward=reward)

            # Increment step.
            self.current_step += 1
            # Check if we are finished.
            if self.finish():
                done = True
            else:
                observation = self.get_observation()

        # Update flag.
        info.update(done=done)
        # save current states
        self.cummulative_profit.append(reward)
        self.cummulative_balance.append(self.balance)
        # Return.
        return observation, reward, done, info

    def get_observation(self) -> "Observation":
        "Returns the observation of the current step."
        # Current game index.
        index = self._get_current_index()
        # Current game id.
        game_id = self._game_ids[index]
        # Current game lineups.
        lineups = self._lineups[index]
        lineups_ids = self._lineups_ids[index]
        # Teams.
        teams_names = self._teams_names.iloc[index]
        teams_ids = self._teams_ids[index]
        # 1X2 and AH odds.
        betting_market = self.get_odds()
        # Chosen line (AH line).
        ah_line = self._lines[index]

        # Observation.
        observation = Observation(
            game_id,
            lineups,
            lineups_ids,
            teams_names,
            teams_ids,
            betting_market,
            ah_line,
            self.observation_space.shape,
        )
        observation = observation()
        return observation

    def get_reward(
        self,
        bet: list,  # The betting matrix for the provided action.
        odds: np.ndarray,  # Odds for the current game.
        results: np.ndarray,  # Game result (real outcome).
    ) -> float:
        "Calculates the reward (the profit)."
        # Agent choice.
        bet_index = np.argmax(np.array(bet))
        # Bet size.
        bet_size_matrix = self.bet_size_matrix
        # Balance.
        balance = self.balance
        # If the action is a AH bet.
        if bet_index in [3, 4]:
            # Game goal_difference.
            obs_gd = (
                self._gd[self.current_step]
                if bet_index == 3
                else -self._gd[self.current_step]
            )
            # Ah line.
            ah_line = float(
                self._lines[self.current_step]
                if bet_index == 3
                else -self._lines[self.current_step]
            )
            # Ah side odds.
            ah_odds = (
                odds[:, 3:4][0].item() if bet_index == 3 else odds[:, 4:][0].item()
            )
            # Calculate profit.
            profit = ah_pnl(obs_gd, ah_line, ah_odds)
            profit = (
                0 if profit is None else numexpr.evaluate("sum(bet * balance * profit)")
            )
        else:  # Case 1X2.
            reward = numexpr.evaluate("sum(bet * balance * results * odds)")
            expense = numexpr.evaluate("sum(bet * balance)")
            profit = reward - expense

        return profit

    def reset(self) -> "Observation":
        "Resets the state of the environment and returns an initial observation."
        # Reset balance to initial starting bank.
        self.balance = self.starting_bank
        # Reset initial step to 0.
        self.current_step = 0
        # Reset cumm profit and balance
        self.cummulative_profit = [0]
        self.cummulative_balance = [self.balance]

        # Return the first observation.
        return self.get_observation()

    def render(
        self,
    ) -> None:
        "Outputs the current balance, profit and the current step."
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
        # Display Graph.
        trace1 = px.line(
            x=list(range(self.current_step + 1)),
            y=self.cummulative_profit,
            title="Cumulative Profit",
            markers=True,
        )
        trace2 = px.line(
            x=list(range(self.current_step + 1)),
            y=self.cummulative_balance,
            title="Cumulative Balance",
            markers=True,
        )

        if self.current_step == 1:
            trace1.data[0]["showlegend"] = True
            trace2.data[0]["showlegend"] = True

        trace1.data[0]["name"] = "Cumulative Profit"
        trace1.data[0]["line"] = {"color": "red", "dash": "solid"}
        trace2.data[0]["name"] = "Cumulative Balance"

        self.fig.add_trace(
            trace1.data[0],
        )
        self.fig.add_trace(trace2.data[0])

        return self.fig.show()

    def finish(self) -> bool:
        "Checks if the episode has reached an end."
        # If no more games left to bet.
        return self.current_step == self._odds.shape[0]

    def get_results(self) -> np.ndarray:
        "Returns the results matrix for the current step."
        result = np.zeros(shape=(1, self._odds.shape[1]))
        result[
            np.arange(result.shape[0], dtype=np.int32),
            np.array([self._results[self.current_step]], dtype=np.int32),
        ] = 1
        return result

    def legal_bet(
        self,
        bet: list,  # The betting matrix for the provided action.
    ) -> bool:
        "Checks that the bet does not exceed the current balance."
        bet_size = sum([b * self.balance for b in bet])
        return bet_size <= self.balance

    def create_info(
        self,
        action: int,  # The chosen action by the agent.
    ) -> dict:
        "Creates the info dictionary for the given action."
        return {
            "current_step": self.current_step,
            "odds": self.get_odds(),
            "bet_action": ACTIONS_LIST[action],
            "balance": self.balance,
            "reward": 0,
            "legal_bet": False,
            "results": None,
            "done": False,
        }