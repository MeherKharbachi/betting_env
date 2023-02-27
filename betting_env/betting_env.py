# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/Environment/03_betting_env.ipynb.

# %% auto 0
__all__ = ['SMALL_BET', 'MEDIUM_BET', 'LARGE_BET', 'Observation', 'BettingEnv']

# %% ../nbs/Environment/03_betting_env.ipynb 3
import json
import gym
import numexpr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from fastcore.basics import *
from .config.mongo import mongo_init
from .datastructure.fixtures import *
from .utils.asian_1x2_pnl import *

# %% ../nbs/Environment/03_betting_env.ipynb 5
# Bet size (small, medium, large) -> range[0,1].
SMALL_BET, MEDIUM_BET, LARGE_BET = 0.05, 0.2, 0.7

# %% ../nbs/Environment/03_betting_env.ipynb 11
class Observation:
    def __init__(
        self,
        game_id: int,  # Game Id.
        lineups: np.ndarray,  # Lineups(playerName:position), shape=(2,).
        lineups_ids: np.ndarray,  # Lineups opta Ids [list(11 home players Ids),list(11 away players Ids)], shape=(2,).
        lineups_slots: np.ndarray,  # Lineups slots [list(11 home positions Ids),list(11 away positions Ids)], shape=(2,).
        lineups_formation: np.ndarray,  # Lineups formations [home team formation, away team formation], shape=(2,).
        teams_names: np.ndarray,  # Team names (homeTeam name, awayteam name), shape=(2,).
        teams_ids: np.ndarray,  # Teams opta Ids [homeTeam Id, awayTeam Id], shape=(2,).
        betting_market: np.ndarray,  # Odds [[1X2 and Asian Handicap]], shape=(1,5).
        ah_line: float,  # Asian handicap line.
        observation_shape: tuple,  # Observation shape = (30,).
    ):
        # Checks on objects shape compatibilites.
        assert isinstance(
            game_id, np.int64
        ), f"game_id must be an integer. Got {type(game_id)}."
        assert lineups.shape == (
            2,
        ), f"Invalid shape for lineups: {lineups.shape}. Expected (2,)."
        assert lineups_ids.shape == (
            2,
        ), f"Invalid shape for lineups_ids: {lineups_ids.shape}. Expected (2,)."
        assert lineups_slots.shape == (
            2,
        ), f"Invalid shape for lineups_slots: {lineups_slots.shape}. Expected (2,)."
        assert lineups_formation.shape == (
            2,
        ), f"Invalid shape for lineups_formation: {lineups_formation.shape}. Expected (2,)."
        assert teams_names.shape == (
            2,
        ), f"Invalid shape for teams_names: {teams_names.shape}. Expected 2."
        assert teams_ids.shape == (
            2,
        ), f"Invalid shape for teams_ids: {teams_ids.shape}. Expected (2,)."
        assert betting_market.shape == (
            1,
            5,
        ), f"Invalid shape for betting_market: {betting_market.shape}. Expected (1, 5)."
        assert isinstance(
            ah_line, float
        ), f"ah_line must be a float. Got {type(ah_line)}."
        assert observation_shape == (
            30,
        ), f"Invalid observation_shape: {observation_shape}. Expected (30,)."

        store_attr()

# %% ../nbs/Environment/03_betting_env.ipynb 13
@patch
def __call__(self: Observation) -> Observation:
    "Numpy encoder."
    self.numerical_observation = np.array(
        [self.game_id]
        + list(self.teams_ids)
        + self.lineups_ids[0]
        + self.lineups_ids[1]
        + list(self.betting_market.flatten())
    )
    self.dtype = self.numerical_observation.dtype
    return self


@patch
def reshape(
    self: Observation,
    new_shape: tuple,  # new shape to transform the object in
) -> Observation:
    "Reshape observation."
    self.numerical_observation = self.numerical_observation.reshape(new_shape)
    return self


@patch
def astype(
    self: Observation,
    data_type: str,  # new type to convert to
) -> Observation:
    "Cast observation type."
    self.numerical_observation = self.numerical_observation.astype(data_type)
    return self

# %% ../nbs/Environment/03_betting_env.ipynb 15
@patch
def pretty(self: Observation) -> pd.DataFrame:
    "User-friendly output"
    self.observation = {
        "gameId": [self.game_id],
        "homeTeam": [self.teams_names[0]],
        "awayTeam": [self.teams_names[1]],
        "homeLineup": self.lineups[0],
        "awayLineup": self.lineups[1],
        "homeFormation":[self.lineups_formation[0]],
        "awayFormation":[self.lineups_formation[1]],
        "odds1": self.betting_market[:, 0:3][0][0],
        "oddsX": self.betting_market[:, 0:3][0][1],
        "odds2": self.betting_market[:, 0:3][0][2],
        "oddsAhHome": self.betting_market[:, 3:][0][0],
        "oddsAhAway": self.betting_market[:, 3:][0][1],
        "ahLine": [self.ah_line],
    }

    return pd.DataFrame(self.observation, index=[0])

# %% ../nbs/Environment/03_betting_env.ipynb 23
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
        small_bet: float = SMALL_BET,  # Small bet proportion value.
        medium_bet: float = MEDIUM_BET,  # Medium bet proportion value.
        large_bet: float = LARGE_BET,  # Large bet proportion value.
    ) -> None:
        "Initializes a new environment."

        super().__init__()
        # Games dataframe.
        self._game = game_odds.copy()
        # Sort data by date.
        if "gameDate" in self._game.columns:
            self._game["gameDate"] = pd.to_datetime(self._game["gameDate"])
            self._game = self._game.sort_values(by="gameDate").reset_index()

        # Games ids.
        self._game_ids = self._game["optaGameId"].values
        # Odds (1X2 and Asian handicap) values.
        self._odds = self._game[odds_column_names].values
        # Ah lines.
        self._lines = self._game["lineId"].values
        # Teams names.
        self._teams_names = self._game[["homeTeamName", "awayTeamName"]].values
        # Teams opta id.
        self._teams_ids = self._game[["homeTeamOptaId", "awayTeamOptaId"]].values
        # Teams lineups (names and positions).
        self._lineups = self._game[["homeTeamLineup", "awayTeamLineup"]].values
        # Teams lineups (players opta ids).
        self._lineups_ids = self._game[
            ["homeTeamLineupIds", "awayTeamLineupIds"]
        ].values
        # Teams lineups slots (players positions ids).
        self._lineups_slots = self._game[
            ["homeTeamLineupSlots", "awayTeamLineupSlots"]
        ].values
        # Teams formation.
        self._lineups_formations = self._game[
            ["homeTeamFormation", "awayTeamFormation"]
        ].values
        # Results (homewin -> 0 , draw -> 1, awaywin -> 2).
        self._results = self._game["result"].values
        # Game goal-difference.
        self._gd = self._game["postGameGd"].values
        # Env balance.
        self.balance = self.starting_bank = starting_bank
        # Current step (game).
        self.current_step = 0
        # Cummulative reward.
        self.cummulative_profit = [0]
        # Cummulative balance.
        self.cummulative_balance = [self.balance]
        # Actions.
        self.actions_list = [
            [0, 0, 0, 0, 0],  # No bets.
            [small_bet, 0, 0, 0, 0],  # Betting on home team (1x2).
            [medium_bet, 0, 0, 0, 0],  # Betting on home team (1x2).
            [large_bet, 0, 0, 0, 0],  # Betting on home team (1x2).
            [0, 0, small_bet, 0, 0],  # Betting on away team (1x2).
            [0, 0, medium_bet, 0, 0],  # Betting on away team (1x2).
            [0, 0, large_bet, 0, 0],  # Betting on away team (1x2).
            [0, small_bet, 0, 0, 0],  # Betting on draw (1x2).
            [0, medium_bet, 0, 0, 0],  # Betting on draw (1x2).
            [0, large_bet, 0, 0, 0],  # Betting on draw (1x2).
            [0, 0, 0, small_bet, 0],  # Betting on home (Asian Handicap).
            [0, 0, 0, medium_bet, 0],  # Betting on home (Asian Handicap).
            [0, 0, 0, large_bet, 0],  # Betting on home (Asian Handicap).
            [0, 0, 0, 0, small_bet],  # Betting on away (Asian Handicap).
            [0, 0, 0, 0, medium_bet],  # Betting on away (Asian Handicap).
            [0, 0, 0, 0, large_bet],  # Betting on away (Asian Handicap).
        ]

        # Plotly figure.
        # Init figure.
        self.fig = go.Figure()
        # Set titles.
        self.fig.update_layout(
            title="Cumulative Profit and Balance over time",
            xaxis_title="Step",
            yaxis_title="Cumulative Profit and Balance",
        )
        # Hide x axis grid.
        self.fig.update_xaxes(showgrid=False)
        # Init figure with initial data.
        self.fig.add_scatter(
            x=[self.current_step], y=self.cummulative_profit, name="Profit"
        )
        self.fig.add_bar(
            x=[self.current_step], y=self.cummulative_balance, name="Balance"
        )

        # Gym action space.
        self.action_space = gym.spaces.Discrete(
            len(self.actions_list)
        )  # Betting action
        # Gym observation space.
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self._odds.shape[1] + 25,
            ),  # 25 = 22(players Ids) + 2(home and away team ids) + 1(gameId).
            dtype=np.float64,
        )

    def _get_current_index(
        self,
    ) -> int:  # Current step index.
        "Returns the current index of the current game."
        return self.current_step % self._odds.shape[0]

    def get_odds(
        self,
    ) -> np.ndarray:  # Current step (1X2 and Asian Handicap) odds, shape=(1,5).
        "Returns odds for the current step"
        return pd.DataFrame([self._odds[self.current_step]]).values

    def get_bet(
        self,
        action: int,  # The chosen action (integer value) by the agent.
    ) -> list:  # Betting choice list of 5 values (4 are 0 and 1 takes (small/medium/large bet size)).
        "Returns the betting matrix for the provided action."
        return self.actions_list[action]

    def step(
        self,
        action: int,  # The chosen action by the agent.
    ) -> tuple:  # Returns (observation, reward, done, info).
        "Run one timestep of the environment's dynamics. It accepts an action and returns a tuple (observation, reward, done, info)"
        # Init observation.
        observation = np.ones(shape=self.observation_space.shape)
        # Reward.
        reward = 0
        # finish flag.
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
                reward = -numexpr.evaluate("sum(bet)")
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
        # save current states.
        self.cummulative_profit.append(round(reward, 2))
        self.cummulative_balance.append(round(self.balance, 2))
        # Return results.
        return observation, reward, done, info

    def get_observation(
        self,
    ) -> Observation:  # Current Observation instance.
        "Returns the observation of the current step."
        # Current game index.
        index = self._get_current_index()
        # Current game id.
        game_id = self._game_ids[index]
        # Current game lineups.
        lineups = self._lineups[index]
        lineups_ids = self._lineups_ids[index]
        lineups_slots = self._lineups_slots[index]
        lineups_formation = self._lineups_formations[index]
        # Teams.
        teams_names = self._teams_names[index]
        teams_ids = self._teams_ids[index]
        # 1X2 and AH odds.
        betting_market = self.get_odds()
        # Chosen line (AH line).
        ah_line = self._lines[index]

        # Observation.
        observation = Observation(
            game_id=game_id,
            lineups=lineups,
            lineups_ids=lineups_ids,
            lineups_slots=lineups_slots,
            lineups_formation=lineups_formation,
            teams_names=teams_names,
            teams_ids=teams_ids,
            betting_market=betting_market,
            ah_line=ah_line,
            observation_shape=self.observation_space.shape,
        )
        observation = observation()
        return observation

    def get_reward(
        self,
        bet: list,  # The betting matrix for the provided action.
        odds: np.ndarray,  # Odds (1X2 and Asian Handicap) for the current game.
        results: np.ndarray,  # Game result (Binary side outcome).
    ) -> float:  # Profit from the action.
        "Calculates the reward (the profit)."
        # Agent choice.
        bet_index = np.argmax(np.array(bet))
        # Balance.
        balance = self.balance
        # If the action is an Asian Handicap bet.
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
            # Calculate Asian Handicap PnL.
            profit = pnl_ah(
                bet=bet,
                balance=balance,
                obs_gd=obs_gd,
                ah_line=ah_line,
                ah_odds=ah_odds,
            )
        # Case 1X2.
        else:
            # Calculate 1X2 PnL.
            profit = pnl_1X2(
                bet=bet,
                balance=balance,
                results=results,
                odds=odds,
            )

        return profit

    def reset(
        self,
    ) -> Observation:  # Initial Observation instance.
        "Resets the state of the environment and returns an initial observation."
        # Reset balance to initial starting bank.
        self.balance = self.starting_bank
        # Reset initial step to 0.
        self.current_step = 0
        # Reset cumm profit and balance.
        self.cummulative_profit = [0]
        self.cummulative_balance = [self.balance]

        # Return the first observation.
        return self.get_observation()

    def render(
        self,
    ) -> None:
        "Updates the figure with the current step data."
        # Display Graph.
        # Get current fig data.
        scatter = self.fig.data[0]
        bar = self.fig.data[1]
        # Update X-axis (0-> current step).
        scatter.x = bar.x = list(range(self.current_step + 1))
        # Update Y-axis (profit and current balance).
        scatter.y = self.cummulative_profit
        bar.y = self.cummulative_balance
        # Add hover-text to the fig.
        scatter.text = self.cummulative_balance
        scatter.hovertemplate = (
            "<br><b>Step:</b> %{x}<br><b>Profit:</b> %{y}<br><b>Balance:</b>%{text}"
        )

    def finish(
        self,
    ) -> bool:  # Boolean output to verify the end of the episode.
        "Checks if the episode has reached an end."
        # If no more games left to bet.
        return self.current_step == self._odds.shape[0]

    def get_results(
        self,
    ) -> np.ndarray:  # Binary outcome.It Puts a 1 on the outcome side (home win, draw, away win), shape=(1,5).
        "Returns the results matrix (binary outcome) for the current step."
        # Init results to zeros array.
        result = np.zeros(shape=(1, self._odds.shape[1]))
        result[
            np.arange(result.shape[0], dtype=np.int32),
            np.array([self._results[self.current_step]], dtype=np.int32),
        ] = 1
        return result

    def legal_bet(
        self,
        bet: list,  # The betting matrix for the provided action.
    ) -> bool:  # Boolean output to verify current balance size.
        "Checks that the bet does not exceed the current balance."
        bet_size = sum([b * self.balance for b in bet])
        return bet_size <= self.balance

    def create_info(
        self,
        action: int,  # The chosen action by the agent.
    ) -> dict:  # Current step information.
        "Creates the info dictionary for the given action."
        return {
            "current_step": self.current_step,
            "odds": self.get_odds(),
            "bet_action": self.actions_list[action],
            "balance": self.balance,
            "reward": 0,
            "legal_bet": False,
            "results": None,
            "done": False,
        }
