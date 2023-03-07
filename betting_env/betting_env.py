# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/Environment/04_betting_env.ipynb.

# %% auto 0
__all__ = ['SMALL_BET', 'MEDIUM_BET', 'LARGE_BET', 'Actions', 'STEP', 'Observation', 'BettingEnv']

# %% ../nbs/Environment/04_betting_env.ipynb 3
import json
import gym
import numexpr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime
from pathlib import Path
from typing import Dict, Tuple
from fastcore.basics import *
from .utils.data_extractor import * 
from .utils.asian_1x2_pnl import *
from collections import namedtuple

# %% ../nbs/Environment/04_betting_env.ipynb 5
# Bet size (small, medium, large) -> range[0,1].
SMALL_BET, MEDIUM_BET, LARGE_BET = 0.05, 0.2, 0.7

# Named Tuple for actions
Actions = namedtuple(
    "Actions",
    [
        "no_bets",
        "small_bet_on_home_team_1x2",
        "medium_bet_on_home_team_1x2",
        "large_bet_on_home_team_1x2",
        "small_bet_on_away_team_1x2",
        "medium_bet_on_away_team_1x2",
        "large_bet_on_away_team_1x2",
        "small_bet_on_draw_1x2",
        "medium_bet_on_draw_1x2",
        "large_bet_on_draw_1x2",
        "small_bet_on_home_team_asian_handicap",
        "medium_bet_on_home_team_asian_handicap",
        "large_bet_on_home_team_asian_handicap",
        "small_bet_on_away_team_asian_handicap",
        "medium_bet_on_away_team_asian_handicap",
        "large_bet_on_away_team_asian_handicap",
    ],
)

# %% ../nbs/Environment/04_betting_env.ipynb 11
class Observation:
    def __init__(
        self,
        game_id: int,  # Game Id.
        game_date: datetime.datetime,  # Game Date
        lineups: np.ndarray,  # Lineups(playerName:position), shape=(2,).
        lineups_ids: np.ndarray,  # Lineups opta Ids [list(11 home players Ids),list(11 away players Ids)], shape=(2,).
        lineups_slots: np.ndarray,  # Lineups slots [list(11 home positions Ids),list(11 away positions Ids)], shape=(2,).
        lineups_formation: np.ndarray,  # Lineups formations [home team formation, away team formation], shape=(2,).
        teams_names: np.ndarray,  # Team names (homeTeam name, awayteam name), shape=(2,).
        ra_teams_ids: np.ndarray,  # Teams Real-Analytics Ids [homeTeam Id, awayTeam Id], shape=(2,).
        opta_teams_ids: np.ndarray,  # Teams opta Ids [homeTeam Id, awayTeam Id], shape=(2,).
        betting_market: np.ndarray,  # Odds [[1X2 and Asian Handicap]], shape=(1,5).
        ah_line: float,  # Asian handicap line.
        shape: tuple,  # Observation shape = (30,).
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
        assert (
            len(lineups_ids[0]) == 11
        ), f"Invalid Home lineups_ids length: {len(lineups_ids[0])}. Expected 11 players."
        assert (
            len(lineups_ids[1]) == 11
        ), f"Invalid Away lineups_ids length: {len(lineups_ids[1])}. Expected 11 players."
        assert lineups_slots.shape == (
            2,
        ), f"Invalid shape for lineups_slots: {lineups_slots.shape}. Expected (2,)."
        assert (
            len(lineups_slots[0]) == 11
        ), f"Invalid Home lineups_slots length: {len(lineups_slots[0])}. Expected 11 players."
        assert (
            len(lineups_slots[1]) == 11
        ), f"Invalid Away lineups_slots length: {len(lineups_slots[1])}. Expected 11 players."

        assert lineups_formation.shape == (
            2,
        ), f"Invalid shape for lineups_formation: {lineups_formation.shape}. Expected (2,)."
        assert teams_names.shape == (
            2,
        ), f"Invalid shape for teams_names: {teams_names.shape}. Expected 2."
        assert ra_teams_ids.shape == (
            2,
        ), f"Invalid shape for ra_teams_ids: {ra_teams_ids.shape}. Expected (2,)."

        assert opta_teams_ids.shape == (
            2,
        ), f"Invalid shape for opta_teams_ids: {opta_teams_ids.shape}. Expected (2,)."
        assert betting_market.shape == (
            1,
            5,
        ), f"Invalid shape for betting_market: {betting_market.shape}. Expected (1, 5)."
        assert isinstance(
            ah_line, float
        ), f"ah_line must be a float. Got {type(ah_line)}."
        assert shape == (30,), f"Invalid observation_shape: {shape}. Expected (30,)."

        store_attr()

# %% ../nbs/Environment/04_betting_env.ipynb 13
@patch
def __call__(self: Observation) -> Observation:
    "Numpy encoder."
    self.numerical_observation = np.array(
        [self.game_id]
        + list(self.opta_teams_ids)
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

# %% ../nbs/Environment/04_betting_env.ipynb 15
@patch
def pretty(self: Observation) -> pd.DataFrame:
    "User-friendly output"
    self.observation = {
        "gameId": [self.game_id],
        "gameDate": [self.game_date],
        "homeTeam": [self.teams_names[0]],
        "awayTeam": [self.teams_names[1]],
        "homeLineup": self.lineups[0],
        "awayLineup": self.lineups[1],
        "homeFormation": [self.lineups_formation[0]],
        "awayFormation": [self.lineups_formation[1]],
        "odds1": self.betting_market[:, 0:3][0][0],
        "oddsX": self.betting_market[:, 0:3][0][1],
        "odds2": self.betting_market[:, 0:3][0][2],
        "oddsAhHome": self.betting_market[:, 3:][0][0],
        "oddsAhAway": self.betting_market[:, 3:][0][1],
        "ahLine": [self.ah_line],
    }

    return pd.DataFrame(self.observation, index=[0])

# %% ../nbs/Environment/04_betting_env.ipynb 26
class BettingEnv(gym.Env):
    """OpenAI Gym class for football betting environments."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        game_info: pd.DataFrame,  # Games with betting odds and other info.
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
        self._game = game_info.copy()

        # Sort data by date.
        if set(
            ["home_team_lineup_received_at", "away_team_lineup_received_at", "gameDate"]
        ).issubset(set(self._game.columns)):
            # Get max lineup timestamp.
            self._game["lineupReceivedAt"] = self._game[
                ["home_team_lineup_received_at", "away_team_lineup_received_at"]
            ].max(axis=1)
            # Sort.
            self._game = self._game.sort_values(
                by=["lineupReceivedAt", "gameDate"]
            ).reset_index()
            # Get gameDate date part.
            self._game["gameDate"] = pd.to_datetime(self._game["gameDate"]).dt.date
            # Shift the timestamp values by adding an offset based on each row index.
            offset = pd.Timedelta("1 second")
            self._game["lineupReceivedAt"] = (
                self._game["lineupReceivedAt"] + self._game.index.to_series() * offset
            )

        # Games ids.
        self._game_ids = self._game["game_optaId"].values

        # Odds (1X2 and Asian handicap) values.
        self._odds = self._game[odds_column_names].values

        # Ah lines.
        self._lines = self._game["LineId"].values

        # Teams names.
        self._teams_names = self._game[["homeTeamName", "awayTeamName"]].values

        # Teams RA id.
        self._ra_teams_ids = self._game[["homeTeamId", "awayTeamId"]].values

        # Teams Opta id.
        self._teams_ids = self._game[["homeTeam_optaId", "awayTeam_optaId"]].values

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
        self._results = self._game["tgt_outcome"].values

        # Game goal-difference.
        self._gd = self._game["tgt_gd"].values

        # Env balance.
        self.balance, self.starting_bank = starting_bank, starting_bank

        # Current step (game).
        self.current_step = self._game.index[0]

        # Cummulative reward.
        self.cummulative_profit = [0]

        # Cummulative balance.
        self.cummulative_balance = [self.balance]

        # Cummulative bets.
        self.bets = []

        # Actions.
        self.actions_list = Actions(
            *np.array(
                [
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
            )
        )

        # Plotly figure.
        # Init figure.
        self.fig = go.Figure()

        # Set titles.
        self.fig.update_layout(
            title="Cumulative performance over time",
            xaxis_title="Date",
            yaxis_title="Profit & Bank",
            xaxis=dict(type="category", tickangle=50, tickfont=dict(size=12)),
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
        return self.current_step % self._game.shape[0]

    def get_odds(
        self,
    ) -> np.ndarray:  # Current step (1X2 and Asian Handicap) odds, shape=(1,5).
        "Returns odds for the current step"
        return self._odds[self.current_step].reshape((1, -1))

    def get_bet(
        self,
        action: int,  # The chosen action (integer value) by the agent.
    ) -> (
        np.ndarray
    ):  # Betting choice list of 5 values (4 are 0 and 1 takes (small/medium/large bet size)).
        "Returns the betting matrix for the provided action."
        bet = np.array(self.actions_list[action])
        req_bet_size = bet.max()
        possible_bet_size = min(req_bet_size, self.balance)
        bet[np.argmax(bet)] = possible_bet_size

        return bet

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
            "bet_placed": False,
            "gd": None,
            "done": False,
        }

# %% ../nbs/Environment/04_betting_env.ipynb 28
@patch
def get_observation(
    self: BettingEnv,
) -> Observation:  # Current Observation instance.
    "Returns the observation of the current step."
    # Current game index.
    index = self._get_current_index()

    # Observation.
    return Observation(
        game_id=self._game_ids[index],
        game_date= self._game["gameDate"][index],
        lineups=self._lineups[index],
        lineups_ids=self._lineups_ids[index],
        lineups_slots=self._lineups_slots[index],
        lineups_formation=self._lineups_formations[index],
        teams_names=self._teams_names[index],
        ra_teams_ids = self._ra_teams_ids[index],
        opta_teams_ids=self._teams_ids[index],
        betting_market=self.get_odds(),
        ah_line=self._lines[index],
        shape=self.observation_space.shape,
    )()

# %% ../nbs/Environment/04_betting_env.ipynb 30
@patch
def reset(
    self: BettingEnv,
) -> Observation:  # Initial Observation instance.
    "Resets the state of the environment and returns an initial observation."

    # Reset balance to initial starting bank.
    self.balance = self.starting_bank

    # Reset initial step to 0.
    self.current_step = self._game.index[0]

    # Reset cumm profit and balance.
    self.cummulative_profit = [0]
    self.cummulative_balance = [self.balance]
    self.bets = []

    # Init figure with initial data.
    self.fig = go.Figure()
    # Set titles.
    self.fig.update_layout(
        title="Cumulative performance over time",
        xaxis_title="Date",
        yaxis_title="Profit & Bank",
        xaxis=dict(type="category", tickangle=50, tickfont=dict(size=12)),
    )

    # Hide x axis grid.
    self.fig.update_xaxes(showgrid=False)

    self.fig.add_scatter(
        x=[self.current_step], y=self.cummulative_profit, name="Profit"
    )
    self.fig.add_bar(x=[self.current_step], y=self.cummulative_balance, name="Balance")

    # Return the first observation.
    return self.get_observation()

# %% ../nbs/Environment/04_betting_env.ipynb 33
STEP = Tuple[Observation, float, bool, Dict]


@patch
def step(
    self: BettingEnv,
    action: int,  # The chosen action by the agent.
) -> STEP:  # Returns (observation, reward, done, info).
    "Run one timestep of the environment's dynamics. It accepts an action and returns a tuple (observation, reward, done, info)"

    # Init observation.
    observation = np.ones(shape=self.observation_space.shape)

    # Reward.
    reward = 0.0

    # Finish flag.
    done = False

    # Initialise info.
    info = self.create_info(action)

    # If no more money.
    if self.balance <= 0.0:
        done = True
    else:
        # Reward (positive or negative).
        _obs_gd = np.array(listify(self._gd[self.current_step]))
        _ah_line = np.array(listify(self._lines[self.current_step]))

        _reward = pnl(
            selection=self.get_bet(action).reshape((1, -1)) * self.starting_bank,
            odds=self.get_odds().reshape((1, -1)),
            obs_gd=_obs_gd,
            ah_line=_ah_line,
        ).squeeze(0)
        reward = _reward[0]

        # Update balance.
        self.balance += reward
        info.update(bet_placed=True)
        # Update info.
        info.update(gd=self._gd[self.current_step])
        info.update(reward=reward)

        # Increment step.
        _next_it = np.where(self._game.index == self.current_step)[0][0] + 1
        if _next_it < self._odds.shape[0]:
            self.current_step = self._game.index[_next_it]
            observation = self.get_observation()
            # Save the action.
            self.bets.append(
                [
                    name.replace("_", " ").capitalize()
                    for name in self.actions_list._fields
                    if (
                        getattr(self.actions_list, name) == self.actions_list[action]
                    ).all()
                ]
            )
            # save current states.
            self.cummulative_profit.append(round(reward, 2))
            self.cummulative_balance.append(round(self.balance, 2))

        else:
            done = True

    # Update flag.
    info.update(done=done)
    # Return results.
    return observation, reward, done, info

# %% ../nbs/Environment/04_betting_env.ipynb 34
@patch
def render(
    self: BettingEnv,
) -> None:
    "Updates the figure with the current step data."
    # Display Graph.
    # Get current fig data.
    scatter = self.fig.data[0]
    bar = self.fig.data[1]

    if "lineupReceivedAt" in self._game.columns:
        # Fig x-axis is lineups timestamp.
        fig_x_axis = list(
            self._game["lineupReceivedAt"][: self.current_step]
            .dt.strftime("%Y-%m-%d %H:%M:%S")
            .values
        )
        # When the bet has not yet begun, provide an empty value for the first initial step.
        fig_x_axis.insert(0, "Inital Step")
    else:
        fig_x_axis = list(range(self.current_step + 1))

    # Update X-axis (0-> current step).
    scatter.x, bar.x = fig_x_axis, fig_x_axis
    # Update Y-axis (profit and current balance).
    scatter.y = self.cummulative_profit
    bar.y = self.cummulative_balance
    # Add hover-text to the fig.
    scatter.text = self.cummulative_balance
    # We want to viz game and bet info (game date, teams, 1X2 and AH odds and the performed action).
    custom_data = np.hstack(
        (
            self._teams_names[: self.current_step],
            self._odds[: self.current_step],
            self.bets,
            self._game["gameDate"][: self.current_step].values.reshape(-1, 1),
            self._game["tgt_outcome"][: self.current_step]
            .map({0.0: "Home Win", 2.0: "Away Win", 1.0: "Draw"})
            .values.reshape(-1, 1),
        )
    )
    # Add this row to Viz Initial state before starting bet.
    initial_step_infos = np.full_like(custom_data[0], "")
    custom_data = np.concatenate([[initial_step_infos], custom_data])
    # Add this info to the figure.
    scatter.customdata = custom_data
    scatter.hovertemplate = "<br><b>Game: </b>%{customdata[0]} VS %{customdata[1]}\
        <br><b>Game Date: </b>%{customdata[8]}\
        <br><b>Game Result: </b>%{customdata[9]}\
        <br><b>1X2 Odds: </b>%{customdata[2]} %{customdata[3]} %{customdata[4]}\
        <br><b>Asian Handicap Odds: </b>%{customdata[5]} %{customdata[6]}\
        <br><b>Bet Action: </b>%{customdata[7]}\
        <br><b>Balance: </b>%{text}\
        <br><b>Profit: </b> %{y}\
        "
    # Display fig.
    self.fig.update_layout(hovermode="x")
    self.fig.show()
