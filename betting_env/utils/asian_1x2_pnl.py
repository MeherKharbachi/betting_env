# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/Utils/01_pnl_calculator.ipynb.

# %% auto 0
__all__ = ['pnl_1X2', 'pnl_ah', 'pnl']

# %% ../../nbs/Utils/01_pnl_calculator.ipynb 3
import mongoengine
import numexpr
import numpy as np
import pandas as pd

# %% ../../nbs/Utils/01_pnl_calculator.ipynb 13
def pnl_1X2(
    selection: np.ndarray,  # The amount invested on each selection.
    outcome: np.ndarray,  # Game result (Binary side outcome), shape=(1,5).
    odds: np.ndarray,  # odds for the current game, shape=(1,5)
) -> np.ndarray:  # 1X2 PnL
    "Returns the 1X2 PnL."
    assert selection.shape == odds.shape, "odds and selection should be same shape!"
    assert outcome.shape == odds.shape, "odds and outcome should be same shape!"
    n = selection.shape[0]

    pnl = selection[:, :3] * outcome[:, :3] * odds[:, :3] - selection[:, :3]
    return pnl.sum(axis=1).reshape((n, -1))

# %% ../../nbs/Utils/01_pnl_calculator.ipynb 18
def pnl_ah(
    selection: np.ndarray,  # The amount invested on each selection; shape n x 5; last 2 are for home/away asian handicap
    odds: np.ndarray,  # market odds in 1|X|2|A1|A2 order; shape n x 5
    obs_gd: np.ndarray,  # Game goal-difference; shape (n,)
    ah_line: np.ndarray,  # Asian line could be integer, half or quarter line; shape (n,)
) -> np.ndarray:  # Asian Handicap PnL.
    "Returns the Asian Handicap PnL"

    # check dimension
    n = selection.shape[0]
    obs_gd = obs_gd.reshape((n, -1))
    ah_line = ah_line.reshape((n, -1))
    assert selection.shape == odds.shape, "odds and selection should be same shape!"

    def _pnl_ah(obs_gd, ah_line, ah_odds):
        "provides the asian outcome given for a unit bet."
        # Team advantage.
        gd_advantage = obs_gd + ah_line

        if gd_advantage == 0:
            return 0.0
        elif gd_advantage == 0.25:
            return (ah_odds - 1) * 0.5
        elif gd_advantage == -0.25:
            return -0.5
        elif gd_advantage >= 0.5:
            return ah_odds - 1
        elif gd_advantage <= -0.5:
            return -1.0

    ah_selection = selection[:, -2:]
    ah_odds = odds[:, -2:]

    # change line sign if betting on away
    ah_idx = np.where(ah_selection > 0)
    flip_sign = np.zeros_like(ah_line)
    flip_sign[ah_idx[0], 0] = np.where(ah_idx[1] == 0, 1, -1)

    # reshape odds
    _ah_odds = np.zeros_like(ah_line)
    _ah_odds[ah_idx[0], 0] = ah_odds[ah_idx]

    # reshape selection
    _ah_sel = np.zeros_like(ah_line)
    _ah_sel[ah_idx[0], 0] = ah_selection[ah_idx]

    _pnl_ah_v = np.vectorize(_pnl_ah)
    _unit_pnl = _pnl_ah_v(
        obs_gd * flip_sign, ah_line * flip_sign, _ah_odds.reshape((n, -1))
    )
    
    return _unit_pnl * _ah_sel.reshape((n, -1))

# %% ../../nbs/Utils/01_pnl_calculator.ipynb 22
def pnl(
    selection: np.ndarray,  # The amount invested on each selection; shape n x 5; last 2 are for home/away asian handicap
    odds: np.ndarray,  # market odds in 1|X|2|A1|A2 order; shape n x 5
    obs_gd: np.ndarray,  # Game goal-difference; shape (n,)
    ah_line: np.ndarray,  # Asian line could be integer, half or quarter line; shape (n,)
) -> np.ndarray:  # Asian Handicap PnL.
    "Returns the total PnL"
    # check dimesnion
    n = selection.shape[0]
    assert selection.shape == odds.shape, "odds and selection should be same shape!"

    if len(obs_gd.shape) > 1 and obs_gd.shape[1] == 1:
        obs_gd = obs_gd.squeeze(1)
    binary_result = np.zeros_like(selection)
    binary_result[obs_gd > 0, 0] = 1
    binary_result[obs_gd == 0, 1] = 1
    binary_result[obs_gd < 0, 2] = 1

    # 1x2 pnl
    _pnl_1x2 = pnl_1X2(selection, binary_result, odds)

    # ah-line
    _pnl_ah = pnl_ah(selection, odds, obs_gd, ah_line)
    
    _total_pnl = _pnl_1x2 + _pnl_ah

    return _total_pnl
