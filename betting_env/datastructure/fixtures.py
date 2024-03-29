# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/dataStrcuture/01_fixtures.ipynb.

# %% auto 0
__all__ = ['Fixture']

# %% ../../nbs/dataStrcuture/01_fixtures.ipynb 3
import mongoengine
import pandas as pd
from ..config.localconfig import CONFIG

# %% ../../nbs/dataStrcuture/01_fixtures.ipynb 4
class Fixture(mongoengine.Document):
    "Extract all Fixtures information"

    # game-id
    game_id = mongoengine.StringField(db_field="gameId", required=True)
    game_opta_id = mongoengine.IntField(db_field="optaGameId", required=False)

    # game-date/time
    game_date = mongoengine.DateTimeField(db_field="gameDate", required=True)

    # home team
    home_team_id = mongoengine.StringField(db_field="homeTeamId", required=True)
    home_team_opta_id = mongoengine.IntField(db_field="homeTeamOptaId", required=True)
    home_team_name = mongoengine.StringField(db_field="homeTeamName", required=True)
    home_team_lineup = mongoengine.StringField(db_field="homeTeamLineup", required=True)
    home_team_lineup_ids = mongoengine.ListField(
        db_field="homeTeamLineupIds", required=True
    )
    home_team_lineup_slots = mongoengine.ListField(
        db_field="homeTeamLineupSlots", required=True
    )
    home_team_formation = mongoengine.IntField(
        db_field="homeTeamFormation", required=True
    )

    # away team
    away_team_id = mongoengine.StringField(db_field="awayTeamId", required=True)
    away_team_opta_id = mongoengine.IntField(db_field="awayTeamOptaId", required=True)
    away_team_name = mongoengine.StringField(db_field="awayTeamName", required=True)
    away_team_lineup = mongoengine.StringField(db_field="awayTeamLineup", required=True)
    away_team_lineup_ids = mongoengine.ListField(
        db_field="awayTeamLineupIds", required=True
    )
    away_team_lineup_slots = mongoengine.ListField(
        db_field="awayTeamLineupSlots", required=True
    )
    away_team_formation = mongoengine.IntField(
        db_field="awayTeamFormation", required=True
    )

    # 1X2 odds
    pre_game_odds_1 = mongoengine.FloatField(db_field="preGameOdds1", required=True)
    pre_game_odds_x = mongoengine.FloatField(db_field="preGameOddsX", required=True)
    pre_game_odds_2 = mongoengine.FloatField(db_field="preGameOdds2", required=True)

    # Asian Handicap odds
    pre_game_ah_home = mongoengine.FloatField(db_field="preGameAhHome", required=True)
    pre_game_ah_away = mongoengine.FloatField(db_field="preGameAhAway", required=True)
    line_id = mongoengine.FloatField(db_field="lineId", required=True)

    # targets
    result = mongoengine.FloatField(db_field="result", required=True)
    post_game_gd = mongoengine.IntField(db_field="postGameGd", required=True)

    meta = {
        "db_alias": "football",
        "collection": CONFIG["connections"]["football"]["fixtures"],
    }

    @classmethod
    def get_all_fixtures(cls, limit=None):
        return cls.objects().order_by("game_date").limit(limit)
