import nba_api
from nba_api.stats import endpoints
from nba_api.stats.static import players, teams

import pandas as pd
import numpy as np
import warnings


def get_plyr_stats_dataframe(plyr_id=None, timeout_s=30):

    assert isinstance(plyr_id, int)

    # collect inputs
    ep = endpoints.PlayerCareerStats(plyr_id, timeout=timeout_s)
    df = ep.season_totals_regular_season.get_data_frame()

    return df


def get_team_roster_dataframe(
    team_id=None, season_id=None, add_season_id=True, timeout_s=30
):
    if isinstance(team_id, str):
        team_id = _get_team_id(full_name=team_id)

    if isinstance(season_id, int):
        season_id = _season_to_season_id(season=season_id)

    assert isinstance(team_id, int)
    assert isinstance(season_id, str)

    # collect inputs
    ep = endpoints.CommonTeamRoster(team_id, season=season_id, timeout=timeout_s)
    df = ep.common_team_roster.get_data_frame()

    # alter data
    if add_season_id:
        df["SEASON_ID"] = df.apply(
            lambda r: _season_to_season_id(season=int(r["SEASON"])), axis=1
        )

    return df


def get_team_plyr_stats_dataframe(team_id=None, season_id=None, timeout_s=30):

    # collect roster
    team_roster_df = get_team_roster_dataframe(
        team_id=team_id, season_id=season_id, add_season_id=True, timeout_s=timeout_s
    )

    # collect plyr stats from roster
    team_plyr_ids = team_roster_df["PLAYER_ID"].tolist()

    team_plyr_stats_df = []
    for plyr_id in team_plyr_ids:
        plyr_stats_df = get_plyr_stats_dataframe(plyr_id=plyr_id, timeout_s=timeout_s)
        team_plyr_stats_df.append(plyr_stats_df)

    team_plyr_stats_df = pd.concat(team_plyr_stats_df)

    # merge correctly
    df = team_roster_df.merge(
        team_plyr_stats_df,
        left_on=["TeamID", "SEASON_ID", "PLAYER_ID"],
        right_on=["TEAM_ID", "SEASON_ID", "PLAYER_ID"],
    )

    return df


def get_default_team_plyr_stats_dataframe():

    # get LAL team_id
    teams_dict = teams.get_teams()
    LAL = [x for x in teams_dict if x["full_name"] == "Los Angeles Lakers"][0]
    LAL_id = LAL["id"]

    # Params for defautl team-plyr stats dataframe
    TEAM_ID = LAL_id
    SEAS_ID = "2018-19"

    COLUMNS_OF_INTEREST = [
        "SEASON_ID",
        # "TEAM_ID",
        "TEAM_ABBREVIATION",
        "PLAYER_ID",
        "PLAYER",
        "NUM",
        "POSITION",
        "HEIGHT",
        "WEIGHT",
        "BIRTH_DATE",
        "PLAYER_AGE",
        # "EXP",
        "GP",
        "GS",
        "MIN",
        "FGM",
        "FGA",
        "FG_PCT",
        "FG3M",
        "FG3A",
        "FG3_PCT",
        "FTM",
        "FTA",
        "FT_PCT",
        # "OREB",
        # "DREB",
        "REB",
        "AST",
        "STL",
        "BLK",
        # "TOV",
        # "PF",
        "PTS",
    ]

    df = get_team_plyr_stats_dataframe(team_id=TEAM_ID, season_id=SEAS_ID, timeout_s=30)

    return df[COLUMNS_OF_INTEREST]


# Helpers
def _season_to_season_id(season=None):
    assert isinstance(season, int), "Season must be int"
    return "{}-{}".format(season, str(season + 1)[-2:])


def _get_team_id(full_name="Los Angeles Lakers"):
    teams_dict_list = teams.get_teams()
    team = [x for x in teams_dict_list if x["full_name"] == full_name][0]
    return team["id"]
