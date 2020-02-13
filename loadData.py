import boto3
import pandas as pd
from datetime import datetime
import re
import numpy as np
import time

s3 = boto3.client('s3')

def download_data_from_S3():
    """Get a list of all keys in an S3 bucket."""
    keysPlayers = []
    keysTournaments = []

    print("Téléchargement des matches et statistiques de 2020 S3...")

    kwargs = {'Bucket': "pronoo"}
    while True:
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            if obj['Key'].split('/')[0] == 'tennis' and obj['Key'].split('/')[1] == "Players" and int(
                    obj['Key'].split('/')[2]) == 2020:
                keysPlayers.append(obj['Key'])
            if obj['Key'].split('/')[0] == 'tennis' and obj['Key'].split('/')[1] == "Tournaments" and int(
                    obj['Key'].split('/')[2]) == 2020:
                keysTournaments.append(obj['Key'])

        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break

    all_tournaments = pd.DataFrame([], columns=['name', 'phase', 'surfaceInout', 'typeTournament', 'date', 'winnerRate',
                                                'looserRate', 'looserName', 'winnerName', 'winnerRateSurface',
                                                'looserRateSurface', 'url'])

    all_players = pd.DataFrame([], columns=['player', 'aces', 'doubleFautes', 'firstServe', 'firstServePointsWon',
                                            'secondServePointsWon', 'breakPointsFaced', 'breakPointsSaved',
                                            'serviceGamePlayed', 'serviceGamesWon', 'totalServicePointsWon',
                                            'firstServeReturnPointsWon', 'secondServeReturnPointsWon',
                                            'breakPointsOpportunities', 'breakPointsConverted', 'returnGamesPlayed',
                                            'returnGamesWon', 'returnPointsWon', 'totalPointsWon', 'url', 'year',
                                            'weight', 'height', 'age', 'startCareer', 'plays', 'ranking', 'points',
                                            'tournCell', 'pointsDropping', 'nextBest'])

    for i, key in enumerate(keysPlayers):
        if i % 5000 == 0:
            print(i)
        obj = s3.get_object(Bucket='pronoo', Key=key)
        grid_sizes = pd.read_csv(obj['Body']).drop(['Unnamed: 0'], axis=1)
        all_players = pd.concat([all_players, grid_sizes], axis=0)

    for i, key in enumerate(keysTournaments):
        if i % 5000 == 0:
            print(i)
        obj = s3.get_object(Bucket='pronoo', Key=key)
        grid_sizes = pd.read_csv(obj['Body']).drop(['Unnamed: 0'], axis=1)
        all_tournaments = pd.concat([all_tournaments, grid_sizes], axis=0)

    all_tournaments.to_csv('data/all_tournaments2020.csv', index=False)
    all_players.to_csv('data/all_players2020.csv', index=False)

    print("Téléchargement terminé.")


def getPlayStats(player, year, surface, all_players):
    if "Hard" in surface:
        surface = "hard"
    if "Clay" in surface:
        surface = "clay"
    if "Grass" in surface or "Carpet" in surface:
        surface = "grass"

    df = all_players[(all_players.year == year) & (all_players.player == player) & (all_players.surface == surface)]

    if df.empty:
        columns = ['player', 'aces', 'doubleFautes', 'firstServe', 'firstServePointsWon',
                   'secondServePointsWon', 'breakPointsFaced', 'breakPointsSaved',
                   'serviceGamePlayed', 'serviceGamesWon', 'totalServicePointsWon',
                   'firstServeReturnPointsWon', 'secondServeReturnPointsWon',
                   'breakPointsOpportunities', 'breakPointsConverted', 'returnGamesPlayed',
                   'returnGamesWon', 'returnPointsWon', 'totalPointsWon', 'url', 'year',
                   'weight', 'height', 'age', 'startCareer', 'plays', 'ranking', 'points',
                   'tournCell', 'pointsDropping', 'nextBest']
        a = np.empty(len(columns))
        a[:] = np.nan
        newDF = pd.DataFrame([a], columns=columns)
        return newDF
    else:
        return df.iloc[-1]

def p2f(x):
    return float(x.strip('%'))/100

def merge_with_local_data():

    all_tournaments = pd.read_csv('data/all_tournaments2020.csv')

    year = [match.replace(' ', '').split('-')[0].split('.')[0] for match in all_tournaments.date]
    duree_tournois = [(datetime.strptime(match.replace(' ', '').split('-')[1], '%Y.%m.%d').date() - datetime.strptime(
        match.replace(' ', '').split('-')[0], '%Y.%m.%d').date()).days for match in all_tournaments.date]
    year = [match.replace(' ', '').split('-')[0].split('.')[0] for match in all_tournaments.date]
    all_tournaments.insert(2, "year", year, True)
    all_tournaments.insert(3, "duree", duree_tournois, True)
    all_tournaments.winnerRate = all_tournaments.winnerRate.str.replace(r'[^0-9]+', '')
    all_tournaments.looserRate = all_tournaments.looserRate.str.replace(r'[^0-9]+', '')

    # Load and add surface all_players

    all_players = pd.read_csv('data/all_players2020.csv')

    surface = [url.split('=')[2] for url in all_players.url]
    all_players.insert(3, "surface", surface, True)

    label = []

    for (winner, looser) in zip(all_tournaments.winnerName, all_tournaments.looserName):
        if min(winner, looser) == winner:
            label.append("joueur0")
        else:
            label.append("joueur1")

    all_tournaments['label'] = label

    df_0 = pd.DataFrame([], columns=['player', 'aces', 'doubleFautes', 'firstServe', 'firstServePointsWon',
                                     'secondServePointsWon', 'breakPointsFaced', 'breakPointsSaved',
                                     'serviceGamePlayed', 'serviceGamesWon', 'totalServicePointsWon',
                                     'firstServeReturnPointsWon', 'secondServeReturnPointsWon',
                                     'breakPointsOpportunities', 'breakPointsConverted', 'returnGamesPlayed',
                                     'returnGamesWon', 'returnPointsWon', 'totalPointsWon', 'url', 'year',
                                     'weight', 'height', 'age', 'startCareer', 'plays', 'ranking', 'points',
                                     'tournCell', 'pointsDropping', 'nextBest'])

    df_1 = pd.DataFrame([], columns=['player', 'aces', 'doubleFautes', 'firstServe', 'firstServePointsWon',
                                     'secondServePointsWon', 'breakPointsFaced', 'breakPointsSaved',
                                     'serviceGamePlayed', 'serviceGamesWon', 'totalServicePointsWon',
                                     'firstServeReturnPointsWon', 'secondServeReturnPointsWon',
                                     'breakPointsOpportunities', 'breakPointsConverted', 'returnGamesPlayed',
                                     'returnGamesWon', 'returnPointsWon', 'totalPointsWon', 'url', 'year',
                                     'weight', 'height', 'age', 'startCareer', 'plays', 'ranking', 'points',
                                     'tournCell', 'pointsDropping', 'nextBest'])

    total_size = len(all_tournaments)
    start_time = time.time()

    for i, (winnerName, looserName, year, surface) in enumerate(
            zip(all_tournaments.winnerName, all_tournaments.looserName, all_tournaments.year,
                all_tournaments.surfaceInout)):

        if min(winnerName, looserName) == winnerName:
            df_0 = df_0.append(getPlayStats(winnerName, int(year), surface, all_players), ignore_index=True)
            df_1 = df_1.append(getPlayStats(looserName, int(year), surface, all_players), ignore_index=True)
        else:
            df_0 = df_0.append(getPlayStats(looserName, int(year), surface, all_players), ignore_index=True)
            df_1 = df_1.append(getPlayStats(winnerName, int(year), surface, all_players), ignore_index=True)

    df_1.columns = ['aces1', 'age1', 'breakPointsConverted1', 'breakPointsFaced1',
           'breakPointsOpportunities1', 'breakPointsSaved1', 'doubleFautes1',
           'firstServe1', 'firstServePointsWon1', 'firstServeReturnPointsWon1',
           'height1', 'nextBest1', 'player1', 'plays1', 'points1', 'pointsDropping1',
           'ranking1', 'returnGamesPlayed1', 'returnGamesWon1', 'returnPointsWon1',
           'secondServePointsWon1', 'secondServeReturnPointsWon1',
           'serviceGamePlayed1', 'serviceGamesWon1', 'startCareer1', 'surface1',
           'totalPointsWon1', 'totalServicePointsWon1', 'tournCell1', 'url1', 'weight1',
           'year1']

    df_0.columns = ['aces0', 'age0', 'breakPointsConverted0', 'breakPointsFaced0',
           'breakPointsOpportunities0', 'breakPointsSaved0', 'doubleFautes0',
           'firstServe0', 'firstServePointsWon0', 'firstServeReturnPointsWon0',
           'height0', 'nextBest0', 'player0', 'plays0', 'points0', 'pointsDropping0',
           'ranking0', 'returnGamesPlayed0', 'returnGamesWon0', 'returnPointsWon0',
           'secondServePointsWon0', 'secondServeReturnPointsWon0',
           'serviceGamePlayed0', 'serviceGamesWon0', 'startCareer0', 'surface0',
           'totalPointsWon0', 'totalServicePointsWon0', 'tournCell0', 'url0', 'weight0',
           'year0']



    all_tournaments_with_stats = pd.concat([all_tournaments.reset_index(), df_0.reset_index(), df_1.reset_index()], axis=1)

    all_tournaments_with_stats['breakPointsConverted0'] = all_tournaments_with_stats['breakPointsConverted0'].replace('[]',np.NaN )
    all_tournaments_with_stats['breakPointsConverted0'] = pd.to_numeric(all_tournaments_with_stats['breakPointsConverted0'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['breakPointsSaved0'] = all_tournaments_with_stats['breakPointsSaved0'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['breakPointsSaved0'] = pd.to_numeric(
        all_tournaments_with_stats['breakPointsSaved0'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['firstServe0'] = all_tournaments_with_stats['firstServe0'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['firstServe0'] = pd.to_numeric(
        all_tournaments_with_stats['firstServe0'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['firstServePointsWon0'] = all_tournaments_with_stats['firstServePointsWon0'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['firstServePointsWon0'] = pd.to_numeric(
        all_tournaments_with_stats['firstServePointsWon0'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['firstServeReturnPointsWon0'] = all_tournaments_with_stats['firstServeReturnPointsWon0'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['firstServeReturnPointsWon0'] = pd.to_numeric(
        all_tournaments_with_stats['firstServeReturnPointsWon0'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['serviceGamesWon0'] = all_tournaments_with_stats[
        'serviceGamesWon0'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['serviceGamesWon0'] = pd.to_numeric(
        all_tournaments_with_stats['serviceGamesWon0'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['totalPointsWon0'] = all_tournaments_with_stats[
        'totalPointsWon0'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['totalPointsWon0'] = pd.to_numeric(
        all_tournaments_with_stats['totalPointsWon0'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['totalServicePointsWon0'] = all_tournaments_with_stats[
        'totalServicePointsWon0'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['totalServicePointsWon0'] = pd.to_numeric(
        all_tournaments_with_stats['totalServicePointsWon0'].str.rstrip('%').astype('float') / 100.0)

    height0 = all_tournaments_with_stats.height0.str.extract('(\d+)')
    height0.columns = ['height0']
    all_tournaments_with_stats.height0 = height0


    all_tournaments_with_stats['returnGamesWon0'] = all_tournaments_with_stats[
        'returnGamesWon0'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['returnGamesWon0'] = pd.to_numeric(
        all_tournaments_with_stats['returnGamesWon0'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['returnPointsWon0'] = all_tournaments_with_stats[
        'returnPointsWon0'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['returnPointsWon0'] = pd.to_numeric(
        all_tournaments_with_stats['returnPointsWon0'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['secondServePointsWon0'] = all_tournaments_with_stats[
        'secondServePointsWon0'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['secondServePointsWon0'] = pd.to_numeric(
        all_tournaments_with_stats['secondServePointsWon0'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['secondServeReturnPointsWon0'] = all_tournaments_with_stats[
        'secondServeReturnPointsWon0'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['secondServeReturnPointsWon0'] = pd.to_numeric(
        all_tournaments_with_stats['secondServeReturnPointsWon0'].str.rstrip('%').astype('float') / 100.0)



    date = []
    for y, s in zip(all_tournaments_with_stats.year, all_tournaments_with_stats.age0):
        try:
            date.append(int(y) - datetime.strptime(s[s.find("(") + 1:s.find(")")], '%Y.%m.%d').date().year)
        except:
            date.append("NaN")

    all_tournaments_with_stats.age0 = date

    all_tournaments_with_stats['breakPointsConverted1'] = all_tournaments_with_stats['breakPointsConverted1'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['breakPointsConverted1'] = pd.to_numeric(
        all_tournaments_with_stats['breakPointsConverted1'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['breakPointsSaved1'] = all_tournaments_with_stats['breakPointsSaved1'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['breakPointsSaved1'] = pd.to_numeric(
        all_tournaments_with_stats['breakPointsSaved1'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['firstServe1'] = all_tournaments_with_stats['firstServe1'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['firstServe1'] = pd.to_numeric(
        all_tournaments_with_stats['firstServe1'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['firstServePointsWon1'] = all_tournaments_with_stats['firstServePointsWon1'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['firstServePointsWon1'] = pd.to_numeric(
        all_tournaments_with_stats['firstServePointsWon1'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['firstServeReturnPointsWon1'] = all_tournaments_with_stats[
        'firstServeReturnPointsWon1'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['firstServeReturnPointsWon1'] = pd.to_numeric(
        all_tournaments_with_stats['firstServeReturnPointsWon1'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['serviceGamesWon1'] = all_tournaments_with_stats[
        'serviceGamesWon1'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['serviceGamesWon1'] = pd.to_numeric(
        all_tournaments_with_stats['serviceGamesWon1'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['totalPointsWon1'] = all_tournaments_with_stats[
        'totalPointsWon1'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['totalPointsWon1'] = pd.to_numeric(
        all_tournaments_with_stats['totalPointsWon1'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['totalServicePointsWon1'] = all_tournaments_with_stats[
        'totalServicePointsWon1'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['totalServicePointsWon1'] = pd.to_numeric(
        all_tournaments_with_stats['totalServicePointsWon1'].str.rstrip('%').astype('float') / 100.0)

    height1 = all_tournaments_with_stats.height1.str.extract('(\d+)')
    height1.columns = ['height1']
    all_tournaments_with_stats.height1 = height1

    all_tournaments_with_stats['returnGamesWon1'] = all_tournaments_with_stats[
        'returnGamesWon1'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['returnGamesWon1'] = pd.to_numeric(
        all_tournaments_with_stats['returnGamesWon1'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['returnPointsWon1'] = all_tournaments_with_stats[
        'returnPointsWon1'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['returnPointsWon1'] = pd.to_numeric(
        all_tournaments_with_stats['returnPointsWon1'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['secondServePointsWon1'] = all_tournaments_with_stats[
        'secondServePointsWon1'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['secondServePointsWon1'] = pd.to_numeric(
        all_tournaments_with_stats['secondServePointsWon1'].str.rstrip('%').astype('float') / 100.0)

    all_tournaments_with_stats['secondServeReturnPointsWon1'] = all_tournaments_with_stats[
        'secondServeReturnPointsWon1'].replace(
        '[]', np.NaN)
    all_tournaments_with_stats['secondServeReturnPointsWon1'] = pd.to_numeric(
        all_tournaments_with_stats['secondServeReturnPointsWon1'].str.rstrip('%').astype('float') / 100.0)


    date = []
    for y, s in zip(all_tournaments_with_stats.year, all_tournaments_with_stats.age1):
        try:
            date.append(int(y) - datetime.strptime(s[s.find("(") + 1:s.find(")")], '%Y.%m.%d').date().year)
        except:
            date.append("NaN")

    all_tournaments_with_stats.age1 = date


    print(all_tournaments_with_stats)
    all_tournaments_with_stats.to_csv('data/all_tournaments_with_stats2020.csv', index=False)
