import boto3
import pandas as pd
from datetime import datetime
import re
import numpy as np
import time
import difflib
from fuzzywuzzy import process


def count(lst):
    return sum(bool(x) for x in lst)


def getFormPlayer(player, year, df):
    counter = 0
    df = df[(df.player0 == player) | (df.player1 == player)]
    df = df[df.year <= year]
    forme = 0

    if len(df) > 3:
        try:
            df = df.iloc[:3]
            rank0 = df[((df.label == 1) & (df.player0 != player))].ranking0
            rank1 = df[((df.label == 0) & (df.player1 != player))].ranking1
            df = df[((df.label == 0) & (df.player0 == player)) | ((df.label == 1) & (df.player1 == player))]
            forme = len(df) / 3
            rank = (rank0.mean() + rank1.mean()) / (len(rank0) + len(rank1))
            return forme, rank
        except:
            return 0.5, 150

    else:
        try:
            rank0 = df[((df.label == 1) & (df.player0 != player))].ranking0
            rank1 = df[((df.label == 0) & (df.player1 != player))].ranking1
            initLenDf = len(df)
            df = df[((df.label == 0) & (df.player0 == player)) | ((df.label == 1) & (df.player1 == player))]
            forme = len(df) / initLenDf
            rank = (rank0.mean() + rank1.mean()) / (len(rank0) + len(rank1))
            return forme, rank
        except:
            return 0, 0


def getGapRank(rank0, rank1):
    beta = abs(rank0-rank1)/300
    res = [(1-beta)/2,(1+beta)/2]
    return max(res),min(res)


def preprocessingForDNN():
    # Load all_tournaments
    all_tournaments_with_stats = pd.read_csv('data/all_tournaments_with_stats2020.csv').dropna(
        subset=['player0', 'player1']).reset_index()

    # Load all_odds
    all_tournaments_with_stats.head()
    all_tournaments_with_stats_drop = all_tournaments_with_stats.drop(
        ['index', 'name', 'url', 'date', 'url0', 'url1', 'year0',
         'year1', 'surface0', 'surface1'], axis=1)
    all_tournaments_with_stats_drop['plays0'] = [play.lstrip().rstrip() for play in
                                                 all_tournaments_with_stats_drop.plays0]

    all_tournaments_with_stats_drop['plays1'] = [play.lstrip().rstrip() for play in
                                                 all_tournaments_with_stats_drop.plays1]

    all_tournaments_with_stats_drop.breakPointsConverted0 = all_tournaments_with_stats_drop.breakPointsConverted0.replace(
        '[]', np.NaN)
    all_tournaments_with_stats_drop.breakPointsFaced0 = all_tournaments_with_stats_drop.breakPointsFaced0.replace('[]',
                                                                                                                  np.NaN)
    all_tournaments_with_stats_drop.breakPointsOpportunities0 = all_tournaments_with_stats_drop.breakPointsOpportunities0.replace(
        '[]', np.NaN)
    all_tournaments_with_stats_drop.breakPointsSaved0 = all_tournaments_with_stats_drop.breakPointsSaved0.replace('[]',
                                                                                                                  np.NaN)
    all_tournaments_with_stats_drop.doubleFautes0 = all_tournaments_with_stats_drop.doubleFautes0.replace('[]', np.NaN)
    all_tournaments_with_stats_drop.firstServe0 = all_tournaments_with_stats_drop.firstServe0.replace('[]', np.NaN)
    all_tournaments_with_stats_drop.firstServePointsWon0 = all_tournaments_with_stats_drop.firstServePointsWon0.replace(
        '[]', np.NaN)
    all_tournaments_with_stats_drop.firstServeReturnPointsWon0 = all_tournaments_with_stats_drop.firstServeReturnPointsWon0.replace(
        '[]', np.NaN)
    all_tournaments_with_stats_drop.breakPointsConverted1 = all_tournaments_with_stats_drop.breakPointsConverted1.replace(
        '[]', np.NaN)
    all_tournaments_with_stats_drop.breakPointsFaced1 = all_tournaments_with_stats_drop.breakPointsFaced1.replace('[]',
                                                                                                                  np.NaN)
    all_tournaments_with_stats_drop.breakPointsOpportunities1 = all_tournaments_with_stats_drop.breakPointsOpportunities1.replace(
        '[]', np.NaN)
    all_tournaments_with_stats_drop.breakPointsSaved1 = all_tournaments_with_stats_drop.breakPointsSaved1.replace('[]',
                                                                                                                  np.NaN)
    all_tournaments_with_stats_drop.doubleFautes1 = all_tournaments_with_stats_drop.doubleFautes1.replace('[]', np.NaN)
    all_tournaments_with_stats_drop.firstServe1 = all_tournaments_with_stats_drop.firstServe1.replace('[]', np.NaN)
    all_tournaments_with_stats_drop.firstServePointsWon1 = all_tournaments_with_stats_drop.firstServePointsWon1.replace(
        '[]', np.NaN)
    all_tournaments_with_stats_drop.firstServeReturnPointsWon1 = all_tournaments_with_stats_drop.firstServeReturnPointsWon1.replace(
        '[]', np.NaN)
    all_tournaments_with_stats_drop.winnerRate = all_tournaments_with_stats_drop.winnerRate.replace('', 50)
    all_tournaments_with_stats_drop.looserRate = all_tournaments_with_stats_drop.looserRate.replace('', 50)
    all_tournaments_with_stats_drop.winnerName = all_tournaments_with_stats_drop.winnerName.replace('', np.NaN)
    all_tournaments_with_stats_drop.looserName = all_tournaments_with_stats_drop.looserName.replace('', np.NaN)
    all_tournaments_with_stats_drop = all_tournaments_with_stats_drop.dropna()

    all_tournaments_with_stats_drop.breakPointsFaced0 = pd.to_numeric(all_tournaments_with_stats_drop.breakPointsFaced0)
    all_tournaments_with_stats_drop.breakPointsConverted0 = pd.to_numeric(
        all_tournaments_with_stats_drop.breakPointsConverted0)
    all_tournaments_with_stats_drop.returnGamesWon0 = pd.to_numeric(all_tournaments_with_stats_drop.returnGamesWon0)
    all_tournaments_with_stats_drop.serviceGamesWon0 = pd.to_numeric(all_tournaments_with_stats_drop.serviceGamesWon0)

    all_tournaments_with_stats_drop.breakPointsFaced1 = pd.to_numeric(all_tournaments_with_stats_drop.breakPointsFaced1)
    all_tournaments_with_stats_drop.breakPointsConverted1 = pd.to_numeric(
        all_tournaments_with_stats_drop.breakPointsConverted1)
    all_tournaments_with_stats_drop.returnGamesWon1 = pd.to_numeric(all_tournaments_with_stats_drop.returnGamesWon1)
    all_tournaments_with_stats_drop.serviceGamesWon1 = pd.to_numeric(all_tournaments_with_stats_drop.serviceGamesWon1)

    all_tournaments_with_stats_drop[
        'mental0'] = all_tournaments_with_stats_drop.breakPointsFaced0 + all_tournaments_with_stats_drop.breakPointsConverted0 + all_tournaments_with_stats_drop.returnGamesWon0 + all_tournaments_with_stats_drop.serviceGamesWon0
    all_tournaments_with_stats_drop[
        'mental1'] = all_tournaments_with_stats_drop.breakPointsFaced1 + all_tournaments_with_stats_drop.breakPointsConverted1 + all_tournaments_with_stats_drop.returnGamesWon1 + all_tournaments_with_stats_drop.serviceGamesWon1

    all_tournaments_with_stats_for_DNN = all_tournaments_with_stats_drop.drop(['winnerName', 'looserName'], axis=1)
    all_tournaments_with_stats_for_DNN.typeTournament = all_tournaments_with_stats_for_DNN.typeTournament.replace(
        'finals', 2000)
    all_tournaments_with_stats_for_DNN.typeTournament = all_tournaments_with_stats_for_DNN.typeTournament.replace(
        'grandslam', 2000)

    all_tournaments_with_stats_for_DNN.phase = all_tournaments_with_stats_for_DNN.phase.replace('Qualification', 1)
    all_tournaments_with_stats_for_DNN.phase = all_tournaments_with_stats_for_DNN.phase.replace('Robin', 1)
    all_tournaments_with_stats_for_DNN.phase = all_tournaments_with_stats_for_DNN.phase.replace('Round Robin', 1)
    all_tournaments_with_stats_for_DNN.phase = all_tournaments_with_stats_for_DNN.phase.replace('Round of 128', 2)
    all_tournaments_with_stats_for_DNN.phase = all_tournaments_with_stats_for_DNN.phase.replace('Round of 64', 3)
    all_tournaments_with_stats_for_DNN.phase = all_tournaments_with_stats_for_DNN.phase.replace('Round of 32', 4)
    all_tournaments_with_stats_for_DNN.phase = all_tournaments_with_stats_for_DNN.phase.replace('Round of 16', 5)
    all_tournaments_with_stats_for_DNN.phase = all_tournaments_with_stats_for_DNN.phase.replace('Quarter-Finals', 6)
    all_tournaments_with_stats_for_DNN.phase = all_tournaments_with_stats_for_DNN.phase.replace('Semi-Finals', 7)
    all_tournaments_with_stats_for_DNN.phase = all_tournaments_with_stats_for_DNN.phase.replace('Finals', 8)
    all_tournaments_with_stats_for_DNN.phase = all_tournaments_with_stats_for_DNN.phase.replace('Quarterfinals', 6)
    all_tournaments_with_stats_for_DNN.phase = all_tournaments_with_stats_for_DNN.phase.replace('Semifinals', 7)
    all_tournaments_with_stats_for_DNN.phase = all_tournaments_with_stats_for_DNN.phase.replace('Final', 8)

    all_tournaments_with_stats_for_DNN['inOut'] = [surface.split(' ')[0] for surface in
                                                   all_tournaments_with_stats_for_DNN.surfaceInout]
    all_tournaments_with_stats_for_DNN['surface'] = [surface.split(' ')[1] for surface in
                                                     all_tournaments_with_stats_for_DNN.surfaceInout]

    all_tournaments_with_stats_for_DNN = pd.concat([all_tournaments_with_stats_for_DNN,
                                                    pd.get_dummies(all_tournaments_with_stats_for_DNN.inOut),
                                                    pd.get_dummies(all_tournaments_with_stats_for_DNN.surface)], axis=1)

    all_tournaments_with_stats_for_DNN = all_tournaments_with_stats_for_DNN.drop(['surfaceInout'], axis=1)

    all_tournaments_with_stats_for_DNN.plays0 = all_tournaments_with_stats_for_DNN.plays0.replace('Eric Lambert',
                                                                                                  'Right-Handed, Two-Handed Backhand')
    all_tournaments_with_stats_for_DNN.plays0 = all_tournaments_with_stats_for_DNN.plays0.replace('',
                                                                                                  'Right-Handed, Two-Handed Backhand')

    all_tournaments_with_stats_for_DNN.plays1 = all_tournaments_with_stats_for_DNN.plays1.replace('Brent Larkham',
                                                                                                  'Right-Handed, Two-Handed Backhand')
    all_tournaments_with_stats_for_DNN.plays1 = all_tournaments_with_stats_for_DNN.plays1.replace('Eric Lambert',
                                                                                                  'Right-Handed, Two-Handed Backhand')
    all_tournaments_with_stats_for_DNN.plays1 = all_tournaments_with_stats_for_DNN.plays1.replace('Nelen Brodar',
                                                                                                  'Right-Handed, Two-Handed Backhand')
    all_tournaments_with_stats_for_DNN.plays1 = all_tournaments_with_stats_for_DNN.plays1.replace('Tony Graham',
                                                                                                  'Right-Handed, Two-Handed Backhand')
    all_tournaments_with_stats_for_DNN.plays1 = all_tournaments_with_stats_for_DNN.plays1.replace('',
                                                                                                  'Right-Handed, Two-Handed Backhand')

    all_tournaments_with_stats_for_DNN.points0 = all_tournaments_with_stats_for_DNN.points0.replace(',', '.')
    all_tournaments_with_stats_for_DNN.points1 = all_tournaments_with_stats_for_DNN.points1.replace(',', '.')

    all_tournaments_with_stats_for_DNN.plays0 = [play.split(',')[0] for play in all_tournaments_with_stats_for_DNN.plays0]
    all_tournaments_with_stats_for_DNN.plays1 = [play.split(',')[0] for play in all_tournaments_with_stats_for_DNN.plays1]

    dummiesPlays0 = pd.get_dummies(all_tournaments_with_stats_for_DNN.plays0)
    dummiesPlays0.columns = ['Left-Handed', 'Right-Handed']

    dummiesPlays1 = pd.get_dummies(all_tournaments_with_stats_for_DNN.plays1)
    dummiesPlays1.columns = ['Left-Handed.1', 'Right-Handed.1']

    all_tournaments_with_stats_for_DNN = pd.concat([all_tournaments_with_stats_for_DNN,
                                                    dummiesPlays0,
                                                    dummiesPlays1],axis=1)

    all_tournaments_with_stats_for_DNN = all_tournaments_with_stats_for_DNN.drop(['plays0', 'plays1'], axis=1)

    all_tournaments_with_stats_for_DNN.points0 = [point.replace(",", ".") for point in
                                                  all_tournaments_with_stats_for_DNN.points0]
    all_tournaments_with_stats_for_DNN.points1 = [point.replace(",", ".") for point in
                                                  all_tournaments_with_stats_for_DNN.points1]

    player0_face2face = []
    player1_face2face = []

    for (label, W, L, Wsurface, Lsurface) in zip(all_tournaments_with_stats_for_DNN.label,
                                                 all_tournaments_with_stats_for_DNN.winnerRate,
                                                 all_tournaments_with_stats_for_DNN.looserRate,
                                                 all_tournaments_with_stats_for_DNN.winnerRateSurface,
                                                 all_tournaments_with_stats_for_DNN.looserRateSurface):
        if label == 0:
            player0_face2face.append([W, Wsurface])
            player1_face2face.append([L, Lsurface])
        else:
            player0_face2face.append([W, Wsurface])
            player1_face2face.append([L, Lsurface])

    all_tournaments_with_stats_for_DNN = pd.concat([all_tournaments_with_stats_for_DNN,
                                                    pd.DataFrame(player0_face2face,
                                                                 columns=['RateFace2Face0', 'RateFace2FaceSurface0']),
                                                    pd.DataFrame(player1_face2face,
                                                                 columns=['RateFace2Face1', 'RateFace2FaceSurface1'])],
                                                   axis=1)

    all_tournaments_with_stats_for_DNN.RateFace2Face0 = all_tournaments_with_stats_for_DNN.RateFace2Face0 / 100
    all_tournaments_with_stats_for_DNN.RateFace2Face1 = all_tournaments_with_stats_for_DNN.RateFace2Face1 / 100
    all_tournaments_with_stats_for_DNN.ranking0 = all_tournaments_with_stats_for_DNN.ranking0.fillna(300)
    all_tournaments_with_stats_for_DNN.ranking1 = all_tournaments_with_stats_for_DNN.ranking1.fillna(300)

    RateFace2Face0All = []
    RateFace2FaceSurface0All = []
    RateFace2Face1All = []
    RateFace2FaceSurface1All = []

    counter = 0

    for RateFace2Face0, RateFace2FaceSurface0, RateFace2Face1, RateFace2FaceSurface1, ranking0, ranking1 in zip(
            all_tournaments_with_stats_for_DNN.RateFace2Face0,
            all_tournaments_with_stats_for_DNN.RateFace2FaceSurface0,
            all_tournaments_with_stats_for_DNN.RateFace2Face1,
            all_tournaments_with_stats_for_DNN.RateFace2FaceSurface1,
            all_tournaments_with_stats_for_DNN.ranking0,
            all_tournaments_with_stats_for_DNN.RateFace2FaceSurface0):
        if np.isnan(ranking0):
            ranking0 = 300

        if np.isnan(ranking1):
            ranking1 = 300

        if np.isnan(RateFace2Face0) or np.isnan(RateFace2Face1):

            ranking = getGapRank(ranking0, ranking1)
            player0ranking = ranking[0]
            player1ranking = ranking[1]

            counter += 1

            if ranking1 < ranking0:
                player0ranking = ranking[1]
                player1ranking = ranking[0]

            RateFace2Face0All.append(player0ranking)
            RateFace2Face1All.append(player1ranking)

        else:
            RateFace2Face0All.append(RateFace2Face0)
            RateFace2Face1All.append(RateFace2Face1)

        if np.isnan(RateFace2FaceSurface0) or np.isnan(RateFace2FaceSurface1):

            ranking = getGapRank(ranking0, ranking1)
            player0ranking = ranking[0]
            player1ranking = ranking[1]

            if ranking1 < ranking0:
                player0ranking = ranking[1]
                player1ranking = ranking[0]

            RateFace2FaceSurface0All.append(player0ranking)
            RateFace2FaceSurface1All.append(player1ranking)
        else:
            RateFace2FaceSurface0All.append(RateFace2FaceSurface0)
            RateFace2FaceSurface1All.append(RateFace2FaceSurface1)

    all_tournaments_with_stats_for_DNN = all_tournaments_with_stats_for_DNN.drop(
        ['RateFace2FaceSurface0', 'RateFace2FaceSurface1',
         'RateFace2Face0', 'RateFace2Face1'], axis=1)

    all_tournaments_with_stats_for_DNN = pd.concat([all_tournaments_with_stats_for_DNN.reset_index(),
                                                    pd.DataFrame(RateFace2FaceSurface0All,
                                                                 columns=['RateFace2FaceSurface0']),
                                                    pd.DataFrame(RateFace2FaceSurface1All,
                                                                 columns=['RateFace2FaceSurface1']),
                                                    pd.DataFrame(RateFace2Face0All, columns=['RateFace2Face0']),
                                                    pd.DataFrame(RateFace2Face1All, columns=['RateFace2Face1'])],
                                                   axis=1)
    Allforme0 = []
    Allforme1 = []
    AllrankingMean0 = []
    AllrankingMean1 = []

    for player0, player1, year in zip(all_tournaments_with_stats_for_DNN.player0,
                                      all_tournaments_with_stats_for_DNN.player1,
                                      all_tournaments_with_stats_for_DNN.year):
        forme0, rank0 = getFormPlayer(player0, year, all_tournaments_with_stats_for_DNN)
        forme1, rank1 = getFormPlayer(player1, year, all_tournaments_with_stats_for_DNN)

        Allforme0.append(forme0)
        Allforme1.append(forme1)
        AllrankingMean0.append(rank0)
        AllrankingMean1.append(rank1)

    all_tournaments_with_stats_for_DNN = pd.concat([all_tournaments_with_stats_for_DNN.reset_index(drop=True),
                                                    pd.DataFrame(Allforme0, columns=['Forme0']),
                                                    pd.DataFrame(Allforme1, columns=['Forme1']),
                                                    pd.DataFrame(AllrankingMean0, columns=['WinMean0']),
                                                    pd.DataFrame(AllrankingMean1, columns=['WinMean1'])],
                                                   axis=1)

    if 'Carpet' not in all_tournaments_with_stats_for_DNN.columns:
        all_tournaments_with_stats_for_DNN = pd.concat([all_tournaments_with_stats_for_DNN.reset_index(drop=True),
                                                        pd.DataFrame(np.zeros(len(all_tournaments_with_stats_for_DNN)),
                                                                     columns=['Carpet'])],
                                                       axis=1)

    if 'Grass' not in all_tournaments_with_stats_for_DNN.columns:
        all_tournaments_with_stats_for_DNN = pd.concat([all_tournaments_with_stats_for_DNN.reset_index(drop=True),
                                                        pd.DataFrame(np.zeros(len(all_tournaments_with_stats_for_DNN)),
                                                                     columns=['Grass'])],
                                                       axis=1)

    all_historic_for_DNN = pd.read_csv('data/all_tournaments_with_stats_for_DNN.csv')
    all_tournaments_with_stats_for_DNN = all_tournaments_with_stats_for_DNN[all_historic_for_DNN.columns]
    all_tournaments_with_stats_for_DNN.ranking0 = all_tournaments_with_stats_for_DNN.ranking0.fillna(300)
    all_tournaments_with_stats_for_DNN.ranking1 = all_tournaments_with_stats_for_DNN.ranking1.fillna(300)
    all_tournaments_with_stats_for_DNN.WinMean0 = all_tournaments_with_stats_for_DNN.WinMean0.fillna(200)
    all_tournaments_with_stats_for_DNN.WinMean1 = all_tournaments_with_stats_for_DNN.WinMean1.fillna(200)
    all_tournaments_with_stats_for_DNN.to_csv('data/all_tournaments_with_stats_for_DNN2020.csv', index=False)


