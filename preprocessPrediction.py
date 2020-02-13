import psycopg2
import requests
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import json
from datetime import datetime
import json
from pickle import load
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
import os
import re




def get_corresponding_player(player, all_players_odds):
    best_ratio = 0
    best_word = ""
    for playerodds in all_players_odds:
        if isinstance(playerodds, str):
            current_ratio = fuzz.ratio(playerodds, player)
            if current_ratio>best_ratio:
                best_ratio = current_ratio
                best_word = playerodds

    return best_word

def getInfoTournois(name):
    with open('data/tournaments.json') as json_file:
        data = json.load(json_file)
    return data[name]['surface'], data[name]['inOut'], data[name]['points']


def getCorrespondingStats(player, surface, all_players, all_players_name):
    df = all_players[
        (all_players.player == get_corresponding_player(player, all_players_name)) & (all_players.surface == surface)]
    df = df.replace('[]', np.NaN)
    df = df.sort_values(by=['year'], ascending=False)

    try:

        df = df.dropna()

        if len(df) > 0:
            df = all_players[(all_players.player == get_corresponding_player(player, all_players_name)) & (
                        all_players.surface == surface)]
            if list(df.isnull().sum(axis=1))[0] > 0:
                df = all_players[(all_players.player == get_corresponding_player(player, all_players_name)) & (
                            all_players.surface == "hard")]
        return df.iloc[[-2]]

    except:
        pass


def getPlayersConfrontation(player0, player1, surface, ranking0, ranking1, all_players_name, all_tournements):
    player0DictName = get_corresponding_player(player0, all_players_name)
    player1DictName = get_corresponding_player(player1, all_players_name)

    df0 = all_tournements[
        (all_tournements.winnerName == player0DictName) & (all_tournements.looserName == player1DictName) & (
                    all_tournements.surface == surface)]
    df1 = all_tournements[
        (all_tournements.winnerName == player1DictName) & (all_tournements.looserName == player0DictName) & (
                    all_tournements.surface == surface)]
    try:
        df = pd.concat([df0, df1], axis=0).sort_values(by=['date'], ascending=False).reset_index().iloc[0]
    except:

        ranking = getGapRank(ranking0, ranking1)
        player0ranking = ranking[0]
        player1ranking = ranking[1]

        if ranking1.values[0] < ranking0.values[0]:
            player0ranking = ranking[1]
            player1ranking = ranking[0]

        return {
            "player1": {
                "player1Rate": player1ranking,
                "player1RateSurface": player1ranking,
            },
            "player0": {
                "player0Rate": player0ranking,
                "player0RateSurface": player0ranking
            }
        }

    df = df.replace('[]', np.NaN)

    if list(df.isnull())[0] > 0:

        ranking = getGapRank(ranking0, ranking1)
        player0ranking = ranking[0]
        player1ranking = ranking[1]

        if ranking1.values[0] < ranking0.values[0]:
            player0ranking = ranking[1]
            player1ranking = ranking[0]

        return {
            "player1": {
                "player1Rate": player1ranking,
                "player1RateSurface": player1ranking,
            },
            "player0": {
                "player0Rate": player0ranking,
                "player0RateSurface": player0ranking,
            }
        }

    else:
        if df.winnerName == player0DictName:
            return {
                "player0": {
                    "player0Rate": df.winnerRate,
                    "player0RateSurface": df.winnerRateSurface,
                },
                "player1": {
                    "player1Rate": df.looserRate,
                    "player1RateSurface": df.looserRateSurface,
                }
            }
        else:
            return {
                "player1": {
                    "player1Rate": df.winnerRate,
                    "player1RateSurface": df.winnerRateSurface,
                },
                "player0": {
                    "player0Rate": df.looserRate,
                    "player0RateSurface": df.looserRateSurface,
                }
            }

    return


def getGapRank(rank0, rank1):
    beta = abs(rank0.values[0] - rank1.values[0]) / 300
    res = [(1 - beta) / 2, (1 + beta) / 2]
    return max(res), min(res)


def returnInOut(inOut):
    dummies = pd.get_dummies(['Indoor', 'Outdoor'])

    if 'Indoor' in inOut:
        return dummies.iloc[0]
    else:
        return dummies.iloc[1]


def returnPlays(play, playerIndex):
    dummies = pd.get_dummies(['Left-Handed', 'Right-Handed', 'Left-Handed.1', 'Right-Handed.1'])

    if 'left' in play.lower():
        if playerIndex == 0:
            return dummies.iloc[0][['Left-Handed', 'Right-Handed']]
        else:
            return dummies.iloc[2][['Left-Handed.1', 'Right-Handed.1']]
    else:
        if playerIndex == 0:
            return dummies.iloc[1][['Left-Handed', 'Right-Handed']]
        else:
            return dummies.iloc[3][['Left-Handed.1', 'Right-Handed.1']]


def returnSurface(surface):
    dummies = pd.get_dummies(['Carpet', 'Clay', 'Grass', 'Hard'])

    if surface.lower() == 'carpet':
        return dummies.iloc[0]
    elif surface.lower() == 'clay':
        return dummies.iloc[1]
    elif surface.lower() == 'grass':
        return dummies.iloc[2]
    else:
        return dummies.iloc[3]


def getFormeAndRankMean(player, all_tournements_forDNN):
    df = all_tournements_forDNN[(all_tournements_forDNN.player0 == player) | (all_tournements_forDNN.player1 == player)].iloc[-1]

    if (df.label == 0 and df.player0 == player):
        return df.Forme0, df.WinMean0

    else:
        return df.Forme1, df.WinMean1

def df_to_dataset(dataframe, shuffle=True, batch_size=2):
  dataframe = dataframe.drop(['label'],axis=1)
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe)))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


def upload_prediction_to_AWS(row):

    date = row['Date']
    id_1 = row['joueur1']
    id_0 = row['joueur0']

    id_match = str(re.sub("[^0-9]", "", date)) + str(id_1) + str(id_0)

    prob_0 = str(1 - row.prediction)
    prob_1 = str(row.prediction)

    if prob_0 > prob_1:
        winner = id_0
    else:
        winner = id_1

    odd_0 = str(row.Odd0)
    odd_1 = str(row.Odd1)

    id_match = str(id_0)+str(id_1)+str(date)

    try:
        connection = psycopg2.connect(
            database="pronoodb",
            user="pronoodbPostgres",
            password="pronoodb2019",
            host="pronoodb.cutflievswqr.eu-central-1.rds.amazonaws.com",
            port='5432'
        )
        cursor = connection.cursor()

        postgres_insert_query = """ INSERT INTO pronoodb.tennismatches (player_0_name,player_1_name,winner_name,probability_score_player_0, 
                                                                   probability_score_player_1, odd_player_0, odd_player_1, is_visible_to_users,
                                                                   match_date, complete_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        record_to_insert = (id_0, id_1, winner, prob_0, prob_1, odd_0, odd_1, True, date, id_match)
        cursor.execute(postgres_insert_query, record_to_insert)

        connection.commit()
        count = cursor.rowcount
        print(count, "Record inserted successfully into match table")


    except (Exception, psycopg2.Error) as error:

        if error.pgcode == "23505":

            if (connection):

                cursor.close()

                connection.close()

                try:

                    connection = psycopg2.connect(

                        database="pronoodb",

                        user="pronoodbPostgres",

                        password="pronoodb2019",

                        host="pronoodb.cutflievswqr.eu-central-1.rds.amazonaws.com",

                        port='5432'

                    )

                    cursor = connection.cursor()

                    postgres_insert_query = """Update pronoodb.tennismatches set (player_0_name,player_1_name,winner_name,probability_score_player_0, 
                                                                   probability_score_player_1, odd_player_0, odd_player_1, is_visible_to_users,
                                                                   match_date)  = (%s,%s,%s,%s,%s,%s,%s,%s,%s) where complete_id = %s"""
                    record_to_insert = (
                    id_0, id_1, winner, prob_0, prob_1, odd_0, odd_1, True, date, id_match)
                    cursor.execute(postgres_insert_query, record_to_insert)

                    connection.commit()

                    count = cursor.rowcount

                    print(count, "Record inserted successfully into match table")


                except (Exception, psycopg2.Error) as error:

                    print("Error in update operation", error)


    finally:

        # closing database connection.

        if (connection):
            cursor.close()

            connection.close()

            print("PostgreSQL connection is closed")


def buildDatasetForPrediction(all_tournements_forDNN, matchesToPredict, all_players, all_players_name, all_tournements):
    all_match_to_predict = pd.DataFrame()
    all_plays = pd.DataFrame()
    allSurface = pd.DataFrame()
    allInOut = pd.DataFrame()
    allTypeTournement = pd.DataFrame()
    matchSelected = []
    AllForme0 = []
    AllForme1 = []
    AllWinMean0 = []
    AllWinMean1 = []

    allPlayers = list(set(list(set(all_tournements_forDNN.player0)) + list(set(all_tournements_forDNN.player1))))

    for currentMatch in matchesToPredict.values():

        try:
            surface, inout, typeTournement = getInfoTournois(currentMatch['tournement'])

            if min(currentMatch['joueur0'], currentMatch['joueur1']) == currentMatch['joueur0']:

                joueur0 = currentMatch['joueur0']
                joueur1 = currentMatch['joueur1']
                odd0 = currentMatch['oddJoueur0']
                odd1 = currentMatch['oddJoueur1']

            if min(currentMatch['joueur0'], currentMatch['joueur1']) == currentMatch['joueur1']:
                joueur0 = currentMatch['joueur1']
                joueur1 = currentMatch['joueur0']
                odd0 = currentMatch['oddJoueur1']
                odd1 = currentMatch['oddJoueur0']

            print("joueur 0", joueur0)
            print("joueur 1", joueur1)
            print("Odd0", odd0)
            print("Odd1", odd1)

            joueur0Stats = getCorrespondingStats(joueur0, surface, all_players, all_players_name)

            forme0, winMean0 = getFormeAndRankMean(get_corresponding_player(joueur0, allPlayers), all_tournements_forDNN)

            joueur0Stats.columns = ['player0', 'aces0', 'doubleFautes0', 'firstServe0', 'firstServePointsWon0',
                                    'secondServePointsWon0', 'breakPointsFaced0', 'breakPointsSaved0',
                                    'serviceGamePlayed0', 'serviceGamesWon0', 'totalServicePointsWon0',
                                    'firstServeReturnPointsWon0', 'secondServeReturnPointsWon0',
                                    'breakPointsOpportunities0', 'breakPointsConverted0', 'returnGamesPlayed0',
                                    'returnGamesWon0', 'returnPointsWon0', 'totalPointsWon0', 'url0', 'year0',
                                    'weight0', 'height0', 'age0', 'startCareer0', 'plays0', 'ranking0', 'points0',
                                    'tournCell0', 'pointsDropping0', 'nextBest0', 'surface0']

            joueur1Stats = getCorrespondingStats(joueur1, surface, all_players, all_players_name)

            forme1, winMean1 = getFormeAndRankMean(get_corresponding_player(joueur1, allPlayers), all_tournements_forDNN)

            joueur1Stats.columns = ['player1', 'aces1', 'doubleFautes1', 'firstServe1', 'firstServePointsWon1',
                                    'secondServePointsWon1', 'breakPointsFaced1', 'breakPointsSaved1',
                                    'serviceGamePlayed1', 'serviceGamesWon1', 'totalServicePointsWon1',
                                    'firstServeReturnPointsWon1', 'secondServeReturnPointsWon1',
                                    'breakPointsOpportunities1', 'breakPointsConverted1', 'returnGamesPlayed1',
                                    'returnGamesWon1', 'returnPointsWon1', 'totalPointsWon1', 'url1', 'year1',
                                    'weight1', 'height1', 'age1', 'startCareer1', 'plays1', 'ranking1', 'points1',
                                    'tournCell1', 'pointsDropping1', 'nextBest1', 'surface1']

            versus = getPlayersConfrontation(joueur0, joueur1, surface, joueur0Stats.ranking0, joueur1Stats.ranking1,
                                             all_players_name, all_tournements)
            versus = pd.DataFrame([versus['player0']['player0Rate'], versus['player0']['player0RateSurface'],
                                   versus['player1']['player1Rate'], versus['player1']['player1RateSurface']]).T

            versus.columns = ['RateFace2Face0', 'RateFace2FaceSurface0', 'RateFace2Face1', 'RateFace2FaceSurface1']

            current_match = pd.concat([joueur0Stats.reset_index(), joueur1Stats.reset_index(), versus], axis=1)

            current_match['plays0'] = [str(play.split(',')[0].lstrip().rstrip()) for play in current_match.plays0]

            current_match['plays1'] = [str(play.split(',')[0].lstrip().rstrip()) for play in current_match.plays1]

            current_match['aces0'] = (int(current_match['aces0'].iloc[0])) / (
                    3.5 * int(current_match.serviceGamePlayed0.iloc[0]))
            current_match['aces1'] = (int(current_match['aces1'].iloc[0])) / (
                    3.5 * int(current_match.serviceGamePlayed1.iloc[0]))

            current_match['doubleFautes0'] = (int(current_match['doubleFautes0'].iloc[0])) / (
                    3.5 * int(current_match.serviceGamePlayed0.iloc[0]))
            current_match['doubleFautes1'] = (int(current_match['doubleFautes1'].iloc[0])) / (
                    3.5 * int(current_match.serviceGamePlayed0.iloc[0]))

            all_plays = all_plays.append(pd.concat([returnPlays(current_match.plays0.iloc[0], 0),
                                                    returnPlays(current_match.plays1.iloc[0], 1)], axis=0),
                                         ignore_index=True)

            allInOut = allInOut.append(returnInOut(inout), ignore_index=True)
            allSurface = allSurface.append(returnSurface(surface), ignore_index=True)
            AllForme0.append(forme0)
            AllForme1.append(forme1)
            AllWinMean0.append(winMean0)
            AllWinMean1.append(winMean1)

            allTypeTournement = allTypeTournement.append(pd.DataFrame([typeTournement], columns=['typeTournament']),
                                                         ignore_index=True)

            all_match_to_predict = all_match_to_predict.append(current_match)

            matchSelected.append(
                [joueur0, joueur1, currentMatch['start'], odd0, odd1])

        except Exception as e:
            print(e)

    all_match_to_predict = pd.concat([all_match_to_predict.reset_index(), all_plays.reset_index(),
                                      allInOut.reset_index(), allSurface.reset_index(),
                                      allTypeTournement.reset_index(),
                                      pd.DataFrame(AllForme0, columns=['Forme0']),
                                      pd.DataFrame(AllForme1, columns=['Forme1']),
                                      pd.DataFrame(AllWinMean0, columns=['WinMean0']),
                                      pd.DataFrame(AllWinMean1, columns=['WinMean1'])], axis=1)

    all_match_to_predict.RateFace2Face0 = pd.to_numeric(all_match_to_predict.RateFace2Face0.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.RateFace2Face1 = pd.to_numeric(all_match_to_predict.RateFace2Face1.str.replace(r'[^0-9]+', ''))

    all_match_to_predict.age0 = [2020 - datetime.strptime(age[age.find("(") + 1:age.find(")")], '%Y.%m.%d').date().year
                                 for age in all_match_to_predict.age0]
    all_match_to_predict.age1 = [2020 - datetime.strptime(age[age.find("(") + 1:age.find(")")], '%Y.%m.%d').date().year
                                 for age in all_match_to_predict.age1]

    all_match_to_predict.breakPointsConverted0 = pd.to_numeric(
        all_match_to_predict.breakPointsConverted0.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.breakPointsSaved0 = pd.to_numeric(
        all_match_to_predict.breakPointsSaved0.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.firstServe0 = pd.to_numeric(all_match_to_predict.firstServe0.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.firstServePointsWon0 = pd.to_numeric(
        all_match_to_predict.firstServePointsWon0.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.firstServeReturnPointsWon0 = pd.to_numeric(
        all_match_to_predict.firstServeReturnPointsWon0.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.serviceGamesWon0 = pd.to_numeric(
        all_match_to_predict.serviceGamesWon0.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.totalPointsWon0 = pd.to_numeric(
        all_match_to_predict.totalPointsWon0.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.totalServicePointsWon0 = pd.to_numeric(
        all_match_to_predict.totalServicePointsWon0.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.height0 = pd.to_numeric(all_match_to_predict.height0.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.returnGamesWon0 = pd.to_numeric(
        all_match_to_predict.returnGamesWon0.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.returnPointsWon0 = pd.to_numeric(
        all_match_to_predict.returnPointsWon0.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.secondServePointsWon0 = pd.to_numeric(
        all_match_to_predict.secondServePointsWon0.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.secondServeReturnPointsWon0 = pd.to_numeric(
        all_match_to_predict.secondServeReturnPointsWon0.str.replace(r'[^0-9]+', ''))

    all_match_to_predict.breakPointsConverted1 = pd.to_numeric(
        all_match_to_predict.breakPointsConverted1.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.breakPointsSaved1 = pd.to_numeric(
        all_match_to_predict.breakPointsSaved1.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.firstServe1 = pd.to_numeric(all_match_to_predict.firstServe1.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.firstServePointsWon1 = pd.to_numeric(
        all_match_to_predict.firstServePointsWon1.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.firstServeReturnPointsWon1 = pd.to_numeric(
        all_match_to_predict.firstServeReturnPointsWon1.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.serviceGamesWon1 = pd.to_numeric(
        all_match_to_predict.serviceGamesWon1.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.totalPointsWon1 = pd.to_numeric(
        all_match_to_predict.totalPointsWon1.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.totalServicePointsWon1 = pd.to_numeric(
        all_match_to_predict.totalServicePointsWon1.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.height1 = pd.to_numeric(all_match_to_predict.height1.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.returnGamesWon1 = pd.to_numeric(
        all_match_to_predict.returnGamesWon1.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.returnPointsWon1 = pd.to_numeric(
        all_match_to_predict.returnPointsWon1.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.secondServePointsWon1 = pd.to_numeric(
        all_match_to_predict.secondServePointsWon1.str.replace(r'[^0-9]+', ''))
    all_match_to_predict.secondServeReturnPointsWon1 = pd.to_numeric(
        all_match_to_predict.secondServeReturnPointsWon1.str.replace(r'[^0-9]+', ''))

    all_match_to_predict.breakPointsConverted0 = all_match_to_predict.breakPointsConverted0.replace('[]', np.NaN)
    all_match_to_predict.breakPointsFaced0 = all_match_to_predict.breakPointsFaced0.replace('[]', np.NaN)
    all_match_to_predict.breakPointsOpportunities0 = all_match_to_predict.breakPointsOpportunities0.replace('[]',
                                                                                                            np.NaN)
    all_match_to_predict.breakPointsSaved0 = all_match_to_predict.breakPointsSaved0.replace('[]', np.NaN)
    all_match_to_predict.doubleFautes0 = all_match_to_predict.doubleFautes0.replace('[]', np.NaN)
    all_match_to_predict.firstServe0 = all_match_to_predict.firstServe0.replace('[]', np.NaN)
    all_match_to_predict.firstServePointsWon0 = all_match_to_predict.firstServePointsWon0.replace('[]', np.NaN)
    all_match_to_predict.firstServeReturnPointsWon0 = all_match_to_predict.firstServeReturnPointsWon0.replace('[]',
                                                                                                              np.NaN)

    all_match_to_predict.breakPointsConverted1 = all_match_to_predict.breakPointsConverted1.replace('[]', np.NaN)
    all_match_to_predict.breakPointsFaced1 = all_match_to_predict.breakPointsFaced1.replace('[]', np.NaN)
    all_match_to_predict.breakPointsOpportunities1 = all_match_to_predict.breakPointsOpportunities1.replace('[]',
                                                                                                            np.NaN)
    all_match_to_predict.breakPointsSaved1 = all_match_to_predict.breakPointsSaved1.replace('[]', np.NaN)
    all_match_to_predict.doubleFautes1 = all_match_to_predict.doubleFautes1.replace('[]', np.NaN)
    all_match_to_predict.firstServe1 = all_match_to_predict.firstServe1.replace('[]', np.NaN)
    all_match_to_predict.firstServePointsWon1 = all_match_to_predict.firstServePointsWon1.replace('[]', np.NaN)
    all_match_to_predict.firstServeReturnPointsWon1 = all_match_to_predict.firstServeReturnPointsWon1.replace('[]',
                                                                                                              np.NaN)

    all_match_to_predict.breakPointsFaced0 = pd.to_numeric(all_match_to_predict.breakPointsFaced0)
    all_match_to_predict.breakPointsConverted0 = pd.to_numeric(all_match_to_predict.breakPointsConverted0)
    all_match_to_predict.returnGamesWon0 = pd.to_numeric(all_match_to_predict.returnGamesWon0)
    all_match_to_predict.serviceGamesWon0 = pd.to_numeric(all_match_to_predict.serviceGamesWon0)

    all_match_to_predict.breakPointsFaced1 = pd.to_numeric(all_match_to_predict.breakPointsFaced1)
    all_match_to_predict.breakPointsConverted1 = pd.to_numeric(all_match_to_predict.breakPointsConverted1)
    all_match_to_predict.returnGamesWon1 = pd.to_numeric(all_match_to_predict.returnGamesWon1)
    all_match_to_predict.serviceGamesWon1 = pd.to_numeric(all_match_to_predict.serviceGamesWon1)

    all_match_to_predict.points0 = [point.replace(",", ".") for point in all_match_to_predict.points0]
    all_match_to_predict.points1 = [point.replace(",", ".") for point in all_match_to_predict.points1]

    all_match_to_predict.pointsDropping0 = [point.replace(",", ".") for point in all_match_to_predict.pointsDropping0]
    all_match_to_predict.pointsDropping1 = [point.replace(",", ".") for point in all_match_to_predict.pointsDropping1]

    all_match_to_predict[
        'mental0'] = all_match_to_predict.breakPointsFaced0 + all_match_to_predict.breakPointsConverted0 + all_match_to_predict.returnGamesWon0 + all_match_to_predict.serviceGamesWon0
    all_match_to_predict[
        'mental1'] = all_match_to_predict.breakPointsFaced1 + all_match_to_predict.breakPointsConverted1 + all_match_to_predict.returnGamesWon1 + all_match_to_predict.serviceGamesWon1

    all_match_to_predict = all_match_to_predict.drop(['index', 'player0', 'url0', 'year0', 'weight0', 'startCareer0',
                                                      'index', 'player1', 'url1', 'year1', 'weight1', 'startCareer1'],
                                                     axis=1)

    return all_match_to_predict, matchSelected


def create_model(X_predict_scaled):
    feature_columns = []
    for header in X_predict_scaled.columns:
        if header != 'label':
            feature_columns.append(feature_column.numeric_column(header))
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(1000, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2000, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2000, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2000, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2000, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2000, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2000, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    opt = tf.keras.optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

    return model

def loadNextMatch():

    authorized_type_tournement = ['Grand Chelem', 'ATP']

    with open('data/tournaments.json') as json_file:
        data = json.load(json_file)

    "Récupération des matchs à venir..."
    authorized_tournement = list(data.keys())

    all_ids = {}

    matches = requests.get(
        'http://api.unicdn.net/v1/feeds/sportsbook/group/1000093193.json?app_id=506c35d4&app_key=9d47ace92665dd700a19194f822f858a&local=fr_CH&rangeStart=0&outComeSortBy=lexical&outComeSortDir=desc&type=2&rangeSize=100&includeparticipants=true&site=fr.unibet.com&BetOfferType=name').json()
    all_tournementsKindred = [tournement['name'] for tournement in matches['group']['groups']]

    for typeOfTournement in matches['group']['groups']:
        if typeOfTournement['name'] in authorized_type_tournement:
            for tournement in typeOfTournement['groups']:
                if 'Femmes' not in tournement['name'] and 'Doubles' not in tournement['name']:
                    all_ids[tournement['name']] = tournement['id']

    id_events = {}
    for tournement, _id in all_ids.items():
        detailsTournement = requests.get('http://api.unicdn.net/v1/feeds/sportsbookv2/event/group/' + str(
            _id) + '.json?app_id=506c35d4&app_key=9d47ace92665dd700a19194f822f858a&includeparticipants=false').json()
        for events in detailsTournement['events']:
            id_events[events['id']] = tournement

    matchesToPredict = {}

    for id_event, tournement in id_events.items():

        currentRequestEvent = requests.get('http://api.unicdn.net/v1/feeds/sportsbookv2/betoffer/event/' + str(
            id_event) + '.json?app_id=506c35d4&app_key=9d47ace92665dd700a19194f822f858a&includeparticipants=false&outComeSortBy=lexical&outComeSortDir=desc').json()
        betOffers = currentRequestEvent['betOffers']
        events = currentRequestEvent['events']
        matchs = {}

        for event in events:
            matchs[event['id']] = event["englishName"]

        ids = list(matchs.keys())

        for bet in betOffers:
            if bet['eventId'] in ids:
                if 'Match Odds' in bet['criterion']['label']:

                    joueur0 = matchs[bet['eventId']].split('-')[0].rstrip().lstrip().lower().replace(',', '').split(' ')
                    joueur1 = matchs[bet['eventId']].split('-')[1].rstrip().lstrip().lower().replace(',', '').split(' ')

                    try:
                        matchesToPredict[joueur0[0] + joueur1[0]] = {
                            'tournement': tournement,
                            'start': events[0]['start'],
                            'joueur0': joueur0[1] + " " + joueur0[0],
                            'joueur1': joueur1[1] + " " + joueur1[0],
                            'oddJoueur0': 1 + eval(bet['outcomes'][0]["oddsFractional"]),
                            'oddJoueur1': 1 + eval(bet['outcomes'][1]["oddsFractional"])}
                    except:
                        pass
    print("Match récupérés")
    print(matchesToPredict)


    all_players = pd.read_csv('data/all_players.csv')
    all_tournements = pd.read_csv('data/all_tournaments.csv')
    df2020 = pd.read_csv('data/all_tournaments_with_stats_for_DNN2020.csv').dropna()
    all_tournaments_with_stats_for_DNN = pd.read_csv('data/all_tournaments_with_stats_for_DNN.csv').dropna()

    df2020.label = df2020.label.replace('joueur0', 0)
    df2020.label = df2020.label.replace('joueur1', 1)

    df2020.pointsDropping0 = df2020.pointsDropping0.apply(lambda x: x.replace(',', '.'))
    df2020.pointsDropping1 = df2020.pointsDropping1.apply(lambda x: x.replace(',', '.'))

    all_tournaments_with_stats_for_DNN.RateFace2Face0 = all_tournaments_with_stats_for_DNN.RateFace2Face0 / 100
    all_tournaments_with_stats_for_DNN.RateFace2Face1 = all_tournaments_with_stats_for_DNN.RateFace2Face1 / 100

    all_tournaments_with_stats_for_DNN = pd.concat([all_tournaments_with_stats_for_DNN, df2020], axis=0)

    all_tournements['surface'] = [surface.split(' ')[1].lower() for surface in all_tournements.surfaceInout]
    all_tournements['date'] = [(datetime.strptime(date.split('-')[0].lstrip().rstrip(), '%Y.%m.%d').date()) for date in
                               all_tournements.date]
    all_tournements = all_tournements[(all_tournements['date'] > datetime.strptime('2017.01.01', '%Y.%m.%d').date())]

    all_players['surface'] = [url.split('=')[2] for url in all_players.url]

    all_players_name = list(all_players.player)

    all_match_to_predict = pd.DataFrame([])

    all_match_to_predict_clean, matchSelected = buildDatasetForPrediction(all_tournaments_with_stats_for_DNN, matchesToPredict,
                                                                          all_players, all_players_name, all_tournements)


    all_match_to_predict_clean = pd.concat([pd.DataFrame(np.zeros(len(all_match_to_predict_clean)), columns=['label']),
                                            all_match_to_predict_clean.reset_index()], axis=1).drop('index', axis=1)

    all_match_to_predict_clean = all_match_to_predict_clean[
        ['typeTournament', 'label', 'aces0', 'age0', 'breakPointsConverted0',
         'breakPointsFaced0', 'breakPointsOpportunities0', 'breakPointsSaved0',
         'doubleFautes0', 'firstServe0', 'firstServePointsWon0',
         'firstServeReturnPointsWon0', 'height0', 'nextBest0', 'points0',
         'pointsDropping0', 'ranking0', 'returnGamesPlayed0', 'returnGamesWon0',
         'returnPointsWon0', 'secondServePointsWon0',
         'secondServeReturnPointsWon0', 'serviceGamePlayed0', 'serviceGamesWon0',
         'totalPointsWon0', 'totalServicePointsWon0', 'tournCell0', 'aces1',
         'age1', 'breakPointsConverted1', 'breakPointsFaced1',
         'breakPointsOpportunities1', 'breakPointsSaved1', 'doubleFautes1',
         'firstServe1', 'firstServePointsWon1', 'firstServeReturnPointsWon1',
         'height1', 'nextBest1', 'points1', 'pointsDropping1', 'ranking1',
         'returnGamesPlayed1', 'returnGamesWon1', 'returnPointsWon1',
         'secondServePointsWon1', 'secondServeReturnPointsWon1',
         'serviceGamePlayed1', 'serviceGamesWon1', 'totalPointsWon1',
         'totalServicePointsWon1', 'tournCell1', 'mental0', 'mental1', 'Indoor',
         'Outdoor', 'Carpet', 'Clay', 'Grass', 'Hard', 'Left-Handed',
         'Right-Handed', 'Left-Handed.1', 'Right-Handed.1', 'RateFace2Face0',
         'RateFace2FaceSurface0', 'RateFace2Face1', 'RateFace2FaceSurface1',
         'Forme0', 'Forme1', 'WinMean0', 'WinMean1']]

    all_match_to_predict_clean.RateFace2Face0 = all_match_to_predict_clean.RateFace2Face0.fillna(50)
    all_match_to_predict_clean.RateFace2Face1 = all_match_to_predict_clean.RateFace2Face1.fillna(50)

    f2f0_list = []
    f2f1_list = []

    for f2f0, f2f1 in zip(all_match_to_predict_clean.RateFace2Face0, all_match_to_predict_clean.RateFace2Face1):
        if f2f0 + f2f1 == 1:
            f2f0_list.append(f2f0)
            f2f1_list.append(f2f1)
        elif f2f0 + f2f1 == 100:
            f2f0_list.append(f2f0 / 100)
            f2f1_list.append(f2f1 / 100)
        else:
            f2f0_list.append(0.50)
            f2f1_list.append(0.50)

    all_match_to_predict_clean.RateFace2Face0 = f2f0_list
    all_match_to_predict_clean.RateFace2Face1 = f2f1_list

    f2f0S_list = []
    f2f1S_list = []

    for f2f0S, f2f1S in zip(all_match_to_predict_clean.RateFace2FaceSurface0,
                            all_match_to_predict_clean.RateFace2FaceSurface1):
        if f2f0S + f2f1S == 1:
            f2f0S_list.append(f2f0S)
            f2f1S_list.append(f2f1S)
        elif f2f0S + f2f1S == 100:
            f2f0S_list.append(f2f0S / 100)
            f2f1S_list.append(f2f1S / 100)
        else:
            f2f0S_list.append(0.50)
            f2f1S_list.append(0.50)

    all_match_to_predict_clean.RateFace2FaceSurface0 = f2f0S_list
    all_match_to_predict_clean.RateFace2FaceSurface1 = f2f1S_list

    all_match_to_predict_clean = all_match_to_predict_clean[
        ['typeTournament', 'label', 'aces0', 'age0', 'breakPointsConverted0',
       'breakPointsFaced0', 'breakPointsOpportunities0', 'breakPointsSaved0',
       'doubleFautes0', 'firstServe0', 'firstServePointsWon0',
       'firstServeReturnPointsWon0', 'height0', 'nextBest0', 'points0',
       'pointsDropping0', 'ranking0', 'returnGamesPlayed0', 'returnGamesWon0',
       'returnPointsWon0', 'secondServePointsWon0',
       'secondServeReturnPointsWon0', 'serviceGamePlayed0', 'serviceGamesWon0',
       'totalPointsWon0', 'totalServicePointsWon0', 'tournCell0', 'aces1',
       'age1', 'breakPointsConverted1', 'breakPointsFaced1',
       'breakPointsOpportunities1', 'breakPointsSaved1', 'doubleFautes1',
       'firstServe1', 'firstServePointsWon1', 'firstServeReturnPointsWon1',
       'height1', 'nextBest1', 'points1', 'pointsDropping1', 'ranking1',
       'returnGamesPlayed1', 'returnGamesWon1', 'returnPointsWon1',
       'secondServePointsWon1', 'secondServeReturnPointsWon1',
       'serviceGamePlayed1', 'serviceGamesWon1', 'totalPointsWon1',
       'totalServicePointsWon1', 'tournCell1', 'mental0', 'mental1', 'Indoor',
       'Outdoor', 'Carpet', 'Clay', 'Grass', 'Hard', 'Left-Handed',
       'Right-Handed', 'Left-Handed.1', 'Right-Handed.1', 'RateFace2Face0',
       'RateFace2FaceSurface0', 'RateFace2Face1', 'RateFace2FaceSurface1',
       'Forme0', 'Forme1', 'WinMean0', 'WinMean1']]

    all_match_to_predict_clean.RateFace2Face0 = all_match_to_predict_clean.RateFace2Face0.fillna(50)
    all_match_to_predict_clean.RateFace2Face1 = all_match_to_predict_clean.RateFace2Face1.fillna(50)

    f2f0_list = []
    f2f1_list = []

    for f2f0, f2f1 in zip(all_match_to_predict_clean.RateFace2Face0, all_match_to_predict_clean.RateFace2Face1):
        if f2f0 + f2f1 == 1:
            f2f0_list.append(f2f0)
            f2f1_list.append(f2f1)
        elif f2f0 + f2f1 == 100:
            f2f0_list.append(f2f0 / 100)
            f2f1_list.append(f2f1 / 100)
        else:
            f2f0_list.append(0.50)
            f2f1_list.append(0.50)

    all_match_to_predict_clean.RateFace2Face0 = f2f0_list
    all_match_to_predict_clean.RateFace2Face1 = f2f1_list

    f2f0S_list = []
    f2f1S_list = []

    for f2f0S, f2f1S in zip(all_match_to_predict_clean.RateFace2FaceSurface0,
                            all_match_to_predict_clean.RateFace2FaceSurface1):
        if f2f0S + f2f1S == 1:
            f2f0S_list.append(f2f0S)
            f2f1S_list.append(f2f1S)
        elif f2f0S + f2f1S == 100:
            f2f0S_list.append(f2f0S / 100)
            f2f1S_list.append(f2f1S / 100)
        else:
            f2f0S_list.append(0.50)
            f2f1S_list.append(0.50)

    all_match_to_predict_clean.RateFace2FaceSurface0 = f2f0S_list
    all_match_to_predict_clean.RateFace2FaceSurface1 = f2f1S_list

    scaler = load(open('data/scaler.pkl', 'rb'))
    # transform the test dataset
    X_predict_scaled = pd.DataFrame(scaler.transform(all_match_to_predict_clean), columns=all_match_to_predict_clean.columns)

    X_predict_scaled.RateFace2Face0 = X_predict_scaled.RateFace2Face0 / 100
    X_predict_scaled.RateFace2Face1 = X_predict_scaled.RateFace2Face1 / 100

    batch_size = 2  # A small batch sized is used for demonstration purposes
    predict_ds = df_to_dataset(X_predict_scaled, batch_size=batch_size)

    model = create_model(X_predict_scaled)

    model.load_weights('./checkpoint/my_checkpoint')

    prediction = model.predict(predict_ds)

    prediction_with_names = pd.concat(
        [pd.DataFrame(matchSelected, columns=['joueur0', 'joueur1', 'Date', 'Odd0', 'Odd1']),
         pd.DataFrame(prediction, columns=['prediction'])], axis=1)

    for index, row in prediction_with_names.iterrows():
        if 1.2 < row.Odd0 and row.Odd0 < 3.5 and 1.2 < row.Odd1 and row.Odd1 < 3.5:
            if row.prediction < 0.4 or row.prediction > 0.6:
                upload_prediction_to_AWS(row)







