# -*- coding: utf-8 -*-
import scrapy
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import logging
from io import StringIO
import pandas as pd

surface = "hard"
bucket = 'pronoo'

"ACCESS AWS : https://795298734616.signin.aws.amazon.com/console"


class GetAllTournamentSpider(scrapy.Spider):
    name = 'getAllTournamentCrawler'

    def __init__(self, year=0, **kwargs):
        super().__init__(**kwargs)  # python3
        self.year = year

    def start_requests(self):
        start_urls = 'https://www.atptour.com/en/scores/results-archive?year={}'.format(self.year)
        yield scrapy.Request(url=start_urls, callback=self.parse)

    def parse(self, response):
        urls = ["https://www.atptour.com"+clean_string(rate) for rate in response.xpath('//*[@class="tourney-details"]/a/@href').extract()]
        surface = [clean_string(player) for player in response.xpath('//*[@class="results-archive-table mega-table"]/tbody/tr/td[5]/div/div/span/text()').extract()]
        intout = [clean_string(player) for player in response.xpath('//*[@class="results-archive-table mega-table"]/tbody/tr/td[5]/div/div/text()').extract()][0::2]
        surfaceInout = [io+" "+sur for io, sur in zip(intout, surface)]

        for i, url in enumerate(urls):
            yield scrapy.Request(url=url, callback=self.parse_tournament, meta={'surfaceInout': surfaceInout[i]})

    def parse_tournament(self, response):

        name = [clean_string(player).split("/")[3] for player in response.xpath('//*[@class="tourney-result with-icons"]/td[2]/a/@href').extract()]
        date = [clean_string(rate) for rate in response.xpath('//*[@class="tourney-dates"]/text()').extract()]
        typeTournament = [clean_string(rate).split('/')[-1].split('_')[-1].split('.')[0]  for rate in response.xpath('//*[@class="tourney-badge-wrapper"]/img/@src').extract()]
        rounds = [clean_string(player) for player in response.xpath('//*[@class="day-table"]/thead/tr/th/text()').extract()]
        surfaceInout = response.meta['surfaceInout']

        links = ["https://www.atptour.com" + clean_string(player) for player in response.xpath('//*[@class="day-table-button"]/a/@href').extract()]
        links = [link for link in links if 'players' in link]
        for i, link in enumerate(links):
            phase = self.get_current_round(rounds, i)
            yield scrapy.Request(url=link, callback=self.parse_versus, meta={'phase': phase, 'surface':surface, 'typeTournament':typeTournament,
                                                                             'date':date, 'surfaceInout':surfaceInout, 'name':name})


    def parse_versus(self, response):

        phase = response.meta['phase']
        name = response.meta['name']
        typeTournament = response.meta['typeTournament']
        date = response.meta['date']
        winnerRate = [clean_string(player) for player in response.xpath('//*[@class="player-left-wins"]/div/div[2]/text()').extract()]
        looserRate = [clean_string(player) for player in response.xpath('//*[@class="player-right-wins"]/div/div[2]/text()').extract()]
        looserName = [clean_string(player).split('/')[3] for player in response.xpath('//*[@class="player-right-name"]/a/@href').extract()]
        winnerName = [clean_string(player).split('/')[3] for player in response.xpath('//*[@class="player-left-name"]/a/@href').extract()]
        surfaceBreakdown = [clean_string(player) for player in response.xpath('//*[@class="modal-event-breakdown-table"]/tbody/tr/td[3]/span/text()').extract()]
        winnerBreakdown = [clean_string(player).split('/')[3] for player in response.xpath('//*[@class="modal-event-breakdown-table"]/tbody/tr/td[5]/a/@href').extract()]
        surfaceInout = response.meta['surfaceInout']

        checkIfTrue = [surf == surfaceInout for surf in surfaceBreakdown]
        indexTrue = [player for player, isGoodSurface in zip(winnerBreakdown, checkIfTrue) if isGoodSurface]

        if len(typeTournament)==0:
            typeTournament=[250]

        if len(indexTrue)>0:
            winnerRateSurface = indexTrue.count(winnerName[0])/len(indexTrue)
            looserRateSurface = indexTrue.count(looserName[0])/len(indexTrue)
        else:
            winnerRateSurface = 'NaN'
            looserRateSurface = 'NaN'

        data = [[name[0], phase, surfaceInout, typeTournament[0], date[0], winnerRate[0],looserRate[0], looserName[0], winnerName[0],
                 winnerRateSurface, looserRateSurface, response.request.url]]

        labels = ["name", "phase", "surfaceInout", "typeTournament", "date", "winnerRate","looserRate", "looserName", "winnerName",
                 "winnerRateSurface", "looserRateSurface", "url"]

        csv_buffer = StringIO()
        pd.DataFrame(data, columns=labels).to_csv(csv_buffer)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(bucket, 'tennis/Tournaments/{}/{}/{}/{}/{}.csv'.format(self.year, typeTournament[0], surfaceInout, name[0], "_".join([winnerName[0], looserName[0]]))).put(
            Body=csv_buffer.getvalue())



    def get_current_round(self, rounds, i):

        try:

            if len(["Qualifying" in r for r in rounds])>0:
                finalrounds = rounds[:-sum(["Qualifying" in r for r in rounds])]

                if i > 2**len(finalrounds):
                    return "Qualification"
                elif i == 0:
                    return finalrounds[0]
                elif i > 0 and i < 3:
                    return finalrounds[1]
                elif i > 2 and i <7:
                    return finalrounds[2]
                elif i > 6 and i <15:
                    return finalrounds[3]
                elif i > 14 and i <31:
                    return finalrounds[3]
                elif i > 30 and i <63:
                    return finalrounds[4]
                elif i > 62 and i <127:
                    return finalrounds[5]
                elif i > 126 and i <191:
                    return finalrounds[6]

            else:
                finalrounds = rounds

                if i > 2 ** len(finalrounds):
                    return "Qualification"
                elif i == 0:
                    return finalrounds[0]
                elif i > 0 and i < 3:
                    return finalrounds[1]
                elif i > 2 and i < 7:
                    return finalrounds[2]
                elif i > 6 and i < 15:
                    return finalrounds[3]
                elif i > 14 and i < 31:
                    return finalrounds[3]
                elif i > 30 and i < 63:
                    return finalrounds[4]
                elif i > 62 and i < 127:
                    return finalrounds[5]
                elif i > 126 and i < 191:
                    return finalrounds[6]
        except:
            return "Qualification"

def clean_string(response):
    if isinstance(response, list) and len(response) > 0:
        res = response[0]
        res = res.replace('<td>', "").lstrip()
        res = res.replace('</td>', "").lstrip().rstrip()
        return res

    if isinstance(response, str):
        res = response.replace('<td>', "").lstrip()
        res = res.replace('</td>', "").lstrip().rstrip()
        return res

    else:
        return response


def upload_file(file_name, bucket, object_name):
    """Upload a file to an S3 bucket
    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then same as file_name
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True
