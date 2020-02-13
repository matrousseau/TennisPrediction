# -*- coding: utf-8 -*-
import scrapy
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import logging
from io import StringIO
import pandas as pd

surface = "clay"
bucket = 'pronoo'
current_year = 2018

"ACCESS AWS : https://795298734616.signin.aws.amazon.com/console"


class GetfullhistoriquecrawlerClaySpider(scrapy.Spider):
    name = 'getAllPlayersClayCrawler'

    def __init__(self, year=0, **kwargs):
        super().__init__(**kwargs)  # python3
        self.year = year

    def start_requests(self):
        start_urls = 'https://www.atptour.com/en/rankings/singles'
        yield scrapy.Request(url=start_urls, callback=self.parse)

    def parse(self, response):
        all_dates = []

        dates = [datetime.strptime(date.rstrip().lstrip().replace('.', '-'), '%Y-%m-%d').date() for date in
                 response.xpath('//*[@id="filterHolder"]/div/div/div[1]/div/ul/li/text()').extract()]

        current_year = self.year
        all_dates.append([date for date in dates if date.year == int(current_year)][0])

        for date in all_dates:
            url = "https://www.atptour.com/en/rankings/singles?rankDate={}&rankRange=0-300".format(date)
            yield scrapy.Request(url=url, callback=self.parse_player, meta={'date': date})

    def parse_player(self, response):

        date = response.meta['date']
        playersLink = response.xpath('//*[@class="player-cell"]/a/@href').extract()
        rates =  [ clean_string(rate) for rate in response.xpath('//*[@class="rank-cell"]/text()').extract() ]
        points = [clean_string(rate) for rate in response.xpath('//*[@class="points-cell"]/a/text()').extract()]
        tournCell = [clean_string(rate) for rate in response.xpath('//*[@class="tourn-cell"]/a/text()').extract()]
        pointsDropping = [clean_string(rate) for rate in response.xpath('//*[@class="pts-cell"]/text()').extract()]
        nextBest = [clean_string(rate) for rate in response.xpath('//*[@class="next-cell"]/text()').extract()]


        for i,link in enumerate(playersLink):
            url = ("https://www.atptour.com" + link).replace('overview',
                                                             'player-stats')

            yield scrapy.Request(url=url, callback=self.parse_current_year, meta={'date': date, 'rate':rates[i], 'points':points[i],
                                                                                  'tournCell': tournCell[i],'pointsDropping':pointsDropping[i],
                                                                                  'nextBest':nextBest[i]})

    def parse_current_year(self, response):
        date = response.meta['date']
        rate = response.meta['rate']
        points = response.meta['points']
        tournCell = response.meta['tournCell']
        pointsDropping = response.meta['pointsDropping']
        nextBest = response.meta['nextBest']
        year = int(date.year)
        url = response.request.url
        dates_availables = [int(clean_string(date)) for date in response.xpath(
            '//*[@id="playerMatchFactsFilter"]/div/div/div[1]/div/ul/li/text()').extract()[1:]]
        if len(dates_availables) != 0:
            selected_year = min(dates_availables, key=lambda x: abs(x - year))
            url = url + "?year={}&surfaceType={}".format(selected_year, surface)
            yield scrapy.Request(url=url, callback=self.parse_stats, meta={'date': date,  'rate':rate, 'points':points,
                                                                                  'tournCell': tournCell,'pointsDropping':pointsDropping,
                                                                                  'nextBest':nextBest})

    def parse_stats(self, response):

        date = response.meta['date']

        year = date.year
        week = '{}-{}'.format(date.day, date.month)
        aces = clean_string(response.xpath('//*[@id="playerMatchFactsContainer"]/table[1]/tbody/tr[1]/td[2]').extract())
        doubleFautes = clean_string(
            response.xpath('//*[@id="playerMatchFactsContainer"]/table[1]/tbody/tr[2]/td[2]').extract())
        firstServe = clean_string(
            response.xpath('//*[@id="playerMatchFactsContainer"]/table[1]/tbody/tr[3]/td[2]').extract())
        firstServePointsWon = clean_string(
            response.xpath('//*[@id="playerMatchFactsContainer"]/table[1]/tbody/tr[4]/td[2]').extract())
        secondServePointsWon = clean_string(
            response.xpath('//*[@id="playerMatchFactsContainer"]/table[1]/tbody/tr[5]/td[2]').extract())
        breakPointsFaced = clean_string(
            response.xpath('//*[@id="playerMatchFactsContainer"]/table[1]/tbody/tr[6]/td[2]').extract())
        breakPointsSaved = clean_string(response.xpath(
            '/html/body/div[3]/div[2]/div[1]/div/div[4]/div/div[2]/table[1]/tbody/tr[7]/td[2]').extract())
        serviceGamePlayed = clean_string(
            response.xpath('//*[@id="playerMatchFactsContainer"]/table[1]/tbody/tr[8]/td[2]').extract())
        serviceGamesWon = clean_string(
            response.xpath('//*[@id="playerMatchFactsContainer"]/table[1]/tbody/tr[9]/td[2]').extract())
        totalServicePointsWon = clean_string(
            response.xpath('//*[@id="playerMatchFactsContainer"]/table[1]/tbody/tr[10]/td[2]').extract())
        firstServeReturnPointsWon = clean_string(
            response.xpath('//*[@id="playerMatchFactsContainer"]/table[2]/tbody/tr[1]/td[2]').extract())
        secondServeReturnPointsWon = clean_string(
            response.xpath('//*[@id="playerMatchFactsContainer"]/table[2]/tbody/tr[2]/td[2]').extract())
        breakPointsOpportunities = clean_string(
            response.xpath('//*[@id="playerMatchFactsContainer"]/table[2]/tbody/tr[3]/td[2]').extract())
        breakPointsConverted = clean_string(
            response.xpath('//*[@id="playerMatchFactsContainer"]/table[2]/tbody/tr[4]/td[2]').extract())
        returnGamesPlayed = clean_string(
            response.xpath('//*[@id="playerMatchFactsContainer"]/table[2]/tbody/tr[5]/td[2]').extract())
        returnGamesWon = clean_string(
            response.xpath('//*[@id="playerMatchFactsContainer"]/table[2]/tbody/tr[6]/td[2]').extract())
        returnPointsWon = clean_string(
            response.xpath('//*[@id="playerMatchFactsContainer"]/table[2]/tbody/tr[7]/td[2]').extract())
        totalPointsWon = clean_string(
            response.xpath('//*[@id="playerMatchFactsContainer"]/table[2]/tbody/tr[8]/td[2]').extract())
        player = response.request.url.split('/')[5]
        weight = clean_string(response.xpath('//*[@class="table-weight-lbs"]/text()').extract())
        height = clean_string(response.xpath('//*[@class="table-height-cm-wrapper"]/text()').extract())
        age = clean_string(response.xpath('//*[@class="table-birthday"]/text()').extract())
        startCareer = response.xpath('//*[@class="table-big-value"]/text()').extract()[2]
        plays = response.xpath('//*[@class="table-value"]/text()').extract()[2]
        ranking = response.meta['rate']
        points = response.meta['points']
        tournCell = response.meta['tournCell']
        pointsDropping = response.meta['pointsDropping']
        nextBest = response.meta['nextBest']
        print('tennis/Players/{}/{}/{}/{}.csv'.format(year, week, surface, player))

        data = [[player, aces, doubleFautes, firstServe, firstServePointsWon, secondServePointsWon, breakPointsFaced,
                 breakPointsSaved,
                 serviceGamePlayed, serviceGamesWon, totalServicePointsWon, firstServeReturnPointsWon,
                 secondServeReturnPointsWon,
                 breakPointsOpportunities, breakPointsConverted, returnGamesPlayed, returnGamesWon, returnPointsWon,
                 totalPointsWon, response.request.url, year, weight, height, age, startCareer, plays, ranking,points,
                 tournCell, pointsDropping, nextBest]]

        labels = ["player", "aces", "doubleFautes", "firstServe", "firstServePointsWon", "secondServePointsWon",
                  "breakPointsFaced", "breakPointsSaved",
                  "serviceGamePlayed", "serviceGamesWon", "totalServicePointsWon", "firstServeReturnPointsWon",
                  "secondServeReturnPointsWon",
                  "breakPointsOpportunities", "breakPointsConverted", "returnGamesPlayed", "returnGamesWon",
                  "returnPointsWon",
                  "totalPointsWon", "url", "year", "weight", "height", "age", "startCareer", "plays", "ranking",
                  "points","tournCell", "pointsDropping", "nextBest"
                  ]

        csv_buffer = StringIO()
        pd.DataFrame(data, columns=labels).to_csv(csv_buffer)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(bucket, 'tennis/Players/{}/{}/{}/{}.csv'.format(year, week, surface, player)).put(
            Body=csv_buffer.getvalue())


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
