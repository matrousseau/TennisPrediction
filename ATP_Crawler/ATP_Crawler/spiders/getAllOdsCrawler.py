# -*- coding: utf-8 -*-
import scrapy
from scrapy_splash import SplashRequest
from scrapy.http import HtmlResponse
import logging

" ACCESS AWS : https://795298734616.signin.aws.amazon.com/console"

custom_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36'


class GetAllOdsCrawler(scrapy.Spider):
    name = 'getAllOdsCrawler'



    def start_requests(self):
        yield SplashRequest(
            "https://checkbestodds.com/historical-tennis-odds",
            self.parse,
            args={'wait': 3})

    def parse(self, response):

        urls = ["https://checkbestodds.com/" + url for url in response.xpath('//*[@id="content"]/div/ul/li/a/@href').extract() if "atp" in url]

        for url in urls:

            yield SplashRequest(url, self.parseTournements, args={'wait': 3})

    def parseTournements(self, response):

        html = response.body

        response2 = HtmlResponse(url="my HTML string", body=html, encoding='utf-8')

        odds = response2.xpath('//div[@id="main"]/div/article/div[2]/table/tr/td/b/text()').extract()
        players = response2.xpath('//div[@id="main"]/div/article/div[2]/table/tr/td/a/text()').extract()
        tournement = response2.xpath('//div[@id="main"]/div/article/div/span/span[2]/text()').extract()

        yield{
            "tournement":tournement,
            "odds":odds,
            "player":players
        }






