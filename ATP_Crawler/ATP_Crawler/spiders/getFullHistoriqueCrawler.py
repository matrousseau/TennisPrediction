# -*- coding: utf-8 -*-
import scrapy

" ACCESS AWS : https://795298734616.signin.aws.amazon.com/console"

class GetfullhistoriquecrawlerSpider(scrapy.Spider):
    name = 'getFullHistoriqueCrawler'
    allowed_domains = ['https://www.atptour.com/']
    start_urls = ['http://https://www.atptour.com//']

    def parse(self, response):
        pass
