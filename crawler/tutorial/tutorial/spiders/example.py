import scrapy
from scrapy.item import Item, Field


# scrapy crawl example -O myfile.csv:csv
class CustomItem(Item):
    header = Field()
    paragraph = Field()
    link = Field()


class ExampleSpider(scrapy.Spider):
    name = 'example'
    allowed_domains = ['example.com']
    start_urls = ['http://example.com/']

    def parse(self, response):
        text = [s for s in response.css('div ::text').getall() if s.strip()]
        return CustomItem(header=text[0], paragraph=text[1], link=text[2])
