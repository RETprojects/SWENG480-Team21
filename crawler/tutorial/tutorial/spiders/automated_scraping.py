import scrapy


# scrapy crawl auto -a start_urls=[url,url2] -a range=[str1,str2]
class AutoSpider(scrapy.Spider):
    name = "auto"

    def __init__(self, **kwargs):
        if 'start_urls' in kwargs:
            self.start_urls = kwargs.pop('start_urls').split(',')
            for i in self.start_urls:
                print("Provided URL " + i)
        if 'range' in kwargs:
            self.range = kwargs.pop('range').split(',')
            for i in self.range:
                print("Provided Range " + i)
        super().__init__(**kwargs)

    #def parse(self, response, **kwargs):

