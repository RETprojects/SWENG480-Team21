from scrapy import signals
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class SourceMakingSpider(CrawlSpider):
    name = 'sourcemaking'
    allowed_domains = ['sourcemaking.com']
    start_urls = ['https://sourcemaking.com/']

    rules = (
        # https://regex101.com/r/c5EFB6/1
        Rule(LinkExtractor(allow=('/design_patterns/[^/]+(?<!patterns)/?$')), callback='parse_item'),
    )

    links = []

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(SourceMakingSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def spider_closed(self, spider):
        spider.logger.info('Spiderdsfasd closed: %s', spider.name)
        with open('your_file.txt', 'w') as f:
            for line in self.links:
                f.write(f"{line}\n")
        # spider.logger.info(self.links)

    def parse_item(self, response):
        # page = response.url.split("/")[-1]
        # filename = f'patterns/{page}.html'
        # with open(filename, 'wb') as f:
        #     f.write(response.body)
        # self.log(f'Saved file {filename}')
        self.logger.info('A response from %s just arrived!', response.url)
        self.links.append(response.url)