# scrapy runspider crawler/tutorial/tutorial/spiders/sourcemaking_spider.py

import os

from scrapy import signals
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor


class SourceMakingSpider(CrawlSpider):
    name = "sourcemaking"
    allowed_domains = ["sourcemaking.com"]
    start_urls = ["https://sourcemaking.com/"]

    rules = (
        # https://regex101.com/r/c5EFB6/1
        Rule(
            LinkExtractor(allow=("/design_patterns/[^/]+(?<!patterns)/?$")),
            callback="parse_item",
        ),
    )

    rows = []

    # TODO: write the extracted data to a DB rather than a CSV file

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(SourceMakingSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def spider_closed(self, spider):
        with open(
            os.path.dirname(os.path.abspath(__file__)) + "/sourcemaking.csv", "w"
        ) as f:
            f.write("pattern_name,text\n")
            for line in self.rows:
                f.write(f"{line}\n")

    def parse_item(self, response):
        text_nodes = response.xpath(
            "//article/*[not(self::script or contains(@class,'banner'))]/descendant::*/text()"
        ).getall()
        text_nodes = [x.replace("\n", " ").replace(",", "") for x in text_nodes]
        full_text = response.url.split("/")[-1] + "," + " ".join(text_nodes)
        self.rows.append(full_text)
