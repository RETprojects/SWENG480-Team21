import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            'https://sourcemaking.com/design_patterns/creational_patterns',
            'https://sourcemaking.com/design_patterns/structural_patterns',
            'https://sourcemaking.com/design_patterns/behavioral_patterns',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-1]
        filename = f'{page}.html'
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log(f'Saved file {filename}')