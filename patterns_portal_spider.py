# Remi T, 11/17/22, 11:43 am CST
# This spider will be used to crawl all design patterns on the design pattern catalog page of Refactoring Guru.
import scrapy

# Source: https://docs.scrapy.org/en/latest/intro/tutorial.html

class PatternsPortalSpider(scrapy.Spider):
    name = "portal"
    
    # The starting URLs are derived from the patterns featured on the main design patterns portal.
    start_urls = [
        'https://refactoring.guru/design-patterns/creational-patterns',
    ]

    def parse(self, response):
        page = response.url[response.url.rfind("/") + 1:]
        filename = f'patterns-{page}.html'
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log(f'Saved file {filename}')

        next_page = response.css('div.next a::attr(href)').get()
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse)
