import scrapy


class SourceMakingSpider(scrapy.Spider):
    name = "sourcemaking"

    creational_urls = [
        'https://sourcemaking.com/design_patterns/abstract_factory',
        'https://sourcemaking.com/design_patterns/builder',
        'https://sourcemaking.com/design_patterns/factory_method',
        'https://sourcemaking.com/design_patterns/object_pool',
        'https://sourcemaking.com/design_patterns/prototype',
        'https://sourcemaking.com/design_patterns/singleton'
    ]

    structural_urls = [
        'https://sourcemaking.com/design_patterns/adapter',
        'https://sourcemaking.com/design_patterns/bridge',
        'https://sourcemaking.com/design_patterns/composite',
        'https://sourcemaking.com/design_patterns/decorator',
        'https://sourcemaking.com/design_patterns/facade',
        'https://sourcemaking.com/design_patterns/flyweight',
        'https://sourcemaking.com/design_patterns/proxy',
        'https://sourcemaking.com/design_patterns/private_class_data',
        'https://sourcemaking.com/design_patterns/proxy'
    ]

    behavioral_urls = [
        'https://sourcemaking.com/design_patterns/chain_of_responsibility',
        'https://sourcemaking.com/design_patterns/command',
        'https://sourcemaking.com/design_patterns/interpreter',
        'https://sourcemaking.com/design_patterns/iterator',
        'https://sourcemaking.com/design_patterns/mediator',
        'https://sourcemaking.com/design_patterns/memento',
        'https://sourcemaking.com/design_patterns/null_object',
        'https://sourcemaking.com/design_patterns/observer',
        'https://sourcemaking.com/design_patterns/state',
        'https://sourcemaking.com/design_patterns/state',
        'https://sourcemaking.com/design_patterns/strategy',
        'https://sourcemaking.com/design_patterns/template_method',
        'https://sourcemaking.com/design_patterns/visitor'

    ]

    start_urls = creational_urls + structural_urls + behavioral_urls

    def parse(self, response):
        page = response.url.split("/")[-1]
        filename = f'patterns/{page}.html'
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log(f'Saved file {filename}')