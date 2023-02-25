import scrapy


class RefactoringGuruSpider(scrapy.Spider):
    name = "refactoringguru"

    creational_urls = [
        'https://refactoring.guru/design-patterns/factory-method',
        'https://refactoring.guru/design-patterns/abstract-factory',
        'https://refactoring.guru/design-patterns/builder',
        'https://refactoring.guru/design-patterns/prototype',
        'https://refactoring.guru/design-patterns/singleton'
    ]

    structural_urls = [
        'https://refactoring.guru/design-patterns/adapter',
        'https://refactoring.guru/design-patterns/bridge',
        'https://refactoring.guru/design-patterns/composite',
        'https://refactoring.guru/design-patterns/decorator',
        'https://refactoring.guru/design-patterns/facade',
        'https://refactoring.guru/design-patterns/flyweight',
        'https://refactoring.guru/design-patterns/proxy'
    ]

    behavioral_urls = [
        'https://refactoring.guru/design-patterns/chain-of-responsibility',
        'https://refactoring.guru/design-patterns/command',
        'https://refactoring.guru/design-patterns/iterator',
        'https://refactoring.guru/design-patterns/mediator',
        'https://refactoring.guru/design-patterns/memento',
        'https://refactoring.guru/design-patterns/observer',
        'https://refactoring.guru/design-patterns/state',
        'https://refactoring.guru/design-patterns/strategy',
        'https://refactoring.guru/design-patterns/template-method',
        'https://refactoring.guru/design-patterns/visitor'
    ]

    start_urls = creational_urls + structural_urls + behavioral_urls

    def parse(self, response):
        page = response.url.split("/")[-1]
        filename = f'patterns/{page}.html'
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log(f'Saved file {filename}')