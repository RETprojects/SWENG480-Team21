import scrapy
import mysql.connector
from bs4 import BeautifulSoup
from scrapy.item import Item, Field
import pandas as pd
import csv
import requests

# scrapy crawl example -O myfile.csv:csv
class CustomItem(Item):
    header = Field()
    paragraph = Field()
    link = Field()

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
        # page = response.url.split("/")[-1]
        # filename = f'patterns/{page}.html'
        # with open(filename, 'wb') as f:
        #     f.write(response.body)
        # self.log(f'Saved file {filename}')

        HOST = "localhost"
        # database name, if you want just to connect to MySQL server, leave it empty
        DATABASE = "pattern_recommender"
        # this is the user you create
        USER = "root"
        # user password
        PASSWORD = "THX4theF1$h!"
        # connect to MySQL server
        db_connection = mysql.connector.connect(host=HOST, database=DATABASE, user=USER, password=PASSWORD)
        print("Connected to:", db_connection.get_server_info())

        cursor = db_connection.cursor()

        file = open('temp_text.txt', 'w', encoding="utf-8")
        soup = BeautifulSoup(response.text, 'lxml')

        patternName = response.url.split("/")[-1]
        patternName = patternName.split('.')[0]
        patternName = patternName.replace('-', '_')

        overview = ""

        print(patternName)

        print("=========================================================================================")

        for s in soup.select('h2'):
            imgHit = False
            for ns in s.fetchNextSiblings():
                if ns.text == "Solution":
                    break # stop reading into the overview column when you hit the Solution section
                if ns.name == "figure":
                    imgHit = True
                if ns.name == "p":
                    imgHit = False
                    continue
                if imgHit:
                    continue
                overview += ns.text

        # insert the pieces of text collected into a new DB entry
        add_pattern = ("INSERT INTO pattern_gof "
                       "(category_id, name, overview) "
                       "VALUES (%s, %s, %s)")
        data_pattern = (0, patternName, overview)
        cursor.execute(add_pattern, data_pattern)  # insert new pattern
        db_connection.commit()  # commit the data to the DB

        # now that the data is stored in the DB, export the table as a CSV file
        # source: https://datatofish.com/export-sql-table-to-csv-python/
        query_all = pd.read_sql_query("select * from pattern_recommender.pattern_gof", db_connection)
        #
        df = pd.DataFrame(query_all)
        df.to_csv(r'scraped_pattern_data_guru.csv', index=False)