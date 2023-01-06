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




class ExampleSpider(scrapy.Spider):
    name = 'example'

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

    # Is there a way to just get all those design pattern links from one page automatically?
    # See:
    # https://thepythonscrapyplaybook.com/scrapy-pagination-guide/
    # https://refactoring.guru/design-patterns/catalog (it has all the URLs we need)
    # https://www.geeksforgeeks.org/extract-all-the-urls-from-the-webpage-using-python/
    # https://stackoverflow.com/a/5041056
    # Just get the links from the pattern-card elements on that catalog page and put them into an array that the spider can loop through.
    # url = 'https://refactoring.guru/design-patterns/catalog'
    # grab = requests.get(url)
    # soup = BeautifulSoup(grab.text, 'html.parser')
    #
    # start_urls = []
    # for link in soup.find_all("a", {"class": "pattern-card"}):
    #     data = link.get('href')
    #     start_urls.append('https://refactoring.guru' + data)


    def parse(self, response):
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
        last_heading = ""
        intent = ""
        problem = ""
        discussion = ""
        structure = ""

        for s in soup.select('h3'):
            if s.text == "Example":
                break

            last_heading = s.text

            file.write(s.text + '\n\n')

            for ns in (s.fetchNextSiblings()):
                if ns.name == "h3":
                    file.write('\n\n')
                    break

                if last_heading == "Intent":
                    # write the current text to the intent field
                    intent += ns.text
                elif last_heading == "Problem":
                    problem += ns.text
                elif last_heading == "Discussion":
                    discussion += ns.text
                elif last_heading == "Structure":
                    structure += ns.text
                else:
                    break
                file.write(ns.text + '\n')
                print(ns)

        # insert the pieces of text collected into a new DB entry
        add_pattern = ("INSERT INTO pattern_ML "
                       "(category_id, name, intent, problem, discussion, structure) "
                       "VALUES (%s, %s, %s, %s, %s, %s)")
        data_pattern = (0, patternName, intent, problem, discussion, structure)
        cursor.execute(add_pattern, data_pattern) # insert new pattern
        db_connection.commit() # commit the data to the DB

        # now that the data is stored in the DB, export the table as a CSV file
        # source: https://datatofish.com/export-sql-table-to-csv-python/
        query_all = pd.read_sql_query("select * from pattern_recommender.pattern_ML", db_connection)

        df = pd.DataFrame(query_all)
        df.to_csv(r'scraped_pattern_data.csv', index=False)

        # source: https://stackoverflow.com/questions/4613465/using-python-to-write-mysql-query-to-csv-need-to-show-field-names
        #rows = cursor.fetchall()
        #fp = open('/tmp/file.csv', 'w')
        #myFile = csv.writer(fp)
        #myFile.writerows(rows)
        #fp.close()
