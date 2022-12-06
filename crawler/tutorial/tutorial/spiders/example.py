import scrapy
import mysql.connector
from bs4 import BeautifulSoup
from scrapy.item import Item, Field


# scrapy crawl example -O myfile.csv:csv
class CustomItem(Item):
    header = Field()
    paragraph = Field()
    link = Field()




class ExampleSpider(scrapy.Spider):


    name = 'example'
    #allowed_domains = ['example.com']
    start_urls = ['https://sourcemaking.com/design_patterns/abstract_factory']


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
                else:# last_heading == "Structure":
                    structure += ns.text
                file.write(ns.text + '\n')
                print(ns)

        # insert the pieces of text collected into a new DB entry
        add_pattern = ("INSERT INTO pattern_ML "
                       "(category_id, name, intent, problem, discussion, structure) "
                       "VALUES (%s, %s, %s, %s, %s, %s)")
        data_pattern = (0, 'Abstract Factory', intent, problem, discussion, structure)
        cursor.execute(add_pattern, data_pattern) # insert new pattern
        db_connection.commit() # commit the data to the DB
