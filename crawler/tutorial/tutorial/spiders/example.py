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
    # HOST = "root"
    # # database name, if you want just to connect to MySQL server, leave it empty
    # DATABASE = "pattern_recommender"
    # # this is the user you create
    # USER = ""
    # # user password
    # PASSWORD = "THXfortheF1$h!"
    # # connect to MySQL server
    # db_connection = mysql.connector.connect(host=HOST, database=DATABASE, user=USER, password=PASSWORD)
    # print("Connected to:", db_connection.get_server_info())

    name = 'example'
    #allowed_domains = ['example.com']
    start_urls = ['https://sourcemaking.com/design_patterns/abstract_factory']


    def parse(self, response):
        file = open('temp_text.txt', 'w', encoding="utf-8")
        soup = BeautifulSoup(response.text, 'lxml')

        for s in soup.select('h3'):
            if s.text == "Example":
                break

            file.write(s.text + '\n\n')

            for ns in (s.fetchNextSiblings()):
                if ns.name == "h3":
                    file.write('\n\n')
                    break

                file.write(ns.text + '\n')
                print(ns)

