"""
import csv

import mysql.connector
import pandas as pd
import requests
import scrapy
from bs4 import BeautifulSoup
from scrapy.item import Field, Item


# scrapy crawl example -O myfile.csv:csv
class CustomItem(Item):
    header = Field()
    paragraph = Field()
    link = Field()


class ExampleSpider(scrapy.Spider):
    name = "gof-patterns"

    creational_urls = [
        "https://www.gofpatterns.com/creational/patterns/abstract-factory-pattern.php",
        "https://www.gofpatterns.com/creational/patterns/builder-pattern.php",
        "https://www.gofpatterns.com/creational/patterns/factory-method-pattern.php",
        "https://www.gofpatterns.com/creational/patterns/prototype-pattern.php",
        "https://www.gofpatterns.com/creational/patterns/singleton-pattern.php",
    ]

    structural_urls = [
        "https://www.gofpatterns.com/structural/patterns/adapter-pattern.php",
        "https://www.gofpatterns.com/structural/patterns/bridge-pattern.php",
        "https://www.gofpatterns.com/structural/patterns/composite-pattern.php",
        "https://www.gofpatterns.com/structural/patterns/decorator-pattern.php",
        "https://www.gofpatterns.com/structural/patterns/facade-pattern.php",
        "https://www.gofpatterns.com/structural/patterns/flyweight-pattern.php",
        "https://www.gofpatterns.com/structural/patterns/proxy-pattern.php",
    ]

    behavioral_urls = [
        "https://www.gofpatterns.com/behavioral/patterns/chain-of-responsibility.php",
        "https://www.gofpatterns.com/behavioral/patterns/command-pattern.php",
        "https://www.gofpatterns.com/behavioral/patterns/interpreter-pattern.php",
        "https://www.gofpatterns.com/behavioral/patterns/iterator-pattern.php",
        "https://www.gofpatterns.com/behavioral/patterns/mediator-pattern.php",
        "https://www.gofpatterns.com/behavioral/patterns/memento-pattern.php",
        "https://www.gofpatterns.com/behavioral/patterns/observer-pattern.php",
        "https://www.gofpatterns.com/behavioral/patterns/state-pattern.php",
        "https://www.gofpatterns.com/behavioral/patterns/strategy-pattern.php",
        "https://www.gofpatterns.com/behavioral/patterns/template-pattern.php",
        "https://www.gofpatterns.com/behavioral/patterns/visitor-pattern.php",
        "https://www.gofpatterns.com/behavioral/patterns/externalize-stack-pattern.php",
        "https://www.gofpatterns.com/behavioral/patterns/hierarchical-visitor-pattern.php",
    ]

    start_urls = creational_urls + structural_urls + behavioral_urls

    def parse(self, response):
        HOST = "localhost"
        # database name, if you want just to connect to MySQL server, leave it empty
        DATABASE = "pattern_recommender"
        # this is the user you create
        USER = "root"
        # user password
        PASSWORD = "THX4theF1$h!"
        # connect to MySQL server
        db_connection = mysql.connector.connect(
            host=HOST, database=DATABASE, user=USER, password=PASSWORD
        )
        print("Connected to:", db_connection.get_server_info())

        cursor = db_connection.cursor()

        file = open("temp_text.txt", "w", encoding="utf-8")
        soup = BeautifulSoup(response.text, "lxml")

        patternName = response.url.split("/")[-1]
        patternName = patternName.split(".")[0]
        patternName = patternName.replace("-", "_")

        overview = ""

        print(patternName)

        print(
            "========================================================================================="
        )
        for s in soup.select("h2"):
            imgHit = False
            for ns in s.fetchNextSiblings():
                if ns.name == "figure":
                    imgHit = True
                if ns.name == "h3":
                    imgHit = False
                    continue
                if imgHit:
                    continue
                overview += ns.text

        if overview == "":
            for s in soup.select("h1"):
                imgHit = False
                for ns in s.fetchNextSiblings():
                    if ns.name == "figure":
                        imgHit = True
                    if ns.name == "h3":
                        imgHit = False
                        continue
                    if imgHit:
                        continue
                    overview += ns.text

        # insert the pieces of text collected into a new DB entry
        add_pattern = (
            "INSERT INTO pattern_gof "
            "(category_id, name, overview) "
            "VALUES (%s, %s, %s)"
        )
        data_pattern = (0, patternName, overview)
        cursor.execute(add_pattern, data_pattern)  # insert new pattern
        db_connection.commit()  # commit the data to the DB

        # now that the data is stored in the DB, export the table as a CSV file
        # source: https://datatofish.com/export-sql-table-to-csv-python/
        query_all = pd.read_sql_query(
            "select * from pattern_recommender.pattern_gof", db_connection
        )
        #
        df = pd.DataFrame(query_all)
        df.to_csv(r"scraped_pattern_data_gofpatterns.csv", index=False)
"""
