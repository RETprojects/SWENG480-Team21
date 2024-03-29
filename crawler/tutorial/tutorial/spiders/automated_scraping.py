import scrapy
import sqlite3
import os.path
from bs4 import BeautifulSoup
import re
import os


# scrapy crawl auto -a start_urls=[url] -a range=["str1:+_:str2:+_:str3:+_:str4"]
# scrapy crawl auto -a start_urls=https://www.crummy.com/software/BeautifulSoup/bs4/doc/ -a range="which have been removed.:+_:that no longer exists.:+_:Beautiful Soup will never be as fast as the parsers:+_:searching the document much faster."
class InvalidInputException(BaseException):
    pass


def run(arg1, arg2):
    os.chdir(os.path.dirname(__file__))
    open("automated_scraping_output.txt", "w").close()
    os.system('scrapy crawl auto -a start_urls=' + arg1 + ' -a range="' + arg2 + '"')
    # os.chdir(os.path.dirname(__file__))
    # print('scrapy crawl auto -a start_urls=' + arg1 + ' -a range=' + arg2)
    # print(type(arg2))
    # if arg2 == "":
    #    os.system('scrapy crawl auto -a start_urls=' + arg1)
    # else:
    #    os.system('scrapy crawl auto -a start_urls=' + arg1 + ' -a range="' + arg2 + '"')
    # os.system('scrapy crawl auto -a start_urls=https://www.crummy.com/software/BeautifulSoup/bs4/doc/ -a range="which have been removed.:+_:that no longer exists.:+_:Beautiful Soup will never be as fast as the parsers:+_:searching the document much faster."')
    # os.system('scrapy crawl auto -a start_urls=' + arg1 + ' -a range=' + arg2)


class AutoSpider(scrapy.Spider):
    name = "auto"

    def __init__(self, **kwargs):
        if 'start_urls' in kwargs:
            self.start_urls = kwargs.pop('start_urls').split(',')
            for i in self.start_urls:
                print("Provided URL " + i)
            if len(self.start_urls) == 0 or len(self.start_urls) > 1:
                raise InvalidInputException
        if 'range' in kwargs:
            self.range = kwargs.pop('range').split(':+_:')
            for i in self.range:
                print("Provided Range " + i)
            if len(self.range) % 2 != 0:
                raise InvalidInputException
        super().__init__(**kwargs)

    def parse(self, response, **kwargs):
        # print("TEST" + os.path.split(os.path.split(os.path.split(os.path.split(os.path.dirname(__file__))[0])[0])[0])[0] + "\webserver\sqlitedatabase.db")

        # conn = sqlite3.connect(
        #    os.path.split(os.path.split(os.path.split(os.path.split(os.path.dirname(__file__))[0])[0])[0])[
        #        0] + "\webserver\sqlitedatabase")
        # cur = conn.cursor()
        # open("automated_scraping_output.txt", "w").close()
        file = open("automated_scraping_output.txt", 'a')
        soup = BeautifulSoup(response.text, 'lxml')
        s = soup.get_text().strip()
        s = s.replace("\t", "").replace("\r", "").replace("\n", " ")
        for i in range(0, len(self.range), 1):
            matches = [m.start() for m in re.finditer(self.range[i], s)]
            # print("TEST: " + str(len(matches)))
            if len(matches) == 0 or len(matches) > 1:
                raise InvalidInputException
        # print("TEST: " + s)
        # print("TEST: " + self.range[0] + " BREAK " + self.range[1])
        # print("TEST: " + str(s.find(self.range[0])) + " BREAK " + str(s.find(self.range[1])))
        for i in range(0, len(self.range), 2):
            text_in_range = s[s.find(self.range[i]):s.rfind(self.range[i + 1]) + len(self.range[i + 1])]
            file.write(text_in_range)
            # print(text_in_range)
            # add_pattern = ("INSERT INTO problem "
            #               "(category_id, description) "
            #               "VALUES (?, ?)")
            # data_pattern = (1, text_in_range)
            # cur.execute(add_pattern, data_pattern)
            # conn.commit()

    def errback(self, failure):
        self.logger.error(repr(failure))
