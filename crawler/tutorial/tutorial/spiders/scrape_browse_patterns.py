import scrapy
from bs4 import BeautifulSoup
from scrapy.item import Item, Field
import matplotlib.pyplot as plt
import matplotlib.image as mpim

# scrapy crawl example -O myfile.csv:csv
class CustomItem(Item):
    header = Field()
    paragraph = Field()
    link = Field()

class BrowseSpider(scrapy.Spider):
    name = 'browse'

    creational_urls = [
        'https://sourcemaking.com/design_patterns/abstract_factory',
        ]

    start_urls = creational_urls


    def parse(self, response):

        file = open('temp_text.txt', 'w', encoding="utf-8")
        soup = BeautifulSoup(response.text, 'lxml')

        patternName = response.url.split("/")[-1]
        last_heading = ""
        intent = ""
        problem = ""
        discussion = ""
        structure = ""
        example = ""
        checklist = ""
        rulesOfThumb = ""

        for s in soup.select('h3'):
            if s.text == "Support our free website and own the eBook!":
                break

            last_heading = s.text

            file.write(s.text + '\n')# the name of the pattern

            for ns in (s.fetchNextSiblings()):
                if ns.name == "h3":
                    #ile.write('\n'+ns.text+'\n')
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

            # file.write(intent + '\n')
            # file.write(problem + '\n')
            # file.write(discussion + '\n')
            # file.write(structure + '\n')
            # file.write(example + '\n')
            # file.write(checklist + '\n')
            # file.write(rulesOfThumb + '\n')

        for myp in soup.find_all("p", class_="image"):
            for img in myp.find_all("img"):
                print(img.attrs['src'])
                file.write(img.attrs['src'] + "\n")

            #print(img.attrs['src'])
            #imgtemp = mpim.imread(img.attrs['src'])
            #imgplot = plt.imshow(imgtemp)
            #plt.show()
            #file.write(img)

        #print(ns)