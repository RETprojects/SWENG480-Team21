from html.parser import HTMLParser
import os.path


class NewHTMLParser(HTMLParser):
    text = ""

    # def handle_starttag(self, tag, attrs):
    #     if tag == "h3":
    #         print("Start tag:", tag)

    # def handle_endtag(self, tag):
    #     if tag == "h3":
    #         print("End tag:", tag)

    def handle_data(self, data):
        self.text += data


output = ""
parser = NewHTMLParser()
d = os.getcwd() + r"\tutorial\patterns"
for (path) in os.listdir(d):
    file = open(d + "\\" + path, "r", encoding='utf-8')
    parser.feed(file.read())
    output += parser.text[parser.text.find("Intent"):parser.text.find("Support our free website and own the eBook!")]
    parser.text = ""
    file.close()
print(output)
