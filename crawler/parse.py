from html.parser import HTMLParser
import os.path


class MyHTMLParser(HTMLParser):
    text = ""

    def handle_data(self, data):
        self.text += data.strip()


parser = MyHTMLParser()
d = os.getcwd() + r"\tutorial\patterns"
for (path) in os.listdir(d):
    file = open(d + "\\" + path, "r", encoding='utf-8')
    parser.feed(file.read())
    file.close()
print(parser.text)
