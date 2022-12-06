from html.parser import HTMLParser
import os.path


class NewHTMLParser(HTMLParser):
    text = ""

    def handle_data(self, data):
        self.text += data


parser = NewHTMLParser()
d = os.getcwd() + r"\tutorial\patterns"
for (path) in os.listdir(d):
    file = open(d + "\\" + path, "r", encoding='utf-8')
    file2 = open(os.getcwd() + r"\parsed_patterns\object_oriented\\" + path[0:path.find(".html")] + ".txt", "w", encoding='utf-8')
    parser.feed(file.read())
    file2.write(parser.text[parser.text.find("Intent"):parser.text.find("Support our free website and own the eBook!")])
    parser.text = ""
    file.close()
# print(output)
