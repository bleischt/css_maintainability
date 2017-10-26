import urllib.request
import bs4
import string

class Alexa:

    base_url = 'https://www.alexa.com/siteinfo/'

    #todo: add error checking & remove prints
    @classmethod
    def get_global_rank(cls, url):
        try:
            webpage = urllib.request.urlopen(cls.base_url + url).read().decode('utf-8')
            soup = bs4.BeautifulSoup(webpage, 'html.parser')
            site_base = soup.find("input", id='siteInput')
            #print('Alexa rank for', site_base['value'] + ':')
            global_rank = soup.find("span", class_='globleRank').find("strong", class_='metrics-data').contents[-1].replace(',', '')
        except:
            return None

        try:
            global_rank = int(global_rank)
        except:
            global_rank = -1

        return int(global_rank)

#print('rankkkk:', Alexa.get_global_rank('lsdkfj'))

