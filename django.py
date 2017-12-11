from bs4 import BeautifulSoup
import urllib.request
from time import sleep
import random

#url_first = "https://webcache.googleusercontent.com/search?q=cache:https://www.djangosites.org/%3Fpage%3D" 
#url_second = "&num=1&strip=1&vwsrc=0"
url = "https://www.djangosites.org/?page="
numPages = 264

django_sites = open('django_sites.txt', 'w')

#for index in range(numPages):
for index in range(264):
    print('downloading page {}......'.format(index+1))
    try:
        request = urllib.request.Request(url + str(index+1), headers={'User-Agent': 'Mozilla/5.0'})
        page = urllib.request.urlopen(request).read()
    except:
        print("Couldn't pull sites for page " + str(index+1))

    soup = BeautifulSoup(page, 'html.parser')
    urls = [site.a['href'] for site in soup.find_all('p', class_='header')]
    #print(urls)
    django_sites.write('\n'.join(urls))  
    sleep(random.randint(20, 40))

