import os, sys
from urllib.parse import urlparse
from Alexa import Alexa
from SiteDownloader import SiteDownloader



if len(sys.argv) != 2:
    print('Incorrect args. Usage: python3 DataCollector.py <websiteList.txt>')
    exit() 
    
with open(sys.argv[1], 'r') as f:
    websites = f.readlines()
    websites = [line.strip() for line in websites] 

for website in websites:
    #TODO: website string validation?
    #setup and navigate to new directory for current website
    print('checking for ', website, '...')
    rootDirectory = os.getcwd()
    newDirectory = rootDirectory  + '/' + urlparse(website).netloc.replace('www.', '')
    #skip already-downloaded websites, in case script fails and reruns
    if os.path.exists(newDirectory):
        print('found, skipping...')
        continue
    else:
        os.makedirs(newDirectory)
    os.chdir(newDirectory) 

    #collect Alexa rank and download the website at limited depth
    print('collecting alexa rank...')
    globalRank = Alexa.get_global_rank(website)
    with open('alexa_rank.txt', 'w') as f:
        f.write('global_rank=' + str(globalRank))
    print('alexa rank=', globalRank)
    wget_flags = SiteDownloader.generate_wget_flags(noParent=False, verbose=True, 
        outputFile='wget.log', wait=10)
    #accepted_filetypes = {'.html', '.css', '.js'}
    #wget_flags = SiteDownloader.generate_wget_flags(noParent=False, verbose=True, 
    #    outputFile='wget.log', accepted_filetypes)
    print('downloading site...')
    SiteDownloader.download_website(website, wget_flags, restrictDomain=True)
    print('...done')
    os.chdir(rootDirectory)
