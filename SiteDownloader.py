import os
from urllib.parse import urlparse

class SiteDownloader:

    @staticmethod
    def generate_wget_flags(wait=20, outputFile='', level=3, randomWait=True, 
            convertLinks=True, recursive=True, pageRequisites=True, verbose=False, 
            adjustExtension=True, noParent=True, acceptList={}, quota='500m'):
        flags = []

        flags.append("--wait=" + str(wait))
        flags.append("--level=" + str(level))
    
        if randomWait:
            flags.append("--random-wait")
        if convertLinks:
            flags.append("--convert-links")
        if recursive:
            flags.append("--recursive")
        if pageRequisites:
            flags.append("--page-requisites")
        if verbose:
            flags.append("--verbose")
        if adjustExtension:
            flags.append("--adjust-extension")
        if noParent:
            flags.append("--no-parent")
        if outputFile: 
            flags.append("--output-file='" + outputFile + "'")
        if acceptList: 
            flags.append("--accept='" + ','.join(acceptList))
        if quota: 
            flags.append("--quota='" + quota)

        return flags

    @staticmethod 
    def download_website(url, flags, restrictDomain=False): 
        command = "wget " + " ".join(flags) 

        if restrictDomain:
            domain = urlparse(url).netloc.replace('www.', '') 
            command += " --domains=" + domain
        
        print(command + ' ' +  url)
        os.system(command + ' ' + url)

#print(SiteDownloader.generate_wget_flags(verbose=True, outputFile='wget.log'))
#SiteDownloader.download_website('https://www.crummy.com/software/BeautifulSoup/bs4/doc/#contents-and-children', SiteDownloader.generate_wget_flags(verbose=True, outputFile='wget.log'))
