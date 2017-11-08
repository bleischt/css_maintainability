import os, subprocess
import logging
from urllib.parse import urlparse

logger = logging.getLogger()

class SiteDownloader:

    @staticmethod
    def generate_wget_flags(wait=10, outputFile='', level=3, randomWait=True, 
            convertLinks=True, recursive=True, pageRequisites=True, verbose=False, 
            adjustExtension=True, noParent=True, acceptList={}, rejectList={}, quota='500m'):
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
            flags.append("--accept='" + ','.join(acceptList) + "'")
        if rejectList:
            flags.append("--reject='" + ','.join(rejectList) + "'")
        if quota: 
            flags.append("--quota='" + quota + "'")

        return flags

    @staticmethod 
    def download_website(url, flags, restrictDomain=False): 
        command = "wget " + " ".join(flags) 

        if restrictDomain:
            domain = urlparse(url).netloc.replace('www.', '') 
            command += " --domains=" + domain
        
        logger.debug(command + ' ' +  url.replace('www.', ''))
        os.system(command + ' ' + url.replace('www.', ''))

    @staticmethod
    def get_wget_version():
        return subprocess.check_output(['wget', '--version']).decode('utf-8').split('\n')[0]


#print(SiteDownloader.generate_wget_flags(verbose=True, outputFile='wget.log'))
#SiteDownloader.download_website('https://www.crummy.com/software/BeautifulSoup/bs4/doc/#contents-and-children', SiteDownloader.generate_wget_flags(verbose=True, outputFile='wget.log'))
