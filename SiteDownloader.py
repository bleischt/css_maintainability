import os, subprocess
import logging
from urllib.parse import urlparse

logger = logging.getLogger()

class SiteDownloader:

    @staticmethod
    def generate_wget_flags(wait=20, outputFile='', level=3, randomWait=True, 
            convertLinks=True, recursive=True, pageRequisites=True, verbose=False, 
            adjustExtension=True, spanHosts=True, acceptList={}, rejectList={},
            quota='5m', limitRate='60k'):
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
        if spanHosts:
            flags.append("--span-hosts")
        if verbose:
            flags.append("--verbose")
        if adjustExtension:
            flags.append("--adjust-extension")
        if outputFile: 
            flags.append("--output-file='" + outputFile + "'")
        if acceptList: 
            flags.append("--accept='" + ','.join(acceptList) + "'")
        if rejectList:
            flags.append("--reject='" + ','.join(rejectList) + "'")
        if quota: 
            flags.append("--quota='" + quota + "'")
        if limitRate:
            flags.append("--limit-rate='" + limitRate + "'")

        return flags

    @staticmethod 
    def download_website(url, flags, restrictDomain=False): 
        command = "wget " + " ".join(flags) 

        if restrictDomain:
            parse = urlparse(url)
            if not parse.netloc:
                logger.error("couldn't restrict domain for %s", url)
            else:
                domain = urlparse(url).netloc.replace('www.', '') 
                command += " --domains=" + domain
        
        command += ' ' + url.replace('www.', '').replace('http://', '')
        logger.debug(command)
        #os.system(command)

        try:
            proc = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as error:
            logger.error('wget command FAILED...')
            logger.error(error.output.decode('utf-8'))
            logger.error('....moving on...')


    @staticmethod
    def get_wget_version():
        return subprocess.check_output(['wget', '--version']).decode('utf-8').split('\n')[0]


#print(SiteDownloader.generate_wget_flags(verbose=True, outputFile='wget.log'))
#SiteDownloader.download_website('https://www.crummy.com/software/BeautifulSoup/bs4/doc/#contents-and-children', SiteDownloader.generate_wget_flags(verbose=True, outputFile='wget.log'))
