import os, sys, platform
import datetime
import traceback
import pip
import logging, logging.handlers
from urllib.parse import urlparse
from Alexa import Alexa
from SiteDownloader import SiteDownloader

logger = logging.getLogger()
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger.addHandler(logging.FileHandler('download.log'))

def log_except_hook(*exc_info):
    text = "".join(traceback.format_exception(*exc_info))
    logger.error("Exception halted execution:\n%s", text)
    logger.error("-----Halting Run on %s-----\n\n", datetime.datetime.now()) 

sys.excepthook = log_except_hook

wget_version = SiteDownloader.get_wget_version()
python_version = sys.version
os_version = platform.platform()
python_modules = [{pkg.key : pkg.version} for pkg in pip.get_installed_distributions() if pkg.key in set(sys.modules)]
#print(python_modules)
#print(wget_version)
#print(python_version)
#print(os_version)


def check_args():
    if len(sys.argv) != 2:
        print('Incorrect args. Usage: python3 % <websiteList.txt>', sys.argv[0])
        exit() 

def write_metadata_file(filename, website, datetime, wget_version, wget_flags, python_version, python_modules, os_version):
    #create metadata for this download
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write('url=' + website + '\n');
            f.write('datetime=' + str(datetime) + '\n')
            f.write('wget_version=' + wget_version + '\n')
            f.write('wget_flags=' + "'" + wget_flags + "'" + '\n')
            f.write('python_version=' + python_version + '\n')
            f.write('python_modules_versions=' + str(python_modules) + '\n')
            f.write('os_version=' + os_version)

def write_alexa_rank_file(filename, website):
    globalRank = Alexa.get_global_rank(website)
    with open(filename, 'w') as f:
        f.write('global_rank=' + str(globalRank))


#website = 'www.connorbatch.com'
#write_alexa_rank_file('alexa.txt', website)
#write_metadata_file('meta.txt', website, datetime.datetime.now(), 'wgetttt', 'pythonnn', 'osss')
#exit()


#begin collecting websites + ranks
check_args()
logger.info("-----Starting new run at %s-----", datetime.datetime.now())

with open(sys.argv[1], 'r') as f:
    websites = f.readlines()
    websites = [line.strip() for line in websites] 

for website in websites:
    #TODO: website string validation?

    #setup and navigate to new directory for current website
    parse = urlparse(website)
    domain = parse.netloc.replace('www.', '')

    if not domain:
        logger.debug("couldn't parse domain: <%s>", website)
        logger.debug("doesn't affect anything other than perhaps downloading website when copy already exists", parse.path)
        domain = parse.path

    logger.info('checking for <%s>...', domain)
    rootDirectory = os.getcwd()
    newDirectory = rootDirectory  + '/' + domain

    #skip already-downloaded websites
    if os.path.exists(newDirectory):
        logger.info('found, skipping...')
        continue
    else:
        os.makedirs(newDirectory)
    os.chdir(newDirectory) 

    #collect Alexa rank and write to file
    logger.info('collecting alexa rank...')
    write_alexa_rank_file('alexa_rank.txt', website)

    #specify filetypes to be accepted/rejected while downloading site
    #accepted_filetypes = {'.html', '.css', '.js'}
    rejected_image_extensions = {'jpeg', 'jfif', 'tiff', 'gif', 'bmp', 'png',
            'ppm', 'pgm', 'pbm', 'pnm', 'webp', 'hdr', 'heif', 'bat', 
            'bpg', 'cgm', 'svg', 'PNG', 'gif', 'ico'}
    rejected_archive_extensions = {'zip', 'tar', 'iso', 'mar', 'bz2', 'gz', 
            'z', '7z', 'dmg', 'rar', 'zipx'}
    rejected_filetypes = {'pdf'}.union(rejected_image_extensions).union(rejected_archive_extensions)
    #generate wget flags for the current download
    wget_flags = SiteDownloader.generate_wget_flags(noParent=False, verbose=True, 
        outputFile='wget.log', rejectList=rejected_filetypes)
    
    #collect download meta data and write to file
    write_metadata_file('meta.txt', website, datetime.datetime.now(), wget_version, wget_flags, python_version, python_modules, os_version)


    logger.info('downloading site...')
    SiteDownloader.download_website(website, wget_flags)

    logger.info('...done')
    os.chdir(rootDirectory)

logger.info('-----Finishing Run on %s-----\n\n', datetime.datetime.now())

    #clean up

