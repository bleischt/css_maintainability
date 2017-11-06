import os, sys
import http.server, socketserver
import threading
import logging

logger = logging.getLogger()
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger.addHandler(logging.FileHandler('cssnose.log'))

def check_args():
    if len(sys.argv) != 2:
        print('Incorrect args. Usage: python3 % <pathToWebsites>', argv[0]) 
        exit()

check_args()
sitesDir = sys.argv[1]
try:
    os.chdir(sitesDir)
except FileNotFoundError as e:
    logger.error('Cannot find path to websites: %s', sitesDir)
    exit()

#start code smell collection
port = 8000
print('serving at port', port)
handler = http.server.SimpleHTTPRequestHandler
httpd = socketserver.TCPServer(("", port), handler) 

for siteDir in list(os.walk(sitesDir))[0][1]:
    try:
        os.chdir(os.getcwd() + '/' + siteDir + '/' + siteDir)
        logger.info('current directory: %s', os.getcwd())
        threading.Thread(target = httpd.serve_forever).start()
        wait = input('hit enter')
        httpd.shutdown()
        os.chdir('../..')
    except FileNotFoundError as e:
        logger.error("couldn't find files to serve: %s", siteDir)
    except:
        logger.error('failed trying to serve website: %s', siteDir)
        continue

    logger.info('feeding website to cssNose...')
    try:
        #run css nose java process here
        pass
    except:
        logger.error('failed feeding website to CSSNose java process: %s', siteDir)



