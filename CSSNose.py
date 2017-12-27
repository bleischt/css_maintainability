import os, sys, shutil
import http.server, socketserver
import threading
import time
import logging
import subprocess

logger = logging.getLogger()
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger.addHandler(logging.FileHandler('cssnose.log'))

def check_args():
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print('Incorrect args. Usage: python3 {} <pathToWebsites> <pathToJar> optional:<siteList.txt>'.format(sys.argv[0])) 
        exit()

def read_sites(filepath):
    sites = []
    with open(filepath, 'r') as f:
       sites = f.readlines()
       sites = [line.strip() for line in sites]
    return sites

check_args()
sitesDir = os.path.abspath(sys.argv[1])
pathToJar = os.path.abspath(sys.argv[2])

#try:
#    os.chdir(sitesDir)
#except FileNotFoundError as e:
#    logger.error('Cannot find path to sites: %s', sitesDir)
#    exit()

port = 8000
#print('serving at port', port)
#handler = http.server.SimpleHTTPRequestHandler
#httpd = socketserver.TCPServer(("", port), handler) 
#threading.Thread(target = httpd.serve_forever).start()
#input('lets see if it works')

def run():

    try:
        os.chdir(pathToJar)
    except FileNotFoundError as e:
        logger.error('Cannot find path to jar: %s', pathToJar)
        exit()

    #start code smell collection
    sites = [domain for domain in os.listdir(sitesDir) if os.path.isdir('{}/{}'.format(sitesDir, domain))]

    if len(sys.argv) == 4:
        sites = read_sites(sys.argv[3])

    for siteDir in sites:
        if os.path.isfile(sitesDir + '/' + siteDir + '/' + 'cilla.txt'):
            logger.info('already found cilla.txt for %s.....', siteDir)
            continue
        logger.info('current website: %s', siteDir)
        logger.info('feeding website to cssNose...')
        logger.info('url: %s', 'http://localhost:' + str(port) + '/' + siteDir + '/' + siteDir)

        try:
            #run css nose java process here
            startTime = time.time()
            command = ['java', '-jar', 'CSSNose.jar', 'http://localhost:{}/{}/{}'.format(port, siteDir, siteDir)]
            #output = subprocess.check_output('java -jar CssNose.jar http://localhost:{}/{}/{}'.format(port, siteDir, siteDir), timeout=(60 * 15))
            output = subprocess.check_output(command, timeout=(60 * 15))
            finishTime = (time.time() - startTime) / 60
            shutil.copyfile('CillaOutput/cilla-{}.txt'.format(siteDir), '{}/{}/cilla.txt'.format(sitesDir, siteDir)) 
            with open('{}/{}/cilla.log'.format(sitesDir, siteDir), 'w') as f:
                f.write(str(output))
                f.write('\nfinished in {} minutes'.format(finishTime / 60))
            logger.info('finished with site: %s', siteDir)
            logger.info('time to completion: %s', finishTime)
        except IOError as e:
            logger.error('cssnose (i think) worked but copying over the output failed: %s', siteDir)
            logger.error(e)
        except subprocess.TimeoutExpired as e:
           logger.error('CSSNose is taking too long, moving on.....')
           logger.error(e)
           open('{}/{}/cilla.txt'.format(sitesDir, siteDir), 'w').close()
        except Exception as e:
            logger.error('failed feeding website to CSSNose java process: %s', siteDir)
            logger.error(e)
            open('{}/{}/cilla.txt'.format(sitesDir, siteDir), 'w').close()

        kill_output = ''
        try:
            time.sleep(5)
            logger.info('killing chromedriver processes...')
            kill_output = subprocess.check_output("pkill -f chromedriver", shell=True)
        except Exception as e:
            logger.error('failed to kill chromedriver process, or it died naturally')
            logger.error(e)
            logger.error(str(kill_output))

        try:
            time.sleep(5)
            logger.info('killing Chrome processes...')
            kill_output = subprocess.check_output("pkill -f Chrome", shell=True)
        except Exception as e:
            logger.error('failed to kill chrome processes, or they died naturally')
            logger.error(e)
            logger.error(str(kill_output))

run()
