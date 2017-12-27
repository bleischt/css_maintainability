import os, sys, shutil
import http.server, socketserver
import threading
import time
import logging
import subprocess

import atexit

logger = logging.getLogger()
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger.addHandler(logging.FileHandler('cssnose.log'))

from http.server import HTTPServer as BaseHTTPServer, SimpleHTTPRequestHandler


class HTTPHandler(SimpleHTTPRequestHandler):
    """This handler uses server.base_path instead of always using os.getcwd()"""
    def translate_path(self, path):
        path = SimpleHTTPRequestHandler.translate_path(self, path)
        relpath = os.path.relpath(path, os.getcwd())
        fullpath = os.path.join(self.server.base_path, relpath)
        return fullpath

class HTTPServer(BaseHTTPServer):
    """The main server, you pass in base_path which is the path you want to serve requests from"""
    def __init__(self, base_path, server_address, RequestHandlerClass=HTTPHandler):
        self.base_path = base_path
        BaseHTTPServer.__init__(self, server_address, RequestHandlerClass)

def start_serving(port, path):
    logger.info('serving at port: {}'.format(port))
    #handler = http.server.SimpleHTTPRequestHandler
    #httpd = socketserver.TCPServer(("", port), handler) 
    threading.Thread(target = HTTPServer(path, ("", port)).serve_forever).start()

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


def run():
    port = 8000
    start_serving(port, sitesDir)

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

        killProcessFromString('CSSNose.jar')
        killProcessFromString('Chrome')
        killProcessFromString('chromedriver')

        logger.info('current website: %s', siteDir)
        logger.info('feeding website to cssNose...')
        logger.info('url: %s', 'https://localhost:' + str(port) + '/' + siteDir + '/' + siteDir)

        #try:
            #run css nose java process here
        startTime = time.time()
        command = ['java', '-jar', 'CSSNose.jar', 'http://localhost:{}/{}/{}'.format(port, siteDir, siteDir)]
            #output = subprocess.check_output('java -jar CssNose.jar http://localhost:{}/{}/{}'.format(port, siteDir, siteDir), timeout=(60 * 15))
            #output = subprocess.check_output(command, timeout=(60 * 15))
        with subprocess.Popen(command, preexec_fn=os.setsid) as process:
            try:
                output = process.communicate(timeout=60*15)[0]
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGINT)
                output = process.communicate()[0]
            except Exception as e:
                logger.error('failed feeding website to CSSNose java process: %s', siteDir)
                logger.error(e)
                open('{}/{}/cilla.txt'.format(sitesDir, siteDir), 'w').close()
        try:
            finishTime = (time.time() - startTime) / 60
            shutil.copyfile('CillaOutput/cilla-{}.txt'.format(siteDir), '{}/{}/cilla.txt'.format(sitesDir, siteDir)) 
            with open('{}/{}/cilla.log'.format(sitesDir, siteDir), 'w') as f:
                f.write(str(output))
                f.write('\nfinished in {} minutes'.format(finishTime / 60))
            logger.info('finished with site: %s', siteDir)
            logger.info('time to completion: %s', finishTime)
        except IOError as e:
            logger.error('cssnose finsihed but copying over the output failed: %s', siteDir)
            logger.error(e)
        #except subprocess.TimeoutExpired as e:
        #   logger.error('CSSNose is taking too long, moving on.....')
        #   logger.error(e)
        #   open('{}/{}/cilla.txt'.format(sitesDir, siteDir), 'w').close()
        #except Exception as e:
        #    logger.error('failed feeding website to CSSNose java process: %s', siteDir)
        #    logger.error(e)
        #    open('{}/{}/cilla.txt'.format(sitesDir, siteDir), 'w').close()


def killProcessFromString(string): 
    kill_output = ''
    try:
        time.sleep(5)
        logger.info('killing {} processes...'.format(string))
        command = ['pkill', '-f', string]
        kill_output = subprocess.check_output(command)
    except Exception as e:
        logger.error('failed to kill {} processes, or they died naturally'.format(string))
        #logger.error(str(kill_output))
    if kill_output:
        logger.info(e)

def _cleanup():
    killProcessFromString('CSSNose.jar')
    killProcessFromString('Chrome')
    killProcessFromString('chromedriver')

atexit.register(_cleanup)

run()
