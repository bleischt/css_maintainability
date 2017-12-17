import os, sys, subprocess
import math
import logging

logger = logging.getLogger()

max_threads = 20

def check_args():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print('Incorrect args. Usage: python3 {} <directory> <pathToJar> optional:<websiteList.txt>'.format(sys.argv[0]))
        exit()
    return sys.argv[1], sys.argv[2]

def read_sites(filepath):
    sites = []
    with open(filepath, 'r') as f:
        sites = f.readlines()
        sites = [line.strip() for line in sites]
    return sites

def run():
    directory,pathToJar = check_args() 
    sites = os.listdir(directory)
    if len(sys.argv) == 4:
        sites = read_sites(sys.argv[3])

    # get the directory of the current script
    scriptDir = os.path.realpath(__file__) #os.path.dirname(sys.argv[0])
    scriptDir = '/'.join(scriptDir.split('/')[:-1])
    print('scriptDir:', scriptDir)

    #calculate the number of sites that should be given to each thread
    #based on the # of sites and the # of max threads specified
    sites_per_thread = 1
    if len(sites) // max_threads > 0:
        sites_per_thread  = math.ceil(len(sites) / max_threads)
    print('sites_per_thread:', sites_per_thread)
    print('sites', len(sites))

    commands = []
    #os.chdir(directory)
   
    print('range: ', math.ceil(len(sites) / sites_per_thread))
    for index in range(math.ceil(len(sites) / sites_per_thread)):
        temp_file_name = '{}.txt.tmp'.format(index)
        with open(directory + '/' + temp_file_name, 'w') as f:
            f.write('\n'.join(sites[index * sites_per_thread : (index + 1) * sites_per_thread]))

        commands.append('python3.5 {}/CSSNose.py {} {} {} &'.format(scriptDir, directory, pathToJar, os.path.abspath(directory) + '/' + temp_file_name))

    for command in commands:
       subprocess.call(command, shell=True) 


run()
