import os, sys, subprocess
import math
import logging

logger = logging.getLogger()

max_threads = 20

def check_args():
    if len(sys.argv) != 3:
        print('Incorrect args. Usage: python3 {} <websiteList.txt> <directory>'.format(sys.argv[0]))
        exit()
    return sys.argv[1], sys.argv[2]

def read_sites(filepath):
    sites = []
    with open(filepath, 'r') as f:
        sites = f.readlines()
        sites = [line.strip() for line in sites]
    return sites

def run():
    filepath,directory = check_args() 
    sites = read_sites(filepath)
    #print(sites)

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
    os.chdir(directory)
   
    print('range: ', math.ceil(len(sites) / sites_per_thread))
    for index in range(math.ceil(len(sites) / sites_per_thread)):
        temp_file_name = '{}.txt.tmp'.format(index)
        #print('temp_file_name:', temp_file_name)
        with open(temp_file_name, 'w') as f:
            #print('index * sites_per_thread:', index * sites_per_thread)
            #print('index + 1 * sites_per_thread - 1:', (index + 1) * sites_per_thread)
            #print(sites[index * sites_per_thread : (index + 1) * sites_per_thread])
            f.write('\n'.join(sites[index * sites_per_thread : (index + 1) * sites_per_thread]))
        commands.append('python3.5 {}/DataCollector.py {} &'.format(scriptDir, temp_file_name))

    #print([command.replace('\n', ', ') for command in commands])
    #print(len(sites))

    for command in commands:
       subprocess.call(command, shell=True) 


run()
