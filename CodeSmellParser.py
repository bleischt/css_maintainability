import sys, os
import re
import pprint

def checkArgs():
    if len(sys.argv) != 2:
        print('Incorrect args. Usage: python3 {} <sitesDir>')
        exit()
    return sys.argv[1]

def readCillaFile(filepath):
    with open('{}/cilla.txt'.format(filepath), 'r') as f:
       text = f.read()
       if 'chrome-error://chromewebdata/' in text or not text:
           return None
       text = text.replace(' from which:','').replace('(:link, :hover, etc)', '')
       lines = text.split('\n')
       lines = lines[:lines.index('PERCENTAGE: ') + 1]
    return lines

def extractCodeSmellsFromText(lines):
    if lines is not None and len(lines) > 1:
        smells = [line.replace('->','').replace(' ','').split(':') for line in lines if '->' in line and '%' not in line]
        smells = {split[0]:int(split[1]) for split in smells}
        return smells
    return None

def getCodeSmellsPerSite(sitesDir, sitesList):
    smells = dict()
    for site in sitesList:
        try:
            lines = readCillaFile(sitesDir + '/' + site)
        except IOError:
            print('Error reading file: {}/{}/cilla.txt'.format(sitesDir, site))
            print('moving on...')
            continue
        smells[site] =  extractCodeSmellsFromText(lines)
    return smells

def discardInvalidSmellSets(sitesToSmells):
    toSave = dict()
    for site,smells in sitesToSmells.items():
        if smells is not None and smells['LOC(CSS)'] != 0: 
            toSave[site] = smells
    return toSave

def readCodeSmells(sitesDir):
    #sitesDir = checkArgs()
    sites = [name for name in os.listdir(sitesDir) if os.path.isdir('{}/{}'.format(sitesDir, name))]
    sites = [site for site in sites if 'cilla.txt' in os.listdir('{}/{}'.format(sitesDir, site))]
    sitesToSmells = getCodeSmellsPerSite(sitesDir, sites)
    sitesToSmells = discardInvalidSmellSets(sitesToSmells)
    return sitesToSmells 

      
        
#readCodeSmells(checkArgs())

