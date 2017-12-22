import sys
import CodeSmellParser


#def checkArgs():
#    if 


smells = CodeSmellParser.readCodeSmells('sites/django_sites')
print(len(smells))
