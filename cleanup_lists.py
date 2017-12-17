
import subprocess
import sys

if len(sys.argv) != 2:
    print('incorrect args')
    print('usgae: python3 {} {}'.format(argv[0], '<sites_list.txt>'))
    exit()

with open(sys.argv[1], 'r') as f:
   sites = f.readlines()

failed_sites = []
for site in sites:
    try:
        output = subprocess.check_output('curl -Is {}'.format(site), shell=True)
    except:
        print('failed:', site)
        failed_sites.append(site)
    if 'moved' in str(output):
        print('moved...', site)
        print(str(output).split('\n')[0])

with open(sys.argv[1] + '.fail', 'w') as f:
    f.write('\n'.join(failed_sites))
