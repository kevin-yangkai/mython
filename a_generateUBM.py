import os
import argparse as argp
import re
from difflib import SequenceMatcher, get_close_matches
from __builtin__ import exit
import threading

class Container:
    
    
    def __init__(self,filename):
        self.filename=filename
        self._filecontent=list()
    def addContent(self,content):
        self._filecontent.append(content)
    
    
    def getContent(self):
        res=str()
        #Needs to sort before returning, because HCopy does not sort the data
        #IT could happen that the data is concatinated wrong
        self._filecontent.sort()
        self._filecontent.sort(key=len )
        flen=len(self._filecontent)
        for i in range(flen):
            if (i+1<flen):
                res+=self._filecontent[i]+' + '
            else:
                res+=self._filecontent[i]
        res=res.replace('\n', '')
        return res
#===============================================================================
# This script assembles out of a given .scp file, which already is formatted
# into source \t featurefile , every parttial sequence of the file into one 
# featurefile 
#===============================================================================


parser = argp.ArgumentParser()
parser.add_argument('-i',type=file,help='Input .scp file')
parser.add_argument('-o',help='Output directory ')
parser.add_argument('-glst',type=str,help='also generates a .lst file containing the speakernames, given the path',default='lst/data.lst')
parser.add_argument('-f',action='store_true',help='writes a .scp file which can afterwards be executed with HCopy',default=False)
parser.add_argument('-regex',type=str,help='the regex to split the given data into name and speech segment',default='\w+-[AB]')
args=parser.parse_args()

if(args.i is None or args.o is None):
    parser.print_help()
    exit()

fileending='tmp.prm'
regex=re.compile(args.regex)
sortednames = ([entr for entr in args.i])
sortednames.sort()

i=0
j=0


lists=list()

while i < len(sortednames):
    names=list()
    entry = os.path.basename(sortednames[i])
    match=regex.search(entry)
    spkname=match.group(0)
    cont=Container(spkname)
    #add the current item 
    cont.addContent(sortednames[i])
    j = i+1
    while j < len(sortednames):
        check = os.path.basename(sortednames[j])
        if(check.startswith(spkname) and entry is not check):
            cont.addContent(sortednames[j])
        else:
            break
        j=j+1
    lists.append(cont)
    i=j
print 'Finished gathering Data'

def runHCopy(i):
    cont = lists[i]
    cmd = 'bin/HCopy ' + cont.getContent() +' '+ args.o + cont.filename +'.' + fileending
    os.system(cmd)

def writeSCP(i):
    cont = lists[i]
    with open('concat_Frames.tmp.scp','a') as openl:
        openl.write(cont.getContent() +' '+ args.o + cont.filename +'.' + fileending + '\n')  

runningThreads = list()
for i in range(len(lists)):
    tar=runHCopy
    if(args.f):
        tar=writeSCP
    t = threading.Thread( target=tar,args=(i,))
    runningThreads.append(t)
    t.start()
for thr in runningThreads:
    thr.join()
    
print 'Finished assmbling all files'
    

if args.glst:
    print 'Beginning to generate data.lst file'
    fn = args.glst
    if not os.path.exists(fn):
        with open(fn,'w') as glp:
            glp.writelines([cont.filename + '\n' for cont in lists])
        
    
    
    print 'Finished data.lst'

    
