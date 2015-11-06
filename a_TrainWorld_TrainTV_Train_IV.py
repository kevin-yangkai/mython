
import argparse
import os
import re
import csv
import subprocess

#===============================================================================
# This script trains the World, the Totalvariability matrix and extracts the Ivector
# Make sure to have a folder called "cfg" in your directory, which consists of the config files
# For training the World, TotalVariability and IVector
# The script does generate .ndx files for TV and IV Extraction, but these can also be provided by the user
#===============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-tw', action='store_true', help='Runs only TrainWorld', required=False)
parser.add_argument('-tv', action='store_true', help='runs only Training of Total Variability ', required=False)
parser.add_argument('-iv', action='store_true', help='Runs only Ivector Extraction', required=False)
args = parser.parse_args()


pathkeywords = ['featureFilesPath', 'mixtureFilesPath', 'matrixFilesPath', 'saveVectorFilesPath']
tvkeywords = ['ndxFilename']
ivkeywords = ['targetIdList']
# threadkeyword=['numThread']

pathregex = re.compile('|'.join(pathkeywords))
tvregex = re.compile('|'.join(tvkeywords))
ivregex = re.compile('|'.join(ivkeywords))

# threadregex=re.compile('|'.join(threadkeyword))

def trainWorld():
    #--------------- Read in the config file given and create all necessary dirs
    with open('cfg/TrainWorld.cfg', 'r') as twp:
        for line in twp.readlines():
            if pathregex.search(line):
                key, value = line.partition("\t")[::2]
                value = value.strip()
                if not os.path.exists(value):
                    os.makedirs(value)
    
    #------------------------------------------------- run the TrainWorld Script
    with open('TrainWorld.log','w') as twp:
        with open('TrainWorld.err','w') as twep:
            p1=subprocess.Popen('bin/TrainWorld --config cfg/TrainWorld.cfg',shell=True,stdout=twp,stderr=twep)
            p1.wait()
    

def trainTV():
    #---------------------------------- Prepare data ( ndx file ) for processing
    with open('cfg/TotalVariability_fast.cfg', 'r') as twp:
        paths = dict()
        lines = filter(lambda x: pathregex.search(x) or tvregex.search(x), twp.readlines())
        pathlines = filter(lambda x:pathregex.search(x), lines)
        tvlines = filter(lambda x:tvregex.search(x), lines)
        # Fill into paths all the paths, which are given by the pathkeyword argument
        for line in pathlines:
            key, value = line.partition("\t")[::2]
            value = value.strip()
            paths[key.strip()] = value
            if not os.path.exists(value):
                os.makedirs(value)
        for line in tvlines:
            key, value = line.partition("\t")[::2]
            value = value.strip()
            readir=paths['featureFilesPath']
            createIndexDirs(value, readir)
            rawfiles = list()
            for rawfile in os.listdir(readir):
                rawfiles.append(removeEnding(rawfile))
            with open(value, 'w+') as ndxp:
                ndxp.writelines(line + '\n' for line in rawfiles)
                    
    with open('TotalVariability.log','w') as twp:
        with open('TotalVariability.err','w') as twep:
            p1=subprocess.Popen('bin/TotalVariability --config cfg/TotalVariability_fast.cfg',shell=True,stdout=twp,stderr=twep)
            p1.wait()
                

def extractIV():
    with open('cfg/ivExtractor_fast.cfg', 'r') as twp:
        paths = dict()
        #=======================================================================
        # Dont iterate two times over the whole file, just once 
        #=======================================================================
        lines = filter(lambda x:pathregex.search(x) or ivregex.search(x), twp.readlines())
        pathlines = filter(lambda x:pathregex.search(x), lines)
        ivlines = filter(lambda x:ivregex.search(x), lines)
        for line in pathlines:
            key, value = line.partition("\t")[::2]
            value = value.strip()
            paths[key.strip()] = value
            if not os.path.exists(value):
                os.makedirs(value)
        # lines are not neccessarily ordered
        for line in ivlines:
            key, value = line.partition("\t")[::2]
            value = value.strip()
            readir= paths['featureFilesPath']
            createIndexDirs(value,readir)
            rawfiles = list()
            for rawfile in os.listdir(readir):
                rawfiles.append(removeEnding(rawfile))
            with open(value, 'w+') as ndxp:
                ndxp.writelines(line + '\t '+ line + '\n' for line in rawfiles)
    with open('IvExtract.log','w') as twp:
        with open('IvExtract.err','w') as ivep:
            p1=subprocess.Popen('bin/IvExtractor --config cfg/ivExtractor_fast.cfg ',shell=True,stdout=twp,stderr=ivep)
            p1.wait()

def createIndexDirs(ndxfilename,readir):
    # IF file is existing, but empty ... create the file
    if not os.path.isfile(ndxfilename):
        if not os.path.exists(os.path.dirname(ndxfilename)):
            os.makedirs(os.path.dirname(ndxfilename))
    # if file is not empty delete it and replace
    if os.path.exists(ndxfilename) and os.stat(ndxfilename)[6] == 0 :
        os.remove(ndxfilename)

# removes the extension of the given filename
def removeEnding(text):
    t, ending = os.path.splitext(text)
    while ending is not "":
        t, ending = os.path.splitext(t)
    return t

def main():
    # check if any argument is given
    if args.tw:
        print 'Training World'
        trainWorld()
        
    if args.tv:
        print 'Training TV Matrix'
        trainTV()
        
    if args.iv:
        print 'Extracting I-Vector'
        extractIV()
    
    if not any(vars(args).values()):
        trainWorld()
        trainTV()
        extractIV()
        
         
if __name__ == '__main__': 
    main()



