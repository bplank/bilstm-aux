import os
import pickle
from os.path import expanduser
import pkgutil

def conv(embeds):
    if not embeds.endswith('pickle'):
        return
    print('Loading ' + embeds + '...')
    words, vect = pickle.load(open(dataDir + embeds, 'rb'), encoding='latin1')
    if len(words) != len(vect):
        print("error, words are not same size as embeds for " + embeds + ": " + str(len(words)) + ' - ' + str(len(embeds)))
        return
    print("embeddings of size: " + str(len(words)) + ' * ' +str(len(vect[0])))
    print("writing to: " + dataDir + embeds.replace('pickle','txt'))
    outFile = open(dataDir + embeds.replace('pickle','txt'), 'w')
    for i in range(len(words)):
        outFile.write(words[i])
        for val in vect[i]:
            outFile.write(' ' + str(val))
        outFile.write('\n')
    outFile.close()
    os.remove(dataDir + embeds)

for package in ['polyglot', 'pyicu', 'pycld2', 'dill']:
    if not pkgutil.find_loader(package):
        os.system('pip3 install --user ' + package)

from polyglot.downloader import downloader
#downloader.download("TASK:embeddings2")
downloader.download("embeddings2.nl")

dataDir = 'embeds/polyglot/'
if not os.path.exists(dataDir):
    if not os.path.exists('embeds'):
        os.mkdir('embeds')
    os.mkdir(dataDir)

homedir = expanduser("~")
polyDir = homedir + '/polyglot_data/embeddings2/'
for lang in os.listdir(polyDir):
    cmd = 'tar -xjf ' + polyDir + lang + '/embeddings_pkl.tar.bz2'
    os.system(cmd)
    os.rename('words_embeddings_32.pkl', dataDir + lang + '.pickle')
os.system('rm -rf ~/polyglot_data')

#for embeds in os.listdir(dataDir):
#    conv(embeds)

