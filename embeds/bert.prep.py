import os
import pkgutil

#if not os.path.isdir('bert'):
#    os.system('git clone https://github.com/google-research/bert.git')
#if not os.path.isdir('bert-as-a-service'):
#    os.system('git clone https://github.com/hanxiao/bert-as-service.git')

for package in ['tensorflow', 'bert-serving-server', 'bert-serving-client']:
    if not pkgutil.find_loader(package):
        os.system('pip3 install --user ' + package)

os.system('wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip')
os.system('unzip multi_cased_L-12_H-768_A-12.zip')
if not os.path.exists('embeds/bert'):
    os.mkdir('embeds/bert')
os.system('mv multi_cased_L-12_H-768_A-12/* embeds/bert/')
os.system('rm -rf multi_cased_L-12_H-768_A-12*')


