import os
import json
import pkgutil

if not os.path.isdir('ELMoForManyLangs'):
    os.system('git clone https://github.com/HIT-SCIR/ELMoForManyLangs.git')
for package in ['overrides', 'torch']:
    if not pkgutil.find_loader(package):
        os.system('pip3 install --user ' + package)

#langs = ['en', 'tr', 'fi', 'zh']
#numbers = ['144', '174', '149', '179']
langs = []
numbers = []
for line in open('embeds/langmapping.txt'):
    tok = line.strip().split(' ')
    if tok[0] != 'nl':
        continue
    langs.append(tok[0])
    numbers.append(tok[-1])

print(langs)
print(numbers)
basePath = 'embeds/elmo/'
if not os.path.isdir(basePath):
    os.mkdir(basePath)

for lang, number in zip(langs, numbers):
    embDir = basePath + lang + '/'
    os.system('wget http://vectors.nlpl.eu/repository/11/' + number + '.zip')
    if not os.path.exists(embDir):
        os.mkdir(embDir)
    os.rename(number + '.zip', embDir + number + '.zip')
    cmd = 'cd ' + embDir + ' && unzip ' + number + '.zip && rm ' + number + '.zip && cd ../../'
    print(cmd)
    os.system(cmd)

for lang in langs:
    confPath = basePath + lang + '/config.json'
    data = json.load(open(confPath))
    data['config_path'] = '../../../ELMoForManyLangs/configs/cnn_50_100_512_4096_sample.json'
    json.dump(data, open(confPath, 'w'))

