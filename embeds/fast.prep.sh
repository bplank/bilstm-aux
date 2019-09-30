
mkdir embeds/fasttext
#Newer embeddings
curl https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nl.300.bin.gz | gunzip > embeds/fasttext/nl

#Older embeddings (available in multiple languages)
#wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.bm.zip && unzip wiki.bm && mv wiki.bm.bin embeds/fasttext/bm && rm wiki.bm.*


git clone https://github.com/facebookresearch/fastText.git
#undo cpu specific compilation, for usability reasons
cd fastText
sed -i "s;-march=native;;g" Makefile
make 
cd ..

