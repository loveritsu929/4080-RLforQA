ROOT=$(dirname $(realpath $0))
mkdir $ROOT/data
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json -O $ROOT/data/train_dataset
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O $ROOT/data/develop_dataset
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $ROOT/data/glove_archive.zip
unzip -j $ROOT/data/glove_archive.zip -d $ROOT/data
mv $ROOT/data/$(zipinfo -1 $ROOT/data/glove_archive.zip) $ROOT/data/glove_archive
rm $ROOT/data/glove_archive.zip
