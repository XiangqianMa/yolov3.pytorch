## Dataset
### Pascal VOC data
按照如下的方式下载Pascal VOC数据集：

在任意位置（如`mydata`），执行如下语句：
```shell script
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```
接着，在`mydata`目录下，执行：
```shell script
mkdir voc
cp VOCdevit/VOC2007/JPEGImages ./voc/
cp VOCdevit/VOC2007/Annotations ./voc/
cp VOCdevit/VOC2012/JPEGImages/* ./voc/JPEGImages/
cp VOCdevit/VOC2012/Annotations/* ./voc/Annotations/

mkdir ./voc/ImageSets
mkdir ./voc/ImageSets/Main

cat VOCdevit/VOC2007/ImageSets/Main/train.txt VOCdevit/VOC2007/ImageSets/Main/val.txt VOCdevit/VOC2012/ImageSets/Main/train.txt VOCdevit/VOC2012/ImageSets/Main/val.txt > ./voc/ImageSets/Main/train.txt
cp VOC2007/ImageSets/Main/test.txt voc/ImageSets/Main/

rm -r VOCdevit
```
接着，进入项目根目录（`mydata`即为上述数据集的下载根目录），执行：
```shell script
mkdir data
ln mydata/voc ./data/
```
在项目根目录下执行：
```shell script
python scripts/prepare_voc.py
```
执行完成后，将在`./data/voc`目录下生成三个文件夹：
```shell script
train: 存放训练图片
train_txt: 存放各个图片所对应的标注文件
categories_id_to_name.json: 类别id到名称的转换文件
```
## Train
**下载预训练权重**：
```shell script
cd checkpoints/official_weights
wget https://pjreddie.com/media/files/darknet53.conv.74
```
## Attention
* 在使用Mosaic和训练集数据增强时，如果使用SGD优化器，学习率最高只能是1e-4，否则会导致训练不稳定；但如果使用Adam，则学习率可以调整为1e-3。