### 生成tor-mirai botnet 数据

#### 构建容器

```
cd datasets/tor-botnet/docker-mirai/docker-bot
docker build -t bot .
cd ../docker-cnc
docker build -t cnc 

cd ../../private-tor-network
docker build -t tornode .
```

#### 建立靶场
```
cd datasets/tor-botnet/private-tor-network
./restart.sh
```
等待一段时间，直至`python util/get_consensus.py`不再报错

注意运行restart.sh会删除tor文件夹

#### 抓取流量
```
cd datasets/tor-botnet
python main.py
```
抓取时间、bot重启时间可在程序内设定

可在private-tor-network/tor/pcaps中找到抓取到的.pcap文件

#### 生成数据
运行datasets/data_generator.py，其中提供了将pcap文件分割为一定长度的流，并转化为带时间戳的方向序列或是方向序列的功能。可按需要修改其中各类数据的路径和参数

AWF数据地址：https://github.com/DistriNet/DLWF

ISCX-Tor数据地址：https://www.unb.ca/cic/datasets/tor.html

datasets/tor-botnet中有一些已生成好的数据集，它们的描述在同目录下的readme.md中


### 实验
使用复现的deep fingerprinting模型测试
在参数中指定--data为生成好的数据集（train.csv, valid.csv, test.csv）的目录，--data_type为csv，按情况修改其他参数即可


### References
https://github.com/deep-fingerprinting/df
https://github.com/jmhIcoding/flowcontainer
https://github.com/elexae/private-tor-network
https://github.com/lejolly/docker-mirai
https://github.com/brechtsanders/proxysocket
https://github.com/DistriNet/DLWF