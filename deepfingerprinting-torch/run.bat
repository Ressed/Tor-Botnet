echo dataset, feature, epoch, learning_rate, test_loss, test_acc > log.csv

python main.py --epochs 10 --data_feature direction
python main.py --epochs 30 --data_feature direction
python main.py --epochs 10 --data_feature burst
python main.py --epochs 30 --data_feature burst

python main.py --epochs 10 --data dataset/ISCX-Tor-All --data_type csv --bsz 20 --lr 0.0005 --data_feature direction --classes 7
python main.py --epochs 30 --data dataset/ISCX-Tor-All --data_type csv --bsz 20 --lr 0.0005 --data_feature direction --classes 7
python main.py --epochs 10 --data dataset/ISCX-Tor-All --data_type csv --bsz 20 --lr 0.0005 --data_feature burst --classes 7
python main.py --epochs 30 --data dataset/ISCX-Tor-All --data_type csv --bsz 20 --lr 0.0005 --data_feature burst --classes 7
python main.py --epochs 10 --data dataset/ISCX-Tor-All --data_type csv --bsz 20 --lr 0.0005 --data_feature directional_timestamp --classes 7
python main.py --epochs 30 --data dataset/ISCX-Tor-All --data_type csv --bsz 20 --lr 0.0005 --data_feature directional_timestamp --classes 7