import torch
import argparse
import time
from tqdm import tqdm
from torch.autograd import Variable
from model import DFNet
from utility import load_dataset
from torchmetrics import Recall, Precision, Accuracy

parser = argparse.ArgumentParser()
# parser.add_argument('--input_size', type=float, default=5000)
parser.add_argument('--classes', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--bsz', type=int, default=128)
parser.add_argument('--save', type=str, default='models/df.pt')
parser.add_argument('--data', type=str, default='dataset/ClosedWorld/NoDef')
parser.add_argument('--data_type', type=str, default='pickle')
parser.add_argument('--data_feature', type=str, default='direction')
parser.add_argument('--log', type=str, default='log.csv')
parser.add_argument('--positive_label', type=int, default=1)


# torch.manual_seed(int(time.time()))

class Trainer:
    def __init__(self, args):
        self.device = torch.device('cuda')

        self.batch_size = args.bsz
        self.dataset_args = (args.data, args.data_type, args.data_feature)
        d1, d2, d3, d4, d5, d6 = load_dataset(args.data, args.data_type, args.data_feature)
        self.train_in, self.train_out = self.batchify(d1, d2)
        self.valid_in, self.valid_out = self.batchify(d3, d4)
        self.test_in, self.test_out = self.batchify(d5, d6)

        self.model = DFNet(self.test_in.shape[3], args.classes).to(self.device)
        print(self.model)

        self.learning_rate = args.lr
        self.epochs = args.epochs
        self.save_path = args.save
        self.log_path = args.log
        self.positive_label = args.positive_label
        self.num_classes = args.classes

        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.learning_rate)
        # self.scheduler = ExponentialLR(self.optimizer, gamma=0.5)

    def batchify(self, data_in, data_out):
        nbatch = data_out.shape[0] // self.batch_size
        data_in = data_in.narrow(0, 0, nbatch * self.batch_size)
        data_in = data_in.view(nbatch, self.batch_size, data_in.shape[1], data_in.shape[2]).contiguous()
        data_out = data_out.narrow(0, 0, nbatch * self.batch_size)
        data_out = data_out.view(nbatch, self.batch_size).contiguous()
        return data_in.to(self.device), data_out.to(self.device)

    def evaluate(self, testing=False):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total = 0
        recall = Recall(num_classes=self.num_classes, average=None).to('cuda')
        accuracy = Accuracy().to('cuda')
        precision = Precision(num_classes=self.num_classes, average=None).to('cuda')
        X = self.test_in if testing else self.valid_in
        y = self.test_out if testing else self.valid_out
        with torch.no_grad():
            for batch in tqdm(range(len(X))):
                data = X[batch]
                target = y[batch]
                output = self.model(data)
                # print(output[0])
                loss = self.loss_func(output, target)
                total_loss += loss.item()

                # prediction = torch.argmax(output, 1)
                # total_correct += (prediction == target).sum().float()
                # total += self.batch_size
                accuracy.update(output, target)
                recall.update(output, target)
                precision.update(output, target)

        return total_loss / len(X), accuracy.compute(), \
               recall.compute()[self.positive_label], precision.compute()[self.positive_label]

    def train_once(self):
        self.model.train()
        cur_loss = 0
        bar = tqdm(range(len(self.train_in)))
        for batch in bar:
            data = self.train_in[batch]
            target = self.train_out[batch]
            self.model.zero_grad()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_func(output, Variable(target))
            loss.backward()
            cur_loss += loss.item()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

            if batch % 50 == 0:
                bar.set_postfix(current_loss=cur_loss / 50,
                                learning_rate=self.optimizer.state_dict()['param_groups'][0]['lr'])
                cur_loss = 0
                # print(output)

    def train(self):
        best_val_loss = -1
        best_accuracy = 0
        for epoch in range(1, self.epochs + 1):
            print(f'Start epoch {epoch}')
            time.sleep(0.01)
            start_time = time.time()
            self.train_once()
            val_loss, val_acc, val_recall, val_precision = self.evaluate()
            print(('end of epoch {:3d} | time: {:5.2f}s | valid loss {:8.5f} |' +
                   ' valid accuracy {:5.2f}% | valid recall {:5.2f}% | valid precision {:5.2f}%').format(
                      epoch, time.time() - start_time, val_loss, val_acc * 100, val_recall * 100, val_precision * 100))
            print('-' * 89)

            if best_val_loss == -1 or val_loss < best_val_loss:
                with open(self.save_path, 'wb') as f:
                    torch.save(self.model, f)
                best_val_loss = val_loss
                best_accuracy = val_acc

            # self.scheduler.step()
        print('Loading best model')
        self.model = torch.load(self.save_path)
        test_loss, test_acc, test_recall, test_precision = self.evaluate(True)
        print(f'Test loss: {test_loss} | accuracy: {test_acc} | recall {test_recall} | precision {test_precision}')
        if self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(f'{self.dataset_args[0]}, {self.dataset_args[2]}, {self.epochs},'
                        f' {self.learning_rate}, {test_loss}, {test_acc}, {test_recall}, {test_precision}\n')


if __name__ == '__main__':
    args_ = parser.parse_args()
    trainer = Trainer(args_)
    trainer.train()
