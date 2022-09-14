"""
首先使用restart.sh启动靶场环境，等待各节点就绪（可运行util/get_consensus.py确定是否就绪）
以随机间隔使用telnet向C&C服务器发送随机地攻击指令
同时每60s发送signal newnym重置circuit，然后启动5个bot，同时启动tcpdump抓这60s内的流量，作为一个流
"""
import signal
import string
import subprocess
import telnetlib
import time
import sys
import random
import threading
import os
import docker

import stem
from stem.control import Controller


PCAP_COUNT = 0


class BotLoop(threading.Thread):
    def __init__(self, docker_name, times, duration):
        super(BotLoop, self).__init__()
        self.p = None
        self.times = times
        self.duration = duration
        self.docker_name = docker_name
        client = docker.DockerClient()
        container = client.containers.get(self.docker_name)
        self.ip = container.attrs['NetworkSettings']['Networks']['private-tor-network_default']['IPAddress']
        self.controller = Controller.from_port(address=self.ip)
        self.controller.authenticate(password='password')

    def open(self, count):
        self.controller.signal('NEWNYM')

        # print(container.attrs)

        start_tcpdump(f'{self.docker_name}_{count}', self.duration + 2, self.ip)
        time.sleep(1)
        cmd = ['docker', 'exec', '-it', self.docker_name,
               'timeout', str(self.duration), './runbot.sh']
        print("Run cmd:", cmd)
        # self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stdout)
        self.p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)

    def close(self):
        if self.p is not None:
            os.kill(self.p.pid, signal.SIGINT)
            time.sleep(1)
            self.p = None

    def run(self):
        for i in range(self.times):
            self.open(i)
            time.sleep(self.duration + 1)
            # self.close()


def close_all_circuit(con):
    circuits = con.get_info('circuit-status').split('\n')
    for circuit in circuits:
        cid = circuit.split(' ')[0]
        if cid:
            print(f'Closing {cid}')
            con.close_circuit(cid)


def start_tcpdump(name, duration, ip):
    cmd = ['docker', 'exec', 'private-tor-network_da1_1', 'timeout', str(duration), 'tcpdump',
           '-w', f'tor/pcaps/{name}.pcap', 'host', ip, 'and', 'tcp']
    print("Run cmd:", cmd)
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)


def send_random_attack():
    tn = telnetlib.Telnet('172.20.0.15')
    tn.write(b'root\n')
    print(tn.read_until(b'User'))
    tn.write(b'root\n')
    print(tn.read_until(b'Pass'))
    tn.write(b'root\n')
    print(tn.read_until(b'#'))

    attack = random.choice(['http', 'udp', 'dns', 'ack', 'syn'])
    ip = '.'.join([str(random.randint(1, 255)) for i in range(4)])
    duration = random.randint(3, 15)
    dport = random.randint(1000, 60000)

    if attack == 'http':
        method = random.choice(['get', 'post'])
        path = '/' + ''.join(random.sample(string.ascii_letters + string.digits, 16))
        cmd = f'{attack} {ip} {duration} dport={dport} method={method} domain={ip} path={path}\n'

    else:
        sport = random.randint(1000, 60000)
        cmd = f'{attack} {ip} {duration} dport={dport} sport={sport}\n'

    print(f"Sending {cmd}")
    tn.write(cmd.encode())
    print(tn.read_very_eager())
    tn.write(b'exit\n')
    tn.close()
    return duration + 1


def attack_loop(duration):
    while duration > 0:
        d = send_random_attack()
        duration -= d
        time.sleep(d)


if __name__ == '__main__':

    # close_all_circuit(controller)
    # print('circuit-status:', controller.get_info('circuit-status'))
    run_time = 24 * 60 * 60
    for i in range(1, 6):  # 5个bot的重启时间分别为30,60,180,600,3600s
        d = [30, 60, 180, 600, 3600][i-1]
        BotLoop(f'private-tor-network_client_{i}', run_time // d, d).start()
    time.sleep(5)
    attack_loop(run_time)

