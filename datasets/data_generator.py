import numpy
from flowcontainer.extractor import extract
import glob
import random
import os
import csv


def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def pcaps_to_directional_timestamps(file_glob):
    flows = []
    for filename in glob.glob(file_glob):
        res = extract(filename)
        for key in res:
            # print(key)
            val = res[key]
            flows.append([sign(a) * b for a, b in zip(val.payload_lengths, val.payload_timestamps)])
    return flows


def directional_timestamps_to_direction(flows):
    res = []
    for f in flows:
        res.append([sign(x) for x in f])
    return res


def npz_to_flows(filename):
    print('Loading npz from', filename)
    npz = numpy.load(filename)
    return npz['data'].tolist()


def split_flows(flows, min_length=32, max_length=500, step=-1):
    res_flows = []
    for flow in flows:
        length = len(flow)
        left = 0
        while length - left >= min_length:
            l = min(max_length, length - left)
            res_flows.append(flow[left: left + l])
            left += step if step > 0 else l

    for flow in res_flows:
        start_time = abs(flow[0]) - 1
        for i in range(len(flow)):
            if flow[i] > 0:
                flow[i] -= start_time
            else:
                flow[i] += start_time
    return res_flows


def create_dataset(flows, save_dir='dataset\\ISCX-Tor', num_balance=True):
    # 使各类别的数据条数相同
    min_num = 99999999
    for key in flows:
        # flows[key] = split_flows(flows[key])
        # flows[key] = feature(flows[key])
        min_num = min(min_num, len(flows[key]))
        print(f'Number of label {key}: {len(flows[key])}')
    max_len = 0
    for key in flows:
        random.shuffle(flows[key])
        if num_balance:
            flows[key] = flows[key][:min_num]
        for flow in flows[key]:
            max_len = max(max_len, len(flow))

    print(f"Padding to length {max_len} & labeling & mixing")
    mixed_flows = []
    for key in flows:
        for x in flows[key]:
            x.extend([0] * (max_len - len(x)))  # padding
            x.insert(0, key)  # label
            mixed_flows.append(x)
    random.shuffle(mixed_flows)
    mixed_len = len(mixed_flows)
    print(f"Mixed len: {mixed_len}")

    print(f"Saving to {save_dir}")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 分成train valid test
    split_num = (0.8, 0.1, 0.1)
    filename = ('train', 'valid', 'test')
    left = 0
    for i in range(3):
        l = min(int(mixed_len * split_num[i]), mixed_len - left)
        print(f'Saving {filename[i]}, count: {l}')

        with open(os.path.join(save_dir, f'{filename[i]}.csv'), 'w') as f:
            f_writer = csv.writer(f, lineterminator='\n')
            f_writer.writerows(mixed_flows[left: left + l])

        left += l


if __name__ == '__main__':
    # AWF + tor-mirai
    # create_dataset({
    #     0: npz_to_flows('AWF/tor_open_400000w.npz'),
    #     1: directional_timestamps_to_direction(split_flows(
    #         pcaps_to_directional_timestamps('tor-botnet/pcaps/private-tor-network_client_*.pcap'), max_length=512))
    # }, save_dir='tor-botnet/dataset-unbalance', num_balance=False)

    # ISCX-Tor + tor-mirai
    # create_dataset({
    #     0: split_flows(pcaps_to_directional_timestamps('ISCX-TorPcaps\\BROWSING*.pcap'), max_length=512),
    #     1: split_flows(pcaps_to_directional_timestamps('ISCX-TorPcaps\\P2P_*.pcap'), max_length=512),
    #     2: split_flows(pcaps_to_directional_timestamps('ISCX-TorPcaps\\VOIP_*.pcap'), max_length=512),
    #     3: split_flows(pcaps_to_directional_timestamps('ISCX-TorPcaps\\AUDIO_*.pcap'), max_length=512),
    #     4: split_flows(pcaps_to_directional_timestamps('ISCX-TorPcaps\\VIDEO_*.pcap'), max_length=512),
    #     5: split_flows(pcaps_to_directional_timestamps('ISCX-TorPcaps\\FILE*.pcap'), max_length=512),
    #     6: split_flows(pcaps_to_directional_timestamps('ISCX-TorPcaps\\MAIL_*.pcap'), max_length=512),
    #     7: directional_timestamps_to_direction(split_flows(
    #         pcaps_to_directional_timestamps('tor-botnet/pcaps/private-tor-network_client_*.pcap'), max_length=512))
    #     # 7: pcaps_to_directional_timestamps('dataset\\TorPcaps\\CHAT_*.pcap'),
    # }, save_dir='tor-botnet/dataset-ISCX-512', num_balance=False)

    create_dataset({
        0: directional_timestamps_to_direction(split_flows(
            pcaps_to_directional_timestamps('ISCX-TorPcaps\\*.pcap'), max_length=512)),
        1: directional_timestamps_to_direction(split_flows(
            pcaps_to_directional_timestamps('tor-botnet/pcaps/private-tor-network_client_*.pcap'), max_length=512))
    }, save_dir='tor-botnet/dataset-ISCX-512-2class', num_balance=False)
