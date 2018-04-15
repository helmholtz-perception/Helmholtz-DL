import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--bitrate', type=int, default=128, help='Audio bitrate [default: 128kb/s]')
parser.add_argument('--setting', type=int, default=0, help='Model architecture (0-5) [default: 0]')
parser.add_argument('--all', type=bool, default=False, help='Collect all models [default: False]')

FLAGS = parser.parse_args()

bitrate = FLAGS.bitrate
setting = FLAGS.setting
ALL = FLAGS.all

model_list = ["cnn_x4", "cnn_x3", "cnn_x2", "cnn_x3_mlp_0", "cnn_x3_mlp_64_128", "cnn_x3_mlp_128x2"]

def collect(bitrate=128, setting=0):
    model = model_list[setting]
    history_path = os.path.join(model, "bitrate_" + str(bitrate) + "/history.csv")
    f = open(history_path, "r")
    acc_list = []
    for i, line in enumerate(f.readlines()):
        if i == 0:
            continue
        else:
            acc_list.append(float(line.split(",")[3]))
    max_acc = max(acc_list)
    print max_acc, max_acc * 0.985
    for i, acc in enumerate(acc_list):
        if sum(acc_list[i:i+10]) / 10.0 > 0.99 * max_acc:
            print i
            break

if __name__ == "__main__":
    if not ALL:
        collect(bitrate, setting)
    else:
        for bitrate in [8, 16, 32, 64, 96, 128]:
            collect(bitrate, setting)

