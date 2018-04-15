import os
import glob
import subprocess

def quantitize(input_dir, bitrate=128):
    input_dir = input_dir + str(bitrate)
    train_dir = glob.glob(os.path.join(input_dir, "*"))
    print train_dir
    for path in train_dir:
        files = glob.glob(os.path.join(path, "*.mp3"))
        for wav_name in files:
            name = wav_name[wav_name.rindex("/")+1:]
            out_dir = os.path.join("Bitrate_" + str(bitrate),
                    wav_name[wav_name.index("/")+1:wav_name.rindex("/")])
            out_name = name.split(".")[0] + ".wav"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            # print "lame -b %s %s %s" % (str(bitrate), wav_name, os.path.join(out_dir, out_name))
            # os.system("lame -b %s %s %s" % (str(bitrate), wav_name, os.path.join(out_dir, out_name)))
            os.system('ffmpeg -i %s %s' % (wav_name, os.path.join(out_dir, out_name)))

bitrate_list = [160]

if __name__ == "__main__":
    for bitrate in bitrate_list:
        quantitize("bitrate_", bitrate)
        print ("bitrate_" + str(bitrate))
