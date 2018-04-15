(export CUDA_VISIBLE_DEVICES=0 && python train.py --bitrate 8 --setting 0 > logs/bitrate_0_8.log)&
(export CUDA_VISIBLE_DEVICES=1 && python train.py --bitrate 16 --setting 0 > logs/bitrate_0_16.log)&
(export CUDA_VISIBLE_DEVICES=2 && python train.py --bitrate 32 --setting 0 > logs/bitrate_0_32.log)&
(export CUDA_VISIBLE_DEVICES=3 && python train.py --bitrate 64 --setting 0 > logs/bitrate_0_64.log)&
(export CUDA_VISIBLE_DEVICES=4 && python train.py --bitrate 96 --setting 0 > logs/bitrate_0_96.log)&
(export CUDA_VISIBLE_DEVICES=5 && python train.py --bitrate 128 --setting 0 > logs/bitrate_0_128.log)&
