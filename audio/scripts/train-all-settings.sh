(export CUDA_VISIBLE_DEVICES=1 && python train.py --setting 1 --bitrate 32 > logs/setting_1.log)&
(export CUDA_VISIBLE_DEVICES=2 && python train.py --setting 2 --bitrate 32 > logs/setting_2.log)&
(export CUDA_VISIBLE_DEVICES=3 && python train.py --setting 3 --bitrate 32 > logs/setting_3.log)&
(export CUDA_VISIBLE_DEVICES=4 && python train.py --setting 4 --bitrate 32 > logs/setting_4.log)&
(export CUDA_VISIBLE_DEVICES=5 && python train.py --setting 5 --bitrate 32 > logs/setting_5.log)&
