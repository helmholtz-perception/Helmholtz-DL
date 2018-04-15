(export CUDA_VISIBLE_DEVICES=1 && python train.py --setting 1 --quality 25 > logs/setting_1.log)&
(export CUDA_VISIBLE_DEVICES=2 && python train.py --setting 2 --quality 25 > logs/setting_2.log)&
(export CUDA_VISIBLE_DEVICES=3 && python train.py --setting 3 --quality 25 > logs/setting_3.log)&
(export CUDA_VISIBLE_DEVICES=4 && python train.py --setting 4 --quality 25 > logs/setting_4.log)&
(export CUDA_VISIBLE_DEVICES=5 && python train.py --setting 5 --quality 25 > logs/setting_5.log)&
