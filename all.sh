#当前路径 AIC/ 数据清洗用到的预训练模型会自动从modelscope(国内)下载
python data_cleaner.py --raw_path ./data/webfg400/train --refined_path ./data/webfg400/threshold/refined --discard_path ./data/webfg400/threshold/discard \
--mode threshold --n 20 --batch_size 8 --device cuda:0 --threshold 0.5

python data_cleaner.py --raw_path ./data/webfg5000/train --refined_path ./data/webfg5000/threshold/refined --discard_path ./data/webfg5000/threshold/discard \
--mode threshold --n 20 --batch_size 8 --device cuda:0 --threshold 0.8

#当前路径 AIC/
python data_splite.py --raw_dir ./data/webfg400/threshold/refined --train_dir ./data/webfg400/threshold/train --val_dir ./data/webfg400/threshold/val --split_ratio 0.8

python data_splite.py --raw_dir ./data/webfg5000/threshold/refined --train_dir ./data/webfg5000/threshold/train --val_dir ./data/webfg5000/threshold/val --split_ratio 0.8

#当前路径 AIC/
cd code
python train.py --model_name SimpleNet --batch_size 64 --lr 0.0005 --epochs 100 --val_steps 0 --experiment_name webfg400 \
--train_dir ../data/webfg400/threshold/train --val_dir ../data/webfg400/threshold/val --device 'cuda:0'

python train.py --model_name SimpleNet --batch_size 64 --lr 0.0005 --epochs 20 --val_steps 0 --experiment_name webfg5000 \
--train_dir ../data/webfg5000/threshold/train --val_dir ../data/webfg5000/threshold/val --device 'cuda:0'

python predict.py --model_path ../log/outputs/webfg400/best_model.pth --class_mapping ../log/outputs/webfg400/class_mapping.json \
--test_dir ../data/webfg400/test_B --output_csv ../log/results/pred_results_web400.csv --batch_size 64 --device 'cuda:0'

python predict.py --model_path ../log/outputs/webfg5000/best_model.pth --class_mapping ../log/outputs/webfg5000/class_mapping.json \
--test_dir ../data/webfg5000/test_B --output_csv ../log/results/pred_results_web5000.csv --batch_size 64 --device 'cuda:0'
