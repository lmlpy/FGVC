#当前路径 AIC/
cd code
python train.py --model_name SimpleNet --batch_size 64 --lr 0.0005 --epochs 100 --val_steps 0 --experiment_name webfg400 \
--train_dir ../data/webfg400/threshold/train --val_dir ../data/webfg400/threshold/val --device 'cuda:1'

python train.py --model_name SimpleNet --batch_size 64 --lr 0.0005 --epochs 20 --val_steps 0 --experiment_name webfg5000 \
--train_dir ../data/webfg5000/threshold/train --val_dir ../data/webfg5000/threshold/val --device 'cuda:1'

python predict.py --model_path ../log/outputs/webfg400/best_model.pth --class_mapping ../log/outputs/webfg400/class_mapping.json \
--test_dir ../data/webfg400/test_B --output_csv ../log/results/pred_results_web400.csv --batch_size 64 --device 'cuda:1'

python predict.py --model_path ../log/outputs/webfg5000/best_model.pth --class_mapping ../log/outputs/webfg5000/class_mapping.json \
--test_dir ../data/webfg5000/test_B --output_csv ../log/results/pred_results_web5000.csv --batch_size 64 --device 'cuda:1'
