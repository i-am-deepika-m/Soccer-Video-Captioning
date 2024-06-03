Temporally-Aware Feature Pooling for Dense Video Captioning in Video Broadcasts
This the code for the paper SoccerNet-Caption: Dense Video Captioning for Soccer Broadcasts Commentaries (CVSports2023). The training is divided in two phase : spotting training phase and captioning training phase.

Create Environment
conda create -y -n soccernet-DVC python=3.8
conda activate soccernet-DVC
conda install -y pytorch torchvision torchtext pytorch-cuda -c pytorch -c nvidia
pip install SoccerNet matplotlib scikit-learn spacy wandb
pip install git+https://github.com/Maluuba/nlg-eval.git@master
python -m spacy download en_core_web_sm
pip install torchtext
Download weights

mkdir models

Download and extract in the folder models

Train the model
python src/main.py --SoccerNet_path=path/to/SoccerNet/ --model_name=new_model --features=baidu_soccer_embeddings.npy --framerate 1 --pool=NetVLAD --window_size_caption 45 --window_size_spotting 15 --NMS_window 30 --num_layers 4 --first_stage caption --pretrain --GPU 0
Replace path/to/SoccerNet/ with a local path for the SoccerNet dataset. If you do not have a copy of SoccerNet, this code will automatically download SoccerNet.

Inference
python src/main.py --SoccerNet_path=path/to/SoccerNet/ --model_name=baidu-NetVLAD-pretrain-caption --features=baidu_soccer_embeddings.npy --framerate 1 --pool=NetVLAD --window_size_caption 45 --window_size_spotting 15 --NMS_window 30 --num_layers 4 --first_stage caption --pretrain --GPU 0 --test_only



