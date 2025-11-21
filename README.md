Distributed Deep Learning Training project

Go to configs folder and run:
chmod +x set_up_env.sh
./set_up_env.sh

final_project/
├── configs/ # Set up the environment
├── data/ # Dataset directory  
├── results/ # Training results
│ ├── figures/ # Generated plots an visualizations
│ └── logs/ # Training logs
├── src/ # Source code
├── \*.py # Training scripts
└── README.md

# if environment does not auto activate please run

source csc_420_env/bin/activate

Pipeline Parallelism Scripts
ResNet Models (2 GPU & 4 GPU variants):

\*\_resnet18.py - ResNet18 model

\*\_resnet34.py - ResNet34 model

\*\_resnet50.py - ResNet50 model

\*\_resnet152.py - ResNet152 model

python [2gpu|4gpu]\_train_pipeline_resnet[18|34|50|152].py --global-batch <BATCH_SIZE> --m <MICRO_BATCHES> --epochs <EPOCHS>

--global-batch: 128, 256, 512, 1024, 2048

--m (micro-batches): 4 or 8

--epochs: Number of training epochs

torchrun --nproc_per_node=<NUM_GPUS> ddp_train_vit_l16.py --global-batch <BATCH_SIZE> --epochs <EPOCHS> --out "<LOG_FILE>"

Output

Training logs are saved in logs/ directory

Results stored in results/
