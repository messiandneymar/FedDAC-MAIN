📌 FedDAC Run Guide
1. Recommended Command Format
Use the following command format to train a model using the FedDAC algorithm:

bash
python main.py \
  -data <dataset_name> \
  -m <model_name> \
  -algo FedDAC \
  -gr <global_rounds> \
  -ls <local_steps> \
  -lbs <local_batch_size> \
  -nc <num_classes_per_client> \
  -tau <non_iid_factor> \
  -lr <learning_rate>
2. Example Commands
Here are some example commands for running ResNet models on popular datasets:

CIFAR-100 + ResNet10
python main.py -data Cifar100 -m ResNet10 -algo FedDAC -gr 300 -ls 5 -lbs 100 -nc 40 -tau 0.5 -lr 0.01

CIFAR-10 + ResNet10
python main.py -data Cifar10 -m ResNet10 -algo FedDAC -gr 200 -ls 5 -lbs 64 -nc 10 -tau 0.3 -lr 0.005

TinyImageNet + ResNet18
bash
python main.py -data TinyImageNet -m ResNet18 -algo FedDAC -gr 400 -ls 10 -lbs 128 -nc 100 -tau 0.7 -lr 0.01

3. Execution Steps
🔹 Step 1: Navigate to the Project Directory
Open your terminal and enter:
bash
cd system
🔹 Step 2: Run the Training Script
Example:
bash
python main.py -data Cifar100 -m ResNet10 -algo FedDAC -gr 300 -ls 5 -lbs 100 -nc 40 -tau 0.5 -lr 0.01
