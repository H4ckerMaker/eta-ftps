# Encrypted Traffic Analysis for FTPS connection
## Short Description of the Project/ Documentation of the Project
The goal of our project is to classify encrypted network traffic of different protocols such as file transfers, browsing, chatting and streaming (videos and audio recordings). This was done by capturing multiple files of each protocol and thereafter training and testing two different models (one simpler and one more complex).  

The parameters used for classification were selected by reading previous research and reports at KTH Library regarding the subject and choosing the parameters used in the papers most applicable to our project. Information regarding the previous research used for the project can be found further down in the README. 

The data captured with the audio files were collected by having audio files on a server, then listening to the audio files on the computer whilst recording the network traffic on the server and saving it in .pcap files. 

With the complex model, we achieved an accuracy of 90% on the data provided by us. 
## How to Run the Code/ Documentation of Testing the Project

### Build and test on the Random Forest Classification model

TODO

## ODENet: build, train and test

### Requirements

- Python **3.11 – 3.13**. Very recent Python releases may not have PyTorch wheels yet; in that case `pip` fails with `No matching distribution found for torch`.
- `requirements.txt` must include `torch`, `numpy` and `scapy` (scapy is used to read PCAP files).

### Dataset layout

Put the captures in one folder per class:

```
dataset/
├── BROWSING/        *.pcap
├── CHAT/            *.pcap
├── FILE_TRANSFER/   *.pcap
└── STREAMING/       *.pcap
```

The folder name is the class label.

### 1. Set up the environment

```bash
cd ODENet
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.txt
```

### 2. Convert the PCAP files to CSV

```bash
.venv/bin/python3 pcap_to_csv.py dataset/CHAT --name my_chat
.venv/bin/python3 pcap_to_csv.py dataset/STREAMING --name my_streaming
.venv/bin/python3 pcap_to_csv.py dataset/BROWSING --name my_browsing
.venv/bin/python3 pcap_to_csv.py dataset/FILE_TRANSFER --name my_filetransfer
```

Each CSV row is one unidirectional flow (5-tuple). The files are written to `classes_csvs/<class>/reg/<name>.raw.csv`.

- Flows shorter than 20 packets are discarded (`--min-packets`).
- An existing CSV is not overwritten unless you pass `--force`.

### 3. Train the model

```bash
.venv/bin/python3 train.py
```

`train.py` loads every `classes_csvs/*/reg/my_*.raw.csv`, splits the flows 80/20 (train/test), balances the training set and trains for 100 epochs. At the end it prints the macro-F1 and the confusion matrix.

The best checkpoint is saved as `best_<model>_n<n>.pt` (e.g. `best_ode_n20.pt` for the default ODE-LSTM). The model is selected with `model_name` in `main()`: `ode`, `lstm_size` or `lstm_both`.

The class order is alphabetical and is printed at the start of training (`classes: {...}`). You need it in the next step.

### 4. Classify an external capture

```bash
.venv/bin/python3 predict.py capture.pcap --ckpt best_ode_n20.pt --classes BROWSING,CHAT,FILE_TRANSFER,STREAMING
```

The script prints the longest flows with their predicted class and confidence, and a summary with the share of traffic per class.

- `--classes` must follow the training order. It can be omitted if the checkpoint already stores the class names.
- `--n` must match the value used in training (default 20).
- A `.csv` file in the same format as the training CSVs is also accepted as input.

> **Note:** a capture must be converted the same way as the training data (same packet-size definition and protocols), otherwise the predictions degrade. The model always picks one of its own classes, so unrelated traffic (DNS, telemetry) is also assigned a class.

## Previous Research
Previous research regarding encrypted network traffic analysis has shown that using packet sizes and inter-arrival times as classification parameters only required 10-30 packets and resulted in sufficient classification accuracy (Roy, Shapira & Shavitt, 2022). Similarly, another research investigates traffic classification regarding flow features including packet sizes, inter-arrival times and packet directions (Lu, G. et al., 2012). 
## How It Applies to Our Project
Since the collected data from the FTPS traffic for this project is encrypted, the classification cannot be done using the payload but must instead rely on the characteristic of the traffic. Therefore, the aim of this project is to extract packet sizes and inter-arrival times as the features to be analysed. Although the previous research do not directly focus on FTPS, they still provide guidance regarding what features to extract for the classification. 
Packet size is relevant to this project since the same file size may still have packets of different sizes. Therefore, the distribution of packet sizes might provide necessary information in contrast to the total size of the file. Inter-arrival time, the time interval between the packets, may also record differences between the files and make it possible to differentiate the files.
The packet sizes and inter-arrival times will thus be the features extracted from the FTPS interaction to be used for the classification. 
## References
Roy, S., Shapira, T., & Shavitt, Y. (2022). Fast and lean encrypted Internet traffic classification. Computer Communications, 186, 166–173. https://doi.org/10.1016/j.comcom.2022.02.003

Lu, G. et al. (2012) ‘Comparison and Analysis of Flow Features at the Packet Level for Traffic Classification’, in 2012 International Conference on Connected Vehicles and Expo (ICCVE). [Online]. December 2012 IEEE. pp. 262–267.

