# Encrypted Traffic Analysis for FTPS connection
## Short Description of the Project/ Documentation of the Project
 
This project investigates **metadata leakage from encrypted network traffic**. The main question is:
 
> Can a passive observer infer what a user is doing, even when the content of the communication is encrypted?
 
To answer it, we defined a small classification problem with four traffic categories:
 
| Category | Examples |
|----------|----------|
| Streaming | Video and audio streaming |
| Web browsing | Regular page browsing |
| File transfer | File uploads and downloads (e.g. scp, ftps)|
| Messaging | Chat applications (e.g. WhatsApp, Telegram, Discord) |
 
Application payloads are **never decrypted**. Classification relies only on metadata that is visible to an observer on the network.
 
## Methodology
 
### 1. Data collection
 
For each category we captured repeated experiments and saved the traffic as `.pcap` files.
 
For the **streaming** class, TODO

For the **file transfer** class, we used a script to create and upload/download random files to our server using different protocols such as scp and ftps.

For the **messanging** class, we used tcpdump to record different sessions of chatting using chat web apps such as: discord, whatsapp and telegram.

For the **web browsing**, TODO
 
### 2. Baseline: Random Forest
TODO
 
### 3. Limitations of the baseline
TODO

### 4. A better-suited model? ODE-LSTM
 
After reviewing the literature (including reports available through the KTH Library), we adopted the approach proposed by Roy, Shapira and Shavitt in *"Fast and lean encrypted Internet traffic classification"* (Computer Communications, 2022).
 
Instead of aggregated statistics, this method uses only **two per-packet features**:
 
- packet size
- inter-arrival time

These are fed as a time series to a **Neural ODE combined with an LSTM**. The ODE uses the inter-arrival times to model how the hidden state of the flow evolves between packets, which makes the model well suited to irregularly spaced events such as network traffic. 
 
### 5. Comparison
 
Both models were trained and evaluated on the same data and compared on previously unseen traces.
 
## Results
 
| Model | Accuracy |
|-------|----------|
| Random Forest (baseline) | _[add value]_ |
| ODE-LSTM | **~90%** |
 
The accuracy of both the ODE-LSTM model and the Random Forest model is measured on our own dataset.
 
### Known difficulties
 
 
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

## References
Roy, S., Shapira, T., & Shavitt, Y. (2022). Fast and lean encrypted Internet traffic classification. Computer Communications, 186, 166–173. https://doi.org/10.1016/j.comcom.2022.02.003

Lu, G. et al. (2012) ‘Comparison and Analysis of Flow Features at the Packet Level for Traffic Classification’, in 2012 International Conference on Connected Vehicles and Expo (ICCVE). [Online]. December 2012 IEEE. pp. 262–267.

