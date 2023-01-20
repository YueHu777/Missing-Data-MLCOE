# Missing-Data-MLCOE
## Part 1 a
To run the SSSD model, please run 
```
python3 train_SSSD.py
```
To run the CSDI model, please run 
```
python3 train_CSDI.py
```
## Part 1 b-f
Please first run get_stock_data.ipynb under the /datasets folder in part1b-f.
After getting the data, run
```
# for SSSD
python3 train_SSSD.py
# for CSDI
python3 train_CSDI.py
```
The full data sample requires GPU and a large memory. So I provided a small demo here which consists only 9 stocks. ( Three from each market.)\\
In order to run the full sample, please set 
```
"in_channels": 133,
"out_channels":133,
"train_data_path": "./datasets/train_data_ms.npy"
```
in config_SSSDS4.json.
## Part 1 g
Please first run get_stock_data.ipynb under the /datasets folder in part1g.
After getting the data, run
```
# for SSSD
python3 train_SSSD.py
# for CSDI
python3 train_CSDI.py
```
The full data sample requires GPU and a large memory. So I provided a small demo here which consists only 9 stocks. (Same as part1b-f)\\
In order to run the full sample, please set 
```
"in_channels": 133,
"out_channels":133,
"train_data_path": "./datasets/Changewindow_train.npy"
```
in config_SSSDS4.json.

