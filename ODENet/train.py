import numpy as np
from csv_parser_v2 import extract_features
from data import make_label_map, split_by_flow, build_blocks #,undersample

def main(n=20,seed=3):
    flows = extract_features('csv_file.csv') #TODO: import all the flows for data training
    label_map = make_label_map(flows)
    n_classes = len(label_map)
    print("classes:", label_map)
    train_flows, test_flows = split_by_flow(flows, 0.2, seed) #splitting training set and test set from the csv file
    #Xtr, ytr = undersample(*build_blocks(train_flows, n, label_map=label_map), seed=seed)
    xs, ys = build_blocks(train_flows, n, label_map=label_map)

if __name__ == "__main__":
    main()
