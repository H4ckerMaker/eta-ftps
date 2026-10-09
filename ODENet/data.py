import numpy as np

MAX_SIZE = 1500.00

def make_label_map(flows):
    classes = sorted({f["label"] for f in flows}, key=str)
    return {c: i for i, c in enumerate(classes)}

def split_by_flow(flows, test_threshold=0.2, seed=0):
    #This function generate a random permutation of flows and split that permutation in 2 separate sets.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(flows))
    k = int(len(flows) * (1-test_threshold))
    return [flows[i] for i in order[:k]], [flows[i] for i in order[k:]]


def build_blocks(flows, n=20, stride=None, label_map=None):
    stride = stride or n #if stride is set take it otherwise take n (if stride < n it cause overlapping blocks)
    xs, ys = [], []
    for f in flows:
        t = [float(x) for x in f["times"]]
        s = [float(x) for x in f["sizes"]]
        if len(t) != len(s):
            continue #inconsistent flow
        for start in range(0, len(t) - n+1, stride): #set the start for each block based on the stride
            tb = t[start:start + n] #create a timestamp block starting from start and finishing at start + n
            sb = s[start:start + n] #create a sizes block starting from start and finishing at start + n
            block = np.stack([np.minimum(sb, MAX_SIZE) / MAX_SIZE, tb], axis=1) #normalizing every size in the block using 1500 as a MAX SIZE and stacking sizes with timestamps
            xs.append(block)
            ys.append(label_map[f["label"]] if label_map else f["label"])
    if not xs:
        raise ValueError("No blocks generated: check n and flow length")
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64) #generate an a (N, n, 2) matrix for xs and a (N,) matrix for ys
