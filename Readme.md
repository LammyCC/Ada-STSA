## 📊 Datasets

### Data Download

This project uses two public datasets:

1. **Qingdao Dataset**
   - Download Link: https://aistudio.baidu.com/datasetdetail/27526
   - Description: Contains traffic data and taxi trajectory data from Qingdao city

2. **Chengdu Dataset**
   - Download Link: https://www.pkbigdata.com/common/zhzgbCmptDataDetails.html
   - Description: Contains taxi trajectory data from Chengdu city


## 🚀 Quick Start

### Dependencies

```bash
pip install -r requirements.txt
```

### Execution Steps

Execute the programs in the following order:

#### 1. Data Preprocessing
```bash
python run_preprocessing.py
```
**Functions:**
- Convert and standardize raw traffic flow data
- Process taxi trajectory data and extract spatio-temporal features
- Generate data formats required for model training
- Output: Processed traffic flow data and trajectory data

#### 2. Micro-level Graph Construction
```bash
python run_micro_graph.py
```
**Functions:**
- Extract semantic representations and spatial correlations of trajectories
- Generate adjacency matrices and node embeddings for micro-level graphs
- Output: Micro-level semantic representation files

#### 3. Model Training and Prediction
```bash
python run_main.py
```
**Functions:**
- Load preprocessed data and micro-level semantic representations
- Training
- Output: Trained model and prediction results

## Cite
@article{CAI2026122819,
title = {Semantic-aware adaptive traffic flow prediction driven by real-time trajectories},
journal = {Information Sciences},
volume = {728},
pages = {122819},
year = {2026},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2025.122819},
author = {Shijie Cai and Jie Hu and Min Wei and Xiao Zhang}
}


