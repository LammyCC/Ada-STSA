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

## 📦 Preprocessed Data (Optional)

For quick start, preprocessed intermediate variables are available for download:

**Download Link:** https://pan.baidu.com/s/1B-fOfhN6DA2V7KqNAKF8mA  
**Access Code:** 2u13

Extract the downloaded `data` folder to the project root directory. This allows you to skip the preprocessing steps and directly run `python run_main.py` for model training and testing.
