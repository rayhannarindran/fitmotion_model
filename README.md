# FitMotion Model - Machine Learning
FitMotion model starts off from a dataset we gathered from kaggle (https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset/code).

This dataset is collected with an iPhone 6 gyro sensor, this means that the data from our android needs to be fitted to be properly detected by the model trained from the dataset.

## Data Training
With model_training.ipynb we can define the dataset used to train the model, fitmotion initally uses all data from the dataset, but decided only to use the Gravity and Acceleration data (reduced).

```python
# change these following three lines only
reduced = True
operating_system = 'linux' #windows or linux
subject_data_file = 'data_subjects_info.csv'
```

the model is then saved into the the models folder to be inferenced.

## Data Testing
To test data collected we cab yse the model_tester.ipynb file, by first choosing which model you want to use.

```python
# Specify Model
reduced = False
model_type = 'keras' # keras or h5 or onnx
factor_index = 0
```

we then load the data to predict the movement

```python
# Load Android Data
data_source = 'android_data_latest'
if reduced:
    data_source += '_reduced'
data_type = 'wlk'
data_num = '5'
data_url = '../data/' + data_source + '/' + data_type + '/' + data_type + data_num + '-SensorData.csv'

df = pd.read_csv(data_url, sep=',')
    
df = df.drop(['Unnamed: 0'], axis=1) if 'Unnamed: 0' in df.columns else df
df = df.drop(['id'], axis=1) if 'id' in df.columns else df
df
```

after some preprocessing, the data is then passed to the train model to predic the movement which outputs as such
```
wlk
```