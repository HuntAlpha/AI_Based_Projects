#importing the necessary libraries
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Versions for Tensorflow and decision forests
print("Tensorflow v", + tf.__version__)
print("Tensorflow decision forests v", + tfdf.__version__)

#Loading the Dataset
dataset_df = pd.read_csv("spaceship_titanic_dataset.csv")
print("Full dataset shape: ".format(dataset_df.shape))

#Display first 5 dataset columns
dataset_df.head(5)

#Dataset Exploration
dataset_df.describe()
dataset_df.info()

#Bar Chart for label column: Transported
plot_df = dataset_df.Transported.value_counts()
plot_df.plot(kind="bar")

#Numerical data distribution
fig, ax = plt.subplots(5,1, figsize=(10,10))
plt.subplots_adjust(top=2)

sns.histplot(dataset_df['Age'], color='b', bins=50, ax=ax[0])
sns.histplot(dataset_df['FoodCourt'], color='b', bins=50, ax=ax[1])
sns.histplot(dataset_df['ShoppingMall'], color='b', bins=50, ax=ax[2])
sns.histplot(dataset_df['Spa'], color='b', bins=50, ax=ax[3])
sns.histplot(dataset_df['VRDeck'], color='b', bins=50, ax=ax[4])

#Preparing the dataset
dataset_df = dataset_df.drop(['PassengerId', 'Name'], axis=1) #we drop passengerid and name as they are not relevant
dataset_df.head(5)

dataset_df.isnull().sum().sort_values(ascending=False)

#Changing boolean values into numerical values(null - zero)
dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)
dataset_df.isnull().sum().sort_values(ascending=False)

label = "Transported"
dataset_df[label] = dataset_df[label].astype(int)

dataset_df['VIP'] = dataset_df['VIP'].astype(int)
dataset_df['CryoSleep'] = dataset_df['CryoSleep'].astype(int)

dataset_df[["Deck", "Cabin_num", "Side"]] = dataset_df["Cabin"].str.split("/", expand=True)

try:
    dataset_df = dataset_df.drop('Cabin', axis=1)
except KeyError:
    print("Field does not exist")

dataset_df.head(5)

#Split dataset into training and testing datasets
def split_dataset(dataset, test_ratio=0.20):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples in testing.".format(
    len(train_ds_pd), len(valid_ds_pd)))

#We need to convert the datatset from Pandas to TensorFlow Datasets format to train the model
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label)

#list the all the available models in TensorFlow Decision Forests
tfdf.keras.get_all_models()

#Configure the model
rf = tfdf.keras.RandomForestModel(hyperparameter_template="benchmark_rank1")

#Create a Random Forest
rf = tfdf.keras.RandomForestModel()
rf.compile(metrics=["accuracy"]) # Optional, you can use this to include a list of eval metrics

#Train the model
rf.fit(x=train_ds)

#Visualize the model
tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)


#Evaluate the model on the Out of bag (OOB) data and the validation dataset
import matplotlib.pyplot as plt
logs = rf.make_inspector().training_logs()
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")
plt.show()

#General stats on OBB data
inspector = rf.make_inspector()
inspector.evaluation()

#Running an evaluation 
evaluation = rf.evaluate(x=valid_ds,return_dict=True)

for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")

# Load the test dataset
test_df = pd.read_csv('test.csv')
submission_id = test_df.PassengerId

# Replace NaN values with zero
test_df[['VIP', 'CryoSleep']] = test_df[['VIP', 'CryoSleep']].fillna(value=0)

# Creating New Features - Deck, Cabin_num and Side from the column Cabin and remove Cabin
test_df[["Deck", "Cabin_num", "Side"]] = test_df["Cabin"].str.split("/", expand=True)
test_df = test_df.drop('Cabin', axis=1)

# Convert boolean to 1's and 0's
test_df['VIP'] = test_df['VIP'].astype(int)
test_df['CryoSleep'] = test_df['CryoSleep'].astype(int)

# Convert pd dataframe to tf dataset
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)

# Get the predictions for testdata
predictions = rf.predict(test_ds)
n_predictions = (predictions > 0.5).astype(bool)
output = pd.DataFrame({'PassengerId': submission_id,'Transported': n_predictions.squeeze()})

output.head() 
  

sample_submission_df = pd.read_csv('sample_submission.csv')
sample_submission_df['Transported'] = n_predictions
sample_submission_df.to_csv('submission.csv', index=False)
sample_submission_df.head()