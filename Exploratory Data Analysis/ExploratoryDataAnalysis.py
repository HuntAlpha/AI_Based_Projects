import pandas as pd

url = "https://raw.githubusercontent.com/arora123/Data/master/Automobile_data.csv"
df = pd.read_csv(url)
df.head()

print("\nChecking for missing values:")
print(df.isnull().sum())

df_cleaned = df.dropna().copy()
print("\nDataset before dropping the missing values:")
print(df.shape)
print("\nShape of the dataset after dropping rows with missing values:")
print(df_cleaned.shape)

df_cleaned.loc[:, 'price'] = pd.to_numeric(df_cleaned['price'], errors='coerce')

min_price = df_cleaned['price'].min()
max_price = df_cleaned['price'].max()

print("\nCar(s) with the minimum price:")
print(df_cleaned[df_cleaned['price'] == min_price][['price']])
print((df_cleaned[df_cleaned['price'] == min_price][['price','company','index']]).value_counts())

print("\nCar(s) with the maximum price:")
print(df_cleaned[df_cleaned['price'] == max_price][['price']])
print((df_cleaned[df_cleaned['price'] == max_price][['price','company','index']]).value_counts())

print("\nNumber of different body styles in each and every company given in data set")
distinct_body_styles = df_cleaned.groupby('company')['body-style'].nunique()
print(distinct_body_styles)

avg_price_by_body_style = df_cleaned.groupby('body-style')['price'].mean()
avg_price_by_body_style_sorted = avg_price_by_body_style.sort_values(ascending=False)

print("\nAverage price of all body types of cars (sorted in descending order):")
print(avg_price_by_body_style_sorted)

avg_price_by_company = df_cleaned.groupby('company')['price'].mean()
avg_price_by_company_sorted = avg_price_by_company.sort_values(ascending=False)

print("\nAverage price of cars by different companies (sorted in descending order):")
print(avg_price_by_company_sorted)

print("\nCount of car types with different numbers of cylinders:")
print(df_cleaned['num-of-cylinders'].value_counts())

import matplotlib.pyplot as plt
avg_price_by_company = df_cleaned.groupby('company')['price'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
avg_price_by_company.plot(kind='bar', color='#734F96')
plt.title('Average Price of Cars by Company')
plt.xlabel('Company')
plt.ylabel('Average Price')
plt.xticks(rotation=90)
plt.show()

cylinders_by_company = df_cleaned.groupby('company')['num-of-cylinders'].value_counts().unstack().fillna(0)

cylinders_by_company.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='inferno')
plt.title('Distribution of Number of Cylinders by Company')
plt.xlabel('Company')
plt.ylabel('Number of Cars')
plt.xticks(rotation=90)
plt.show()

avg_price_by_body_style = df_cleaned.groupby('body-style')['price'].mean().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
avg_price_by_body_style.plot(kind='bar', color='#734F96')
plt.title('Average Price of Cars by Body Type')
plt.xlabel('Body Type')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.show()

cylinders_by_body_style = df_cleaned.groupby('body-style')['num-of-cylinders'].value_counts().unstack().fillna(0)

cylinders_by_body_style.plot(kind='bar', stacked=True, figsize=(8, 5), colormap='plasma')
plt.title('Number of Cylinders by Body Style')
plt.xlabel('Body Style')
plt.ylabel('Number of Cars')
plt.xticks(rotation=45)
plt.show()
