<img src="https://www.pngall.com/wp-content/uploads/8/Ecommerce-Retail-Business-Transparent.png">

# 📊 Project Title: Global Retail Store - RFM Analysis  
🤵 Author: Loc Huynh
📆 Date: Jan. 10, 2025  <br> 
💻 Tools Used: Python

## 📑 Table of Contents  
1. [📌 Background & Overview](#-background--overview)  
2. [📂 Dataset Description & Data Structure](#-dataset-description--data-structure)  
3. [🧠 Design Thinking Process](#-design-thinking-process)  
4. [📊 Key Insights & Visualizations](#-key-insights--visualizations)  
5. [🔎 Final Conclusion & Recommendations](#-final-conclusion--recommendations)


## 📌 Background & Overview  

### Objective:
### 📖 What is this project about? 
 
> Understanding customer behavior through **RFM (Recency, Frequency, Monetary) analysis** helps businesses make informed decisions and enhance customer loyalty. The Marketing department needs to segment customers for holiday campaigns, and the Marketing Director suggests using RFM analysis.

RFM Segmentation is a method to analyze customer behavior based on three key metrics:

- **Recency (R)**: Time since the last purchase. Recent buyers are more likely to purchase again.
- **Frequency (F)**: Number of purchases within a period. Frequent buyers are more loyal.
- **Monetary Value (M)**: Total money spent. High spenders are more valuable to the business.

### 👤 Who is this project for?  

- The Marketing Department who need to understand customer behavior and thier values for new marketing statergy. 

###  ❓Business Questions:  

- **Customer Segmentation for Marketing Campaigns**: How can the Marketing department classify customer segments effectively to deploy tailored marketing campaigns , appreciating loyal customers and attracting potential ones?
- **Implementing RFM Model**: How can the RFM (Recency, Frequency, Monetary) model be utilized to analyze and segment customers to enhance the effectiveness of marketing campaigns?

### 🎯Project Outcome:  
Summarize key findings and insights/ trends/ themes in a concise, bullet-point 

## 📂 Dataset Description & Data Structure  


### 📌 Data Source  
- Source: Kaggles  
- Size:  525461 rows × 8 columns
- Format: xlsx

### 📊 Data Structure & Relationships  

#### 1️⃣ Tables Used:  
Mention how many tables are in the dataset.  

#### 2️⃣ Table Schema & Data Snapshot  


| Fields             | DataType | Description                                                                                             |
|--------------------|----------|---------------------------------------------------------------------------------------------------------|
| InvoiceNo          | String   | Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'C', it indicates a cancellation. |
| StockCode          | String   | Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.      |
| Description        | String   | Product (item) name. Nominal.                                                                            |
| Quantity           | Integer  | The quantities of each product (item) per transaction. Numeric.                                          |
| InvoiceDate        | DateTime | Invoice Date and time. Numeric, the day and time when each transaction was generated.                     |
| UnitPrice          | Decimal  | Unit price. Numeric, Product price per unit in sterling.                                                 |
| CustomerID         | Integer  | Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.                  |
| Country            | String   | Country name. Nominal, the name of the country where each customer resides.                              |



## ⚒️ Main Process

1️⃣ Load Dataset 

```python
path = "/content/drive/MyDrive/online_retail_II.xlsx"
df = pd.read_excel(path,sheet_name=0)
df.head(10)
```
2️⃣ Exploratory Data Analysis (EDA)  
```python
df.info()
```
```python
df.describe()
```

```python
df.describe(include = 'O')
```


#### 🔍 
 We have found out that Customer ID column and Description column have NULL value. In addition, Customer ID which has null value we can see InvoiceNo has some legit which is not mentioned in Data Dictionary and Qualiaty Column has negative number


![image](https://github.com/user-attachments/assets/6e09ca51-fbc6-4b24-a211-395f89eafc1f)

```python
df[df["Customer ID"].isna()].head(10)
df[df['Quantity']<0].head(10)
df["Invoice"] = df["Invoice"].astype("str")
df[df["Invoice"].str.match("^\\d{6}$")== True]
df["Invoice"].str.replace("[0-9]","",regex= True).unique()
df[df["Invoice"].str.startswith("A")]
df[df["Invoice"].str.startswith("C")]
```
#### 🔍 
 As Discovered the dataset we know that the column Invoice contain an "A" letter that represent for "Adjust bad debt" Desciption and "C" letter represent for "Cancel" Description that lead to Quanlity Column has negative value	

```python
df["StockCode"] = df["StockCode"].astype("str")

df[(df["StockCode"].str.match("^\\d{5}$") == False) & (df["StockCode"].str.match("^\\d{5}[a-zA-Z]+$") == False)]["StockCode"].unique()
df[df["StockCode"].str.contains('^DOT')]
```
### Notes

#### Stock Code
* StockCode is meant to follow the pattern `[0-9]{5}` but seems to have legit values for `[0-9]{5}[a-zA-Z]+`
    * Also contains other values:
        | **Code**            | **Description**                                                        | **Action**              |
        |---------------------|------------------------------------------------------------------------|-------------------------|
        | DCGS            | Looks valid, some quantities are negative though and customer ID is null | Exclude from clustering |
        | D               | Looks valid, represents discount values                                | Exclude from clustering |
        | DOT             | Looks valid, represents postage charges                                | Exclude from clustering |
        | M or m          | Looks valid, represents manual transactions                            | Exclude from clustering |
        | C2              | Carriage transaction - not sure what this means                        | Exclude from clustering |
        | C3              | Not sure, only 1 transaction                                           | Exclude                 |
        | BANK CHARGES or B | Bank charges                                                        | Exclude from clustering |
        | S               | Samples sent to customer                                               | Exclude from clustering |
        | TESTXXX         | Testing data, not valid                                                | Exclude from clustering |
        | gift__XXX       | Purchases with gift cards, might be interesting for another analysis, but no customer data | Exclude |
        | PADS            | Looks like a legit stock code for padding                              | Include                 |
        | SP1002          | Looks like a special request item, only 2 transactions, 3 look legit, 1 has 0 pricing | Exclude for now|
        | AMAZONFEE       | Looks like fees for Amazon shipping or something                       | Exclude for now         |
        | ADJUSTX         | Looks like manual account adjustments by admins                        | Exclude for now         |

3️⃣ Data Cleaning
```python
cleaned_df = df.copy()
cleaned_df["Invoice"] = cleaned_df["Invoice"].astype("str")
mask = (
    cleaned_df["Invoice"].str.match("^\\d{6}$") == True
)
cleaned_df = cleaned_df[mask]
cleaned_df


cleaned_df["StockCode"] = cleaned_df["StockCode"].astype("str")
mask = (
    (cleaned_df["StockCode"].str.match("^\\d{5}$") == True)
    |(cleaned_df["StockCode"].str.match("^\\d{5}[a-zA-Z]+$") == True)
    |(cleaned_df["StockCode"].str.match("^PADS$") == True)
)
cleaned_df = cleaned_df[mask]
cleaned_df



cleaned_df.dropna(subset=["Customer ID"],inplace=True)
cleaned_df = cleaned_df[cleaned_df["Price"] > 0.0]
len(cleaned_df)/len(df)

#  Dropped about 23% of records during cleaning
```

4️⃣ Feature Engineering

```python

cleaned_df["SalesLineTotal"] = cleaned_df["Quantity"] * cleaned_df["Price"]

cleaned_df

aggregated_df = cleaned_df.groupby(by="Customer ID", as_index=False) \
    .agg(
        MonetaryValue=("SalesLineTotal", "sum"),
        Frequency=("Invoice", "nunique"),
        LastInvoiceDate=("InvoiceDate", "max")
    )

aggregated_df.head(5)

max_invoice_date = aggregated_df["LastInvoiceDate"].max()

aggregated_df["Recency"] = (max_invoice_date - aggregated_df["LastInvoiceDate"]).dt.days

aggregated_df.head(5)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.hist(aggregated_df['MonetaryValue'], bins=10, color='skyblue', edgecolor='black')
plt.title('Monetary Value Distribution')
plt.xlabel('Monetary Value')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
plt.hist(aggregated_df['Frequency'], bins=10, color='lightgreen', edgecolor='black')
plt.title('Frequency Distribution')
plt.xlabel('Frequency')
plt.ylabel('Count')


plt.subplot(1, 3, 3)
plt.hist(aggregated_df['Recency'], bins=20, color='salmon', edgecolor='black')
plt.title('Recency Distribution')
plt.xlabel('Recency')
plt.ylabel('Count')

plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/4e5c596b-4fc3-44b0-9fed-c5a3b81d43e4)

```python

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(data=aggregated_df['MonetaryValue'], color='skyblue')
plt.title('Monetary Value Boxplot')
plt.xlabel('Monetary Value')

plt.subplot(1, 3, 2)
sns.boxplot(data=aggregated_df['Frequency'], color='lightgreen')
plt.title('Frequency Boxplot')
plt.xlabel('Frequency')

plt.subplot(1, 3, 3)
sns.boxplot(data=aggregated_df['Recency'], color='salmon')
plt.title('Recency Boxplot')
plt.xlabel('Recency')

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/7e7510c4-af95-40a0-a784-c2c11833de94)

### Notes: We need to split out NON-OUTLIER AND OUTLIER into 2 groups

```python
M_Q1 = aggregated_df["MonetaryValue"].quantile(0.25)
M_Q3 = aggregated_df["MonetaryValue"].quantile(0.75)
M_IQR = M_Q3 - M_Q1

monetary_outliers_df = aggregated_df[(aggregated_df["MonetaryValue"] > (M_Q3 + 1.5 * M_IQR)) | (aggregated_df["MonetaryValue"] < (M_Q1 - 1.5 * M_IQR))].copy()

monetary_outliers_df.describe()


F_Q1 = aggregated_df['Frequency'].quantile(0.25)
F_Q3 = aggregated_df['Frequency'].quantile(0.75)
F_IQR = F_Q3 - F_Q1

frequency_outliers_df = aggregated_df[(aggregated_df['Frequency'] > (F_Q3 + 1.5 * F_IQR)) | (aggregated_df['Frequency'] < (F_Q1 - 1.5 * F_IQR))].copy()

frequency_outliers_df.describe()

non_outliers_df = aggregated_df[(~aggregated_df.index.isin(monetary_outliers_df.index)) & (~aggregated_df.index.isin(frequency_outliers_df.index))]

non_outliers_df.describe()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(data=non_outliers_df['MonetaryValue'], color='skyblue')
plt.title('Monetary Value Boxplot')
plt.xlabel('Monetary Value')

plt.subplot(1, 3, 2)
sns.boxplot(data=non_outliers_df['Frequency'], color='lightgreen')
plt.title('Frequency Boxplot')
plt.xlabel('Frequency')

plt.subplot(1, 3, 3)
sns.boxplot(data=non_outliers_df['Recency'], color='salmon')
plt.title('Recency Boxplot')
plt.xlabel('Recency')

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/a7224380-2a22-4a17-b4f2-91c3ff1f1da6)

```python
scaler = StandardScaler()

scaled_data = scaler.fit_transform(non_outliers_df[["MonetaryValue", "Frequency", "Recency"]])

scaled_data

scaled_data_df = pd.DataFrame(scaled_data, index=non_outliers_df.index, columns=("MonetaryValue", "Frequency", "Recency"))

scaled_data_df


fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(projection="3d")

scatter = ax.scatter(scaled_data_df["MonetaryValue"], scaled_data_df["Frequency"], scaled_data_df["Recency"])

ax.set_xlabel('Monetary Value')
ax.set_ylabel('Frequency')
ax.set_zlabel('Recency')

ax.set_title('3D Scatter Plot of Customer Data')

plt.show()
```

![image](https://github.com/user-attachments/assets/4ee1e303-b95f-47c8-9994-260c999137ef)


5️⃣ Feature Engineering

```python

max_k = 12

inertia = []
k_values = range(2, max_k + 1)

for k in k_values:

    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=1000)

    cluster_labels = kmeans.fit_predict(scaled_data_df)


    inertia.append(kmeans.inertia_)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(k_values, inertia, marker='o')
plt.title('KMeans Inertia for Different Values of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.grid(True)



plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/66ff03b3-db52-4546-9862-42069ee20371)

```python

kmeans = KMeans(n_clusters=4, random_state=42, max_iter=1000)

cluster_labels = kmeans.fit_predict(scaled_data_df)

cluster_labels

non_outliers_df["Cluster"] = cluster_labels

non_outliers_df

cluster_colors = {0: '#1f77b4',  # Blue
                  1: '#ff7f0e',  # Orange
                  2: '#2ca02c',  # Green
                  3: '#d62728'}  # Red

colors = non_outliers_df['Cluster'].map(cluster_colors)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

scatter = ax.scatter(non_outliers_df['MonetaryValue'], 
                     non_outliers_df['Frequency'], 
                     non_outliers_df['Recency'], 
                     c=colors,  # Use mapped solid colors
                     marker='o')

ax.set_xlabel('Monetary Value')
ax.set_ylabel('Frequency')
ax.set_zlabel('Recency')

ax.set_title('3D Scatter Plot of Customer Data by Cluster')

plt.show()
```
![image](https://github.com/user-attachments/assets/bf6747a8-8d72-455c-9c91-6b473db0d36b)
## 📊 Key Insights & Visualizations  
### 🔍 Dashboard Preview  

#### 1️⃣ Dashboard 1 Preview 
```python
plt.figure(figsize=(12, 18))

plt.subplot(3, 1, 1)
sns.violinplot(x=non_outliers_df['Cluster'], y=non_outliers_df['MonetaryValue'], palette=cluster_colors, hue=non_outliers_df["Cluster"])
sns.violinplot(y=non_outliers_df['MonetaryValue'], color='gray', linewidth=1.0)
plt.title('Monetary Value by Cluster')
plt.ylabel('Monetary Value')

plt.subplot(3, 1, 2)
sns.violinplot(x=non_outliers_df['Cluster'], y=non_outliers_df['Frequency'], palette=cluster_colors, hue=non_outliers_df["Cluster"])
sns.violinplot(y=non_outliers_df['Frequency'], color='gray', linewidth=1.0)
plt.title('Frequency by Cluster')
plt.ylabel('Frequency')


plt.subplot(3, 1, 3)
sns.violinplot(x=non_outliers_df['Cluster'], y=non_outliers_df['Recency'], palette=cluster_colors, hue=non_outliers_df["Cluster"])
sns.violinplot(y=non_outliers_df['Recency'], color='gray', linewidth=1.0)
plt.title('Recency by Cluster')
plt.ylabel('Recency')

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/03af9e3c-94ea-4921-bda2-f26d091bbc3e)
![image](https://github.com/user-attachments/assets/efdfef4e-761d-425a-9426-d13ae876110c)
![image](https://github.com/user-attachments/assets/c0720028-58a4-45ce-a009-f07474163347)

📌 Analysis 1:  
- Cluster 0 (Blue): "Retain" Rationale: This cluster represents high-value customers who purchase regularly, though not always very recently. The focus should be on retention efforts to maintain their loyalty and spending levels. Action: Implement loyalty programs, personalized offers, and regular engagement to ensure they remain active.

- Cluster 1 (Orange): "Re-Engage" Rationale: This group includes lower-value, infrequent buyers who haven’t purchased recently. The focus should be on re-engagement to bring them back into active purchasing behavior. Action: Use targeted marketing campaigns, special discounts, or reminders to encourage them to return and purchase again.

- Cluster 2 (Green): "Nurture" Rationale: This cluster represents the least active and lowest-value customers, but they have made recent purchases. These customers may be new or need nurturing to increase their engagement and spending. Action: Focus on building relationships, providing excellent customer service, and offering incentives to encourage more frequent purchases.

- Cluster 3 (Red): "Reward" Rationale: This cluster includes high-value, very frequent buyers, many of whom are still actively purchasing. They are your most loyal customers, and rewarding their loyalty is key to maintaining their engagement. Action: Implement a robust loyalty program, provide exclusive offers, and recognize their loyalty to keep them engaged and satisfied.

** Summary of Cluster Names:**
- Cluster 0 (Blue): "Retain"

- Cluster 1 (Orange): "Re-Engage"

- Cluster 2 (Green): "Nurture"

- Cluster 3 (Red): "Reward"
- 
#### 2️⃣ Dashboard 2 Preview  

```python
overlap_indices = monetary_outliers_df.index.intersection(frequency_outliers_df.index)

monetary_only_outliers = monetary_outliers_df.drop(overlap_indices)
frequency_only_outliers = frequency_outliers_df.drop(overlap_indices)
monetary_and_frequency_outliers = monetary_outliers_df.loc[overlap_indices]

monetary_only_outliers["Cluster"] = -1
frequency_only_outliers["Cluster"] = -2
monetary_and_frequency_outliers["Cluster"] = -3

outlier_clusters_df = pd.concat([monetary_only_outliers, frequency_only_outliers, monetary_and_frequency_outliers])

outlier_clusters_df


cluster_colors = {-1: '#9467bd',
                  -2: '#8c564b',
                  -3: '#e377c2'}

plt.figure(figsize=(12, 18))

plt.subplot(3, 1, 1)
sns.violinplot(x=outlier_clusters_df['Cluster'], y=outlier_clusters_df['MonetaryValue'], palette=cluster_colors, hue=outlier_clusters_df["Cluster"])
sns.violinplot(y=outlier_clusters_df['MonetaryValue'], color='gray', linewidth=1.0)
plt.title('Monetary Value by Cluster')
plt.ylabel('Monetary Value')

plt.subplot(3, 1, 2)
sns.violinplot(x=outlier_clusters_df['Cluster'], y=outlier_clusters_df['Frequency'], palette=cluster_colors, hue=outlier_clusters_df["Cluster"])
sns.violinplot(y=outlier_clusters_df['Frequency'], color='gray', linewidth=1.0)
plt.title('Frequency by Cluster')
plt.ylabel('Frequency')

plt.subplot(3, 1, 3)
sns.violinplot(x=outlier_clusters_df['Cluster'], y=outlier_clusters_df['Recency'], palette=cluster_colors, hue=outlier_clusters_df["Cluster"])
sns.violinplot(y=outlier_clusters_df['Recency'], color='gray', linewidth=1.0)
plt.title('Recency by Cluster')
plt.ylabel('Recency')

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/7ed3fd78-8795-46cb-bf7c-29df1e778812)
![image](https://github.com/user-attachments/assets/0e965fb4-99e1-4de6-84d4-1fbae348e25f)
![image](https://github.com/user-attachments/assets/ec4d7d8e-fb76-4694-8d63-a69257f816cf)



📌 Analysis 2:  

Cluster -1 (Monetary Outliers) PAMPER: Characteristics: High spenders but not necessarily frequent buyers. Their purchases are large but infrequent. Potential Strategy: Focus on maintaining their loyalty with personalized offers or luxury services that cater to their high spending capacity.

Cluster -2 (Frequency Outliers) UPSELL: Characteristics: Frequent buyers who spend less per purchase. These customers are consistently engaged but might benefit from upselling opportunities. Potential Strategy: Implement loyalty programs or bundle deals to encourage higher spending per visit, given their frequent engagement.

Cluster -3 (Monetary & Frequency Outliers) DELIGHT: Characteristics: The most valuable outliers, with extreme spending and frequent purchases. They are likely your top-tier customers who require special attention. Potential Strategy: Develop VIP programs or exclusive offers to maintain their loyalty and encourage continued engagement.


#### 3️⃣ Dashboard 3 Preview  


```python
cluster_labels = {
    0: "RETAIN",
    1: "RE-ENGAGE",
    2: "NURTURE",
    3: "REWARD",
    -1: "PAMPER",
    -2: "UPSELL",
    -3: "DELIGHT"
}

full_clustering_df = pd.concat([non_outliers_df, outlier_clusters_df])

full_clustering_df


full_clustering_df["ClusterLabel"] = full_clustering_df["Cluster"].map(cluster_labels)

full_clustering_df

cluster_counts = full_clustering_df['ClusterLabel'].value_counts()
full_clustering_df["MonetaryValue per 100 pounds"] = full_clustering_df["MonetaryValue"] / 100.00
feature_means = full_clustering_df.groupby('ClusterLabel')[['Recency', 'Frequency', 'MonetaryValue per 100 pounds']].mean()

fig, ax1 = plt.subplots(figsize=(12, 8))

sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax1, palette='viridis', hue=cluster_counts.index)
ax1.set_ylabel('Number of Customers', color='b')
ax1.set_title('Cluster Distribution with Average Feature Values')

ax2 = ax1.twinx()

sns.lineplot(data=feature_means, ax=ax2, palette='Set2', marker='o')
ax2.set_ylabel('Average Value', color='g')

plt.show()
```

![image](https://github.com/user-attachments/assets/ff6aa6c7-ab54-4250-8b9b-0b2b8e51b3e8)


## 🔎 Final Conclusion & Recommendations

| Segment     | Number of Customers | Average Recency | Average Frequency | Average Monetary Value (per 100 pounds) | Key Observations                                                                 | Recommendations                                                                                                                                                                                                                                                           |
|-------------|---------------------|-----------------|-------------------|-----------------------------------------|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Nurture     | ~1500              | High            | Low               | Low                                     | Largest segment, low engagement and spending.                                    | Implement targeted re-engagement campaigns with discounts or new product highlights. Analyze reasons for low Recency and Frequency.                                                                                                                                      |
| Retain      | ~900               | Moderate        | Moderate          | Low                                     | Large segment with slightly better engagement but still low spending.                | Focus on increasing purchase frequency and average order value through bundles, cross-selling, and loyalty programs.                                                                                                                                                  |
| Re-Engage   | ~900               | Very High       | Very Low          | Low                                     | Significant number of lapsed customers.                                            | Develop compelling re-engagement campaigns with special offers. Understand reasons for churn through surveys or feedback.                                                                                                                                              |
| Reward      | ~500               | Low             | Moderate          | Moderate                                | Smaller segment with good Recency and Frequency, moderate spending.                  | Nurture with personalized offers and excellent customer service to maintain loyalty and potentially increase spending.                                                                                                                                                 |
| Delight     | ~200               | Low             | High            | Very High                               | Smallest but most valuable segment with high engagement and spending.              | Prioritize this segment with exclusive rewards, early access, and personalized communication. Identify factors contributing to their high value and replicate them.                                                                                                       |
| Pamper      | ~200               | Low             | Moderate          | Moderate                                | Moderate-sized segment with good engagement and decent spending.                    | Maintain satisfaction and explore opportunities to increase purchase frequency or average order value.                                                                                                                                                            |
| Upsell      | ~50                | Moderate        | Low               | Moderate                                | Smallest segment with recent high spending but low frequency.                       | Explore upselling or cross-selling related products. Understand their initial purchase and identify needs for repeat purchases. Implement strategies to increase purchase frequency.                                                                                 |

