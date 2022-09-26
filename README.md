# Clustering Cryptocurrencies
*** Application of K-Means Clustering to Identify Groups of Cryptocurrencies ***
#### by Justin R. Papreck
---

## Overview

As cryptocurrencies continue to rise in popularity worldwide, it is difficult to determine which crypocurrencies are similar. Many cryptocurrencies use similar algorithms, yet others rely on unique algorithms that may yield the same types of results. As the unique number of crypocurrencies continues to rise, it is difficult to ascertain how each of them are related, and whether the algorithm itself is even a good predictor of how each crypto can be classified. K-Means clusterding, is applied here to perform unsupervised machine learning in the grouping of over 500 active currencies. It is important to know which cryptocurrencies are similar for potential crypto investors. For example, the periodic table enthusiast who recently discovered cryptocurrencies and wants to sample the elements but at the same time wants to diversify. Are the differences between Osmium, Actinium, Lithium, Einsteinium, and Radium as simple as their locations on the periodic table? Or perhaps it's a cannabis entrepreneur looking to invest in 3 different cryptocurrencies, but they have to choose between Cannabis Industry Coin, Canna Coin, Sativa Coin, GanjaCoin, KushCoin and PotCoin and want to diversify their investments as they diversify their stock. Or perhaps there is just a sci-fi investor willing to put all of their money into one place and wants it to go into whatever algorithm is most similar to BitCoin, but wants it to be in 42, Unobtainium, or Klingon Empire Darsek. This application aims to classify all of these currencies through unsupervised learning.

### Deliverable 1: Preprocessing the Data for Principal Component Analysis (PCA)

Upon first glance of the data, there are cryptocurrencies that are not trading as well as currencies that haven't mined or acquired a supply of coins. 

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CoinName</th>
      <th>Algorithm</th>
      <th>IsTrading</th>
      <th>ProofType</th>
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>42 Coin</td>
      <td>Scrypt</td>
      <td>True</td>
      <td>PoW/PoS</td>
      <td>4.199995e+01</td>
      <td>42</td>
    </tr>
    <tr>
      <th>365</th>
      <td>365Coin</td>
      <td>X11</td>
      <td>True</td>
      <td>PoW/PoS</td>
      <td>NaN</td>
      <td>2300000000</td>
    </tr>
    <tr>
      <th>404</th>
      <td>404Coin</td>
      <td>Scrypt</td>
      <td>True</td>
      <td>PoW/PoS</td>
      <td>1.055185e+09</td>
      <td>532000000</td>
    </tr>
    <tr>
      <th>611</th>
      <td>SixEleven</td>
      <td>SHA-256</td>
      <td>True</td>
      <td>PoW</td>
      <td>NaN</td>
      <td>611000</td>
    </tr>
    <tr>
      <th>808</th>
      <td>808</td>
      <td>SHA-256</td>
      <td>True</td>
      <td>PoW/PoS</td>
      <td>0.000000e+00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>EliteCoin</td>
      <td>X13</td>
      <td>True</td>
      <td>PoW/PoS</td>
      <td>2.927942e+10</td>
      <td>314159265359</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>2015 coin</td>
      <td>X11</td>
      <td>True</td>
      <td>PoW/PoS</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>Bitcoin</td>
      <td>SHA-256</td>
      <td>True</td>
      <td>PoW</td>
      <td>1.792718e+07</td>
      <td>21000000</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>Ethereum</td>
      <td>Ethash</td>
      <td>True</td>
      <td>PoW</td>
      <td>1.076842e+08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LTC</th>
      <td>Litecoin</td>
      <td>Scrypt</td>
      <td>True</td>
      <td>PoW</td>
      <td>6.303924e+07</td>
      <td>84000000</td>
    </tr>
  </tbody>
</table>
</div>

Initially, all currencies that are not trading were removed, thus dropping the "Is Trading Column". Subsequently, any row containing a Null value were removed. Finally, since purchasing a currency requires at least 1 coin, all cryptocurrencies with no coin supply were also removed. From the 1252 currencies listed, only 533 met the criteria that there were a non-zero coin supply, the coin name, algorithm, proof type, number of coins mined and total supply were all accounted for and they were actively trading. In order to process for machine learning, the coin name code was used as the index, and the names were saved in a different data frame, as the name should not influence a currency's grouping. The remaining non-numerical data in the set were the algorithm and proof type. 

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CoinName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>42 Coin</td>
    </tr>
    <tr>
      <th>404</th>
      <td>404Coin</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>EliteCoin</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>Bitcoin</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>Ethereum</td>
    </tr>
  </tbody>
</table>
</div>


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algorithm</th>
      <th>ProofType</th>
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>4.199995e+01</td>
      <td>42</td>
    </tr>
    <tr>
      <th>404</th>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>1.055185e+09</td>
      <td>532000000</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>X13</td>
      <td>PoW/PoS</td>
      <td>2.927942e+10</td>
      <td>314159265359</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>SHA-256</td>
      <td>PoW</td>
      <td>1.792718e+07</td>
      <td>21000000</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>Ethash</td>
      <td>PoW</td>
      <td>1.076842e+08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LTC</th>
      <td>Scrypt</td>
      <td>PoW</td>
      <td>6.303924e+07</td>
      <td>84000000</td>
    </tr>
    <tr>
      <th>DASH</th>
      <td>X11</td>
      <td>PoW/PoS</td>
      <td>9.031294e+06</td>
      <td>22000000</td>
    </tr>
    <tr>
      <th>XMR</th>
      <td>CryptoNight-V7</td>
      <td>PoW</td>
      <td>1.720114e+07</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ETC</th>
      <td>Ethash</td>
      <td>PoW</td>
      <td>1.133597e+08</td>
      <td>210000000</td>
    </tr>
    <tr>
      <th>ZEC</th>
      <td>Equihash</td>
      <td>PoW</td>
      <td>7.383056e+06</td>
      <td>21000000</td>
    </tr>
  </tbody>
</table>
</div>


To address these columns, dummy variables were created for each algorithm and proof type, identifying each as a binary value by applying the pandas function 'get_dummies'. Subsequently, the values were scaled to a value between 0 and 1, with a standard deviation of 1 using StandardScaler from the scikit-learn library.


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
      <th>Algorithm_1GB AES Pattern Search</th>
      <th>Algorithm_536</th>
      <th>Algorithm_Argon2d</th>
      <th>Algorithm_BLAKE256</th>
      <th>Algorithm_Blake</th>
      <th>Algorithm_Blake2S</th>
      <th>Algorithm_Blake2b</th>
      <th>Algorithm_C11</th>
      <th>...</th>
      <th>ProofType_PoW/PoS</th>
      <th>ProofType_PoW/PoS</th>
      <th>ProofType_PoW/PoW</th>
      <th>ProofType_PoW/nPoS</th>
      <th>ProofType_Pos</th>
      <th>ProofType_Proof of Authority</th>
      <th>ProofType_Proof of Trust</th>
      <th>ProofType_TPoS</th>
      <th>ProofType_Zero-Knowledge Proof</th>
      <th>ProofType_dPoW/PoW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>4.199995e+01</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>404</th>
      <td>1.055185e+09</td>
      <td>532000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>2.927942e+10</td>
      <td>314159265359</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>1.792718e+07</td>
      <td>21000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>1.076842e+08</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 100 columns</p>
</div>


```python
# Standardize the data with StandardScaler().
X = StandardScaler().fit_transform(X)
```
---

### Deliverable 2: Reducing Data Dimensions Using PCA

Due to the use of dummies, the data expaned to 100 columns, a number of dimensions that would be visually impossible to present. Using PCA() from scikit-learn, the number of dimensions were reduced to 3 for the ability to present in a 3-dimensional graph. The components were named PC 1, PC 2, and PC 3, and the dataframe maintains the indices established in the previous dataframe. 

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC 1</th>
      <th>PC 2</th>
      <th>PC 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>-0.299625</td>
      <td>1.093503</td>
      <td>-0.495499</td>
    </tr>
    <tr>
      <th>404</th>
      <td>-0.282859</td>
      <td>1.093897</td>
      <td>-0.495606</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>2.317407</td>
      <td>1.718744</td>
      <td>-0.532707</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>-0.152806</td>
      <td>-1.298636</td>
      <td>0.123549</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>-0.162274</td>
      <td>-2.025853</td>
      <td>0.333898</td>
    </tr>
    <tr>
      <th>LTC</th>
      <td>-0.133639</td>
      <td>-1.091527</td>
      <td>-0.035369</td>
    </tr>
    <tr>
      <th>DASH</th>
      <td>-0.411299</td>
      <td>1.238712</td>
      <td>-0.421072</td>
    </tr>
    <tr>
      <th>XMR</th>
      <td>-0.156051</td>
      <td>-2.275822</td>
      <td>0.285527</td>
    </tr>
    <tr>
      <th>ETC</th>
      <td>-0.160707</td>
      <td>-2.025936</td>
      <td>0.333897</td>
    </tr>
    <tr>
      <th>ZEC</th>
      <td>-0.150078</td>
      <td>-2.175185</td>
      <td>0.269426</td>
    </tr>
  </tbody>
</table>
</div>
---


### Deliverable 3: Clustering Crytocurrencies Using K-Means
*** Finding the Best Value for 'k' Using the Elbow Curve ***

To determine the optimal number of clusters for the K-Means Clustering application, an elbow curve was created comparing the inertia with the number of clusters. There is a clear bend at 4 clusters with diminishing returns after, so the analyses were performed with 4 clusters.  


ELBOW CURVE


```python 
# Initialize the K-Means model.
model = KMeans(n_clusters=4, random_state=0)

# Fit the model
model.fit(pcs_df)

# Predict clusters
predictions = model.predict(pcs_df)

```
The outcome of the KMeans analysis with 4 clusters provided an array with clusters labeled 0, 1, 2, and 3.

[0 0 0 3 3 3 0 3 3 3 0 3 0 0 3 0 3 3 0 0 3 3 3 3 3 0 3 3 3 0 3 0 3 3 0 0 3
 3 3 3 3 3 0 0 3 3 3 3 3 0 0 3 0 3 3 3 3 0 3 3 0 3 0 0 0 3 3 3 0 0 0 0 0 3
 3 3 0 0 3 0 3 0 0 3 3 3 3 0 0 3 0 3 3 0 0 3 0 0 3 3 0 0 3 0 0 3 0 3 0 3 0
 3 0 0 3 3 0 3 3 3 0 3 3 3 3 3 0 0 3 3 3 0 3 0 3 3 0 3 0 3 0 0 3 3 0 3 3 0
 0 3 0 3 0 0 0 3 3 3 3 0 0 0 0 0 3 3 0 0 0 0 0 3 0 0 0 0 0 3 0 3 0 0 3 0 3
 0 0 3 0 3 0 3 0 3 0 0 0 0 3 0 0 0 0 0 3 3 0 0 3 3 0 0 0 0 0 3 0 0 0 0 0 0
 0 0 3 0 0 0 0 0 0 3 3 3 0 0 0 0 3 0 3 0 0 3 0 3 3 0 3 3 0 3 0 0 0 3 0 0 3
 0 0 0 0 0 0 0 3 0 3 0 0 0 0 3 0 3 0 3 3 3 3 0 3 0 0 3 0 3 3 3 0 3 0 3 3 3
 0 3 0 3 0 0 1 3 0 3 3 3 3 3 0 0 3 0 0 0 3 0 3 0 3 0 3 0 0 0 0 3 0 0 3 0 0
 0 3 3 3 3 0 0 0 0 3 0 3 3 3 0 0 3 3 0 0 3 0 3 3 3 0 3 3 0 0 0 3 3 3 0 0 0
 3 3 0 3 3 3 3 0 1 1 3 3 3 0 1 0 0 0 0 3 3 3 3 0 0 0 3 0 3 0 0 0 0 3 0 0 3
 0 0 3 3 0 3 0 3 3 3 3 0 0 3 0 3 0 0 0 0 0 0 3 3 3 0 0 0 0 0 0 3 0 3 3 3 3
 0 0 0 0 3 0 0 3 0 0 3 1 3 0 3 3 0 0 3 0 3 3 3 3 3 0 3 0 3 0 0 3 0 0 0 0 0
 3 3 3 0 0 0 3 0 3 0 3 0 0 0 0 3 0 0 0 3 0 3 0 3 0 0 0 3 3 0 0 0 0 0 0 1 3
 0 3 0 3 0 0 1 0 2 0 0 0 3 3 0]

These data were concatenated into a dataframe along with the PCA data and the original cryptocurrency dataframe with the Algorithm, Proof Type, and Coin Names

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algorithm</th>
      <th>ProofType</th>
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
      <th>PC 1</th>
      <th>PC 2</th>
      <th>PC 3</th>
      <th>CoinName</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>4.199995e+01</td>
      <td>42</td>
      <td>-0.299625</td>
      <td>1.093503</td>
      <td>-0.495499</td>
      <td>42 Coin</td>
      <td>0</td>
    </tr>
    <tr>
      <th>404</th>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>1.055185e+09</td>
      <td>532000000</td>
      <td>-0.282859</td>
      <td>1.093897</td>
      <td>-0.495606</td>
      <td>404Coin</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>X13</td>
      <td>PoW/PoS</td>
      <td>2.927942e+10</td>
      <td>314159265359</td>
      <td>2.317407</td>
      <td>1.718744</td>
      <td>-0.532707</td>
      <td>EliteCoin</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>SHA-256</td>
      <td>PoW</td>
      <td>1.792718e+07</td>
      <td>21000000</td>
      <td>-0.152806</td>
      <td>-1.298636</td>
      <td>0.123549</td>
      <td>Bitcoin</td>
      <td>3</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>Ethash</td>
      <td>PoW</td>
      <td>1.076842e+08</td>
      <td>0</td>
      <td>-0.162274</td>
      <td>-2.025853</td>
      <td>0.333898</td>
      <td>Ethereum</td>
      <td>3</td>
    </tr>
    <tr>
      <th>LTC</th>
      <td>Scrypt</td>
      <td>PoW</td>
      <td>6.303924e+07</td>
      <td>84000000</td>
      <td>-0.133639</td>
      <td>-1.091527</td>
      <td>-0.035369</td>
      <td>Litecoin</td>
      <td>3</td>
    </tr>
    <tr>
      <th>DASH</th>
      <td>X11</td>
      <td>PoW/PoS</td>
      <td>9.031294e+06</td>
      <td>22000000</td>
      <td>-0.411299</td>
      <td>1.238712</td>
      <td>-0.421072</td>
      <td>Dash</td>
      <td>0</td>
    </tr>
    <tr>
      <th>XMR</th>
      <td>CryptoNight-V7</td>
      <td>PoW</td>
      <td>1.720114e+07</td>
      <td>0</td>
      <td>-0.156051</td>
      <td>-2.275822</td>
      <td>0.285527</td>
      <td>Monero</td>
      <td>3</td>
    </tr>
    <tr>
      <th>ETC</th>
      <td>Ethash</td>
      <td>PoW</td>
      <td>1.133597e+08</td>
      <td>210000000</td>
      <td>-0.160707</td>
      <td>-2.025936</td>
      <td>0.333897</td>
      <td>Ethereum Classic</td>
      <td>3</td>
    </tr>
    <tr>
      <th>ZEC</th>
      <td>Equihash</td>
      <td>PoW</td>
      <td>7.383056e+06</td>
      <td>21000000</td>
      <td>-0.150078</td>
      <td>-2.175185</td>
      <td>0.269426</td>
      <td>ZCash</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>
---


### Deliverable 4: Visualizing Cryptocurrencies Results

#### 3D-Scatter with Clusters

With the reduced dimensions from the PCA application, the results can be visualized in a 3-D plot. The plot was created using Plotly Express to show the unique points with the name, algorithm used, and principal component data. The plot is interactive allowing the user to hover over the individual points, zoom, and rotate. 




CLUSTER 3D  FIGURE


Something very clear with this graph are the points from Classes 1 and 2. (Note: the class number changes when the application is run, but these classes refer to the above figure).

- BitTorrent stands in a class of its own (Class 2). It had very low influence from PC 3, a higher influence from PC2 than groups 0 and 1, but an extremely high influence from PC1, something none of the other groups shared 
- Groups 0 and 3 are tightly associated, having a low impact from PC 1 and PC3
- Group 0 is more positive than Group 3, but it does contain some negative values in PC 2
- Group 1 is a loosely associated group sharing a non-zero influence by PC 3, but otherwise are greater than 0 in PC 2
- Since Group 2 is such an outlier from the rest of the classes, it would be beneficial to reanalyze these data without the BitTorrent data to see if the other groups remain in their respective classes

Since it is difficult to filter or find individual currencies, such as the ones mentioned in the Overview, hvplot was used to create an interactive table that can be sorted by each of the original parameters. 




HV PLOT INTERACTIVE TABLE




From here the user can inspect the cryptocurrencies organized by any of the individual parameters.

```python
# Print the total number of tradable currencies in the Clustered_df Dataframe
print(f"There are {len(clustered_df)} tradable currencies in the dataframe")
```
There are 533 tradable currencies in the dataframe
    
The final analysis considers the number of total coins mined and the total coin supply of the different classes. A low number of coins mined may suggest that the algorithm to mine them is complex, driving the supply down accordingly, and conversely, a high total number mined may have a high supply if they are readily mined and coin becomes quickly available. These may impact the economics of each of the currencies which can give the investment firm an indicator of which currencies are likely to see an increase or decrease of value over time. To analyze these with differences in numbers up to 12 orders of magnitude, the MinMaxScaler function was used to bring these values between 0 and 1. 

```python
# Select the columns to scale
totals = clustered_df[['TotalCoinSupply', 'TotalCoinsMined']]
scaled_totals = MinMaxScaler(feature_range=(0,1)).fit_transform(totals)
clustered_scaled_df = clustered_scaled_df[["TotalCoinSupply", "TotalCoinsMined", "CoinName", "Class"]]
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TotalCoinSupply</th>
      <th>TotalCoinsMined</th>
      <th>CoinName</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>4.200000e-11</td>
      <td>0.005942</td>
      <td>42 Coin</td>
      <td>0</td>
    </tr>
    <tr>
      <th>404</th>
      <td>5.320000e-04</td>
      <td>0.007002</td>
      <td>404Coin</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>3.141593e-01</td>
      <td>0.035342</td>
      <td>EliteCoin</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>2.100000e-05</td>
      <td>0.005960</td>
      <td>Bitcoin</td>
      <td>3</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>0.000000e+00</td>
      <td>0.006050</td>
      <td>Ethereum</td>
      <td>3</td>
    </tr>
    <tr>
      <th>LTC</th>
      <td>8.400000e-05</td>
      <td>0.006006</td>
      <td>Litecoin</td>
      <td>3</td>
    </tr>
    <tr>
      <th>DASH</th>
      <td>2.200000e-05</td>
      <td>0.005951</td>
      <td>Dash</td>
      <td>0</td>
    </tr>
    <tr>
      <th>XMR</th>
      <td>0.000000e+00</td>
      <td>0.005960</td>
      <td>Monero</td>
      <td>3</td>
    </tr>
    <tr>
      <th>ETC</th>
      <td>2.100000e-04</td>
      <td>0.006056</td>
      <td>Ethereum Classic</td>
      <td>3</td>
    </tr>
    <tr>
      <th>ZEC</th>
      <td>2.100000e-05</td>
      <td>0.005950</td>
      <td>ZCash</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Alternatively, the above steps can be done...
# Create a new dataframe that has the scaled data
plot_df = pd.DataFrame(data=scaled_totals, columns=["TotalCoinSupply", "TotalCoinsMined"])

# Add the indexing from the original df
plot_df.index = clustered_df.index  

# Add the coin name and add the class from the original 
plot_df[["CoinName", "Class"]] = clustered_df[["CoinName", "Class"]] 

plot_df.head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TotalCoinSupply</th>
      <th>TotalCoinsMined</th>
      <th>CoinName</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>4.200000e-11</td>
      <td>0.005942</td>
      <td>42 Coin</td>
      <td>0</td>
    </tr>
    <tr>
      <th>404</th>
      <td>5.320000e-04</td>
      <td>0.007002</td>
      <td>404Coin</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>3.141593e-01</td>
      <td>0.035342</td>
      <td>EliteCoin</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>2.100000e-05</td>
      <td>0.005960</td>
      <td>Bitcoin</td>
      <td>3</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>0.000000e+00</td>
      <td>0.006050</td>
      <td>Ethereum</td>
      <td>3</td>
    </tr>
    <tr>
      <th>LTC</th>
      <td>8.400000e-05</td>
      <td>0.006006</td>
      <td>Litecoin</td>
      <td>3</td>
    </tr>
    <tr>
      <th>DASH</th>
      <td>2.200000e-05</td>
      <td>0.005951</td>
      <td>Dash</td>
      <td>0</td>
    </tr>
    <tr>
      <th>XMR</th>
      <td>0.000000e+00</td>
      <td>0.005960</td>
      <td>Monero</td>
      <td>3</td>
    </tr>
    <tr>
      <th>ETC</th>
      <td>2.100000e-04</td>
      <td>0.006056</td>
      <td>Ethereum Classic</td>
      <td>3</td>
    </tr>
    <tr>
      <th>ZEC</th>
      <td>2.100000e-05</td>
      <td>0.005950</td>
      <td>ZCash</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



These data were plotted in a 2D scatter plot. As can be seen, once again Class 2, the BitTorrent currency really stands out having the highest supply and the highest number of coins mined. Class 1 also has a single currency, TurtleCoin, that matches the maximum total coin supply, but the coins mined is only a fraction of that of BitTorrent. 

```python
# Create a hvplot.scatter plot using x="TotalCoinsMined" and y="TotalCoinSupply".
plot_df.hvplot.scatter(x="TotalCoinsMined"
    , y="TotalCoinSupply"
    , by="Class"
    , hover_cols="Class"
    , width=800
    , height=500
    , s=150
    , alpha = 0.8
    , selection_alpha=0.1
    , line_color='black'
    , line_alpha = 0.5
    , title="K-Means Scatter of Cryptocurrencies, Total Coins Mined vs. Total Coin Supply"
    )
```
---

To finally address the question posed at the beginning regarding someone interested in a particular subset of cryptocurrencies by name, a function was created to do just this, requiring only the input of a list with those coin names. 

```python 
def investor(investments, df=clustered_df):
    results = clustered_df[clustered_df['CoinName'].isin(investments)][["CoinName", "Algorithm", "ProofType", "Class"]]
    return results
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CoinName</th>
      <th>Algorithm</th>
      <th>ProofType</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CCN</th>
      <td>CannaCoin</td>
      <td>Scrypt</td>
      <td>PoW</td>
      <td>1</td>
    </tr>
    <tr>
      <th>POT</th>
      <td>PotCoin</td>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>0</td>
    </tr>
    <tr>
      <th>STV</th>
      <td>Sativa Coin</td>
      <td>X13</td>
      <td>PoW/PoS</td>
      <td>0</td>
    </tr>
    <tr>
      <th>DOPE</th>
      <td>DopeCoin</td>
      <td>Scrypt</td>
      <td>PoW</td>
      <td>1</td>
    </tr>
    <tr>
      <th>GNJ</th>
      <td>GanjaCoin V2</td>
      <td>X14</td>
      <td>PoW/PoS</td>
      <td>0</td>
    </tr>
    <tr>
      <th>XCI</th>
      <td>Cannabis Industry Coin</td>
      <td>CryptoNight</td>
      <td>PoW</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

To answer our scientist investor, while the elements are spread across the periodic table, all but one of the cryptocurrencies fall into group 1. Our science fiction enthusiast may be interested to know that one of the three, Unobtainium, is in class 1 while the others are in class 2. However, for our cannabis entrepreneur trying to diversify across classes, the 6 cannabis-themed coins only fall into two of the groups. This is expected, since group 2 only contained one cryptocurrency and group 3 was much smaller than the others, but this provides users with the capability to choose and compare individual currencies given they know the coin name. 