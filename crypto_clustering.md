# Clustering Crypto


```python
# Initial imports
import pandas as pd
import numpy as np
import hvplot.pandas
from pathlib import Path
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

```






<style>.bk-root, .bk-root .bk:before, .bk-root .bk:after {
  font-family: var(--jp-ui-font-size1);
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color1);
}
</style>






### Deliverable 1: Preprocessing the Data for PCA


```python
# Load the crypto_data.csv dataset.
from operator import index


file = Path("crypto_data.csv")
crypto_df = pd.read_csv(file, index_col=0)
crypto_raw = crypto_df.copy()
crypto_df.head(10)
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




```python
crypto_df.dtypes

```




    CoinName            object
    Algorithm           object
    IsTrading             bool
    ProofType           object
    TotalCoinsMined    float64
    TotalCoinSupply     object
    dtype: object




```python
# Identify unique values in the IsTrading column
np.unique(crypto_df.IsTrading)
```




    array([False,  True])




```python
# Keep all the cryptocurrencies that are being traded.
crypto_df = crypto_df[crypto_df.apply(lambda row: row['IsTrading'] == True, axis=1)]
crypto_df.head()
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
  </tbody>
</table>
</div>




```python
# Verify that only True Values remain in the IsTraining column
np.unique(crypto_df.IsTrading)
```




    array([ True])




```python
# Keep all the cryptocurrencies that have a working algorithm.
np.unique(crypto_df.Algorithm)  # There are no columns without working algorithms
```




    array(['1GB AES Pattern Search', '536', 'Argon2', 'Argon2d', 'BLAKE256',
           'Blake', 'Blake2S', 'Blake2b', 'C11', 'Cloverhash', 'Counterparty',
           'CryptoNight', 'CryptoNight Heavy', 'CryptoNight Heavy X',
           'CryptoNight-Lite', 'CryptoNight-V7', 'CryptoNight-lite',
           'Cryptonight-GPU', 'Curve25519', 'DPoS', 'Dagger',
           'Dagger-Hashimoto', 'ECC 256K1', 'Equihash', 'Equihash+Scrypt',
           'Equihash1927', 'Ethash', 'Exosis', 'Green Protocol', 'Groestl',
           'HMQ1725', 'HybridScryptHash256', 'IMesh', 'Jump Consistent Hash',
           'Keccak', 'Leased POS', 'Lyra2RE', 'Lyra2REv2', 'Lyra2Z', 'M7 POW',
           'Momentum', 'Multiple', 'NIST5', 'NeoScrypt', 'Ouroboros',
           'PHI1612', 'POS 2.0', 'POS 3.0', 'PoS', 'Progressive-n',
           'Proof-of-Authority', 'Proof-of-BibleHash', 'QUAIT', 'QuBit',
           'Quark', 'QuarkTX', 'Rainforest', 'SHA-256', 'SHA-256 + Hive',
           'SHA-256D', 'SHA-512', 'SHA3', 'SHA3-256', 'Scrypt', 'Scrypt-n',
           'Semux BFT consensus', 'Shabal256', 'Skein', 'SkunkHash',
           'SkunkHash v2 Raptor', 'Slatechain', 'Stanford Folding',
           'T-Inside', 'TRC10', 'Time Travel', 'Tribus', 'VBFT',
           'VeChainThor Authority', 'X11', 'X11GOST', 'X13', 'X14', 'X15',
           'X16R', 'XEVAN', 'XG Hash', 'YescryptR16', 'Zhash', 'vDPOS'],
          dtype=object)




```python
# Remove the "IsTrading" column. 
crypto_df = crypto_df.drop(columns=['IsTrading'])
```


```python
len(crypto_df)
```




    1144




```python
# Remove rows that have at least 1 null value.
crypto_df.dropna(inplace=True)
len(crypto_df)
```




    685




```python
crypto_df.head()
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
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>42 Coin</td>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>4.199995e+01</td>
      <td>42</td>
    </tr>
    <tr>
      <th>404</th>
      <td>404Coin</td>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>1.055185e+09</td>
      <td>532000000</td>
    </tr>
    <tr>
      <th>808</th>
      <td>808</td>
      <td>SHA-256</td>
      <td>PoW/PoS</td>
      <td>0.000000e+00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>EliteCoin</td>
      <td>X13</td>
      <td>PoW/PoS</td>
      <td>2.927942e+10</td>
      <td>314159265359</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>Bitcoin</td>
      <td>SHA-256</td>
      <td>PoW</td>
      <td>1.792718e+07</td>
      <td>21000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Keep the rows where coins are mined.
crypto_df = crypto_df[crypto_df.apply(
    lambda row: row["TotalCoinsMined"] != 0, axis=1)]
len(crypto_df)

```




    533




```python
# Create a new DataFrame that holds only the cryptocurrencies names.
crypto_names = crypto_df.iloc[:, 0:1]
crypto_names.head()
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




```python
# Drop the 'CoinName' column since it's not going to be used on the clustering algorithm.
crypto_df.drop(columns="CoinName", inplace=True)
```


```python
crypto_df.head(10)
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




```python
# Use get_dummies() to create variables for text features.
X = pd.get_dummies(crypto_df, columns=["Algorithm", "ProofType"])
X.head()
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

### Deliverable 2: Reducing Data Dimensions Using PCA


```python
# Using PCA to reduce dimension to three principal components.
X_pca = PCA(n_components=3).fit_transform(X)

```


```python
# Create a DataFrame with the three principal components.
pcs_df = pd.DataFrame(data=X_pca, columns=["PC 1", "PC 2", "PC 3"])
pcs_df = pcs_df.set_index(crypto_df.index)
pcs_df.head(10)

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



### Deliverable 3: Clustering Crytocurrencies Using K-Means

#### Finding the Best Value for `k` Using the Elbow Curve


```python
# Create an elbow curve to find the best value for K.
inertia = []
k = list(range(1, 11))

for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(pcs_df)
    inertia.append(km.inertia_)

elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
df_elbow.hvplot.line(x="k", y="inertia", title="Elbow Curve", xticks = k)
```

    c:\Users\justi\anaconda3\envs\mlenv\lib\site-packages\sklearn\cluster\_kmeans.py:1037: UserWarning:
    
    KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=3.
    
    






<div id='1536'>





  <div class="bk-root" id="d8eec9c0-ad4c-41ad-b5e8-d3aeb9017230" data-root-id="1536"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"2dfbf2f5-9cc1-4bdd-9aa4-eff12fd84e97":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"GridStack1","overrides":[],"properties":[{"default":"warn","kind":null,"name":"mode"},{"default":null,"kind":null,"name":"ncols"},{"default":null,"kind":null,"name":"nrows"},{"default":true,"kind":null,"name":"allow_resize"},{"default":true,"kind":null,"name":"allow_drag"},{"default":[],"kind":null,"name":"state"}]},{"extends":null,"module":null,"name":"click1","overrides":[],"properties":[{"default":"","kind":null,"name":"terminal_output"},{"default":"","kind":null,"name":"debug_name"},{"default":0,"kind":null,"name":"clears"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{},"id":"1548","type":"LinearScale"},{"attributes":{"axis_label":"inertia","coordinates":null,"formatter":{"id":"1587"},"group":null,"major_label_policy":{"id":"1588"},"ticker":{"id":"1555"}},"id":"1554","type":"LinearAxis"},{"attributes":{"axis_label":"k","coordinates":null,"formatter":{"id":"1582"},"group":null,"major_label_policy":{"id":"1583"},"ticker":{"id":"1580"}},"id":"1550","type":"LinearAxis"},{"attributes":{},"id":"1572","type":"Selection"},{"attributes":{},"id":"1588","type":"AllLabels"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"1574","type":"Line"},{"attributes":{"below":[{"id":"1550"}],"center":[{"id":"1553"},{"id":"1557"}],"height":300,"left":[{"id":"1554"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"1577"}],"sizing_mode":"fixed","title":{"id":"1542"},"toolbar":{"id":"1564"},"width":700,"x_range":{"id":"1538"},"x_scale":{"id":"1546"},"y_range":{"id":"1539"},"y_scale":{"id":"1548"}},"id":"1541","subtype":"Figure","type":"Plot"},{"attributes":{"axis":{"id":"1554"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"1557","type":"Grid"},{"attributes":{"source":{"id":"1571"}},"id":"1578","type":"CDSView"},{"attributes":{"ticks":[1,2,3,4,5,6,7,8,9,10]},"id":"1580","type":"FixedTicker"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02696","sizing_mode":"stretch_width"},"id":"1537","type":"Spacer"},{"attributes":{},"id":"1582","type":"BasicTickFormatter"},{"attributes":{"end":10.0,"reset_end":10.0,"reset_start":1.0,"start":1.0,"tags":[[["k","k",null]]]},"id":"1538","type":"Range1d"},{"attributes":{"coordinates":null,"data_source":{"id":"1571"},"glyph":{"id":"1574"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1576"},"nonselection_glyph":{"id":"1575"},"selection_glyph":{"id":"1579"},"view":{"id":"1578"}},"id":"1577","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.2,"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"1576","type":"Line"},{"attributes":{"line_alpha":0.1,"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"1575","type":"Line"},{"attributes":{},"id":"1587","type":"BasicTickFormatter"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"1563","type":"BoxAnnotation"},{"attributes":{"callback":null,"renderers":[{"id":"1577"}],"tags":["hv_created"],"tooltips":[["k","@{k}"],["inertia","@{inertia}"]]},"id":"1540","type":"HoverTool"},{"attributes":{},"id":"1558","type":"SaveTool"},{"attributes":{"end":3996.2683533209147,"reset_end":3996.2683533209147,"reset_start":-224.78893709094373,"start":-224.78893709094373,"tags":[[["inertia","inertia",null]]]},"id":"1539","type":"Range1d"},{"attributes":{"data":{"inertia":{"__ndarray__":"/KfX8wZ5rEAWH0kHlEejQJt/qgCkIZdA+wFp/m+2gUB6BTBu/c93QJvFLTBfBnNAuKSQQiSvbUA0XnfIvRloQEvJa0DRkmNA3jl0RtC9X0A=","dtype":"float64","order":"little","shape":[10]},"k":[1,2,3,4,5,6,7,8,9,10]},"selected":{"id":"1572"},"selection_policy":{"id":"1595"}},"id":"1571","type":"ColumnDataSource"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"1579","type":"Line"},{"attributes":{},"id":"1559","type":"PanTool"},{"attributes":{},"id":"1562","type":"ResetTool"},{"attributes":{},"id":"1595","type":"UnionRenderers"},{"attributes":{},"id":"1560","type":"WheelZoomTool"},{"attributes":{},"id":"1583","type":"AllLabels"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02697","sizing_mode":"stretch_width"},"id":"1607","type":"Spacer"},{"attributes":{"children":[{"id":"1537"},{"id":"1541"},{"id":"1607"}],"margin":[0,0,0,0],"name":"Row02692","tags":["embedded"]},"id":"1536","type":"Row"},{"attributes":{"overlay":{"id":"1563"}},"id":"1561","type":"BoxZoomTool"},{"attributes":{},"id":"1555","type":"BasicTicker"},{"attributes":{},"id":"1546","type":"LinearScale"},{"attributes":{"coordinates":null,"group":null,"text":"Elbow Curve","text_color":"black","text_font_size":"12pt"},"id":"1542","type":"Title"},{"attributes":{"tools":[{"id":"1540"},{"id":"1558"},{"id":"1559"},{"id":"1560"},{"id":"1561"},{"id":"1562"}]},"id":"1564","type":"Toolbar"},{"attributes":{"axis":{"id":"1550"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"1553","type":"Grid"}],"root_ids":["1536"]},"title":"Bokeh Application","version":"2.4.2"}};
    var render_items = [{"docid":"2dfbf2f5-9cc1-4bdd-9aa4-eff12fd84e97","root_ids":["1536"],"roots":{"1536":"d8eec9c0-ad4c-41ad-b5e8-d3aeb9017230"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>



Running K-Means with `k=4`


```python
# Initialize the K-Means model.
model = KMeans(n_clusters=4, random_state=0)

# Fit the model
model.fit(pcs_df)

# Predict clusters
predictions = model.predict(pcs_df)
print(predictions)
```

    [0 0 0 3 3 3 0 3 3 3 0 3 0 0 3 0 3 3 0 0 3 3 3 3 3 0 3 3 3 0 3 0 3 3 0 0 3
     3 3 3 3 3 0 0 3 3 3 3 3 0 0 3 0 3 3 3 3 0 3 3 0 3 0 0 0 3 3 3 0 0 0 0 0 3
     3 3 0 0 3 0 3 0 0 3 3 3 3 0 0 3 0 3 3 0 0 3 0 0 3 3 0 0 3 0 0 3 0 3 0 3 0
     3 0 0 3 3 0 3 3 3 0 3 3 3 3 3 0 0 3 3 3 0 3 0 3 3 0 3 0 3 0 0 3 3 0 3 3 0
     0 3 0 3 0 0 0 3 3 3 3 0 0 0 0 0 3 3 0 0 0 0 0 3 0 0 0 0 0 3 0 3 0 0 3 0 3
     0 0 3 0 3 0 3 0 3 0 0 0 0 3 0 0 0 0 0 3 3 0 0 3 3 0 0 0 0 0 3 0 0 0 0 0 0
     0 0 3 0 0 0 0 0 0 3 3 3 0 0 0 0 3 0 3 0 0 3 0 3 3 0 3 3 0 3 0 0 0 3 0 0 3
     0 0 0 0 0 0 0 3 0 3 0 0 0 0 3 0 3 0 3 3 3 3 0 3 0 0 3 0 3 3 3 0 3 0 3 3 3
     0 3 0 3 0 0 0 3 0 3 3 3 3 3 0 0 3 0 0 0 3 0 3 0 3 0 3 0 0 0 0 3 0 0 3 0 0
     0 3 3 3 3 0 0 0 0 3 0 3 3 3 0 0 3 3 0 0 3 0 3 3 3 0 3 3 0 0 0 3 3 3 0 0 0
     3 3 0 3 3 3 3 0 1 1 3 3 3 0 0 0 0 0 0 3 3 3 3 0 0 0 3 0 3 0 0 0 0 3 0 0 3
     0 0 3 3 0 3 0 3 3 3 3 0 0 3 0 3 0 0 0 0 0 0 3 3 3 0 0 0 0 0 0 3 0 3 3 3 3
     0 0 0 0 3 0 0 3 0 0 3 0 3 0 3 3 0 0 3 0 3 3 3 3 3 0 3 0 3 0 0 3 0 0 0 0 0
     3 3 3 0 0 0 3 0 3 0 3 0 0 0 0 3 0 0 0 3 0 3 0 3 0 0 0 3 3 0 0 0 0 0 0 1 3
     0 3 0 3 0 0 0 0 2 0 0 0 3 3 0]
    


```python
# Create a new DataFrame including predicted clusters and cryptocurrencies features.
# Concatentate the crypto_df and pcs_df DataFrames on the same columns.
clustered_df = pd.concat([crypto_df, pcs_df], axis=1)

#  Add a new column, "CoinName" to the clustered_df DataFrame that holds the names of the cryptocurrencies. 
clustered_df["CoinName"] = crypto_names.CoinName

#  Add a new column, "Class" to the clustered_df DataFrame that holds the predictions.
clustered_df['Class'] = predictions

# Print the shape of the clustered_df
print(clustered_df.shape)
clustered_df.head(10)
```

    (533, 9)
    




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



### Deliverable 4: Visualizing Cryptocurrencies Results

#### 3D-Scatter with Clusters


```python
# Creating a 3D-Scatter with the PCA data and the clusters
fig = px.scatter_3d(
    clustered_df
    , x="PC 1"
    , y="PC 2"
    , z="PC 3"
    , color='Class'
    , symbol="Class"
    , width=1200
    , height=1000
    , hover_name="CoinName"
    , hover_data=[clustered_df.Algorithm]
    , title="3D Scatter using PCA Parameters and K-Means Analysis with 4 Clusters"
)

fig.update_layout(legend=dict(x=0, y=1))
fig.show()
```




```python
# Create a table with tradable cryptocurrencies.

clustered_df.hvplot.table(columns=["CoinName", "Algorithm", "ProofType", "TotalCoinsMined", "TotalCoinSupply", "Class"], sortable=True, selectable=True, width=800, height=600)

```






<div id='3265'>





  <div class="bk-root" id="ff8f9e82-c12d-4224-ac85-b118b2f8aedc" data-root-id="3265"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"fee26eef-0880-47bc-8648-f8c3b5f9ef93":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"GridStack1","overrides":[],"properties":[{"default":"warn","kind":null,"name":"mode"},{"default":null,"kind":null,"name":"ncols"},{"default":null,"kind":null,"name":"nrows"},{"default":true,"kind":null,"name":"allow_resize"},{"default":true,"kind":null,"name":"allow_drag"},{"default":[],"kind":null,"name":"state"}]},{"extends":null,"module":null,"name":"click1","overrides":[],"properties":[{"default":"","kind":null,"name":"terminal_output"},{"default":"","kind":null,"name":"debug_name"},{"default":0,"kind":null,"name":"clears"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{"editor":{"id":"3280"},"field":"ProofType","formatter":{"id":"3279"},"title":"ProofType"},"id":"3281","type":"TableColumn"},{"attributes":{},"id":"3269","type":"StringFormatter"},{"attributes":{"format":"0,0.0[00000]"},"id":"3284","type":"NumberFormatter"},{"attributes":{},"id":"3270","type":"StringEditor"},{"attributes":{"editor":{"id":"3270"},"field":"CoinName","formatter":{"id":"3269"},"title":"CoinName"},"id":"3271","type":"TableColumn"},{"attributes":{},"id":"3295","type":"IntEditor"},{"attributes":{"editor":{"id":"3295"},"field":"Class","formatter":{"id":"3294"},"title":"Class"},"id":"3296","type":"TableColumn"},{"attributes":{},"id":"3274","type":"StringFormatter"},{"attributes":{"data":{"Algorithm":["Scrypt","Scrypt","X13","SHA-256","Ethash","Scrypt","X11","CryptoNight-V7","Ethash","Equihash","SHA-512","Multiple","SHA-256","SHA-256","Scrypt","X15","X11","Scrypt","Scrypt","Scrypt","Multiple","Scrypt","SHA-256","Scrypt","Scrypt","Scrypt","Quark","Groestl","Scrypt","Scrypt","Scrypt","Scrypt","Scrypt","X11","Scrypt","Groestl","Multiple","SHA-256","Scrypt","Scrypt","Scrypt","Scrypt","PoS","Scrypt","Scrypt","NeoScrypt","Scrypt","Scrypt","Scrypt","Scrypt","X11","Scrypt","X11","SHA-256","Scrypt","Scrypt","Scrypt","SHA3","Scrypt","HybridScryptHash256","Scrypt","Scrypt","SHA-256","Scrypt","X13","Scrypt","SHA-256","Scrypt","X13","NeoScrypt","Scrypt","Scrypt","Scrypt","Scrypt","Scrypt","Scrypt","X11","X11","SHA-256","Multiple","SHA-256","PHI1612","X11","SHA-256","SHA-256","SHA-256","X11","Scrypt","Scrypt","Scrypt","Scrypt","Lyra2REv2","Scrypt","X11","Multiple","SHA-256","X13","Scrypt","CryptoNight","CryptoNight","Shabal256","Counterparty","Scrypt","SHA-256","Groestl","Scrypt","Scrypt","Scrypt","X13","Scrypt","Scrypt","Scrypt","Scrypt","X13","Scrypt","Stanford Folding","X11","Multiple","QuBit","Scrypt","Scrypt","Scrypt","M7 POW","Scrypt","SHA-256","Scrypt","X11","SHA3","X11","Lyra2RE","SHA-256","QUAIT","X11","X11","Scrypt","Scrypt","Scrypt","Ethash","X13","Blake2b","SHA-256","X15","X11","SHA-256","BLAKE256","Scrypt","1GB AES Pattern Search","SHA-256","X11","Scrypt","SHA-256","SHA-256","NIST5","Scrypt","Scrypt","X11","Dagger","Scrypt","X11GOST","X11","Scrypt","SHA-256","Scrypt","PoS","Scrypt","X11","X11","SHA-256","SHA-256","NIST5","X11","Scrypt","POS 3.0","Scrypt","Scrypt","Scrypt","X13","X11","X11","Equihash","X11","Scrypt","CryptoNight","SHA-256","SHA-256","X11","Scrypt","Multiple","Scrypt","Scrypt","Scrypt","SHA-256","Scrypt","Scrypt","SHA-256D","PoS","Scrypt","X11","Lyra2Z","PoS","X13","X14","PoS","SHA-256D","Ethash","Equihash","DPoS","X11","Scrypt","X11","X13","X11","PoS","Scrypt","Scrypt","X11","PoS","X11","SHA-256","Scrypt","X11","Scrypt","Scrypt","X11","CryptoNight","Scrypt","Scrypt","Scrypt","Scrypt","Quark","QuBit","Scrypt","CryptoNight","Lyra2RE","Scrypt","SHA-256","X11","Scrypt","X11","Scrypt","CryptoNight-V7","Scrypt","Scrypt","Scrypt","X13","X11","Equihash","Scrypt","Scrypt","Lyra2RE","Scrypt","Dagger-Hashimoto","X11","Blake2S","X11","Scrypt","PoS","X11","NIST5","PoS","X11","Scrypt","Scrypt","Scrypt","SHA-256","X11","Scrypt","Scrypt","SHA-256","PoS","Scrypt","X15","SHA-256","Scrypt","POS 3.0","CryptoNight-V7","536","Argon2d","Blake2b","Cloverhash","CryptoNight","NIST5","X11","NIST5","Skein","Scrypt","X13","Scrypt","X11","X11","Scrypt","CryptoNight","X13","Time Travel","Scrypt","Keccak","SkunkHash v2 Raptor","X11","Skein","SHA-256","X11","Scrypt","VeChainThor Authority","Scrypt","PoS","Scrypt","Scrypt","Scrypt","Scrypt","Scrypt","Scrypt","Scrypt","CryptoNight","SHA-512","Ouroboros","X11","Equihash","NeoScrypt","X11","Scrypt","NeoScrypt","Lyra2REv2","Equihash","Scrypt","SHA-256","NIST5","PHI1612","Dagger","Scrypt","Quark","Scrypt","POS 2.0","Scrypt","SHA-256","X11","NeoScrypt","Ethash","NeoScrypt","X11","DPoS","NIST5","X13","Multiple","Scrypt","CryptoNight","CryptoNight","Ethash","NIST5","Quark","X11","CryptoNight-V7","Scrypt","Scrypt","Scrypt","X11","BLAKE256","X11","NeoScrypt","Quark","NeoScrypt","Scrypt","Scrypt","Scrypt","X11","X11","SHA-256","C11","POS 3.0","Ethash","Scrypt","CryptoNight","SkunkHash","Scrypt","CryptoNight","Scrypt","Dagger","Lyra2REv2","X13","Proof-of-BibleHash","SHA-256 + Hive","Scrypt","Scrypt","X11","C11","Proof-of-Authority","X11","XEVAN","Scrypt","VBFT","Ethash","CryptoNight","Scrypt","IMesh","NIST5","Scrypt","Scrypt","Equihash","Scrypt","Lyra2Z","Green Protocol","PoS","Scrypt","Semux BFT consensus","X11","Quark","PoS","CryptoNight","X16R","Scrypt","NIST5","Lyra2RE","XEVAN","Tribus","Scrypt","Lyra2Z","CryptoNight","CryptoNight Heavy","CryptoNight","Scrypt","Scrypt","Jump Consistent Hash","SHA-256D","CryptoNight","Scrypt","X15","Scrypt","Quark","SHA-256","DPoS","X16R","HMQ1725","X11","X16R","Quark","Quark","Scrypt","Lyra2REv2","Quark","Scrypt","Scrypt","CryptoNight-V7","Cryptonight-GPU","XEVAN","CryptoNight Heavy","X11","X11","Scrypt","PoS","SHA-256","Keccak","X11","X11","Scrypt","SHA-512","X16R","ECC 256K1","Equihash","XEVAN","Lyra2Z","SHA-256","XEVAN","X11","CryptoNight","Quark","Blake","Blake","Equihash","Exosis","Scrypt","Scrypt","Equihash","Quark","Equihash","Quark","Scrypt","QuBit","X11","Scrypt","XEVAN","SHA-256D","X11","SHA-256","X13","SHA-256","X11","DPoS","Scrypt","Scrypt","X11","NeoScrypt","Scrypt","Blake","Scrypt","SHA-256","Scrypt","X11","Scrypt","Scrypt","SHA-256","X11","SHA-256","Scrypt","Scrypt","Scrypt","Groestl","X11","Scrypt","PoS","Scrypt","Scrypt","X11","SHA-256","DPoS","Scrypt","Scrypt","NeoScrypt","SHA3-256","Multiple","X13","Equihash+Scrypt","DPoS","Ethash","DPoS","SHA-256","Leased POS","PoS","TRC10","PoS","SHA-256","Scrypt","CryptoNight","Equihash","Scrypt"],"Class":{"__ndarray__":"AAAAAAAAAAAAAAAAAwAAAAMAAAADAAAAAAAAAAMAAAADAAAAAwAAAAAAAAADAAAAAAAAAAAAAAADAAAAAAAAAAMAAAADAAAAAAAAAAAAAAADAAAAAwAAAAMAAAADAAAAAwAAAAAAAAADAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAAAAAAAAAAADAAAAAwAAAAMAAAADAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAMAAAADAAAAAAAAAAMAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAADAAAAAwAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAADAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAAAAAADAAAAAwAAAAMAAAADAAAAAAAAAAAAAAADAAAAAAAAAAMAAAADAAAAAAAAAAAAAAADAAAAAAAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAMAAAAAAAAAAwAAAAMAAAADAAAAAAAAAAMAAAADAAAAAwAAAAMAAAADAAAAAAAAAAAAAAADAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAMAAAADAAAAAAAAAAMAAAADAAAAAAAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAADAAAAAwAAAAMAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAAMAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAMAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAAAAAADAAAAAwAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAMAAAADAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAMAAAADAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAMAAAADAAAAAwAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAMAAAAAAAAAAAAAAAMAAAADAAAAAAAAAAAAAAADAAAAAAAAAAMAAAADAAAAAwAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAAAAAAMAAAADAAAAAwAAAAAAAAAAAAAAAAAAAAMAAAADAAAAAAAAAAMAAAADAAAAAwAAAAMAAAAAAAAAAQAAAAEAAAADAAAAAwAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAAAAwAAAAMAAAADAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAAAAAAAAAAAAAMAAAAAAAAAAAAAAAMAAAADAAAAAAAAAAMAAAAAAAAAAwAAAAMAAAADAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAMAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAMAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAAAAAAAAAAAAAMAAAAAAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAMAAAADAAAAAwAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAADAAAAAwAAAAAAAAAAAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAMAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAADAAAAAwAAAAAAAAA=","dtype":"int32","order":"little","shape":[533]},"CoinName":["42 Coin","404Coin","EliteCoin","Bitcoin","Ethereum","Litecoin","Dash","Monero","Ethereum Classic","ZCash","Bitshares","DigiByte","BitcoinDark","PayCoin","ProsperCoin","KoboCoin","Spreadcoin","Argentum","Aurora Coin","BlueCoin","MyriadCoin","MoonCoin","ZetaCoin","SexCoin","Quatloo","EnergyCoin","QuarkCoin","Riecoin","Digitalcoin ","BitBar","Catcoin","CryptoBullion","CannaCoin","CryptCoin","CasinoCoin","Diamond","Verge","DevCoin","EarthCoin","E-Gulden","Einsteinium","Emerald","Exclusive Coin","FlutterCoin","Franko","FeatherCoin","GrandCoin","GlobalCoin","GoldCoin","HoboNickels","HyperStake","Infinite Coin","IOCoin","IXcoin","KrugerCoin","LuckyCoin","Litebar ","MaxCoin","MegaCoin","MediterraneanCoin","MintCoin","MinCoin","MazaCoin","Nautilus Coin","NavCoin","NobleCoin","Namecoin","NyanCoin","OpalCoin","Orbitcoin","PotCoin","PhoenixCoin","Reddcoin","RonPaulCoin","StableCoin","SmartCoin","SuperCoin","SyncCoin","SysCoin","TeslaCoin","TigerCoin","TittieCoin","TorCoin","TerraCoin","UnbreakableCoin","Unobtanium","UroCoin","UnitaryStatus Dollar","UltraCoin","ViaCoin","VeriCoin","Vertcoin","WorldCoin","X11 Coin","Crypti","JouleCoin","StealthCoin","ZCC Coin","ByteCoin","DigitalNote ","BurstCoin","StorjCoin","MonaCoin","Neutron","FairCoin","Gulden","RubyCoin","PesetaCoin","Kore","Wild Beast Coin","Dnotes","Flo","8BIT Coin","Sativa Coin","ArtByte","Folding Coin","Ucoin","Unitus","CypherPunkCoin","OmniCron","Vtorrent","GreenCoin","Cryptonite","MasterCoin","SoonCoin","1Credit","IslaCoin","Nexus","MarsCoin ","Crypto","Anarchists Prime","Droidz","BowsCoin","Squall Coin","Song Coin","BitZeny","Diggits","Expanse","Paycon","Siacoin","Emercoin","EverGreenCoin","MindCoin","I0coin","Decred","Revolution VR","HOdlcoin","EDRCoin","Hitcoin","Gamecredits","DubaiCoin","CarpeDiemCoin","PWR Coin","BillaryCoin","GPU Coin","Adzcoin","SoilCoin","YoCoin","SibCoin","EuropeCoin","ZeitCoin","SwingCoin","SafeExchangeCoin","Nebuchadnezzar","Francs","BolivarCoin","Ratecoin","Revenu","Clockcoin","VIP Tokens","BitSend","Omni","Let it Ride","PutinCoin","iBankCoin","Frankywillcoin","MudraCoin","PizzaCoin","Lutetium Coin","Komodo","GoldBlocks","CarterCoin","Karbo","BitTokens","ZayedCoin","MustangCoin","ZoneCoin","Circuits of Value","RootCoin","DopeCoin","BitCurrency","DollarCoin","Swiscoin","Shilling","BuzzCoin","Opair","PesoBit","Halloween Coin","ZCoin","CoffeeCoin","RoyalCoin","GanjaCoin V2","TeamUP","LanaCoin","Elementrem","ZClassic","ARK","InsaneCoin","KiloCoin","ArtexCoin","EmberCoin","XenixCoin","FreeCoin","PLNCoin","AquariusCoin","Kurrent","Creatio","Eternity","Eurocoin","BitcoinFast","Stakenet","BitConnect Coin","MoneyCoin","Enigma","Cannabis Industry Coin","Russiacoin","PandaCoin","GameUnits","GAKHcoin","Allsafe","LiteCreed","OsmiumCoin","Bikercoins","HexxCoin","Klingon Empire Darsek","Internet of People","KushCoin","Printerium","PacCoin","Impeach","Citadel","Zilbercoin","FirstCoin","BeaverCoin","FindCoin","VaultCoin","Zero","OpenChat","Canada eCoin","Zoin","RenosCoin","DubaiCoin","VirtacoinPlus","TajCoin","Impact","EB3coin","Atmos","HappyCoin","Coinonat","MacronCoin","Condensate","Independent Money System","ArgusCoin","LomoCoin","ProCurrency","GoldReserve","BenjiRolls","GrowthCoin","ILCoin","Phreak","Degas Coin","HTML5 Coin","Ultimate Secure Cash","EquiTrader","QTUM","Quantum Resistant Ledger","Espers","Dynamic","Nano","ChanCoin","Dinastycoin","Denarius","DigitalPrice","Virta Unique Coin","Bitcoin Planet","Unify","BritCoin","SocialCoin","ArcticCoin","DAS","Linda","LeviarCoin","DeepOnion","Bitcore","gCn Coin","SmartCash","Signatum","Onix","Cream","Bitcoin Cash","Monoeci","Draftcoin","Vechain","Sojourn Coin","Stakecoin","NewYorkCoin","FrazCoin","Kronecoin","AdCoin","Linx","CoinonatX","Ethereum Dark","Sumokoin","Obsidian","Cardano","Regalcoin","BitcoinZ","TrezarCoin","Elements","TerraNovaCoin","VIVO Coin","Rupee","Bitcoin Gold","WomenCoin","Theresa May Coin","NamoCoin","LUXCoin","Pirl","Xios","Bitcloud 2.0","eBoost","KekCoin","BlackholeCoin","Infinity Economics","Pura","Innova","Ellaism","GoByte","Magnet","Lamden Tau","Electra","Bitcoin Diamond","SHIELD","Cash & Back Coin","UltraNote","BitCoal","DaxxCoin","Bulwark","Kalkulus","AC3","Lethean","GermanCoin","LiteCoin Ultra","PopularCoin","PhantomX","Photon","Sucre","SparksPay","Digiwage","GunCoin","IrishCoin","Trollcoin","Litecoin Plus","Monkey Project","Pioneer Coin","UnitedBitcoin","Interzone","TokenPay","1717 Masonic Commemorative Token","My Big Coin","TurtleCoin","MUNcoin","Unified Society USDEX","Niobio Cash","ShareChain","Travelflex","KREDS","Tokyo Coin","BiblePay","LitecoinCash","BitFlip","LottoCoin","Crypto Improvement Fund","Stipend","Poa Network","Pushi","Ellerium","Velox","Ontology","Callisto Network","BitTube","Poseidon","Aidos Kuneen","Bitspace","Briacoin","Ignition","Bitrolium","MedicCoin","Alpenschillling","Bitcoin Green","Deviant Coin","Abjcoin","Semux","FuturoCoin","Carebit","Zealium","Monero Classic","Proton","iDealCash","Jumpcoin","Infinex","Bitcoin Incognito","KEYCO","HollyWoodCoin","GINcoin","PlatinCoin","Loki","Newton Coin","Swisscoin","Xt3ch","MassGrid","TheVig","PluraCoin","EmaratCoin","Dekado","Lynx","Poseidon Quark","BitcoinWSpectrum","Muse","Motion","PlusOneCoin","Axe","Trivechain","Dystem","Giant","Peony Coin","Absolute Coin","Vitae","HexCoin","TPCash","Webchain","Ryo","Urals Coin","Qwertycoin","ARENON","EUNO","MMOCoin","Ketan","Project Pai","XDNA","PAXEX","Azart","ThunderStake","Kcash","Xchange","Acute Angle Cloud","CrypticCoin","Bettex coin","Actinium","Bitcoin SV","BitMoney","Junson Ming Chan Coin","FREDEnergy","HerbCoin","Universal Molecule","Lithium","PirateCash","Exosis","Block-Logic","Oduwa","Beam","Galilel","Bithereum","Crypto Sports","Credit","SLICE","Dash Platinum","Nasdacoin","Beetle Coin","Titan Coin","Award","BLAST","Bitcoin Rhodium","GlobalToken","Insane Coin","ALAX","LiteDoge","SolarCoin","TruckCoin","UFO Coin","OrangeCoin","BlakeCoin","BitstarCoin","NeosCoin","HyperCoin","PinkCoin","Crypto Escudo","AudioCoin","IncaKoin","Piggy Coin","Crown Coin","Genstake","SmileyCoin","XiaoMiCoin","Groestlcoin","CapriCoin"," ClubCoin","Radium","Bata","Pakcoin","Creditbit ","OKCash","Lisk","HiCoin","WhiteCoin","FriendshipCoin","Fiii","JoinCoin","Triangles Coin","Vollar","EOS","Reality Clash","Oxycoin","TigerCash","Waves","Particl","BitTorrent","Nxt","ZEPHYR","Gapcoin","Beldex","Horizen","BitcoinPlus"],"ProofType":["PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW","PoW","PoW/PoS","PoW","PoW","PoW","PoS","PoW","PoW/PoS","PoS","PoW","PoW/PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoW","PoW","PoW","PoW","PoW","PoW/PoS","PoW","PoW","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoW","PoC","PoW/PoS","PoW","PoW","PoW","PoW","PoW","PoW","PoS","PoS/PoW/PoT","PoW","PoW","PoW","PoW","PoW","PoW/PoS","PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoW","PoW/PoS","PoW","PoW","PoS","PoW","PoW/PoS","PoS","PoW/PoS","PoW","PoW","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW","PoW","PoS","PoW/PoS","PoW","PoS","PoW","PoS","PoW/PoS","PoW","PoW","PoW","PoW","PoW/PoS","PoW/PoS","PoW","PoST","PoW","PoW","PoW/PoS","PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoW","PoC","PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoS","PoW","PoW/PoS","PoW","PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoW","PoS","PoW","PoW","PoW","PoW/PoS","PoW","PoW","PoW","PoW","PoW","PoW/PoS","PoW/nPoS","PoW","PoW","PoW","PoW/PoS","PoW","PoS/PoW","PoW","PoW","PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoW","PoW/PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoS","PoW","PoW","PoW","PoW","PoW/PoS","PoS","PoW/PoS","PoC","PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoS","PoW","PoS","dPoW/PoW","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoS","PoW","PoW/PoW","PoW","PoW/PoS","PoS","PoW/PoS","PoW/PoS","PoW","PoS","PoS","PoW/PoS","PoS","PoW/PoS","PoW","PoW","DPoS","PoW/PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoS","PoW/PoS","PoW/PoS","PoW","PoS","PoW/PoS","PoW/PoS","PoW/PoS","TPoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoS","PoW/PoS","PoW/PoS","PoS","PoW/PoS","PoW","PoW","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoS","PoW/PoS ","PoW","PoS","PoW","PoW","PoW/PoS","PoW","PoW","PoS","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoS","PoW/PoS","PoW","PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoS","PoW/PoS","PoW","PoS","PoW","PoS","PoW/PoS","PoW/PoS","PoS","PoW","PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoW","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoS","Proof of Authority","PoW","PoS","PoW","PoW","PoW","PoW","PoW","PoW/PoS","PoW/PoS","PoW","PoS","PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoS","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoS","PoW/PoS","PoS","PoW","PoW","PoW","PoW","PoW/PoS","DPoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoS","PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoS","PoW","PoW","PoW","PoS","PoS","PoW and PoS","PoW","PoW","PoW/PoS","PoW","PoW","PoW","PoW","PoS","POBh","PoW + Hive","PoW","PoW","PoW","PoW/PoS","PoA","PoW/PoS","PoW/PoS","PoS","PoS","PoW","PoW","PoW","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoS","PoW","PoS","PoS","PoW/PoS","DPoS","PoW","PoW/PoS","PoS","PoW","PoS","PoW/PoS","PoW","PoW","PoS/PoW","PoW","PoS","PoW","PoW","PoW","PoW","PoW/PoS","PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoS","HPoW","PoS","PoS","PoS","PoW","PoW","PoW","PoW/PoS","PoS","PoW/PoS","PoS","PoW/PoS","PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoW","PoS","PoW/PoS","PoS","PoS","PoW","PoW/PoS","PoS","PoW","PoW/PoS","Zero-Knowledge Proof","PoW","DPOS","PoW","PoS","PoW","PoW","Pos","PoS","PoW","PoW/PoS","PoW","PoW","PoS","PoW","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoS","PoW/PoS","PoW","PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW","PoW","PoW/PoS","DPoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoS","PoW","PoW","Proof of Trust","PoW/PoS","DPoS","PoS","PoW/PoS","PoW/PoS","DPoC","PoW","PoW/PoS","PoW","DPoS","PoW","DPoS","PoS","LPoS","PoS","DPoS","PoS/LPoS","DPoS","PoW/PoS","PoW","PoW","PoS"],"TotalCoinSupply":["42","532000000","314159265359","21000000","0","84000000","22000000","0","210000000","21000000","3600570502","21000000000","22000000","12500000","21000000","350000000","20000000","64000000","16768584","0","2000000000","384000000000","169795588","250000000","100000000","0","247000000","84000000","48166000","500000","21000000 ","1000000","13140000","18000000","40000000000","4380000","16555000000","21000000000","13500000000","21000000 ","299792458","32000000","0","0","11235813","336000000","1420609614","70000000","72245700","120000000","0","90600000000","22000000","21000000","265420800","20000000","1350000","100000000","42000000","200000000","0","10000000","2419200000","16180000","0","15000000000","21000000","337000000","0","3770000","420000000","98000000","0","21000000","250000000","51200000","0","1000","888000000","100000000","47011968","2300000000","10000000","42000000","80000000","250000","0","1600000000","100000000","23000000","0","84000000","265420800","5500000","0","45000000","0","1000000000","184467440735","10000000000","2158812800","500000000","105120000","68000000","0","1680000000","0","166386000","12000000","2628000","500000000","160000000","0","10000000","1000000000","1000000000","20000000","0","0","3371337","20000000","10000000000","1840000000","619478","21000000","92000000000","0","78000000","33000000","65789100","53760000","5060000","21000000","0","210240000","250000000","100000000","16906397","50000000","0","1000000000","26298000","16000000","21000000","21000000","210000000","81962100","22000000","26550000000","84000000","10500000","21626280000 ","0","42000000","221052632","84000000","30000000","168351300","24000000","384000000"," 99000000000","40000000","2147483647","20000000","20000000","25000000","75000000","222725000","525000000","90000000","139000000","616448","33500000","2000000000","44333333","100000000","200000000","25000000","657000000","200000000","50000000","90000000","10000000","21000000","9736000","3000000","21000000","1200000000","0","200000000","0","10638298","3100000000","30000000","20000000000","74000000","0","1500000000","21400000","39999898","2500124","100000000","301000000","7506000000","26205539","21000000","125000000","30000000","10000000000","500000000","850000000","3853326.77707314","50000000","38540000 ","42000000","228000000","20000000","60000000","20000000","33000000","76500000","28000000","650659833","5000000","21000000","144000000","32514916898","13000000","3315789","15000000","78835200","2714286","25000000","9999999","500000000","21000000","9354000","20000000","100000000000","21933333","185000000","55000000","110000000","3360000","14524851.4827","1000000000","17000000","1000000000","100000000 ","21000000","34426423","2232901","100000000","36900000","110000000","4000000000","110290030","100000000","48252000","400000000","500000000","21212121","28600000","1000000000","75000000000","40000000","35520400","2000000000","2500000000","30000000","105000000","90000000000","200084200","72000000","100000000","105000000","50000000000","0","340282367","30000000","2000000000","10000000","100000000","120000000","100000000","19276800","30000000"," 75000000","60000000","18900000","50000000000","54000000","18898187.6216583","21000000","200000000000","5000000000","137500000","1100000000","100000000","21000000","9507271","17405891.19707116","86712634466","10500000000","61599965","0","20000000","84000000","100000000","100000000","48252000","4200000","88888888","91388946","45000000000","27000000","21000000000","400000000","1800000000","15733333","27000000","24000000","21000000","25000000000","100000000","1200000000","60000000","156306732.71","21000000","200000000","100000000","21000000","14788275.991","9000000000","350000000","45000000","280000000","31800000","144000000","500000000","30000000000","210000000","660000000","210000000","85000000000","12500000","10000000000","27716121","20000000","550000000","999481516","50000000000","150000000","4999999999","50000000"," 90000000000","19800000","21000000","120000000","500000000","64000000","900000000","4000000","21000000","23000000","20166000","23000000","25000000","1618033","30000000","1000000000000","16600000","232000000","336000000","10000000000","100000000","1100000000","800000000","5200000000","840000000","40000000","18406979840","500000000","19340594","252460800","25000000","60000000","124000000","1000000000","6500000000","1000000000","21000000","25000000","50000000","3000000","5000000","70000000","500000000","300000000","21000000","88000000","30000000","100000000","100000000","200000000","80000000","18400000","45000000","5121951220","21000000","26280000","21000000","18000000","26000000","10500000","600000518","150000000","184000000000","10200000000","44000000","168000000","100000000","1000000000","84000000","90000000","92000000000","650000000 ","100262205","18081806 ","22075700","21000000","21000000","82546564","21000000","5151000","16880000000","52500000","100000000","22105263","1000000000","1750000000","88188888","210000000","184470000000","55000000","50000000","260000000","210000000","2100000000","366000000","100000000","25000000","18000000000","1000000000","100000000","1000000000","7600000000","50000000","84000000","21000000","70000000000","0","8080000000","54000000","105120001.44","25228800","105000000","21000000","120000000","21000000","262800000","19035999","30886000","13370000","74800000000","100000000","19700000","84000000","500000000","5000000000","420000000","64000000","2100000","168000000","30000000","1000000000","35000000000","98100000000","0","4000000000","200000000","7000000000","54256119","21000000","0","500000000","1000000000","10500000000","190000000","1000000000","42000000","15000000","50000000000","400000000","105000000","208000000","160000000","9000000","5000000","182000000","16504333","105000000","159918400","10008835635","300000000","60168145","5000000000","2800000","120000","2100000000","0","24487944","0","1000000000","100000000","8634140","990000000000","1000000000","2000000000","250000000","1400222610","21000000","1000000"],"TotalCoinsMined":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[533]}},"selected":{"id":"3268"},"selection_policy":{"id":"3302"}},"id":"3267","type":"ColumnDataSource"},{"attributes":{"source":{"id":"3267"}},"id":"3301","type":"CDSView"},{"attributes":{},"id":"3290","type":"StringEditor"},{"attributes":{},"id":"3275","type":"StringEditor"},{"attributes":{},"id":"3285","type":"NumberEditor"},{"attributes":{},"id":"3289","type":"StringFormatter"},{"attributes":{},"id":"3302","type":"UnionRenderers"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer06305","sizing_mode":"stretch_width"},"id":"3306","type":"Spacer"},{"attributes":{},"id":"3280","type":"StringEditor"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer06304","sizing_mode":"stretch_width"},"id":"3266","type":"Spacer"},{"attributes":{"editor":{"id":"3275"},"field":"Algorithm","formatter":{"id":"3274"},"title":"Algorithm"},"id":"3276","type":"TableColumn"},{"attributes":{},"id":"3294","type":"NumberFormatter"},{"attributes":{},"id":"3279","type":"StringFormatter"},{"attributes":{"editor":{"id":"3290"},"field":"TotalCoinSupply","formatter":{"id":"3289"},"title":"TotalCoinSupply"},"id":"3291","type":"TableColumn"},{"attributes":{"children":[{"id":"3266"},{"id":"3299"},{"id":"3306"}],"margin":[0,0,0,0],"name":"Row06300","tags":["embedded"]},"id":"3265","type":"Row"},{"attributes":{"columns":[{"id":"3271"},{"id":"3276"},{"id":"3281"},{"id":"3286"},{"id":"3291"},{"id":"3296"}],"height":600,"reorderable":false,"source":{"id":"3267"},"view":{"id":"3301"},"width":800},"id":"3299","type":"DataTable"},{"attributes":{},"id":"3268","type":"Selection"},{"attributes":{"editor":{"id":"3285"},"field":"TotalCoinsMined","formatter":{"id":"3284"},"title":"TotalCoinsMined"},"id":"3286","type":"TableColumn"}],"root_ids":["3265"]},"title":"Bokeh Application","version":"2.4.2"}};
    var render_items = [{"docid":"fee26eef-0880-47bc-8648-f8c3b5f9ef93","root_ids":["3265"],"roots":{"3265":"ff8f9e82-c12d-4224-ac85-b118b2f8aedc"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>




```python
# Scaling data to create the scatter plot with tradable cryptocurrencies.
# Print the total number of tradable currencies in the Clustered_df Dataframe
print(f"There are {len(clustered_df)} tradable currencies in the dataframe")
```

    There are 533 tradable currencies in the dataframe
    


```python
# Select the columns to scale
totals = clustered_df[['TotalCoinSupply', 'TotalCoinsMined']]
```


```python
# Use the MinMaxScaler() to scale total coin supply and mined
scaled_totals = MinMaxScaler(feature_range=(0,1)).fit_transform(totals)
scaled_totals

```




    array([[4.20000000e-11, 5.94230127e-03],
           [5.32000000e-04, 7.00182308e-03],
           [3.14159265e-01, 3.53420682e-02],
           ...,
           [1.40022261e-03, 6.92655266e-03],
           [2.10000000e-05, 5.94962775e-03],
           [1.00000000e-06, 5.94243008e-03]])




```python
# Create a new DataFrame that has the scaled data with the clustered_df DataFrame index.
clustered_scaled_df = clustered_df.copy()
clustered_scaled_df.TotalCoinSupply = scaled_totals[:,0]
clustered_scaled_df.TotalCoinsMined = scaled_totals[:,1] 

# Add the "CoinName" column from the clustered_df DataFrame to the new DataFrame.
clustered_scaled_df = clustered_scaled_df[["TotalCoinSupply", "TotalCoinsMined", "CoinName", "Class"]]

# Add the "Class" column from the clustered_df DataFrame to the new DataFrame. 
clustered_scaled_df.head(10)

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






<div id='2849'>





  <div class="bk-root" id="16c7e51d-a01b-4a11-b304-b35161a8e072" data-root-id="2849"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"b3c48fa1-5c5e-4c19-ba0c-f82ebb5ea144":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"GridStack1","overrides":[],"properties":[{"default":"warn","kind":null,"name":"mode"},{"default":null,"kind":null,"name":"ncols"},{"default":null,"kind":null,"name":"nrows"},{"default":true,"kind":null,"name":"allow_resize"},{"default":true,"kind":null,"name":"allow_drag"},{"default":[],"kind":null,"name":"state"}]},{"extends":null,"module":null,"name":"click1","overrides":[],"properties":[{"default":"","kind":null,"name":"terminal_output"},{"default":"","kind":null,"name":"debug_name"},{"default":0,"kind":null,"name":"clears"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#e5ae38"},"line_alpha":{"value":0.1},"size":{"value":12.24744871391589},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"2941","type":"Scatter"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer05500","sizing_mode":"stretch_width"},"id":"3130","type":"Spacer"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#e5ae38"},"line_alpha":{"value":0.2},"size":{"value":12.24744871391589},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"2942","type":"Scatter"},{"attributes":{"fill_alpha":{"value":0.8},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":0.8},"hatch_color":{"value":"#e5ae38"},"line_alpha":{"value":0.5},"size":{"value":12.24744871391589},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"2939","type":"Scatter"},{"attributes":{"coordinates":null,"data_source":{"id":"2936"},"glyph":{"id":"2939"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2942"},"nonselection_glyph":{"id":"2940"},"selection_glyph":{"id":"2941"},"view":{"id":"2944"}},"id":"2943","type":"GlyphRenderer"},{"attributes":{},"id":"2867","type":"BasicTicker"},{"attributes":{"source":{"id":"2914"}},"id":"2922","type":"CDSView"},{"attributes":{"source":{"id":"2936"}},"id":"2944","type":"CDSView"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer05499","sizing_mode":"stretch_width"},"id":"2850","type":"Spacer"},{"attributes":{"label":{"value":"1"},"renderers":[{"id":"2921"}]},"id":"2935","type":"LegendItem"},{"attributes":{},"id":"2888","type":"BasicTickFormatter"},{"attributes":{"label":{"value":"0"},"renderers":[{"id":"2900"}]},"id":"2913","type":"LegendItem"},{"attributes":{},"id":"2862","type":"LinearScale"},{"attributes":{"coordinates":null,"group":null,"text":"K-Means Scatter of Cryptocurrencies, Total Coins Mined vs. Total Coin Supply","text_color":"black","text_font_size":"12pt"},"id":"2858","type":"Title"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.2},"size":{"value":12.24744871391589},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"2899","type":"Scatter"},{"attributes":{"coordinates":null,"data_source":{"id":"2914"},"glyph":{"id":"2917"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2920"},"nonselection_glyph":{"id":"2918"},"selection_glyph":{"id":"2919"},"view":{"id":"2922"}},"id":"2921","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"2866"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"2869","type":"Grid"},{"attributes":{"fill_alpha":{"value":0.8},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.8},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.5},"size":{"value":12.24744871391589},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"2896","type":"Scatter"},{"attributes":{},"id":"2909","type":"UnionRenderers"},{"attributes":{},"id":"2982","type":"UnionRenderers"},{"attributes":{},"id":"2871","type":"BasicTicker"},{"attributes":{},"id":"2915","type":"Selection"},{"attributes":{"click_policy":"mute","coordinates":null,"group":null,"items":[{"id":"2913"},{"id":"2935"},{"id":"2959"},{"id":"2985"}],"location":[0,0],"title":"Class"},"id":"2912","type":"Legend"},{"attributes":{"fill_alpha":{"value":0.8},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#e5ae38"},"line_alpha":{"value":0.5},"size":{"value":12.24744871391589},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"2940","type":"Scatter"},{"attributes":{},"id":"2956","type":"UnionRenderers"},{"attributes":{"label":{"value":"2"},"renderers":[{"id":"2943"}]},"id":"2959","type":"LegendItem"},{"attributes":{},"id":"2864","type":"LinearScale"},{"attributes":{"data":{"Class":{"__ndarray__":"AwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAMAAAA=","dtype":"int32","order":"little","shape":[239]},"TotalCoinSupply":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[239]},"TotalCoinsMined":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[239]}},"selected":{"id":"2961"},"selection_policy":{"id":"2982"}},"id":"2960","type":"ColumnDataSource"},{"attributes":{"axis_label":"TotalCoinSupply","coordinates":null,"formatter":{"id":"2891"},"group":null,"major_label_policy":{"id":"2892"},"ticker":{"id":"2871"}},"id":"2870","type":"LinearAxis"},{"attributes":{"axis_label":"TotalCoinsMined","coordinates":null,"formatter":{"id":"2888"},"group":null,"major_label_policy":{"id":"2889"},"ticker":{"id":"2867"}},"id":"2866","type":"LinearAxis"},{"attributes":{"label":{"value":"3"},"renderers":[{"id":"2967"}]},"id":"2985","type":"LegendItem"},{"attributes":{"data":{"Class":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"int32","order":"little","shape":[290]},"TotalCoinSupply":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[290]},"TotalCoinsMined":{"__ndarray__":"qPiOWfRWeD8cYvKR8a18P0/E+wVbGKI/lL9Wn3ZgeD8Tn0Dcwc6BPzvmnbxPWHg/pWjVi5VjeD87g7sK2XF4P5mzbnMMang/payitfvzej8zTpSIhth4P2/nudD/Vng/0B61awxYeD+zKjBCTJunPxZVqwZvWng/fyE6Pe9ceD80uDFmNT16P3YWvTzxtHg/cTyZkD4+fz/DhKrxiWl4P8B/XF2ml3g/Gc9GqMH6mz9lRtq4I/p+Pzz9xXn9Z3g/3ihhX4mceD/2nvGQ6WZ4P5FuYGtYWng/N6hye2hBeT9h9dM606R4P1SdIsAcHaI/Qg7UBFiMeD++K+Cn9FZ4P/w//qR1q3g/o/seNhQDfz8l/CFHdlh4P8jywagG3Hw/BPTTrNiLeD+fG7dEq3h4PxA/X9tFXng/uQv0Nj7AeD8iyllhzHl4P0vXp3HF8ng/r9R5Qd/Lfz+MrP971Yx4PxFPxiYlgHg/tJBJI/aOeD/4csWck3N4P+ZsoHgVWXg/6Z1aPNYOeT+xD4P6f1h4P41l2jVtXng/QVjcWMlZeD88IMtCLGN4P5arJVeMWHg/Aqw/bfqZeD8FbsjB+V94P001GBJNV3g/uQv0Nj7AeD9OTKU2N294P1uITZ9KhHg/JFESWTNleD8EZ9Qe3mF4P4lKdHbRWng/c1cokzRkkT/hN33OWlx4P6i4LMo7cJA/UXw/2W1geD/pk2R4koF4P2/JSYDpYXg/cnZeMK4Mpj/PUbUhkFt4P4yI4TUBloA/vnUeHQNseD8q5VYcoOR4PzEvL5Q2WHg/nCVInX9oeD9Gxe900a54P879i8ahcHg/Q++pyZV/eD9lfIo9DK97P0dfuFu4W3g/uQv0Nj7AeD+faUYIOFx4P1tWBk6zCns/SFzTZ1dneD/zIiodZ4R4P/wHaNSUV3g/cjTVmKVXeD9cK45HrFl4P+tOuzb/WHg/QV5KvYUJeT9x07mJVRN7P256jjcIN5o/FuawUeekeD8B2icTQHp4P9N9PXGhbXk/LbDvSInzeD+TIgY4lll4P7kL9DY+wHg/v27+KbdpeD8CMm8dWcp8P9F1hf7gyHg/P4IHcERqeD/nymx5Nzi5P1MhnvUCW3g//I3ORpmLeD9oKOyl8mh4P4RhNpiIWXg/vnUeHQNseD9ZTJhMWF14P902wxQHZHg/2yw440RseD/i73nfiqZ4P9bVzwWfYng/P3R9MnJieD86pYbsz1d4P9boAYDGX3g/hojUpwptpD+RaypxnFp4P+o4ehJyWng/a080SQdieD/KRYnjx3Z4PzuAae8vcHg/N8PuNp1ZeD8IhTuv6Vx4P3fvn8BmY3g/bqCaZ0dXeD+216QYCFp4P3i+LprFyng/ycxXWT9meD//4pYZ13N8PybuKk/pfHg/DIqoNdBkeD9Fz5tzyGN4Pz8OKIRvy3g/hrAcwffLeD8OWEMrnW14Pzb4O/ea/Xk/miHH5A3XeD8jNG55m1x4PxZ8OdspWHg/yTKfG3tXeD8sYWhQ0cB4PyaqCbEIaXg/3SldybKNeT/VvCdy8mJ4P2uiclpxbXg/3jRiRwjypz+oslg12GF4P7kL9DY+wHg/g1DK5q6HnT8tKAqwWFt4P/xY3qo5mXg/dwO5YB5eeD+WyufpWG14P/aGQOkdxY4//fgM0wdueD8BbCMvo8h4P+8KDBjKiHg/f/fxPTBkeD9e0BDSmmp4P10Y7JtOja8/jCRafipbeD+4Ps4Dh2t4P8OYvGZgW3g/Hc+7zkZxeD8+VzZaJl+gP2HOXdaZXng/Tgl2d0AXeT/qLpDPJ1h4PwqKz0Q5cHg/r47D0Kf0qz8yGN6Y37d4Py2Aw9oppXo/jvUgEbNfeD87H0TDN1l4PwL8s0pKd3g/ybgB+O9jeD/blrqtLmh4P/JpKph2rY4/ATsoVyx/eD93kcvpR4Z5P4T6kYOApKE/oPpM3zEYeT86l37vzMt4P8L/effmZHg/2xoLk85oeD9FpdsEGkWGP4BHnLHRX3g/lvdMioSGeD9hZBOpsnN4Py9mxOA7xXo/QawGZZdZeD/Z8oVDmFt4P5QWbEp7bHg/+nsMdqhYeD9uf17T2WB4P7mRl4I4Tnk/fXlV025JeT+zKugCzWJ4P+HRjF0/Lnk/HR3mpjxZeD+8xXdZZVd4PyZsvHLgcXg/eDNSRTkEez+ljSSzeGV4P8KuS0XYV3g/C22G+TJYeD//BlJsY3B5Pw5XN9BGYXg/Qms47fxteD/31FKEW2F4P126Ky5AWHg/JViHkxXqeD8mYH2MnWJ4Pw4fVFWXW3g/DdRJOF8dfj+YTxWmb2J4P7Ks8VK1f3g/rVAi7ZiSkD+wWlXr/F54P3d+QFHzeXg/Nr+hMrJteD9IuXEa13Z4PyB6J27VibU/O1QqetBZeD+7WuWk0pN4P6liO5Qcang//Nf7jNR9eD+ZK1hRaV54P2FQ1t4vXXg/xviFMw1YeD8UoO/+/WR4P93FO2uAnHg/5nps36RceD8fceY3Pmt4P3YAhllad3g/itzE4GrIeD9RD61cxmB4P0a/GcPfW3g/eNtpg7JbeD+sQ20M97J8P//ilhnXc3w//+KWGddzfD9xZqvOGV54P99QQ3J/Mnk/hhVYd8LHlz/+KWFXInx4PyAa/HWDZng/lhKPdy9qeD+s2cvQ/Vh4P8cMCgHd46I/T8CJPLVXeD/gup1slmx4P/1XKuFpKHk/UCvkargGfD8V4H6wMGd4P2RbpOH8b3g//+KWGddzfD85fBiM/9WVPzemX3ZbVnk/SuXPvK1aeD9phO+Cnmx4P8vn7DiUW3g/BOT6URhheD8tyNONYiJ6P4P9CcWNX3w/s8fm5tuHmD+1jgFbVV96P7tACaoglng/BObxTKcAej8iU3bU9yp5P9eZX3MOxHg/4uz2T/paeD8yOU/Rv2h4P+sXvW6UpXg/32AYQ1DVeD/wqMt5BF6QPxCyLZtJYHk/6VJhUyJYeD97mZhIGld4P/7hL6V4iXw/fHB53rH0fD//4pYZ13N8P7kL9DY+wHg/pqgAgbpgeD//4pYZ13N8P5TkGO5cSIA/XnGO1axmeD9eTHjtFld4Pw==","dtype":"float64","order":"little","shape":[290]}},"selected":{"id":"2894"},"selection_policy":{"id":"2909"}},"id":"2893","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.8},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":0.8},"hatch_color":{"value":"#6d904f"},"line_alpha":{"value":0.5},"size":{"value":12.24744871391589},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"2963","type":"Scatter"},{"attributes":{"axis":{"id":"2870"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"2873","type":"Grid"},{"attributes":{},"id":"2892","type":"AllLabels"},{"attributes":{"source":{"id":"2893"}},"id":"2901","type":"CDSView"},{"attributes":{"coordinates":null,"data_source":{"id":"2960"},"glyph":{"id":"2963"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2966"},"nonselection_glyph":{"id":"2964"},"selection_glyph":{"id":"2965"},"view":{"id":"2968"}},"id":"2967","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"value":0.8},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":0.8},"hatch_color":{"value":"#fc4f30"},"line_alpha":{"value":0.5},"size":{"value":12.24744871391589},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"2917","type":"Scatter"},{"attributes":{"below":[{"id":"2866"}],"center":[{"id":"2869"},{"id":"2873"}],"height":500,"left":[{"id":"2870"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"2900"},{"id":"2921"},{"id":"2943"},{"id":"2967"}],"right":[{"id":"2912"}],"sizing_mode":"fixed","title":{"id":"2858"},"toolbar":{"id":"2880"},"width":800,"x_range":{"id":"2851"},"x_scale":{"id":"2862"},"y_range":{"id":"2852"},"y_scale":{"id":"2864"}},"id":"2857","subtype":"Figure","type":"Plot"},{"attributes":{"fill_alpha":{"value":0.8},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.5},"size":{"value":12.24744871391589},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"2897","type":"Scatter"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#fc4f30"},"line_alpha":{"value":0.1},"size":{"value":12.24744871391589},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"2919","type":"Scatter"},{"attributes":{},"id":"2937","type":"Selection"},{"attributes":{},"id":"2874","type":"SaveTool"},{"attributes":{"children":[{"id":"2850"},{"id":"2857"},{"id":"3130"}],"margin":[0,0,0,0],"name":"Row05495","tags":["embedded"]},"id":"2849","type":"Row"},{"attributes":{"fill_alpha":{"value":0.8},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#6d904f"},"line_alpha":{"value":0.5},"size":{"value":12.24744871391589},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"2964","type":"Scatter"},{"attributes":{},"id":"2875","type":"PanTool"},{"attributes":{},"id":"2878","type":"ResetTool"},{"attributes":{},"id":"2876","type":"WheelZoomTool"},{"attributes":{},"id":"2894","type":"Selection"},{"attributes":{"coordinates":null,"data_source":{"id":"2893"},"glyph":{"id":"2896"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2899"},"nonselection_glyph":{"id":"2897"},"selection_glyph":{"id":"2898"},"view":{"id":"2901"}},"id":"2900","type":"GlyphRenderer"},{"attributes":{"data":{"Class":{"__ndarray__":"AQAAAAEAAAABAAAA","dtype":"int32","order":"little","shape":[3]},"TotalCoinSupply":{"__ndarray__":"lPYGX5hMdT+8eapDboZLP3sUrkfhenQ/","dtype":"float64","order":"little","shape":[3]},"TotalCoinsMined":{"__ndarray__":"eHs1s8Ogfz9wjSUfGfd6PwAAAAAAAAAA","dtype":"float64","order":"little","shape":[3]}},"selected":{"id":"2915"},"selection_policy":{"id":"2932"}},"id":"2914","type":"ColumnDataSource"},{"attributes":{"overlay":{"id":"2879"}},"id":"2877","type":"BoxZoomTool"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#6d904f"},"line_alpha":{"value":0.2},"size":{"value":12.24744871391589},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"2966","type":"Scatter"},{"attributes":{"fill_alpha":{"value":0.8},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#fc4f30"},"line_alpha":{"value":0.5},"size":{"value":12.24744871391589},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"2918","type":"Scatter"},{"attributes":{},"id":"2961","type":"Selection"},{"attributes":{},"id":"2889","type":"AllLabels"},{"attributes":{"end":1.0625,"reset_end":1.0625,"reset_start":-0.0625,"start":-0.0625,"tags":[[["TotalCoinsMined","TotalCoinsMined",null]]]},"id":"2851","type":"Range1d"},{"attributes":{"source":{"id":"2960"}},"id":"2968","type":"CDSView"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#6d904f"},"line_alpha":{"value":0.1},"size":{"value":12.24744871391589},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"2965","type":"Scatter"},{"attributes":{"callback":null,"renderers":[{"id":"2900"},{"id":"2921"},{"id":"2943"},{"id":"2967"}],"tags":["hv_created"],"tooltips":[["Class","@{Class}"],["TotalCoinsMined","@{TotalCoinsMined}"],["TotalCoinSupply","@{TotalCoinSupply}"]]},"id":"2853","type":"HoverTool"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#fc4f30"},"line_alpha":{"value":0.2},"size":{"value":12.24744871391589},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"2920","type":"Scatter"},{"attributes":{},"id":"2932","type":"UnionRenderers"},{"attributes":{"end":1.1,"reset_end":1.1,"reset_start":-0.1,"start":-0.1,"tags":[[["TotalCoinSupply","TotalCoinSupply",null]]]},"id":"2852","type":"Range1d"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.1},"size":{"value":12.24744871391589},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"2898","type":"Scatter"},{"attributes":{},"id":"2891","type":"BasicTickFormatter"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"2879","type":"BoxAnnotation"},{"attributes":{"data":{"Class":{"__ndarray__":"AgAAAA==","dtype":"int32","order":"little","shape":[1]},"TotalCoinSupply":{"__ndarray__":"rkfhehSu7z8=","dtype":"float64","order":"little","shape":[1]},"TotalCoinsMined":{"__ndarray__":"AAAAAAAA8D8=","dtype":"float64","order":"little","shape":[1]}},"selected":{"id":"2937"},"selection_policy":{"id":"2956"}},"id":"2936","type":"ColumnDataSource"},{"attributes":{"tools":[{"id":"2853"},{"id":"2874"},{"id":"2875"},{"id":"2876"},{"id":"2877"},{"id":"2878"}]},"id":"2880","type":"Toolbar"}],"root_ids":["2849"]},"title":"Bokeh Application","version":"2.4.2"}};
    var render_items = [{"docid":"b3c48fa1-5c5e-4c19-ba0c-f82ebb5ea144","root_ids":["2849"],"roots":{"2849":"16c7e51d-a01b-4a11-b304-b35161a8e072"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>


