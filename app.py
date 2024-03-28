# import crypt
# import imp
from Flask import Flask, request, jsonify, render_template
import pickle
import Flask
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json

app=Flask(__name__)
model=pickle.load(open('kmmodel.pkl','rb'))

def load_clean(path):
    ret=pd.read_csv(path, sep=',', encoding='ISO-8859-1', header=0)
    ret['CustomerID']= ret['CustomerID'].astype(str)
    ret['Amount']=ret['Quantity']*ret['UnitPrice']
    rfm_m=ret.groupby('CustomerID')['Amount'].sum().reset_index()
    rfm_f=ret.groupby('CustomerID')['InvoiceNo'].sum().reset_index()
    rfm_f.columns=['CustomerID','frequency']
    ret['InvoiceDate']=pd.to_datetime(ret['InvoiceDate'],format='%d-%m-%y %H:%M')
    max_date=max(ret['InvoiceDate'])
    ret['Diff']=max_date-ret['InvoiceDate']
    rfm_p=ret.groupby('CustomerID')['Diff'].min().reset_index()
    rfm_p['Diff']=rfm_p['Diff'].dt.days
    rfm=pd.merge(rfm_m,rfm_f, on='CustomerID', how='inner')
    rfm=pd.merge(rfm,rfm_p,on='CustomerID', how='inner')
    rfm.columns=['CustomerID','Amount','Recency']
    Q1=rfm.quantile(0.05)
    Q3=rfm.quantile(0.95)
    IQR=Q3-Q1
    rfm-rfm[(rfm.Amount>=Q1[0]-1.5*IQR[0]) & (rfm.Amount<=Q3[0]+1.5*IQR[0])]
    rfm-rfm[(rfm.Recency>=Q1[2]-1.5*IQR[2]) & (rfm.Recency<=Q3[2]+1.5*IQR[2])]
    rfm-rfm[(rfm.frequency>=Q1[1]-1.5*IQR[1]) & (rfm.frequency<=Q3[1]+1.5*IQR[1])]
    return rfm
def preprocess_data(path):
    rfm=load_clean(path)
    rfm_df=rfm[('Amount','frequency','Recency')]
    scalar=StandardScaler()
    rf_scaled=scalar.fit_transform(rfm_df)
    rf_scaled=pd.DataFrame(rf_scaled)
    rf_col=['Amount','frequency','Recency']
    return rfm, rf_scaled
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    file=request.files['file']
    file_path=os.path.join(os.getcwd(), file.filename)
    file.save(file_path)
    df=preprocess_data(file_path)[1]
    results_df=model.predict(df)
    df_with_id=preprocess_data(file_path)[0]
    df_with_id['Cluster_Id']=results_df
    sns.stripplot(x='Cluster_Id',y='Amount',data=df_with_id,hue='Cluster_Id')
    amount_img_path='./ClusterId_Amount.png'
    plt.savefig(amount_img_path)
    plt.clf()
    sns.stripplot(x='Cluster_Id',y='Frequency',data=df_with_id,hue='Cluster_Id')
    f_img_path='./ClusterId_Frequency.png'
    plt.savefig(f_img_path)
    plt.clf()
if __name__=="__main__":
    app.run(debug=True)
