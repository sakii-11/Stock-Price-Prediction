import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Set background image using HTML/CSS styling
background_image = '''
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/premium-photo/3d-rendering-purple-black-abstract-wave-background_167650-2502.jpg");
    background-size: cover;
}
[data-testid="stFullScreenFrame"] {
    text-align: center;
}
[class="st-emotion-cache-10trblm e1nzilvr1"] {
    color: #BC13FE;
}ś
[class="s1jz82f8"] {
    background-color: #000fff;
}
</style>
'''

# Render the background image using Markdown
st.markdown(background_image, unsafe_allow_html=True)

model = joblib.load('Stock-Model.keras')
st.header('Stock Market Predictor')

st.subheader('Input a Stock Symbol')
stock = st.text_input('Give Stock Symbol','GOOG', placeholder="Enter Stock Symnbol", label_visibility="collapsed")
st.subheader('Start Date')
start = st.text_input('Start Date','2020-12-31', placeholder='Enter The start Date - YYYY-MM-DD', label_visibility="collapsed")
st.subheader('End Date')
end =  st.text_input('End Date','2023-12-31', placeholder='Enter The End Date- YYYY-MM-DD', label_visibility="collapsed")
ticker = yf.Ticker(stock)
stock_info = ticker.info
stock_name = stock_info['longName']

data = yf.download(stock, start ,end)
data1 = pd.DataFrame(data)

st.subheader(stock_name + ' - Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)
a= y[-1]
b= predict[-1]
scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Actual Price vs Predicted Price')

fig1 = plt.figure(figsize=(8,6), facecolor = 'none')

# Customize the color of the scale on both x and y axes
plt.tick_params(axis='x', colors='#ffffff')  
plt.tick_params(axis='y', colors='#ffffff') 

# background color of the plot area
plt.gca().set_facecolor('#000000')

# predict arr = test_val
# y = Predicted values
plt.plot(predict, 'r', label = 'Actual Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time', color = '#BC13FE', fontsize = 15)
plt.ylabel('Closing Price', color = '#BC13FE', fontsize = 15)
plt.legend()
st.pyplot(fig1)

# Adding CSS symbols using st.markdown() with unsafe_allow_html=True
st.markdown(
    """
    <style>
        .gg-arrow-up {
 box-sizing: border-box;
 position: relative;
 display: block;
 transform: scale(var(--ggs,1));
 width: 22px;
 height: 22px;
 color: green;
}

.gg-arrow-up::after,
.gg-arrow-up::before {
 content: "";
 display: block;
 box-sizing: border-box;
 position: absolute;
 top: 4px;
 color: green;
}

.gg-arrow-up::after {
 width: 8px;
 height: 8px;
 border-top: 2px solid;
 border-left: 2px solid;
 transform: rotate(45deg);
 left: 7px;
 color: green;
}

.gg-arrow-up::before {
 width: 2px;
 height: 16px;
 left: 10px;
 background: currentColor;
 color: green;
}

 .gg-arrow-down {
 box-sizing: border-box;
 position: relative;
 display: block;
 transform: scale(var(--ggs,1));
 width: 22px;
 height: 22px;
 color: red;
}

.gg-arrow-down::after,
.gg-arrow-down::before {
 content: "";
 display: block;
 box-sizing: border-box;
 position: absolute;
 bottom: 4px;
 color: red;
}

.gg-arrow-down::after {
 width: 8px;
 height: 8px;
 border-bottom: 2px solid;
 border-left: 2px solid;
 transform: rotate(-45deg);
 left: 7px;
 color: red;
}

.gg-arrow-down::before {
 width: 2px;
 height: 16px;
 left: 10px;
 background: currentColor;
 color: red;
} 
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<br>', unsafe_allow_html=True)


#st.subheader('Actual Price is: ' + str(b))
if(float(a) > float(b)):
   #st.subheader('Predicted Price is: ' + str(a))
   st.subheader('Prediction -> Price will Increase')
   st.markdown('<div class="gg-arrow-up"></div>', unsafe_allow_html=True)
else:
    #st.subheader('Predicted Price is: ' + str(b))
    st.subheader('Prediction -> Price will Decrease')
    st.markdown('<div class="gg-arrow-down"></div>', unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)
st.subheader('Price vs MA50 (Moving Average of Last 50 Days)')
ma_50_days = data.Close.rolling(50).mean()

fig2 = plt.figure(figsize=(8,6), facecolor = 'none')

# Customize the color of the scale on both x and y axes
plt.tick_params(axis='x', colors='#ffffff')  
plt.tick_params(axis='y', colors='#ffffff')

# background color of the plot area
plt.gca().set_facecolor('#000000')
plt.plot(ma_50_days, 'r', label = 'Moving Average of Last 50 Days')
plt.plot(data.Close, 'g', label = 'Closing Price')
plt.xlabel('Time', color = '#BC13FE', fontsize = 15)
plt.ylabel('Closing Price', color = '#BC13FE', fontsize = 15)
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA50 vs MA100 (Moving Average of Last 100 Days)')
ma_100_days = data.Close.rolling(100).mean()

fig3 = plt.figure(figsize=(8,6), facecolor = 'none')

# Customize the color of the scale on both x and y axes
plt.tick_params(axis='x', colors='#ffffff')  
plt.tick_params(axis='y', colors='#ffffff')  

# background color of the plot area
plt.gca().set_facecolor('#000000')

plt.plot(ma_50_days, 'r', label = 'Moving Average of Last 50 Days')
plt.plot(ma_100_days, 'b', label = 'Moving Average of Last 100 Days')
plt.plot(data.Close, 'g', label = 'Closing Price')
plt.xlabel('Time', color = '#BC13FE', fontsize = 15)
plt.ylabel('Closing Price', color = '#BC13FE', fontsize = 15)
plt.legend()
st.pyplot(fig3)

st.subheader('Price vs MA100 vs MA200 (Moving Average of Last 200 Days)')
ma_200_days = data.Close.rolling(200).mean()

fig4 = plt.figure(figsize=(8,6), facecolor = 'none')

#color of the lines of the plot
plt.tick_params(axis='x', colors='#ffffff')  
plt.tick_params(axis='y', colors='#ffffff')  

# background color of the plot area
plt.gca().set_facecolor('#000000')

plt.plot(ma_100_days, 'r', label = 'Moving Average of Last 100 Days')
plt.plot(ma_200_days, 'b', label = 'Moving Average of Last 200 Days')
plt.plot(data.Close, 'g', label = 'Closing Price')
plt.xlabel('Time', color = '#BC13FE', fontsize = 15)
plt.ylabel('Closing Price', color = '#BC13FE', fontsize = 15)
plt.legend()
st.pyplot(fig4)