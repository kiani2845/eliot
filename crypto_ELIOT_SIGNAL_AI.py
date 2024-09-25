from __future__ import annotations
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, LeadingDiagonal
from models.WaveAnalyzer import WaveAnalyzer
from models.WaveOptions import WaveOptionsGenerator5
from models.helpers import plot_pattern
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox, scrolledtext
from Historic_Crypto import HistoricalData
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import requests

def analyze_wave(stock_name, analysis_type):
    # دریافت داده‌های تاریخی
    start_date = input('Enter The Start Date (YYYY-MM-DD): ')
    tkr_response = requests.get('https://api.pro.coinbase.com/products', auth=(input('Enter your user Name:'), input('Enter your password:')))
    df = HistoricalData(stock_name,86400 ,tkr_response,start_date=start_date).retrieve_data()
    #print(df)
    # اضافه کردن ستون تاریخ
    df['date'] = df.index
    df.reset_index(drop=True, inplace=True)
    
    if 'date' not in df.columns:
        raise ValueError("The 'date' column does not exist in the DataFrame.")
    
    idx_start = np.argmin(np.array(list(df['Low'])))
    wa = WaveAnalyzer(df=df, verbose=False)
    wave_options_impulse = WaveOptionsGenerator5(up_to=15)
        
    impulse = Impulse('impulse')
    leading_diagonal = LeadingDiagonal('leading diagonal')
    rules_to_check = [impulse, leading_diagonal]

    wavepatterns_up = set()
    results = []

    for new_option_impulse in wave_options_impulse.options_sorted:
        waves_up = wa.find_impulsive_wave(idx_start=idx_start, wave_config=new_option_impulse.values)
        if waves_up:
            wavepattern_up = WavePattern(waves_up, verbose=True)
            for rule in rules_to_check:
                if wavepattern_up.check_rule(rule):
                    if wavepattern_up in wavepatterns_up:
                        continue
                    wavepatterns_up.add(wavepattern_up)
                    result = f'{rule.name} found: {new_option_impulse.values}'
                    results.append(result)
                    print(result)
                    plot_pattern(df=df, wave_pattern=wavepattern_up, title=str(new_option_impulse))

    result_display = "\n".join(results) if results else "هیچ الگوی معتبری پیدا نشد."
    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, result_display)
    result_text.config(state=tk.DISABLED)

    # تحلیل قیمت با استفاده از شبکه‌های عصبی
    neural_network_results = neural_network_analysis(df)

    # ذخیره تحلیل‌ها در HTML
    save_analysis_to_html(stock_name, results, neural_network_results)

    # نمایش گرافیکی تحلیل‌ها
    plot_analysis(df)

    # معامله خودکار (ایجاد یک مثال ساده)
    if analysis_type == 'خرید':
        print("معامله انجام شد.")  # مثال ساده به جای سفارش واقعی

def neural_network_analysis(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    
    if len(scaled_data) <= 60:
        raise ValueError("Not enough data points to create train_data. Need at least 60.")

    train_data = []
    for i in range(60, len(scaled_data)):
        train_data.append(scaled_data[i-60:i, 0])

    train_data = np.array(train_data)
    
    if len(train_data) == 0:
        raise ValueError("Not enough data to create train_data.")

    X_train = np.reshape(train_data, (train_data.shape[0], train_data.shape[1]))

    model = Sequential()
    model.add(Dense(units=50, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train, scaled_data[60:], epochs=100, batch_size=32)

    predicted_price = model.predict(X_train)
    predicted_price = scaler.inverse_transform(predicted_price)

    return predicted_price.flatten()

def save_analysis_to_html(stock_name, wave_results, neural_network_results):
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write('<html><head><title>تحلیل قیمت</title></head><body>')
        f.write(f'<h1>تحلیل قیمت سهام: {stock_name}</h1>')
        
        f.write('<h2>نتایج تحلیل امواج الیوت</h2><ul>')
        for result in wave_results:
            f.write(f'<li>{result}</li>')
        f.write('</ul>')

        f.write('<h2>نتایج پیش‌بینی با شبکه‌های عصبی</h2><ul>')
        for price in neural_network_results:
            f.write(f'<li>{price:.2f}</li>')
        f.write('</ul>')

        f.write('</body></html>')
    messagebox.showinfo("ذخیره‌سازی", "تحلیل‌ها با موفقیت در فایل index.html ذخیره شد.")

def plot_analysis(df):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(df['Close'], label='Closing price', color='blue')
    plt.title('Stock price chart')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    plt.plot(df['SMA20'], label='20 day moving average', color='orange')
    plt.plot(df['SMA50'], label='50 day moving average', color='red')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('Chart of price changes')
    plt.plot(df['Close'].pct_change(), label='Price changes', color='green')
    plt.xlabel('Date')
    plt.ylabel('Percentage changes')
    plt.legend()

    plt.tight_layout()
    plt.show()

def on_button_click():
    stock_name = entry.get()
    analysis_type = analysis_var.get()
    if stock_name:
        analyze_wave(stock_name, analysis_type)
    else:
        messagebox.showwarning("ورودی نامعتبر", "لطفاً نام سهم را وارد کنید.")

root = tk.Tk()
root.title("تحلیل امواج الیوت")

label = tk.Label(root, text="نام سهم را وارد کنید:")
label.pack()

entry = tk.Entry(root)
entry.pack()

analysis_var = tk.StringVar(value='خرید')
analysis_type_frame = tk.Frame(root)
tk.Radiobutton(analysis_type_frame, text='خرید', variable=analysis_var, value='خرید').pack(side=tk.LEFT)
tk.Radiobutton(analysis_type_frame, text='فروش', variable=analysis_var, value='فروش').pack(side=tk.LEFT)
analysis_type_frame.pack()

button = tk.Button(root, text="تحلیل", command=on_button_click)
button.pack()

result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10)
result_text.pack()
result_text.config(state=tk.DISABLED)

root.mainloop()
