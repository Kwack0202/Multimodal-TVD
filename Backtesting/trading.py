from common_imports import *


'''
01. Up Down Signal
'''
def up_down_signal(output_dir, base_dir, tickers, labels):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Subdirectories
    model_types = []
    # 폴더 이름을 읽어와서 리스트에 추가
    if os.path.exists(base_dir):
        for folder_name in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, folder_name)):
                model_types.append(folder_name)

    for model in tqdm(model_types):
        for label in labels:
            for ticker in tickers:
                input_file_path = os.path.join(base_dir, model, label, f'{ticker}.csv')
                output_file_path = os.path.join(output_dir, model, label, f'{ticker}.csv')
                
                df = pd.read_csv(input_file_path)
                mean_predicted = df['Predicted'].median()
                df['Predicted'] = df['Predicted'].apply(lambda x: 1 if x > mean_predicted else 0)
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                df.to_csv(output_file_path, index=False)
                

'''
02. Buy Sell Position
'''
def buy_sell_signal(output_dir, base_dir, tickers, labels):
    
    # Subdirectories
    model_types = []
    # 폴더 이름을 읽어와서 리스트에 추가
    if os.path.exists(base_dir):
        for folder_name in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, folder_name)):
                model_types.append(folder_name)

    opposite_count = 1
    split_num = 753

    for model in tqdm(model_types):
        for label in labels:
            for ticker in tickers:
                stock_data = pd.read_csv(f'./data/csv/origin_data/{ticker}.csv').iloc[-split_num:, 0:6].reset_index(drop=True)
                
                model_results_data = pd.read_csv(f'./Backtesting/up_down_signal/{model}/{label}/{ticker}.csv', index_col=0).reset_index(drop=True)
                
                trading_data = pd.concat([stock_data, model_results_data], axis=1)
                            
                # ===========================================================================================
                # Buy Sell action 생성
                action = "No action"
                counter = 0
                initial_position_set = False

                for i in range(len(trading_data)):
                    curr_pos = trading_data.loc[i, 'Predicted']
                    if i == 0:
                        prev_pos = 0
                    else:
                        prev_pos = trading_data.loc[i-1, 'Predicted']

                    if not initial_position_set:
                        if curr_pos == 0:
                            action = "No action"
                        else:
                            action = "Buy"
                            initial_position_set = True
                    else:
                        last_action = trading_data.loc[i-1, f'action']

                        if last_action == "sell":
                            if curr_pos == 0:
                                action = "No action"
                                initial_position_set = False
                            else:
                                action = "Buy"
                                counter = 0
                        else:
                            if curr_pos == 1:
                                action = "Holding"
                                counter = 0
                            else:
                                counter += 1
                                if counter == opposite_count:
                                    action = "sell"
                                    counter = 0
                                else:
                                    action = "Holding"
                    
                    if i == len(trading_data) - 1:
                        action = "sell"
                    
                    trading_data.loc[i, f'action'] = action

                if not os.path.exists(os.path.join(output_dir, f'full_period/{model}/{label}/')):
                    os.makedirs(os.path.join(output_dir, f'full_period/{model}/{label}/'))
                    
                output_file_path = os.path.join(output_dir, f'full_period/{model}/{label}/{ticker}.csv')
                trading_data.to_csv(output_file_path, index=True)

def buy_sell_signal_YoY(output_dir, base_dir, tickers, labels):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Subdirectories
    model_types = []
    # 폴더 이름을 읽어와서 리스트에 추가
    if os.path.exists(base_dir):
        for folder_name in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, folder_name)):
                model_types.append(folder_name)

    opposite_count = 1
    split_num = 753
    years= ['2021', '2022', '2023']

    for model in tqdm(model_types):
        for label in labels:
            for ticker in tickers:
                stock_data = pd.read_csv(f'./data/csv/origin_data/{ticker}.csv').iloc[-split_num:, 0:6].reset_index(drop=True)
                
                model_results_data = pd.read_csv(f'./Backtesting/up_down_signal/{model}/{label}/{ticker}.csv', index_col=0).reset_index(drop=True)
                
                trading_data = pd.concat([stock_data, model_results_data], axis=1)
                            
                for year in years:
                    trading_df = trading_data[(trading_data['Date'] >= f'{year}-01-01') & (trading_data['Date'] <= f'{year}-12-31')]
            
                    trading_df = trading_df.reset_index(drop=True)  
                
                    # ===========================================================================================
                    # Buy Sell action 생성
                    action = "No action"
                    counter = 0
                    initial_position_set = False

                    for i in range(len(trading_df)):
                        curr_pos = trading_df.loc[i, 'Predicted']
                        if i == 0:
                            prev_pos = 0
                        else:
                            prev_pos = trading_df.loc[i-1, 'Predicted']

                        if not initial_position_set:
                            if curr_pos == 0:
                                action = "No action"
                            else:
                                action = "Buy"
                                initial_position_set = True
                        else:
                            last_action = trading_df.loc[i-1, f'action']

                            if last_action == "sell":
                                if curr_pos == 0:
                                    action = "No action"
                                    initial_position_set = False
                                else:
                                    action = "Buy"
                                    counter = 0
                            else:
                                if curr_pos == 1:
                                    action = "Holding"
                                    counter = 0
                                else:
                                    counter += 1
                                    if counter == opposite_count:
                                        action = "sell"
                                        counter = 0
                                    else:
                                        action = "Holding"
                        
                        if i == len(trading_df) - 1:
                            action = "sell"
                        
                        trading_df.loc[i, f'action'] = action

                    if not os.path.exists(os.path.join(output_dir, f'YOY_{year}/{model}/{label}/')):
                        os.makedirs(os.path.join(output_dir, f'YOY_{year}/{model}/{label}/'))
                    
                    output_file_path = os.path.join(output_dir, f'YOY_{year}/{model}/{label}/{ticker}.csv')
                    trading_df.to_csv(output_file_path, index=True)
            

'''
03. simulation
'''
def simulation(output_dir, base_dir, tickers, labels):
    
    # Subdirectories
    model_types = []
    # 폴더 이름을 읽어와서 리스트에 추가
    if os.path.exists(base_dir):
        for folder_name in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, folder_name)):
                model_types.append(folder_name)

    commission_rate = 0.0005

    for model in tqdm(model_types):
        for label in labels:
            for ticker in tickers:
                df = pd.read_csv(f"./Backtesting/buy_sell_signal/full_period/{model}/{label}/{ticker}.csv", index_col=0)
                
                # 새로운 데이터프레임 생성
                new_data = {
                    'Date': [],
                    'Margin_Profit': [],
                    'Margin_Return': [],
                    'Cumulative_Profit': [],
                }

                signal_results = []
                
                buy_price = None
                cumulative_profit = 0
                cumulative_profit_ratio = 0     

                for index, row in df.iterrows():
                    if row[f'action'] == 'Buy':
                        buy_price = row['Close']
                        
                        new_data['Date'].append(row['Date'])
                        new_data['Margin_Profit'].append(0)
                        new_data['Cumulative_Profit'].append(cumulative_profit)
                        new_data['Margin_Return'].append(0)
                        
                    elif row[f'action'] == 'sell' and buy_price is not None:
                        if index + 1 < len(df):
                            next_row = df.iloc[index + 1]  # 다음 행을 가져오기
                            # sell_price = next_row['Open']
                            sell_price = row['Close']
                                                    
                            profit = sell_price - buy_price - (sell_price * commission_rate)
                            return_ = profit / buy_price * 100
                            
                            cumulative_profit += profit
                            cumulative_profit_ratio += return_
                            
                            new_data['Date'].append(row['Date'])
                            new_data['Margin_Profit'].append(profit)
                            new_data['Cumulative_Profit'].append(cumulative_profit)
                            new_data['Margin_Return'].append(return_)
                            
                        else:
                            # 다음 행이 없는 경우 해당 행의 Close로 매도
                            sell_price = row['Close']
                            
                            profit = sell_price - buy_price - (sell_price * commission_rate)
                            return_ = profit / buy_price * 100
                            
                            cumulative_profit += profit
                            cumulative_profit_ratio += return_
                            
                            new_data['Date'].append(row['Date'])
                            new_data['Margin_Profit'].append(profit)
                            new_data['Cumulative_Profit'].append(cumulative_profit)
                            new_data['Margin_Return'].append(return_)
                            
                    else:
                        new_data['Date'].append(row['Date'])
                        new_data['Margin_Profit'].append(0)
                        new_data['Cumulative_Profit'].append(cumulative_profit)
                        new_data['Margin_Return'].append(0)
                        
                # 새로운 데이터프레임 생성
                new_df = pd.DataFrame(new_data)

                # "Date" 열을 기준으로 두 데이터프레임 병합
                merged_df = pd.merge(df, new_df, on='Date', how='outer')
                
                merged_df['Holding_Period'] = merged_df.groupby((merged_df['action'] != 'Holding').cumsum()).cumcount()
                
                position_mask = (merged_df['action'] == 'Buy')
                if position_mask.sum() == 0:
                    initial_investment = merged_df.iloc[0]['Close'] 
                else:
                    initial_investment = merged_df.iloc[merged_df[position_mask].index[0] + 1]['Open']
            
                merged_df['Cumulative_Return'] = (merged_df['Cumulative_Profit'] / initial_investment) * 100
                
                merged_df = merged_df[['Date', 'Open', 'High', 'Low', 'Close', f'Predicted', f'action', 
                                       'Margin_Profit', 'Cumulative_Profit', 'Margin_Return', 'Cumulative_Return']]
                
                # Drawdown 계산
                merged_df['Drawdown'] = 0.0
                merged_df['Drawdown_rate'] = 0.0
                peak_profit = merged_df['Cumulative_Profit'].iloc[0]
                peak_profit_rate = merged_df['Cumulative_Return'].iloc[0]
                
                for index, row in merged_df.iterrows():
                    current_profit = row['Cumulative_Profit']
                    # Update peak profit if current profit is higher
                    if current_profit > peak_profit:
                        peak_profit = current_profit
                    # Calculate drawdown as the difference from peak to current
                    drawdown = -(peak_profit - current_profit)
                    merged_df.at[index, 'Drawdown'] = drawdown
                
                for index, row in merged_df.iterrows():
                    current_profit_rate = row['Cumulative_Return']
                    # Update peak profit if current profit is higher
                    if current_profit_rate > peak_profit_rate:
                        peak_profit_rate = current_profit_rate
                    # Calculate drawdown as the difference from peak to current
                    drawdown_rate = -(peak_profit_rate - current_profit_rate)
                    merged_df.at[index, 'Drawdown_rate'] = drawdown_rate
                
                merged_df['Holding_Period'] = merged_df.groupby((df['action'] != 'Holding').cumsum()).cumcount()  
                    
                column_names = [
                    "Date", "Open", "High", "Low", "Close", "Predicted", "action",
                    "Margin_Profit", "Cumulative_Profit", "Margin_Return", "Cumulative_Return",
                    "Drawdown", "Drawdown_rate", "Holding_Period"
                ]
                
                merged_df = merged_df[column_names]
                # 숫자형 변수들을 소숫점 3자리로 반올림
                merged_df = merged_df.round(3)
    
                if not os.path.exists(os.path.join(output_dir, f'full_period/{model}/{label}/')):
                    os.makedirs(os.path.join(output_dir, f'full_period/{model}/{label}/'))
                    
                output_file_path = os.path.join(output_dir, f'full_period/{model}/{label}/{ticker}.csv')
                merged_df.to_csv(output_file_path, index=True)
                              
def simulation_YoY(output_dir, base_dir, tickers, labels):
    
    # Subdirectories
    model_types = []
    # 폴더 이름을 읽어와서 리스트에 추가
    if os.path.exists(base_dir):
        for folder_name in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, folder_name)):
                model_types.append(folder_name)

    commission_rate = 0.0005
    years= ['2021', '2022', '2023']

    for model in tqdm(model_types):
        for label in labels:
            for ticker in tickers:
                for year in years:
                    df = pd.read_csv(f"./Backtesting/buy_sell_signal/YOY_{year}/{model}/{label}/{ticker}.csv", index_col=0)
                    
                    # 새로운 데이터프레임 생성
                    new_data = {
                        'Date': [],
                        'Margin_Profit': [],
                        'Margin_Return': [],
                        'Cumulative_Profit': [],
                    }

                    signal_results = []
                    
                    buy_price = None
                    cumulative_profit = 0
                    cumulative_profit_ratio = 0     

                    for index, row in df.iterrows():
                        if row[f'action'] == 'Buy':
                            buy_price = row['Close']
                            
                            new_data['Date'].append(row['Date'])
                            new_data['Margin_Profit'].append(0)
                            new_data['Cumulative_Profit'].append(cumulative_profit)
                            new_data['Margin_Return'].append(0)
                            
                        elif row[f'action'] == 'sell' and buy_price is not None:
                            if index + 1 < len(df):
                                next_row = df.iloc[index + 1]  # 다음 행을 가져오기
                                # sell_price = next_row['Open']
                                sell_price = row['Close']
                                                        
                                profit = sell_price - buy_price - (sell_price * commission_rate)
                                return_ = profit / buy_price * 100
                                
                                cumulative_profit += profit
                                cumulative_profit_ratio += return_
                                
                                new_data['Date'].append(row['Date'])
                                new_data['Margin_Profit'].append(profit)
                                new_data['Cumulative_Profit'].append(cumulative_profit)
                                new_data['Margin_Return'].append(return_)
                                
                            else:
                                # 다음 행이 없는 경우 해당 행의 Close로 매도
                                sell_price = row['Close']
                                
                                profit = sell_price - buy_price - (sell_price * commission_rate)
                                return_ = profit / buy_price * 100
                                
                                cumulative_profit += profit
                                cumulative_profit_ratio += return_
                                
                                new_data['Date'].append(row['Date'])
                                new_data['Margin_Profit'].append(profit)
                                new_data['Cumulative_Profit'].append(cumulative_profit)
                                new_data['Margin_Return'].append(return_)
                                
                        else:
                            new_data['Date'].append(row['Date'])
                            new_data['Margin_Profit'].append(0)
                            new_data['Cumulative_Profit'].append(cumulative_profit)
                            new_data['Margin_Return'].append(0)
                            
                    # 새로운 데이터프레임 생성
                    new_df = pd.DataFrame(new_data)

                    # "Date" 열을 기준으로 두 데이터프레임 병합
                    merged_df = pd.merge(df, new_df, on='Date', how='outer')
                    
                    merged_df['Holding_Period'] = merged_df.groupby((merged_df['action'] != 'Holding').cumsum()).cumcount()
                    
                    position_mask = (merged_df['action'] == 'Buy')
                    if position_mask.sum() == 0:
                        initial_investment = merged_df.iloc[0]['Close'] 
                    else:
                        initial_investment = merged_df.iloc[merged_df[position_mask].index[0] + 1]['Open']
                
                    merged_df['Cumulative_Return'] = (merged_df['Cumulative_Profit'] / initial_investment) * 100
                    
                    merged_df = merged_df[['Date', 'Open', 'High', 'Low', 'Close', f'Predicted', f'action', 
                                        'Margin_Profit', 'Cumulative_Profit', 'Margin_Return', 'Cumulative_Return']]
                    
                    # Drawdown 계산
                    merged_df['Drawdown'] = 0.0
                    merged_df['Drawdown_rate'] = 0.0
                    peak_profit = merged_df['Cumulative_Profit'].iloc[0]
                    peak_profit_rate = merged_df['Cumulative_Return'].iloc[0]
                    
                    for index, row in merged_df.iterrows():
                        current_profit = row['Cumulative_Profit']
                        # Update peak profit if current profit is higher
                        if current_profit > peak_profit:
                            peak_profit = current_profit
                        # Calculate drawdown as the difference from peak to current
                        drawdown = -(peak_profit - current_profit)
                        merged_df.at[index, 'Drawdown'] = drawdown
                    
                    for index, row in merged_df.iterrows():
                        current_profit_rate = row['Cumulative_Return']
                        # Update peak profit if current profit is higher
                        if current_profit_rate > peak_profit_rate:
                            peak_profit_rate = current_profit_rate
                        # Calculate drawdown as the difference from peak to current
                        drawdown_rate = -(peak_profit_rate - current_profit_rate)
                        merged_df.at[index, 'Drawdown_rate'] = drawdown_rate
                    
                    merged_df['Holding_Period'] = merged_df.groupby((df['action'] != 'Holding').cumsum()).cumcount()  
                    
                    column_names = [
                        "Date", "Open", "High", "Low", "Close", "Predicted", "action",
                        "Margin_Profit", "Cumulative_Profit", "Margin_Return", "Cumulative_Return",
                        "Drawdown", "Drawdown_rate", "Holding_Period"
                    ]
                        
                    merged_df = merged_df[column_names]
                    # 숫자형 변수들을 소숫점 3자리로 반올림
                    merged_df = merged_df.round(3)
                    
                    if not os.path.exists(os.path.join(output_dir, f'YOY_{year}/{model}/{label}/')):
                        os.makedirs(os.path.join(output_dir, f'YOY_{year}/{model}/{label}/'))
                        
                    output_file_path = os.path.join(output_dir, f'YOY_{year}/{model}/{label}/{ticker}.csv')
                    merged_df.to_csv(output_file_path, index=True)               
                    
'''
04. Backtesting
'''
def backtesting(output_dir, base_dir, tickers, labels):
    
    model_types = []
    if os.path.exists(base_dir):
        for folder_name in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, folder_name)):
                model_types.append(folder_name)

    summary_data = []
    for model in tqdm(model_types):
        for label in labels:
            for ticker in tickers:
                backtesting_df = pd.read_csv(f"./Backtesting/simulation/full_period/{model}/{label}/{ticker}.csv", index_col=0)
                
                backtesting_df['action'] = backtesting_df['action'].replace('No action', 0)
                backtesting_df['action'] = backtesting_df['action'].replace('Buy', 1)
                backtesting_df['action'] = backtesting_df['action'].replace('sell', -1)            

                # 거래 횟수
                no_trade = len(backtesting_df[backtesting_df['Margin_Profit'] > 0]) + len(backtesting_df[backtesting_df['Margin_Profit'] < 0])
                
                # 가장 긴 Holding 기간
                max_holding_period = backtesting_df[backtesting_df['action'] == 'Holding']['Holding_Period'].max() if no_trade > 0 else 0
                
                # 평균 Holding 기간
                mean_holding_period = backtesting_df[backtesting_df['action'] == 'Holding']['Holding_Period'].mean() if no_trade > 0 else 0
                
                # 승률
                winning_ratio = len(backtesting_df[(backtesting_df['action'] == -1) & (backtesting_df['Margin_Profit'] > 0)]) / no_trade if no_trade > 0 else 0
                
                # 수익 평균, 손실 평균
                profit_average = backtesting_df[backtesting_df['Margin_Profit'] > 0]['Margin_Profit'].mean() if len(backtesting_df[backtesting_df['Margin_Profit'] > 0]) > 0 else 0
                loss_average = backtesting_df[backtesting_df['Margin_Profit'] < 0]['Margin_Profit'].mean() if len(backtesting_df[backtesting_df['Margin_Profit'] < 0]) > 0 else 0
                
                # payoff_ratio, profit_factor
                payoff_ratio = profit_average / -loss_average if loss_average < 0 else 0
                loss_sum = backtesting_df[backtesting_df['Margin_Profit'] < 0]['Margin_Profit'].sum()
                profit_sum = backtesting_df[backtesting_df['Margin_Profit'] > 0]['Margin_Profit'].sum()
                profit_factor = -profit_sum / loss_sum if loss_sum < 0 else 0
                
                final_cumulative_profit = backtesting_df['Cumulative_Profit'].iloc[-1]
                final_cumulative_return = backtesting_df['Cumulative_Return'].iloc[-1]
                max_realized_profit = backtesting_df['Margin_Profit'].max() if no_trade > 0 else 0
                max_realized_return = backtesting_df['Margin_Return'].max() if no_trade > 0 else 0

                # Maximum Drawdown (MDD)
                MDD = backtesting_df['Drawdown'].min() if no_trade > 0 else 0
                MDD_rate = backtesting_df['Drawdown_rate'].min() if no_trade > 0 else 0

                summary_data.append([model, label, ticker, no_trade, max_holding_period, mean_holding_period, winning_ratio,
                                     profit_average, loss_average, payoff_ratio, profit_factor, 
                                     final_cumulative_profit, final_cumulative_return,
                                     max_realized_profit, max_realized_return,
                                     MDD, MDD_rate])
    
    summary_df = pd.DataFrame(summary_data, columns=[
        "model", "label", "ticker", 
        "no_trade", "max_holding_period", "mean_holding_period", "winning_ratio", 
        "profit_average", "loss_average", "payoff_ratio", "profit_factor", 
        "final_cumulative_profit", "final_cumulative_return", 
        "max_realized_profit", "max_realized_return", 
        "MaxDrawdown", "MaxDrawdown_rate"
    ])
    
    summary_df = summary_df.round(3)
    
    if not os.path.exists(os.path.join(output_dir, f'full_period/')):
        os.makedirs(os.path.join(output_dir, f'full_period/'))
        
    summary_df.to_csv(f"{output_dir}/full_period/results_summary.csv", encoding='utf-8-sig', index=False)


def backtesting_YoY(output_dir, base_dir, tickers, labels):
    
    model_types = []
    if os.path.exists(base_dir):
        for folder_name in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, folder_name)):
                model_types.append(folder_name)

    years = ['2021', '2022', '2023']
    
    for year in years:
        summary_data = []
        
        for model in tqdm(model_types):
            for label in labels:
                for ticker in tickers:
                    backtesting_df = pd.read_csv(f"./Backtesting/simulation/YOY_{year}/{model}/{label}/{ticker}.csv", index_col=0)
                    
                    backtesting_df['action'] = backtesting_df['action'].replace('No action', 0)
                    backtesting_df['action'] = backtesting_df['action'].replace('Buy', 1)
                    backtesting_df['action'] = backtesting_df['action'].replace('sell', -1)            

                    # 거래 횟수
                    no_trade = len(backtesting_df[backtesting_df['Margin_Profit'] > 0]) + len(backtesting_df[backtesting_df['Margin_Profit'] < 0])
                    
                    # 가장 긴 Holding 기간
                    max_holding_period = backtesting_df[backtesting_df['action'] == 'Holding']['Holding_Period'].max() if no_trade > 0 else 0
                    
                    # 평균 Holding 기간
                    mean_holding_period = backtesting_df[backtesting_df['action'] == 'Holding']['Holding_Period'].mean() if no_trade > 0 else 0
                    
                    # 승률
                    winning_ratio = len(backtesting_df[(backtesting_df['action'] == -1) & (backtesting_df['Margin_Profit'] > 0)]) / no_trade if no_trade > 0 else 0
                    
                    # 수익 평균, 손실 평균
                    profit_average = backtesting_df[backtesting_df['Margin_Profit'] > 0]['Margin_Profit'].mean() if len(backtesting_df[backtesting_df['Margin_Profit'] > 0]) > 0 else 0
                    loss_average = backtesting_df[backtesting_df['Margin_Profit'] < 0]['Margin_Profit'].mean() if len(backtesting_df[backtesting_df['Margin_Profit'] < 0]) > 0 else 0
                    
                    # payoff_ratio, profit_factor
                    payoff_ratio = profit_average / -loss_average if loss_average < 0 else 0
                    loss_sum = backtesting_df[backtesting_df['Margin_Profit'] < 0]['Margin_Profit'].sum()
                    profit_sum = backtesting_df[backtesting_df['Margin_Profit'] > 0]['Margin_Profit'].sum()
                    profit_factor = -profit_sum / loss_sum if loss_sum < 0 else 0
                    
                    final_cumulative_profit = backtesting_df['Cumulative_Profit'].iloc[-1]
                    final_cumulative_return = backtesting_df['Cumulative_Return'].iloc[-1]
                    max_realized_profit = backtesting_df['Margin_Profit'].max() if no_trade > 0 else 0
                    max_realized_return = backtesting_df['Margin_Return'].max() if no_trade > 0 else 0

                    # Maximum Drawdown (MDD)
                    MDD = backtesting_df['Drawdown'].min() if no_trade > 0 else 0
                    MDD_rate = backtesting_df['Drawdown_rate'].min() if no_trade > 0 else 0

                    summary_data.append([model, label, ticker, no_trade, max_holding_period, mean_holding_period, winning_ratio,
                                        profit_average, loss_average, payoff_ratio, profit_factor, 
                                        final_cumulative_profit, final_cumulative_return,
                                        max_realized_profit, max_realized_return,
                                        MDD, MDD_rate])
        
        summary_df = pd.DataFrame(summary_data, columns=[
            "model", "label", "ticker", 
            "no_trade", "max_holding_period", "mean_holding_period", "winning_ratio", 
            "profit_average", "loss_average", "payoff_ratio", "profit_factor", 
            "final_cumulative_profit", "final_cumulative_return", 
            "max_realized_profit", "max_realized_return", 
            "MaxDrawdown", "MaxDrawdown_rate"
        ])
        
        summary_df = summary_df.round(3)
        
        if not os.path.exists(os.path.join(output_dir, f'YOY_{year}/')):
            os.makedirs(os.path.join(output_dir, f'YOY_{year}/'))
        
        summary_df.to_csv(f"{output_dir}/YOY_{year}/results_summary.csv", encoding='utf-8-sig', index=False)