from common_imports import *

model_name_mapping = {
    'benchmark_MM_single_120': 'Cross_single_120',
    'benchmark_MM_single_20': 'Cross_single_20',
    'benchmark_MM_single_5': 'Cross_single_5',
    'benchmark_MM_single_60': 'Cross_single_60',
    'benchmark_only_IMG': 'Self_only_IMG',
    'benchmark_only_TA': 'Self_only_TA',
    'MM_Causal_ViT_LSTM_ExcludeWindow120_(LSTM_25_256_4)_(ViT_256_4_16_512)_(MHAL_256_16)_(MLP_768_512)': 'Cross_exclude_120',
    'MM_Causal_ViT_LSTM_ExcludeWindow20_(LSTM_25_256_4)_(ViT_256_4_16_512)_(MHAL_256_16)_(MLP_768_512)': 'Cross_exclude_20',
    'MM_Causal_ViT_LSTM_ExcludeWindow5_(LSTM_25_256_4)_(ViT_256_4_16_512)_(MHAL_256_16)_(MLP_768_512)': 'Cross_exclude_5',
    'MM_Causal_ViT_LSTM_ExcludeWindow60_(LSTM_25_256_4)_(ViT_256_4_16_512)_(MHAL_256_16)_(MLP_768_512)': 'Cross_exclude_60',
    'MM_to_seq_(LSTM_25_512_12)_(ViT_512_12_16_1024)_(MHAL_512_16)_(MLP_2048_512)': 'Cross_Multi_Modal',
    'Seq_to_MM_(LSTM_25_512_12_512)_(ViT_512_12_16_1024_512)_(MHAL_2048_16)_(MLP_1024_512)': 'Self_Multi_Modal',
    '1D-CNN': '1D-CNN',
    'GRU': 'GRU',
    'LSTM': 'LSTM',
    # 'SVM': 'SVM',
    'TCN': 'TCN',
    'Transformer': 'Transformer',
    # 'XGB': 'XGB'
}

# Color and style mapping for models
model_style_mapping = {
    
    # baselines
    '1D-CNN': {'color': '#1f77b4', 'linestyle': 'solid'},  # Blue
    'GRU': {'color': '#ff7f0e', 'linestyle': 'solid'},    # Orange
    'LSTM': {'color': '#2ca02c', 'linestyle': 'solid'},   # Green
    'SVM': {'color': '#d62728', 'linestyle': 'solid'},    # Red
    'TCN': {'color': '#9467bd', 'linestyle': 'solid'},    # Purple
    'Transformer': {'color': '#8c564b', 'linestyle': 'solid'},  # Brown
    'XGB': {'color': '#e377c2', 'linestyle': 'solid'},    # Pink
    
    # Multimodal
    'Cross_Multi_Modal': {'color': '#800080', 'linestyle': 'solid'},  # Bold Purple
    'Self_Multi_Modal': {'color': '#ff4500', 'linestyle': 'solid'},   # Bold Orange
    
    # Multimodla (Ablation)
    'Cross_single_5': {'color': '#1e90ff', 'linestyle': 'solid'},     # Dodger Blue
    'Cross_single_20': {'color': '#87cefa', 'linestyle': 'solid'},   # Light Sky Blue
    'Cross_single_60': {'color': '#4682b4', 'linestyle': 'solid'},    # Steel Blue
    'Cross_single_120': {'color': '#b0e0e6', 'linestyle': 'solid'},  # Powder Blue
    
    'Cross_exclude_5': {'color': '#228b22', 'linestyle': 'dashed'},    # Forest Green
    'Cross_exclude_20': {'color': '#90ee90', 'linestyle': 'dashed'},  # Light Green
    'Cross_exclude_60': {'color': '#3cb371', 'linestyle': 'dashed'},   # Medium Sea Green
    'Cross_exclude_120': {'color': '#98fb98', 'linestyle': 'dashed'}, # Pale Green
    
    'Self_only_IMG': {'color': '#808080', 'linestyle': 'solid'},      # Gray
    'Self_only_TA': {'color': '#a9a9a9', 'linestyle': 'solid'}       # Dark Gray
}


'''
00. Backtesting summary
'''
def plot_backtesting_metrics(base_dir, tickers, labels):
    # Color mapping for models and metrics
    palette_mapping = {
        'Cross-Attention': {
            'payoff_ratio': {'Signal_origin': '#1e90ff', 'Signal_trend': '#4682b4'},  # Dodger Blue, Steel Blue
            'profit_factor': {'Signal_origin': '#228b22', 'Signal_trend': '#3cb371'}  # Forest Green, Medium Sea Green
        },
        'Self-Attention': {
            'payoff_ratio': {'Signal_origin': '#DA70D6', 'Signal_trend': '#800080'},  # Light Purple, Purple
            'profit_factor': {'Signal_origin': '#FFDAB9', 'Signal_trend': '#FF8C00'}   # Light Peach, Orange
        },
        'default': {
            'payoff_ratio': {'Signal_origin': '#FFB6C1', 'Signal_trend': '#DC143C'},  # Light Pink, Crimson
            'profit_factor': {'Signal_origin': '#FF4040', 'Signal_trend': '#8B0000'}  # Red, Dark Red
        }
    }

    # Get model types
    model_types = []
    if os.path.exists(base_dir):
        for folder_name in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, folder_name)):
                model_types.append(folder_name)

    # Load and preprocess data
    df = pd.read_csv(f"./Backtesting/backtesting/full_period/results_summary.csv")
    df = df[['model', 'label', 'ticker', 'no_trade', 'winning_ratio', 'payoff_ratio', 'profit_factor', 'MaxDrawdown_rate']]
    df.replace([0, np.inf, -np.inf], 1, inplace=True)

    # Set seaborn style
    sns.set(style="whitegrid")

    # Plot for each model and metric
    for model in model_types:
        model_label = model_name_mapping.get(model, model)
        palette_key = 'Cross-Attention' if 'MM_to_seq' in model else 'Self-Attention' if 'Seq_to_MM' in model else 'default'

        for metric in ['payoff_ratio', 'profit_factor']:
            palette = palette_mapping[palette_key][metric]

            filtered_df = df[df['model'] == model].reset_index(drop=True)

            # Set up figure and gridspec
            fig = plt.figure(figsize=(20, 14))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

            # KDE plot
            ax0 = plt.subplot(gs[0])
            for label in labels:
                subset = filtered_df[filtered_df['label'] == label]
                sns.kdeplot(
                    data=subset,
                    x=metric,
                    fill=True,
                    label=label,
                    ax=ax0,
                    color=palette.get(label, 'gray'),
                    alpha=0.6
                )

            ax0.axvline(x=1, color='red', linestyle='--', linewidth=1)
            ax0.set_title(f'Distribution of {metric} ({model_label})', fontsize=35)
            ax0.set_ylabel('Density', fontsize=30)
            ax0.legend(title='Label', fontsize=26)
            ax0.tick_params(axis='both', which='major', labelsize=24)

            # Box plot
            ax1 = plt.subplot(gs[1], sharex=ax0)
            sns.boxplot(
                x=metric,
                y='label',
                data=filtered_df,
                palette=palette,
                orient='h',
                ax=ax1
            )

            ax1.axvline(x=1, color='red', linestyle='--', linewidth=1)
            ax1.set_xlabel(metric, fontsize=30)
            ax1.set_ylabel('Label', fontsize=30)
            ax1.tick_params(axis='both', which='major', labelsize=24)

            plt.tight_layout()

            # Save plot
            output_directory = f'./Backtesting/plot/backtesting_summary/full_period/'
            os.makedirs(output_directory, exist_ok=True)
            plt.savefig(os.path.join(output_directory, f'{metric}_{model}_comparison.png'), bbox_inches='tight')
            plt.close()

'''
01. cumulative_profit
'''
def plot_cumulative_profit(base_dir, tickers, labels):
    
    model_types = []
    if os.path.exists(base_dir):
        for folder_name in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, folder_name)):
                model_types.append(folder_name)
    
    for ticker in tqdm(tickers):
        for label in labels:
            for model in model_types:             
                trading_df = pd.read_csv(f'./Backtesting/simulation/full_period/{model}/{label}/{ticker}.csv')

                cumulative_profits = {
                    'Multimodal trading': trading_df['Cumulative_Profit']
                }
                            
                trading_df['Investment_Value'] = trading_df['Close'] - trading_df.iloc[0]['Close']
                trading_df['Close_Relative'] = trading_df['Investment_Value']
                        
                # 날짜를 인덱스로 설정
                trading_df['Date'] = pd.to_datetime(trading_df['Date'])
                trading_df.set_index('Date', inplace=True)
                    
                plt.figure(figsize=(25, 12))
                plt.tight_layout()

                for i, (profit_model, cumulative_profit) in enumerate(cumulative_profits.items()):
                    if profit_model == 'Multimodal trading':
                        plt.plot(trading_df.index, cumulative_profit, label=profit_model, color='purple', linewidth=3)

                plt.plot(trading_df.index, trading_df['Close_Relative'], label='Buy & Hold', linestyle='--')

                # plt.title(f'Cumulative Profit Comparison: {ticker}', fontsize=20)  # 제목 폰트 크기 키움
                plt.xlabel('Date', fontsize=20)  # x축 폰트 크기 키움
                plt.ylabel('Cumulative Profit', fontsize=20)  # y축 폰트 크기 키움
                plt.legend(fontsize=14)  # 범례 폰트 크기 키움
                plt.grid(True)

                output_directory = f'./Backtesting/plot/Cumulative_plot/{model}/'
                os.makedirs(output_directory, exist_ok=True)

                plt.savefig(os.path.join(output_directory, f'{ticker}_{label}.png'))
                plt.close()

def plot_cumulative_profit_by_model(base_dir, tickers, labels):
    """Plot cumulative profits for all models."""
    model_types = []
    if os.path.exists(base_dir):
        for folder_name in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, folder_name)):
                model_types.append(folder_name)

    def extract_number(model_label):
        """Extract numerical value from model label for sorting."""
        match = re.search(r'(\d+)', model_label)
        return int(match.group(1)) if match else float('inf')

    for ticker in tqdm(tickers):
        for label in labels:
            plt.figure(figsize=(25, 12))
            plt.tight_layout()

            # Store plot handles and labels for legend
            handles = []
            labels_list = []

            # Group models by prefix for sorting
            model_groups = {}
            for model in model_types:
                model_label = model_name_mapping.get(model, model)
                prefix = re.match(r'^(Cross_single|Cross_exclude|[^\d]+)', model_label)
                if prefix:
                    prefix = prefix.group(0)
                    if prefix not in model_groups:
                        model_groups[prefix] = []
                    model_groups[prefix].append((model, model_label))

            # Plot sorted models
            for prefix, models in model_groups.items():
                if prefix in ['Cross_single', 'Cross_exclude']:
                    # Sort by numerical value
                    models.sort(key=lambda x: extract_number(x[1]))
                for model, model_label in models:
                    try:
                        trading_df = pd.read_csv(f'./Backtesting/simulation/full_period/{model}/{label}/{ticker}.csv')
                        
                        trading_df['Investment_Value'] = trading_df['Close'] - trading_df.iloc[0]['Close']
                        trading_df['Close_Relative'] = trading_df['Investment_Value']
                        
                        trading_df['Date'] = pd.to_datetime(trading_df['Date'])
                        trading_df.set_index('Date', inplace=True)
                        
                        style = model_style_mapping.get(model_label, {'color': 'gray', 'linestyle': 'solid'})
                        
                        line, = plt.plot(
                            trading_df.index, 
                            trading_df['Cumulative_Profit'], 
                            label=model_label, 
                            linewidth=2.0,
                            color=style['color'],
                            linestyle=style['linestyle']
                        )
                        handles.append(line)
                        labels_list.append(model_label)
                        
                    except FileNotFoundError:
                        print(f"File not found for {model}/{label}/{ticker}. Skipping...")
                        continue

            # Plot Buy & Hold
            if 'trading_df' in locals() and not trading_df.empty:
                line, = plt.plot(
                    trading_df.index, 
                    trading_df['Close_Relative'], 
                    label='Buy & Hold', 
                    linestyle='--', 
                    color='black'
                )
                handles.append(line)
                labels_list.append('Buy & Hold')

            plt.xlabel('Date', fontsize=20)
            plt.ylabel('Cumulative Profit', fontsize=20)
            plt.legend(handles, labels_list, fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
            plt.grid(True)

            output_directory = f'./Backtesting/plot/Cumulative_plot_by_Models/{label}/'
            os.makedirs(output_directory, exist_ok=True)

            plt.savefig(os.path.join(output_directory, f'{ticker}_{label}.png'), bbox_inches='tight')
            plt.close()
    
def plot_cumulative_profit_baselines_mm(base_dir, tickers, labels):
    """Plot cumulative profits comparing Cross_Multi_Modal, Self_Multi_Modal, and baseline models."""
    baseline_models = ['1D-CNN', 'GRU', 'LSTM', 'SVM', 'TCN', 'Transformer', 'XGB']
    target_models = baseline_models + [
        'MM_to_seq_(LSTM_25_512_12)_(ViT_512_12_16_1024)_(MHAL_512_16)_(MLP_2048_512)',
        'Seq_to_MM_(LSTM_25_512_12_512)_(ViT_512_12_16_1024_512)_(MHAL_2048_16)_(MLP_1024_512)'
    ]
    
    for ticker in tqdm(tickers):
        for label in labels:
            plt.figure(figsize=(25, 12))
            plt.tight_layout()

            for model in target_models:
                try:
                    trading_df = pd.read_csv(f'./Backtesting/simulation/full_period/{model}/{label}/{ticker}.csv')
                    
                    trading_df['Investment_Value'] = trading_df['Close'] - trading_df.iloc[0]['Close']
                    trading_df['Close_Relative'] = trading_df['Investment_Value']
                    
                    trading_df['Date'] = pd.to_datetime(trading_df['Date'])
                    trading_df.set_index('Date', inplace=True)
                    
                    model_label = model_name_mapping.get(model, model)
                    style = model_style_mapping.get(model_label, {'color': 'gray', 'linestyle': 'solid'})
                    
                    plt.plot(
                        trading_df.index, 
                        trading_df['Cumulative_Profit'], 
                        label=model_label, 
                        linewidth=2.0,
                        color=style['color'],
                        linestyle=style['linestyle']
                    )
                    
                except FileNotFoundError:
                    print(f"File not found for {model}/{label}/{ticker}. Skipping...")
                    continue

            if not trading_df.empty:
                plt.plot(trading_df.index, trading_df['Close_Relative'], 
                        label='Buy & Hold', linestyle='--', color='black')

            # plt.title(f'Cumulative Profit: {ticker} - Baselines vs Multi-Modal', fontsize=22)
            plt.xlabel('Date', fontsize=20)
            plt.ylabel('Cumulative Profit', fontsize=20)
            plt.legend(fontsize=14, loc='upper left')  # Legend inside the plot
            plt.grid(True)

            output_directory = f'./Backtesting/plot/Cumulative_plot_Baselines_with_MM/{label}/'
            os.makedirs(output_directory, exist_ok=True)

            plt.savefig(os.path.join(output_directory, f'{ticker}_{label}.png'), bbox_inches='tight')
            plt.close()

def plot_cumulative_profit_cross_comparisons(base_dir, tickers, labels):
    """Plot cumulative profits comparing Cross_Multi_Modal, Cross_single, and Cross_exclude models."""
    cross_models = [
        'MM_to_seq_(LSTM_25_512_12)_(ViT_512_12_16_1024)_(MHAL_512_16)_(MLP_2048_512)',
        'benchmark_MM_single_120',
        'benchmark_MM_single_20',
        'benchmark_MM_single_5',
        'benchmark_MM_single_60',
        'MM_Causal_ViT_LSTM_ExcludeWindow120_(LSTM_25_256_4)_(ViT_256_4_16_512)_(MHAL_256_16)_(MLP_768_512)',
        'MM_Causal_ViT_LSTM_ExcludeWindow20_(LSTM_25_256_4)_(ViT_256_4_16_512)_(MHAL_256_16)_(MLP_768_512)',
        'MM_Causal_ViT_LSTM_ExcludeWindow5_(LSTM_25_256_4)_(ViT_256_4_16_512)_(MHAL_256_16)_(MLP_768_512)',
        'MM_Causal_ViT_LSTM_ExcludeWindow60_(LSTM_25_256_4)_(ViT_256_4_16_512)_(MHAL_256_16)_(MLP_768_512)'
    ]
    def extract_number(model_label):
        """Extract numerical value from model label for sorting."""
        match = re.search(r'(\d+)', model_label)
        return int(match.group(1)) if match else float('inf')

    for ticker in tqdm(tickers):
        for label in labels:
            plt.figure(figsize=(25, 12))
            plt.tight_layout()

            # Store plot handles and labels for legend
            handles = []
            labels_list = []

            # Group models by prefix for sorting
            model_groups = {}
            for model in cross_models:
                model_label = model_name_mapping.get(model, model)
                prefix = re.match(r'^(Cross_single|Cross_exclude|Cross_Multi_Modal)', model_label)
                if prefix:
                    prefix = prefix.group(0)
                else:
                    prefix = model_label  # Use full label for unique models
                if prefix not in model_groups:
                    model_groups[prefix] = []
                model_groups[prefix].append((model, model_label))

            # Plot sorted models
            for prefix, models in model_groups.items():
                if prefix in ['Cross_single', 'Cross_exclude']:
                    # Sort by numerical value
                    models.sort(key=lambda x: extract_number(x[1]))
                for model, model_label in models:
                    try:
                        trading_df = pd.read_csv(f'./Backtesting/simulation/full_period/{model}/{label}/{ticker}.csv')
                        
                        trading_df['Investment_Value'] = trading_df['Close'] - trading_df.iloc[0]['Close']
                        trading_df['Close_Relative'] = trading_df['Investment_Value']
                        
                        trading_df['Date'] = pd.to_datetime(trading_df['Date'])
                        trading_df.set_index('Date', inplace=True)
                        
                        style = model_style_mapping.get(model_label, {'color': 'gray', 'linestyle': 'solid'})
                        
                        line, = plt.plot(
                            trading_df.index, 
                            trading_df['Cumulative_Profit'], 
                            label=model_label, 
                            linewidth=2.0,
                            color=style['color'],
                            linestyle=style['linestyle']
                        )
                        handles.append(line)
                        labels_list.append(model_label)
                        
                    except FileNotFoundError:
                        print(f"File not found for {model}/{label}/{ticker}. Skipping...")
                        continue

            # Plot Buy & Hold
            if 'trading_df' in locals() and not trading_df.empty:
                line, = plt.plot(
                    trading_df.index, 
                    trading_df['Close_Relative'], 
                    label='Buy & Hold', 
                    linestyle='--', 
                    color='black'
                )
                handles.append(line)
                labels_list.append('Buy & Hold')

            plt.xlabel('Date', fontsize=20)
            plt.ylabel('Cumulative Profit', fontsize=20)
            plt.legend(handles, labels_list, fontsize=14, loc='upper left')  # Legend inside the plot
            plt.grid(True)

            output_directory = f'./Backtesting/plot/Cumulative_plot_Cross_Comparisons/{label}/'
            os.makedirs(output_directory, exist_ok=True)

            plt.savefig(os.path.join(output_directory, f'{ticker}_{label}.png'), bbox_inches='tight')
            plt.close()

'''
02. Drawdown
'''
def plot_drawdown(base_dir, tickers, labels):
    
    model_types = []
    if os.path.exists(base_dir):
        for folder_name in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, folder_name)):
                model_types.append(folder_name)
    
    for ticker in tqdm(tickers):
        for label in labels:
            for model in model_types:             
                trading_df = pd.read_csv(f'./Backtesting/simulation/full_period/{model}/{label}/{ticker}.csv')
                        
                # 날짜를 인덱스로 설정
                trading_df['Date'] = pd.to_datetime(trading_df['Date'])
                trading_df.set_index('Date', inplace=True)
                    
                plt.figure(figsize=(25, 12))
                plt.tight_layout()

                # 시각화
                plt.plot(trading_df.index, trading_df['Drawdown_rate'], label='Drawdown', linewidth=3, color='darkblue')
                plt.fill_between(trading_df.index, 0, trading_df['Drawdown_rate'], color='darkblue', alpha=0.3)
                plt.title(f'Drawdown : {ticker}', fontsize=20)  # 제목 폰트 크기 키움
                plt.xlabel('Date', fontsize=20)  # x축 폰트 크기 키움
                plt.ylabel('Drawdown', fontsize=20)  # y축 폰트 크기 키움
                plt.legend(fontsize=14)  # 범례 폰트 크기 키움
                plt.grid(True)

                output_directory = f'./Backtesting/plot/Drawdown/{model}/'
                os.makedirs(output_directory, exist_ok=True)

                plt.savefig(os.path.join(output_directory, f'{ticker}_{label}.png'))
                plt.close()

'''
03. trading signal
'''
def plot_trading_signal(base_dir, tickers, labels):
    
    model_types = []
    if os.path.exists(base_dir):
        for folder_name in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, folder_name)):
                model_types.append(folder_name)
    
    for ticker in tqdm(tickers):
        for label in labels:
            for model in model_types:             
                trading_df = pd.read_csv(f'./Backtesting/simulation/full_period/{model}/{label}/{ticker}.csv')
                        
                # 날짜를 인덱스로 설정
                trading_df['Date'] = pd.to_datetime(trading_df['Date'])
                trading_df.set_index('Date', inplace=True)
                
                # 매수 (Buy)와 매도 (sell) 신호에 대한 인덱스 추출
                buy_signals = trading_df[trading_df['action'] == 'Buy']
                sell_signals = trading_df[trading_df['action'] == 'sell']

                # 주식 가격과 신호를 시각화
                plt.figure(figsize=(25, 12))
                plt.tight_layout()
                plt.plot(trading_df.index, trading_df['Close'], label='Close', color='black', alpha = 0.5)
                plt.scatter(buy_signals.index, buy_signals['Close'], label='Buy', marker='^', color='g', lw=2, s = 50)
                plt.scatter(sell_signals.index, sell_signals['Close'], label='Sell', marker='v', color='r', lw=2, s = 50)

                plt.title(f'Buy Sell Signal : {ticker}', fontsize=20)  # 제목 폰트 크기 키움
                plt.xlabel('Date', fontsize=20)  # x축 폰트 크기 키움
                plt.ylabel('Price', fontsize=20)  # y축 폰트 크기 키움
                plt.legend(fontsize=14)  # 범례 폰트 크기 키움
                plt.grid(True)

                output_directory = f'./Backtesting/plot/trading_signal/{model}/'
                os.makedirs(output_directory, exist_ok=True)

                plt.savefig(os.path.join(output_directory, f'{ticker}_{label}.png'))
                plt.close()

'''
04. return size
'''
def plot_return_size(base_dir, tickers, labels):
    
    model_types = []
    if os.path.exists(base_dir):
        for folder_name in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, folder_name)):
                model_types.append(folder_name)
    
    for ticker in tqdm(tickers):
        for label in labels:
            for model in model_types:             
                trading_df = pd.read_csv(f'./Backtesting/simulation/full_period/{model}/{label}/{ticker}.csv')
                        
                # 날짜를 인덱스로 설정
                trading_df['Date'] = pd.to_datetime(trading_df['Date'])
                trading_df.set_index('Date', inplace=True)
                
                # 시각화
                fig, ax = plt.subplots(figsize=(25, 12))
                
                # Y축 0에 선 추가
                ax.axhline(y=0, color='gray', linestyle='--')
                
                # return 값에 비례한 원 그리기
                marker_size = 100 * abs(trading_df['Margin_Return'])
                colors = ['red' if x >= 0 else 'blue' for x in trading_df['Margin_Return']]
                ax.scatter(trading_df.index, trading_df['Margin_Return'], s=marker_size, alpha=0.5, color=colors, label='Sell Signal Return')
                
                ax.set_xlabel('Date', fontsize=20)  # x축 폰트 크기 키움
                ax.set_ylabel('Margin_Return', fontsize=20)  # y축 폰트 크기 키움
                ax.set_title('Sell Signal Return Visualization', fontsize=20)  # 제목 폰트 크기 키움
                ax.legend(fontsize=14)  # 범례 폰트 크기 키움
                
                plt.xticks(rotation=45)

                output_directory = f'./Backtesting/plot/return_size/{model}/'
                os.makedirs(output_directory, exist_ok=True)

                plt.savefig(os.path.join(output_directory, f'{ticker}_{label}.png'))
                plt.close()
    