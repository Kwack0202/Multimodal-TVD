from common_imports import *

# 이미지 생성용 함수
def plot_candles(pricing, title=None, trend_line=False, color_function=None):
    def default_color(index, open_price, close_price, low, high):
        return 'red' if open_price[index] > close_price[index] else 'green'
    
    color_function = color_function or default_color
    open_price = pricing['Open']
    close_price = pricing['Close']
    low = pricing['Low']
    high = pricing['High']
    oc_min = pd.concat([open_price, close_price], axis=1).min(axis=1)
    oc_max = pd.concat([open_price, close_price], axis=1).max(axis=1)
    
    def plot_trendline(pricing, linewidth=5):
        x = np.arange(len(pricing))
        y = pricing.values
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), 'g--', linewidth=2)
        mean_price = np.mean(y)
        std_price = np.std(y)
        plt.plot(x, [mean_price + std_price] * len(x), 'k--', linewidth=2)
        plt.plot(x, [mean_price - std_price] * len(x), 'k--', linewidth=2)
    
    fig = plt.figure(figsize=(10,10))
    
    # 배경을 투명하게 설정
    fig.patch.set_alpha(0.0)  # Figure 배경 투명
    plt.gca().patch.set_alpha(0.0)   # Axes 배경 투명
    
    if title:
        plt.title(title)
    plt.tight_layout()
    x = np.arange(len(pricing))
    candle_colors = [color_function(i, open_price, close_price, low, high) for i in x]
    plt.bar(x, oc_max-oc_min, bottom=oc_min, color=candle_colors, linewidth=0)
    plt.vlines(x, low, high, color=candle_colors, linewidth=1)
    
    if trend_line:
        plot_trendline(pricing['Close'])
    
    plt.grid(True)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    
    return fig