from common_imports import *

# 기술적 지표 생성 함수
def calculate_indicators(df):
       
    # ADX (Average Directional Movement Index)
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    # AROON
    df['AROON_down'], df['AROON_up'] = talib.AROON(df['High'], df['Low'], timeperiod=14)

    # AROONOSC (Aroon Oscillator)
    df['AROONOSC'] = talib.AROONOSC(df['High'], df['Low'], timeperiod=14)

    # BOP (Balance Of Power)
    df['BOP'] = talib.BOP(df['Open'], df['High'], df['Low'], df['Close'])

    # CCI (Commodity Channel Index)
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)

    # CMO (Chande Momentum Oscillator)
    df['CMO'] = talib.CMO(df['Close'], timeperiod=14)

    # DX (Directional Movement Index)
    df['DX'] = talib.DX(df['High'], df['Low'], df['Close'], timeperiod=14)

    # MFI (Money Flow Index)
    df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)

    # MINUS_DI (Minus Directional Indicator)
    df['MINUS_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)

    # MOM (Momentum)
    df['MOM'] = talib.MOM(df['Close'], timeperiod=10)

    # PLUS_DI (Plus Directional Indicator)
    df['PLUS_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)

    # PPO (Percentalibge Price Oscillator)
    df['PPO'] = talib.PPO(df['Close'], fastperiod=12, slowperiod=26, matype=0)

    # ROC (Rate of Change)
    df['ROC'] = talib.ROC(df['Close'], timeperiod=10)

    # ROCR (Rate of Change Ratio)
    df['ROCR'] = talib.ROCR(df['Close'], timeperiod=10)

    # RSI (Relative Strength Index)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

    # STOCH (Stochastic)
    df['STOCH_slowk'], df['STOCH_slowd'] = talib.STOCH(
        df['High'], df['Low'], df['Close'], 
        fastk_period=5, slowk_period=3, slowk_matype=0, 
        slowd_period=3, slowd_matype=0)

    # STOCHF (Stochastic Fast)
    df['STOCHF_fastk'], df['STOCHF_fastd'] = talib.STOCHF(
        df['High'], df['Low'], df['Close'], 
        fastk_period=5, fastd_period=3, fastd_matype=0)

    # STOCHRSI (Stochastic Relative Strength Index)
    df['STOCHRSI_fastk'], df['STOCHRSI_fastd'] = talib.STOCHRSI(df['Close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)

    # TRIX (1-day Rate of Change of a Triple Smooth EMA)
    df['TRIX'] = talib.TRIX(df['Close'], timeperiod=30)

    # ULTOSC (Ultimate Oscillator)
    df['ULTOSC'] = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)

    # WILLR (Williams' %R)
    df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    return df