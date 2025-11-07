from common_imports import *

class Backtesting:
    def __init__(self, args):
        self.args = args
    
    def UpDown_Signal(self):
        in_root   = "./stock_prediction/pred_results"
        data_root = "./data/numeric_modal"
        out_root  = "./Backtesting/up_down_signal"

        if not os.path.exists(in_root):
            print(f"[WARNING] Not found: {in_root}")
            return

        for model in sorted(d for d in os.listdir(in_root)
                            if os.path.isdir(os.path.join(in_root, d)) and not d.startswith(".")):
            for lbl in sorted(d for d in os.listdir(os.path.join(in_root, model))
                            if os.path.isdir(os.path.join(in_root, model, d)) and not d.startswith(".")):

                in_dir  = os.path.join(in_root, model, lbl)
                out_dir = os.path.join(out_root, model, lbl)
                os.makedirs(out_dir, exist_ok=True)

                for fname in sorted(f for f in os.listdir(in_dir) if f.lower().endswith(".csv")):
                    ticker  = os.path.splitext(fname)[0]
                    pred_fp = os.path.join(in_dir, fname)
                    px_fp   = os.path.join(data_root, f"{ticker}.csv")

                    # 예측 읽기
                    try:
                        df_pred = pd.read_csv(pred_fp)
                    except Exception as e:
                        print(f"[ERROR] read fail: {pred_fp} -> {e}"); continue

                    if not {"Actual", "Predicted"}.issubset(df_pred.columns):
                        print(f"[WARNING] Need columns ['Actual','Predicted'] in {pred_fp}. Skip."); continue
                    n = len(df_pred)
                    if n == 0:
                        print(f"[WARNING] Empty predictions: {pred_fp}. Skip."); continue

                    # 가격 데이터 읽기 (마지막 n행 사용)
                    if not os.path.exists(px_fp):
                        print(f"[WARNING] Missing price file: {px_fp}. Skip {ticker}."); continue
                    try:
                        df_px = pd.read_csv(px_fp)
                    except Exception as e:
                        print(f"[ERROR] read fail: {px_fp} -> {e}"); continue

                    needed_cols = ["Date","Open","High","Low","Close","Adj Close","Volume"]
                    if not set(needed_cols).issubset(df_px.columns):
                        print(f"[WARNING] {ticker}: missing {set(needed_cols)-set(df_px.columns)} in price file. Skip.")
                        continue

                    tail_n = min(n, len(df_px))
                    if tail_n < n:
                        print(f"[INFO] {ticker}: trim preds {n}->{tail_n} to match price rows {len(df_px)}")

                    px_tail   = df_px.iloc[-tail_n:][needed_cols].copy().reset_index(drop=True)
                    preds_tail= df_pred.iloc[-tail_n:].copy().reset_index(drop=True)

                    thr = 0.5
                    preds_bin = (preds_tail["Predicted"] >= thr).astype(int)

                    # 출력 조립
                    px_tail["Date"] = pd.to_datetime(px_tail["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
                    out = px_tail.assign(
                        label = preds_tail["Actual"].astype(int),
                        pred  = preds_bin.astype(int)
                    )

                    out_fp = os.path.join(out_dir, f"{ticker}.csv")
                    try:
                        out.to_csv(out_fp, index=False)
                        print(f"[INFO] Saved: {out_fp} (thr={thr:.6f}, rows={len(out)})")
                    except Exception as e:
                        print(f"[ERROR] save fail: {out_fp} -> {e}")

    def BuySell_Signal(self):
        in_root  = "./Backtesting/up_down_signal"
        out_root = "./Backtesting/buy_sell_signal"

        # 기간 정의
        periods = {
            "Full_period": ("2021-01-01", "2023-12-31"),
            "2021": ("2021-01-01", "2021-12-31"),
            "2022": ("2022-01-01", "2022-12-31"),
            "2023": ("2023-01-01", "2023-12-31"),
        }

        # 기존 로직: 매수 후 pred==1이면 Holding, pred==0이면 Sell
        def make_actions(pred_series):
            actions, holding = [], False
            n = len(pred_series)
            for i, p in enumerate(pred_series):
                if i == n - 1:
                    actions.append("Sell" if holding else "No action")  # 마지막날 포지션 정리
                    holding = False
                    continue
                if not holding:
                    if p == 1:
                        actions.append("Buy")
                        holding = True
                    else:
                        actions.append("No action")
                else:
                    if p == 1:
                        actions.append("Holding")
                    else:
                        actions.append("Sell")
                        holding = False
            return actions

        ''' def make_actions(pred_series, sell_confirm: int = 1):
            actions = []
            holding = False
            zero_streak = 0
            n = len(pred_series)

            for i, p in enumerate(map(int, pred_series)):
                # 마지막 날 처리: 보유 중이면 청산
                if i == n - 1:
                    actions.append("Sell" if holding else "No action")
                    holding = False
                    continue

                if not holding:
                    if p == 1:
                        actions.append("Buy")
                        holding = True
                        zero_streak = 0
                    else:
                        actions.append("No action")
                else:
                    if p == 0:
                        zero_streak += 1
                        if zero_streak >= sell_confirm:   # ← 0이 연속 sell_confirm회면 매도
                            actions.append("Sell")
                            holding = False
                            zero_streak = 0
                        else:
                            actions.append("Holding")      # 첫 0은 일단 보유 유지
                    else:
                        zero_streak = 0                    # 다시 상승(1) 나오면 카운터 리셋
                        actions.append("Holding")
            return actions '''
        
        if not os.path.exists(in_root):
            print(f"[WARNING] Not found: {in_root}")
            return

        # ./Backtesting/up_down_signal/<model>/<label>/<ticker>.csv 순회
        for model in sorted(d for d in os.listdir(in_root)
                            if os.path.isdir(os.path.join(in_root, d)) and not d.startswith(".")):
            model_dir = os.path.join(in_root, model)
            for lbl in sorted(d for d in os.listdir(model_dir)
                            if os.path.isdir(os.path.join(model_dir, d)) and not d.startswith(".")):
                lbl_dir = os.path.join(model_dir, lbl)
                csv_files = sorted(f for f in os.listdir(lbl_dir) if f.lower().endswith(".csv"))
                if not csv_files:
                    print(f"[INFO] No CSV under {lbl_dir}")
                    continue

                for fname in csv_files:
                    ticker = os.path.splitext(fname)[0]
                    fp = os.path.join(lbl_dir, fname)

                    try:
                        df = pd.read_csv(fp)
                    except Exception as e:
                        print(f"[ERROR] read fail: {fp} -> {e}")
                        continue

                    need = ["Date","Open","High","Low","Close","Adj Close","Volume","label","pred"]
                    if not set(need).issubset(df.columns):
                        print(f"[WARNING] Missing columns in {fp}. Need {need}. Skip {ticker}.")
                        continue

                    # 날짜 파싱/정렬
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

                    # 각 기간별로 신호 생성 & 저장
                    for period_name, (start_s, end_s) in periods.items():
                        start = pd.to_datetime(start_s)
                        end   = pd.to_datetime(end_s)

                        seg = df[(df["Date"] >= start) & (df["Date"] <= end)].copy().reset_index(drop=True)
                        if seg.empty:
                            # 해당 연도/기간에 데이터 없으면 스킵
                            continue

                        # pred 이진(안전) & 액션 생성
                        seg["pred"] = seg["pred"].astype(int)
                        seg["action"] = make_actions(seg["pred"].tolist())
                        # seg["action"] = make_actions(seg["pred"].astype(int).tolist(), sell_confirm=2)

                        # 컬럼 순서 맞춰 저장
                        out_cols = ["Date","Open","High","Low","Close","Adj Close","Volume","label","pred","action"]
                        out_df = seg[out_cols].copy()
                        out_df["Date"] = out_df["Date"].dt.strftime("%Y-%m-%d")

                        out_dir = os.path.join(out_root, period_name, model, lbl)
                        os.makedirs(out_dir, exist_ok=True)
                        out_fp = os.path.join(out_dir, f"{ticker}.csv")
                        try:
                            out_df.to_csv(out_fp, index=False)
                            print(f"[INFO] Saved: {out_fp} (rows={len(out_df)})")
                        except Exception as e:
                            print(f"[ERROR] save fail: {out_fp} -> {e}")
                            
    def Simulation(self, commission_rate=0.0005):
        # 입력/출력 루트
        in_root  = "./Backtesting/buy_sell_signal"
        out_root = "./Backtesting/simulation"

        periods = ["Full_period", "2021", "2022", "2023"]

        for period in periods:
            period_in = os.path.join(in_root, period)
            if not os.path.exists(period_in):
                continue

            for model in sorted(d for d in os.listdir(period_in)
                                if os.path.isdir(os.path.join(period_in, d)) and not d.startswith(".")):
                for lbl in sorted(d for d in os.listdir(os.path.join(period_in, model))
                                if os.path.isdir(os.path.join(period_in, model, d)) and not d.startswith(".")):

                    in_dir  = os.path.join(period_in, model, lbl)
                    out_dir = os.path.join(out_root, period, model, lbl)
                    os.makedirs(out_dir, exist_ok=True)

                    csvs = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(".csv"))
                    for fname in csvs:
                        fp = os.path.join(in_dir, fname)
                        try:
                            df = pd.read_csv(fp)
                        except Exception as e:
                            print(f"[ERROR] read fail: {fp} -> {e}")
                            continue

                        need = ["Date","Open","High","Low","Close","Adj Close","Volume","label","pred","action"]
                        if not set(need).issubset(df.columns):
                            print(f"[WARNING] Missing columns in {fp}. Need {need}. Skip.")
                            continue

                        # 정렬
                        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

                        n = len(df)
                        actions = df["action"].astype(str).tolist()

                        # 다음날 체결 인덱스 (신호일 기준)
                        buy_exec_idx  = {i+1 for i,a in enumerate(actions) if a == "Buy"  and (i+1) < n}
                        sell_exec_idx = {i+1 for i,a in enumerate(actions) if a == "Sell" and (i+1) < n}
                        last_day_sell = (n > 0 and actions[-1] == "Sell")

                        # 결과 버퍼
                        new_data = {
                            "Date": [],
                            "Margin_Profit": [],
                            "Margin_Return": [],
                            "Cumulative_Profit": [],
                            "Unrealized_Profit": [],
                            "Unrealized_Return": [],
                            "Cumulative_Profit_Including_Unrealized": [],
                            "Cumulative_Return_Including_Unrealized": [],
                        }

                        buy_price = None          # 보유 포지션의 체결가
                        cumulative_profit = 0.0   # 실현 누적 손익

                        # 초기 투자금: 첫 매수 체결일의 Open(없으면 첫날 Close)
                        if buy_exec_idx:
                            first_buy_j = min(buy_exec_idx)
                            initial_investment = float(df.iloc[first_buy_j]["Open"])
                        else:
                            initial_investment = float(df.iloc[0]["Close"])

                        # “다음날 아침에 체결/청산 반영” 스케줄
                        pending_buy_exec_idx  = None
                        pending_sell_exec_idx = None

                        for j in range(n):
                            # 아침: 보류 체결/청산 반영
                            if pending_buy_exec_idx is not None and j == pending_buy_exec_idx:
                                buy_price = float(df.at[j, "Open"])
                                pending_buy_exec_idx = None

                            if pending_sell_exec_idx is not None and j == pending_sell_exec_idx:
                                # 청산 체결 완료 → 포지션 해제
                                buy_price = None
                                pending_sell_exec_idx = None

                            date_str = df.at[j, "Date"].strftime("%Y-%m-%d")
                            margin_profit = 0.0
                            margin_return = 0.0

                            act = actions[j]

                            # ===== 신호일에 "실현 손익" 기록 =====
                            if act == "Sell":
                                if buy_price is not None:
                                    # 1) entry_price는 'buy_price'를 변경하기 전에 따로 보관
                                    entry_price = float(buy_price)

                                    # 2) 청산 가격 결정(다음날 시가, 마지막 날이면 당일 종가)
                                    if j+1 < n:
                                        exec_price = float(df.at[j+1, "Open"])
                                        pending_sell_exec_idx = j+1  # 내일 아침에 포지션 해제
                                        # buy_price는 아직 유지 (익일 아침에 해제됨)
                                    else:
                                        exec_price = float(df.at[j, "Close"])
                                        pending_sell_exec_idx = None
                                        # 마지막 날은 즉시 청산되므로 여기서 포지션 해제
                                        buy_price = None

                                    # 3) 손익 계산 (entry_price 사용 — None 안전)
                                    profit = exec_price - entry_price - (exec_price * commission_rate)
                                    margin_profit = profit
                                    margin_return = (profit / entry_price) * 100 if entry_price != 0 else 0.0
                                    cumulative_profit += profit
                                else:
                                    # 포지션이 없는데 Sell 신호 → 손익 없음, 스킵
                                    margin_profit = 0.0
                                    margin_return = 0.0
                            elif act == "Buy":
                                # 다음날 시가에 매수 체결(마지막 날 Buy는 체결 불가)
                                if j+1 < n:
                                    pending_buy_exec_idx = j+1

                            # ===== EOD 미실현(MTM) 계산 =====
                            if buy_price is not None:
                                mtm_price = float(df.at[j, "Close"])
                                unreal = mtm_price - float(buy_price)     # 수수료 미차감
                                unreal_ret = (unreal / float(buy_price)) * 100 if float(buy_price) != 0 else 0.0
                            else:
                                unreal = 0.0
                                unreal_ret = 0.0

                            cum_incl_unreal = cumulative_profit + unreal
                            cum_ret_incl_unreal = (cum_incl_unreal / initial_investment) * 100 if initial_investment != 0 else 0.0

                            # 기록
                            new_data["Date"].append(date_str)
                            new_data["Margin_Profit"].append(margin_profit)
                            new_data["Cumulative_Profit"].append(cumulative_profit)
                            new_data["Margin_Return"].append(margin_return)
                            new_data["Unrealized_Profit"].append(unreal)
                            new_data["Unrealized_Return"].append(unreal_ret)
                            new_data["Cumulative_Profit_Including_Unrealized"].append(cum_incl_unreal)
                            new_data["Cumulative_Return_Including_Unrealized"].append(cum_ret_incl_unreal)

                        new_df = pd.DataFrame(new_data)

                        # 병합
                        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                        new_df["Date"] = pd.to_datetime(new_df["Date"], errors="coerce")
                        merged_df = pd.merge(df, new_df, on="Date", how="outer")

                        # Holding_Period
                        merged_df["Holding_Period"] = merged_df.groupby(
                            (merged_df["action"] != "Holding").cumsum()
                        ).cumcount()

                        # 누적 수익률(실현)
                        merged_df["Cumulative_Return"] = (
                            merged_df["Cumulative_Profit"] / initial_investment * 100
                            if initial_investment != 0 else 0.0
                        )

                        # 드로우다운(실현)
                        merged_df["Drawdown"] = 0.0
                        merged_df["Drawdown_rate"] = 0.0
                        peak_profit = float(merged_df["Cumulative_Profit"].iloc[0])
                        peak_profit_rate = float(merged_df["Cumulative_Return"].iloc[0])

                        for idx, row in merged_df.iterrows():
                            cur_profit = float(row["Cumulative_Profit"])
                            if cur_profit > peak_profit:
                                peak_profit = cur_profit
                            merged_df.at[idx, "Drawdown"] = -(peak_profit - cur_profit)

                            cur_rate = float(row["Cumulative_Return"])
                            if cur_rate > peak_profit_rate:
                                peak_profit_rate = cur_rate
                            merged_df.at[idx, "Drawdown_rate"] = -(peak_profit_rate - cur_rate)

                        # 드로우다운(미실현 포함)
                        merged_df["Drawdown_unrealized"] = 0.0
                        merged_df["Drawdown_rate_unrealized"] = 0.0
                        peak_pnl_incl = float(merged_df["Cumulative_Profit_Including_Unrealized"].iloc[0])
                        peak_rate_incl = float(merged_df["Cumulative_Return_Including_Unrealized"].iloc[0])

                        for idx, row in merged_df.iterrows():
                            cur_pnl_incl = float(row["Cumulative_Profit_Including_Unrealized"])
                            if cur_pnl_incl > peak_pnl_incl:
                                peak_pnl_incl = cur_pnl_incl
                            merged_df.at[idx, "Drawdown_unrealized"] = -(peak_pnl_incl - cur_pnl_incl)

                            cur_rate_incl = float(row["Cumulative_Return_Including_Unrealized"])
                            if cur_rate_incl > peak_rate_incl:
                                peak_rate_incl = cur_rate_incl
                            merged_df.at[idx, "Drawdown_rate_unrealized"] = -(peak_rate_incl - cur_rate_incl)

                        # 출력
                        out_cols = [
                            "Date","Open","High","Low","Close","Adj Close","Volume",
                            "label","pred","action",
                            "Margin_Profit","Cumulative_Profit","Margin_Return","Cumulative_Return",
                            "Unrealized_Profit","Unrealized_Return",
                            "Cumulative_Profit_Including_Unrealized","Cumulative_Return_Including_Unrealized",
                            "Drawdown","Drawdown_rate",
                            "Drawdown_unrealized","Drawdown_rate_unrealized",
                            "Holding_Period",
                        ]
                        merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
                        merged_df = merged_df[out_cols].copy().round(3)

                        out_fp = os.path.join(out_dir, fname)
                        try:
                            merged_df.to_csv(out_fp, index=False)
                            print(f"[INFO] Saved simulation (signal-day realized): {out_fp}")
                        except Exception as e:
                            print(f"[ERROR] save fail: {out_fp} -> {e}")

    def Benchmark_BuyAndHold(self, commission_rate=0.0000):
        data_root = "./data/numeric_modal"
        out_root  = "./Backtesting/simulation"

        periods = {
            "Full_period": ("2021-01-01", "2023-12-31"),
            "2021": ("2021-01-01", "2021-12-31"),
            "2022": ("2022-01-01", "2022-12-31"),
            "2023": ("2023-01-01", "2023-12-31"),
        }

        if not os.path.exists(data_root):
            print(f"[WARNING] Not found: {data_root}")
            return

        # ticker 파일들 순회
        for fname in sorted(f for f in os.listdir(data_root) if f.lower().endswith(".csv")):
            ticker = os.path.splitext(fname)[0]
            fp = os.path.join(data_root, fname)

            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"[ERROR] read fail: {fp} -> {e}")
                continue

            need = ["Date","Open","High","Low","Close","Adj Close","Volume"]
            if not set(need).issubset(df.columns):
                print(f"[WARNING] {ticker}: missing {set(need)-set(df.columns)}. Skip.")
                continue

            # 마지막 753행만 사용
            tail_n = min(753, len(df))
            if tail_n <= 0:
                print(f"[INFO] Empty after trimming: {ticker}")
                continue

            df = df.iloc[-tail_n:].copy()
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

            # 기간별 저장
            for period_name, (start_s, end_s) in periods.items():
                start = pd.to_datetime(start_s)
                end   = pd.to_datetime(end_s)

                seg = df[(df["Date"] >= start) & (df["Date"] <= end)].copy().reset_index(drop=True)
                if seg.empty:
                    continue

                # 초기 투자금: 해당 기간 첫날 Open
                initial_investment = float(seg.iloc[0]["Open"])

                # 결과 버퍼
                rows = {
                    "Date": [],
                    "Open": [], "High": [], "Low": [], "Close": [], "Adj Close": [], "Volume": [],
                    "label": [], "pred": [], "action": [],
                    "Margin_Profit": [], "Cumulative_Profit": [], "Margin_Return": [], "Cumulative_Return": [],
                    "Unrealized_Profit": [], "Unrealized_Return": [],
                    "Cumulative_Profit_Including_Unrealized": [], "Cumulative_Return_Including_Unrealized": [],
                    "Drawdown": [], "Drawdown_rate": [],
                    "Drawdown_unrealized": [], "Drawdown_rate_unrealized": [],
                    "Holding_Period": [],
                }

                cumulative_profit = 0.0
                peak_profit = 0.0
                peak_rate   = 0.0

                for _, r in seg.iterrows():
                    o = float(r["Open"]); c = float(r["Close"])
                    profit = (c - o) - (c * commission_rate)   # 매도 시 수수료
                    ret_pct = (profit / o) * 100 if o != 0 else 0.0
                    cumulative_profit += profit
                    cum_ret = (cumulative_profit / initial_investment) * 100 if initial_investment != 0 else 0.0

                    # Buy&Hold(당일 청산) → 미실현 지표는 0, 포함 누적 = 실현 누적
                    unreal = 0.0
                    unreal_ret = 0.0
                    cum_incl_unreal = cumulative_profit
                    cum_ret_incl_unreal = cum_ret

                    # DD(실현 기준)
                    if cumulative_profit > peak_profit:
                        peak_profit = cumulative_profit
                    dd = -(peak_profit - cumulative_profit)

                    if cum_ret > peak_rate:
                        peak_rate = cum_ret
                    dd_rate = -(peak_rate - cum_ret)

                    # 미실현 포함 DD (동일)
                    dd_u = dd
                    dd_rate_u = dd_rate

                    rows["Date"].append(r["Date"].strftime("%Y-%m-%d"))
                    rows["Open"].append(o); rows["High"].append(float(r["High"])); rows["Low"].append(float(r["Low"]))
                    rows["Close"].append(c); rows["Adj Close"].append(float(r["Adj Close"])); rows["Volume"].append(r["Volume"])

                    rows["label"].append(0); rows["pred"].append(0)
                    rows["action"].append("Sell")  # 매일 당일 청산

                    rows["Margin_Profit"].append(profit)
                    rows["Cumulative_Profit"].append(cumulative_profit)
                    rows["Margin_Return"].append(ret_pct)
                    rows["Cumulative_Return"].append(cum_ret)

                    rows["Unrealized_Profit"].append(unreal)
                    rows["Unrealized_Return"].append(unreal_ret)
                    rows["Cumulative_Profit_Including_Unrealized"].append(cum_incl_unreal)
                    rows["Cumulative_Return_Including_Unrealized"].append(cum_ret_incl_unreal)

                    rows["Drawdown"].append(dd)
                    rows["Drawdown_rate"].append(dd_rate)
                    rows["Drawdown_unrealized"].append(dd_u)
                    rows["Drawdown_rate_unrealized"].append(dd_rate_u)

                    rows["Holding_Period"].append(0)

                out_df = pd.DataFrame(rows).round(3)

                out_dir = os.path.join(out_root, period_name, "Buy&Hold", "benchmark")
                os.makedirs(out_dir, exist_ok=True)
                out_fp = os.path.join(out_dir, f"{ticker}.csv")
                try:
                    out_df.to_csv(out_fp, index=False)
                    print(f"[INFO] Saved Buy&Hold: {out_fp} (rows={len(out_df)})")
                except Exception as e:
                    print(f"[ERROR] save fail: {out_fp} -> {e}")

    def BacktestSummary(self):
        in_root = "./Backtesting/simulation"
        out_dir = "./Backtesting/backtesting"
        os.makedirs(out_dir, exist_ok=True)

        periods = ["Full_period", "2021", "2022", "2023"]

        for period in periods:
            period_dir = os.path.join(in_root, period)
            if not os.path.exists(period_dir):
                print(f"[INFO] Skip period {period} (no folder).")
                continue

            rows = []

            # <model>/<label>/<ticker>.csv 순회
            for model in sorted(d for d in os.listdir(period_dir)
                                if os.path.isdir(os.path.join(period_dir, d)) and not d.startswith(".")):
                model_dir = os.path.join(period_dir, model)

                for lbl in sorted(d for d in os.listdir(model_dir)
                                if os.path.isdir(os.path.join(model_dir, d)) and not d.startswith(".")):
                    lbl_dir = os.path.join(model_dir, lbl)

                    for fname in sorted(f for f in os.listdir(lbl_dir) if f.lower().endswith(".csv")):
                        ticker = os.path.splitext(fname)[0]
                        fp = os.path.join(lbl_dir, fname)

                        try:
                            df = pd.read_csv(fp)
                        except Exception as e:
                            print(f"[ERROR] read fail: {fp} -> {e}")
                            continue
                        if df.empty:
                            continue

                        # 안전한 타입 보정
                        df["action"] = df.get("action", "No action").fillna("No action").astype(str)

                        num_cols = [
                            # 기존 실현 지표
                            "Margin_Profit","Cumulative_Profit","Margin_Return","Cumulative_Return",
                            "Drawdown","Drawdown_rate","Holding_Period",
                            # 새 미실현/포함 지표
                            "Unrealized_Profit","Unrealized_Return",
                            "Cumulative_Profit_Including_Unrealized","Cumulative_Return_Including_Unrealized",
                            "Drawdown_unrealized","Drawdown_rate_unrealized",
                        ]
                        for c in num_cols:
                            if c in df.columns:
                                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                            else:
                                df[c] = 0.0

                        # 거래(청산) 기준: action == 'Sell'
                        sells_mask = (df["action"] == "Sell")
                        no_trade = int(sells_mask.sum())

                        # 승률
                        winners = int(((sells_mask) & (df["Margin_Profit"] > 0)).sum())
                        winning_ratio = (winners / no_trade) if no_trade > 0 else 0.0

                        # 평균 이익/손실, 페이오프
                        profit_avg = df.loc[(sells_mask) & (df["Margin_Profit"] > 0), "Margin_Profit"].mean()
                        profit_avg = 0.0 if pd.isna(profit_avg) else float(profit_avg)

                        loss_avg = df.loc[(sells_mask) & (df["Margin_Profit"] < 0), "Margin_Profit"].mean()
                        loss_avg = 0.0 if pd.isna(loss_avg) else float(loss_avg)

                        payoff_ratio = (profit_avg / -loss_avg) if loss_avg < 0 else 0.0

                        # 프로핏 팩터
                        profit_sum = df.loc[(sells_mask) & (df["Margin_Profit"] > 0), "Margin_Profit"].sum()
                        loss_sum   = df.loc[(sells_mask) & (df["Margin_Profit"] < 0), "Margin_Profit"].sum()
                        profit_factor = (-profit_sum / loss_sum) if loss_sum < 0 else 0.0

                        # 최종 누적(실현만)
                        final_cum_profit = float(df["Cumulative_Profit"].iloc[-1]) if "Cumulative_Profit" in df.columns else 0.0
                        final_cum_return = float(df["Cumulative_Return"].iloc[-1]) if "Cumulative_Return" in df.columns else 0.0

                        # 최대 실현 이익/수익률 (Sell 시점)
                        max_realized_profit = df.loc[sells_mask, "Margin_Profit"].max() if no_trade > 0 else 0.0
                        max_realized_profit = 0.0 if pd.isna(max_realized_profit) else float(max_realized_profit)

                        max_realized_return = df.loc[sells_mask, "Margin_Return"].max() if no_trade > 0 else 0.0
                        max_realized_return = 0.0 if pd.isna(max_realized_return) else float(max_realized_return)

                        # MDD (실현 기준)
                        MDD      = df["Drawdown"].min() if "Drawdown" in df.columns else 0.0
                        MDD_rate = df["Drawdown_rate"].min() if "Drawdown_rate" in df.columns else 0.0
                        MDD      = 0.0 if pd.isna(MDD) else float(MDD)
                        MDD_rate = 0.0 if pd.isna(MDD_rate) else float(MDD_rate)

                        # ===== 미실현 포함 요약 =====
                        # 최종 누적(미실현 포함)
                        final_cum_profit_incl_unreal = float(
                            df["Cumulative_Profit_Including_Unrealized"].iloc[-1]
                        ) if "Cumulative_Profit_Including_Unrealized" in df.columns else 0.0

                        final_cum_return_incl_unreal = float(
                            df["Cumulative_Return_Including_Unrealized"].iloc[-1]
                        ) if "Cumulative_Return_Including_Unrealized" in df.columns else 0.0

                        # 홀딩 중 최대/최소 미실현 손익/수익률
                        max_unrealized_profit = float(df["Unrealized_Profit"].max()) if "Unrealized_Profit" in df.columns else 0.0
                        min_unrealized_profit = float(df["Unrealized_Profit"].min()) if "Unrealized_Profit" in df.columns else 0.0
                        if pd.isna(max_unrealized_profit): max_unrealized_profit = 0.0
                        if pd.isna(min_unrealized_profit): min_unrealized_profit = 0.0

                        max_unrealized_return = float(df["Unrealized_Return"].max()) if "Unrealized_Return" in df.columns else 0.0
                        min_unrealized_return = float(df["Unrealized_Return"].min()) if "Unrealized_Return" in df.columns else 0.0
                        if pd.isna(max_unrealized_return): max_unrealized_return = 0.0
                        if pd.isna(min_unrealized_return): min_unrealized_return = 0.0

                        # MDD (미실현 포함 기준)
                        MDD_unreal      = df["Drawdown_unrealized"].min() if "Drawdown_unrealized" in df.columns else 0.0
                        MDD_rate_unreal = df["Drawdown_rate_unrealized"].min() if "Drawdown_rate_unrealized" in df.columns else 0.0
                        MDD_unreal      = 0.0 if pd.isna(MDD_unreal) else float(MDD_unreal)
                        MDD_rate_unreal = 0.0 if pd.isna(MDD_rate_unreal) else float(MDD_rate_unreal)

                        # 보유기간 통계: 'Holding' 행 기준
                        hold_mask = (df["action"] == "Holding")
                        if hold_mask.any():
                            max_holding_period  = df.loc[hold_mask, "Holding_Period"].max()
                            mean_holding_period = df.loc[hold_mask, "Holding_Period"].mean()
                        else:
                            max_holding_period = 0.0
                            mean_holding_period = 0.0

                        max_holding_period  = 0.0 if pd.isna(max_holding_period) else float(max_holding_period)
                        mean_holding_period = 0.0 if pd.isna(mean_holding_period) else float(mean_holding_period)

                        rows.append([
                            period, model, lbl, ticker,
                            no_trade, max_holding_period, mean_holding_period, winning_ratio,
                            profit_avg, loss_avg, payoff_ratio, profit_factor,
                            final_cum_profit, final_cum_return,
                            max_realized_profit, max_realized_return,
                            MDD, MDD_rate,
                            # --- New (Unrealized/Included) ---
                            final_cum_profit_incl_unreal, final_cum_return_incl_unreal,
                            max_unrealized_profit, max_unrealized_return,
                            min_unrealized_profit, min_unrealized_return,
                            MDD_unreal, MDD_rate_unreal,
                        ])

            if not rows:
                print(f"[INFO] No summaries to write for period {period}.")
                continue

            cols = [
                "period","model","label","ticker",
                "no_trade","max_holding_period","mean_holding_period","winning_ratio",
                "profit_average","loss_average","payoff_ratio","profit_factor",
                "final_cumulative_profit","final_cumulative_return",
                "max_realized_profit","max_realized_return",
                "MaxDrawdown","MaxDrawdown_rate",
                # --- New columns (unrealized / included) ---
                "final_cumulative_profit_incl_unrealized","final_cumulative_return_incl_unrealized",
                "max_unrealized_profit","max_unrealized_return",
                "min_unrealized_profit","min_unrealized_return",
                "MaxDrawdown_unrealized","MaxDrawdown_rate_unrealized",
            ]

            out_fp = os.path.join(out_dir, f"results_summary_{period}.csv")
            pd.DataFrame(rows, columns=cols).round(3).to_csv(out_fp, encoding="utf-8-sig", index=False)
            print(f"[INFO] Saved backtesting summary -> {out_fp}")

    def PlotResults(self):
        plt.rcParams.update({
            'axes.titlesize': 30,
            'axes.labelsize': 30,
            'xtick.labelsize': 25,
            'ytick.labelsize': 25,
            'legend.fontsize': 30
        })

        in_base  = "./Backtesting/simulation"
        out_base = "./Backtesting/plots"
        os.makedirs(out_base, exist_ok=True)

        periods = ["Full_period"] # "2021", "2022", "2023"]

        # 표시명 매핑 (요청 반영)
        model_name_mapping = {
            # Multimodal full
            'MM_Causal_ViT_LSTM_(LSTM_25_256_4)_(ViT_256_4_16_512_img320_patch16)_(MHAL_256_16)_(MLP_1024_512)' : 'Fusion A',
            'MM_Causal_ViT_LSTM_FusedWindows_(LSTM_25_256_4)_(ViT_256_4_16_512_img320_patch16)_(MHAL_256_16)_(MLP_256_512)' : 'Fusion B',

            # Multimodal ablations (ONLY → L=, EXCLUDE → exc.)
            'MM_Causal_ViT_LSTM_OnlyW[5]_(LSTM_25_256_4)_(ViT_256_4_16_512_img320_patch16)_(MHAL_256_16)_(MLP_256_512)' :  'Fusion A (L=5)',
            'MM_Causal_ViT_LSTM_OnlyW[20]_(LSTM_25_256_4)_(ViT_256_4_16_512_img320_patch16)_(MHAL_256_16)_(MLP_256_512)' : 'Fusion A (L=20)',
            'MM_Causal_ViT_LSTM_OnlyW[60]_(LSTM_25_256_4)_(ViT_256_4_16_512_img320_patch16)_(MHAL_256_16)_(MLP_256_512)' : 'Fusion A (L=60)',
            'MM_Causal_ViT_LSTM_OnlyW[120]_(LSTM_25_256_4)_(ViT_256_4_16_512_img320_patch16)_(MHAL_256_16)_(MLP_256_512)' : 'Fusion A (L=120)',

            'MM_Causal_ViT_LSTM_ExcludeW[5]_(LSTM_25_256_4)_(ViT_256_4_16_512_img320_patch16)_(MHAL_256_16)_(MLP_768_512)'  : 'Fusion A exc. 5',
            'MM_Causal_ViT_LSTM_ExcludeW[20]_(LSTM_25_256_4)_(ViT_256_4_16_512_img320_patch16)_(MHAL_256_16)_(MLP_768_512)' : 'Fusion A exc. 20',
            'MM_Causal_ViT_LSTM_ExcludeW[60]_(LSTM_25_256_4)_(ViT_256_4_16_512_img320_patch16)_(MHAL_256_16)_(MLP_768_512)' : 'Fusion A exc. 60',
            'MM_Causal_ViT_LSTM_ExcludeW[120]_(LSTM_25_256_4)_(ViT_256_4_16_512_img320_patch16)_(MHAL_256_16)_(MLP_768_512)': 'Fusion A exc. 120',

            # Fusion B 단일모달 표기 수정
            'MM_Causal_ViT_LSTM_FusedWindows_Ablation_IMGOnly_(LSTM_25_256_4)_(ViT_256_4_16_512_img320_patch16)_(MHAL_256_16)_(MLP_256_512)' : 'Fusion B (IMG only)',
            'MM_Causal_ViT_LSTM_FusedWindows_Ablation_TAOnly_(LSTM_25_256_4)_(ViT_skipped)_(MHAL_256_16)_(MLP_256_512)' : 'Fusion B (TI only)',

            # Baselines / Benchmark
            '1D-CNN': '1D-CNN',
            'GRU': 'GRU',
            'LSTM': 'LSTM',
            'TCN': 'TCN',
            'Transformer': 'Transformer',
            'Buy&Hold': 'Buy&Hold',
        }

        # 색/스타일 (표시명 기준)
        model_style_mapping = {
            # baselines
            '1D-CNN': {'color': '#1f77b4', 'linestyle': 'solid'},
            'GRU': {'color': '#ff7f0e', 'linestyle': 'solid'},
            'LSTM': {'color': '#2ca02c', 'linestyle': 'solid'},
            'TCN': {'color': '#9467bd', 'linestyle': 'solid'},
            'Transformer': {'color': '#8c564b', 'linestyle': 'solid'},
            # Multimodal full
            'Fusion A': {'color': '#800080', 'linestyle': 'solid'},
            'Fusion B': {'color': '#ff4500', 'linestyle': 'solid'},
            # Multimodal ablations
            'Fusion A (L=5)':   {'color': '#1e90ff', 'linestyle': 'solid'},
            'Fusion A (L=20)':  {'color': '#87cefa', 'linestyle': 'solid'},
            'Fusion A (L=60)':  {'color': '#4682b4', 'linestyle': 'solid'},
            'Fusion A (L=120)': {'color': '#b0e0e6', 'linestyle': 'solid'},

            'Fusion A exc. 5':   {'color': '#228b22', 'linestyle': 'dashed'},
            'Fusion A exc. 20':  {'color': '#90ee90', 'linestyle': 'dashed'},
            'Fusion A exc. 60':  {'color': '#3cb371', 'linestyle': 'dashed'},
            'Fusion A exc. 120': {'color': '#98fb98', 'linestyle': 'dashed'},

            'Fusion B (IMG only)': {'color': '#808080', 'linestyle': 'solid'},
            'Fusion B (TI only)':  {'color': '#a9a9a9', 'linestyle': 'solid'},
            # Benchmark
            'Buy&Hold': {'color': '#000000', 'linestyle': 'dashdot'},
        }

        # 분류 집합(표시명 기준)
        baselines = {'1D-CNN', 'GRU', 'LSTM', 'TCN', 'Transformer'}
        multimodal_full = {'Fusion A', 'Fusion B', 'Fusion A Ver2'}  # Ver2 없으면 자연히 제외
        ablation_prefixes = ('Fusion A (L=', 'Fusion A exc.', 'Fusion B (')  # 단일모달/어블레이션 식별

        def disp_name(model: str) -> str:
            """원본 모델명을 표시명으로 변환(언더바 제거 포함)."""
            name = model_name_mapping.get(model, model)
            return name.replace("_", " ")

        def classify_display_name(display_name: str) -> str:
            if display_name in baselines:
                return 'baseline'
            if display_name in multimodal_full:
                return 'multimodal'
            if any(display_name.startswith(p) for p in ablation_prefixes):
                return 'ablation'
            if display_name == 'Buy&Hold':
                return 'other'
            return 'other'

        def style_for(display_name: str):
            if display_name in model_style_mapping:
                s = model_style_mapping[display_name]
                return s.get('color', None), s.get('linestyle', 'solid')
            return None, 'solid'

        def load_df(period, model, lbl, ticker):
            fp = os.path.join(in_base, period, model, lbl, f"{ticker}.csv")
            if not os.path.exists(fp):
                return None
            df = pd.read_csv(fp)
            if "Date" not in df.columns:
                return None
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
            return df

        # (1) 거래 신호 / (2) 실현 수익 사이즈 — 개별 모델
        for period in periods:
            period_dir = os.path.join(in_base, period)
            if not os.path.exists(period_dir):
                continue

            models = [m for m in os.listdir(period_dir) if os.path.isdir(os.path.join(period_dir, m))]
            for model in sorted(models):
                shown_model = disp_name(model)
                model_dir = os.path.join(period_dir, model)
                labels = [l for l in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, l))]

                for lbl in sorted(labels):
                    in_dir = os.path.join(model_dir, lbl)
                    tickers = sorted([f[:-4] for f in os.listdir(in_dir) if f.lower().endswith(".csv")])

                    out_ts = os.path.join(out_base, "Trading_signal", period, shown_model, lbl)
                    out_rs = os.path.join(out_base, "Realized_return_size", period, shown_model, lbl)
                    os.makedirs(out_ts, exist_ok=True)
                    os.makedirs(out_rs, exist_ok=True)

                    for t in tickers:
                        df = load_df(period, model, lbl, t)
                        if df is None or df.empty:
                            continue

                        # 거래 신호
                        buy  = df[df["action"] == "Buy"]  if "action" in df.columns else pd.DataFrame()
                        sell = df[df["action"] == "Sell"] if "action" in df.columns else pd.DataFrame()

                        plt.figure(figsize=(24, 12))
                        if "Close" in df.columns:
                            plt.plot(df.index, df["Close"], alpha=0.85, linewidth=2.5, label="Close")
                        if not buy.empty:
                            plt.scatter(buy.index, buy["Close"], marker="^", c="green", s=160, label="Buy")
                        if not sell.empty:
                            plt.scatter(sell.index, sell["Close"], marker="v", c="red", s=160, label="Sell")
                        plt.title(f"Trading signals : {t}")
                        plt.xlabel("Date"); plt.ylabel("Price")
                        plt.grid(True, alpha=0.3); plt.legend(ncol=2)
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_ts, f"{t}.png")); plt.close()

                        # 실현 수익 사이즈 (Sell 시점 등)
                        plt.figure(figsize=(24, 12))
                        plt.axhline(y=0, color="gray", linestyle="--", linewidth=1)

                        mret = pd.to_numeric(df.get("Margin_Return", pd.Series(index=df.index, dtype=float)),
                                            errors="coerce").fillna(0.0)
                        realized = df.loc[mret != 0].copy()
                        if realized.empty and "action" in df.columns:
                            realized = df.loc[(df["action"] == "Sell") & (mret != 0)].copy()

                        if not realized.empty:
                            realized["Margin_Return"] = pd.to_numeric(realized["Margin_Return"], errors="coerce")
                            pos = realized[realized["Margin_Return"] >= 0]
                            neg = realized[realized["Margin_Return"] < 0]

                            def bubble_sizes(s, base=60, k=500, cap=1200):
                                import numpy as np
                                sizes = base + k * s.abs()
                                return np.clip(sizes, base, cap)

                            if not pos.empty:
                                plt.scatter(pos.index, pos["Margin_Return"],
                                            s=bubble_sizes(pos["Margin_Return"]),
                                            alpha=0.5, c="green", label="Positive")
                            if not neg.empty:
                                plt.scatter(neg.index, neg["Margin_Return"],
                                            s=bubble_sizes(neg["Margin_Return"]),
                                            alpha=0.5, c="red", label="Negative")

                        plt.title(f"Realized return : {t}")
                        plt.xlabel("Date"); plt.ylabel("Realized return (%)")
                        plt.grid(True, alpha=0.3); plt.legend(ncol=2)
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_rs, f"{t}.png"))
                        plt.close()

        # (3)(4)(5)(6) 비교 차트 — 실현/미실현
        for period in periods:
            period_dir = os.path.join(in_base, period)
            if not os.path.exists(period_dir):
                continue

            models = [m for m in os.listdir(period_dir) if os.path.isdir(os.path.join(period_dir, m))]
            if not models:
                continue

            # label 수집
            all_labels = set()
            for m in models:
                mdir = os.path.join(period_dir, m)
                for l in os.listdir(mdir):
                    if os.path.isdir(os.path.join(mdir, l)):
                        all_labels.add(l)

            for lbl in sorted(all_labels):
                # 이 label의 모든 티커 수집
                tickers = set()
                for m in models:
                    in_dir = os.path.join(period_dir, m, lbl)
                    if os.path.exists(in_dir):
                        for f in os.listdir(in_dir):
                            if f.lower().endswith(".csv"):
                                tickers.add(f[:-4])

                # (1) 벤치마크 폴더의 티커도 합치기
                bh_dir = os.path.join(period_dir, "Buy&Hold", "benchmark")
                if os.path.exists(bh_dir):
                    for f in os.listdir(bh_dir):
                        if f.lower().endswith(".csv"):
                            tickers.add(f[:-4])

                # 폴더 준비 — 실현 기준
                out_crc_base = os.path.join(out_base, "Cumulative_return_comparison", period, lbl)
                out_dd_base  = os.path.join(out_base, "Drawdown_comparison", period, lbl)
                os.makedirs(os.path.join(out_crc_base, "with_baselines"), exist_ok=True)
                os.makedirs(os.path.join(out_crc_base, "with_ablations"), exist_ok=True)
                os.makedirs(os.path.join(out_dd_base, "with_baselines"), exist_ok=True)
                os.makedirs(os.path.join(out_dd_base, "with_ablations"), exist_ok=True)

                # 폴더 준비 — 미실현 포함
                out_crcU_base = os.path.join(out_base, "Cumulative_return_including_unrealized_comparison", period, lbl)
                out_ddU_base  = os.path.join(out_base, "Drawdown_unrealized_comparison", period, lbl)
                os.makedirs(os.path.join(out_crcU_base, "with_baselines"), exist_ok=True)
                os.makedirs(os.path.join(out_crcU_base, "with_ablations"), exist_ok=True)
                os.makedirs(os.path.join(out_ddU_base,  "with_baselines"), exist_ok=True)
                os.makedirs(os.path.join(out_ddU_base,  "with_ablations"), exist_ok=True)

                for t in sorted(tickers):
                    # 공통 로딩 + 분류
                    data_by_model = {}
                    class_by_model = {}

                    # 라벨 내부 모델들
                    for m in models:
                        shown = disp_name(m)  # 표시명
                        df = load_df(period, m, lbl, t)
                        if df is None or df.empty:
                            continue
                        data_by_model[shown] = df
                        class_by_model[shown] = classify_display_name(shown)

                    # (2) 벤치마크 강제 합류(label="benchmark")
                    bh_df = load_df(period, "Buy&Hold", "benchmark", t)
                    if bh_df is not None and not bh_df.empty:
                        data_by_model["Buy&Hold"] = bh_df
                        class_by_model["Buy&Hold"] = "other"

                    def plot_compare(group_name, metric_col, out_path, fill_below_zero=False, title_text=None, y_label=None):
                        # 대상군 정의 (두 그룹 모두에 벤치마크 포함)
                        if group_name == 'with_baselines':
                            target_classes = {'baseline', 'multimodal', 'other'}
                        else:
                            target_classes = {'ablation', 'multimodal', 'other'}

                        plotted = 0
                        plt.figure(figsize=(24, 12))
                        ax = plt.gca()
                        for shown, df in sorted(data_by_model.items()):
                            klass = class_by_model.get(shown, 'other')
                            if klass not in target_classes:
                                continue
                            if metric_col not in df.columns:
                                continue

                            color, linestyle = style_for(shown)
                            y = pd.to_numeric(df[metric_col], errors="coerce")
                            x = df.index
                            ax.plot(x, y, label=shown, linewidth=3.0, color=color, linestyle=linestyle)
                            plotted += 1

                        if plotted > 0:
                            if y_label:
                                ax.set_ylabel(y_label)
                            else:
                                ax.set_ylabel(metric_col)

                            if fill_below_zero:
                                for line in ax.get_lines():
                                    x = line.get_xdata()
                                    y = line.get_ydata()
                                    ax.fill_between(x, 0, y, alpha=0.18, color=line.get_color())

                            # 메인 타이틀: 요청 형식 고정
                            if title_text is None:
                                if metric_col.lower().startswith("cumulative_return"):
                                    title_text = "Cumulative return"
                                elif "drawdown" in metric_col.lower():
                                    title_text = "Drawdown"
                                else:
                                    title_text = metric_col.replace("_", " ").capitalize()
                            ax.set_title(f"{title_text} : {t}")

                            ax.set_xlabel("Date")
                            ax.grid(True, alpha=0.3)
                            ax.legend(ncol=2)
                            plt.tight_layout()
                            plt.savefig(out_path)
                            plt.close()
                        else:
                            plt.close()

                    # (3) 누적 수익률 — 실현
                    plot_compare('with_baselines', "Cumulative_Return",
                                os.path.join(out_crc_base, "with_baselines", f"{t}.png"),
                                fill_below_zero=False, title_text="Cumulative return", y_label="Cumulative return (%)")
                    plot_compare('with_ablations', "Cumulative_Return",
                                os.path.join(out_crc_base, "with_ablations", f"{t}.png"),
                                fill_below_zero=False, title_text="Cumulative return", y_label="Cumulative return (%)")

                    # (4) 드로우다운 — 실현
                    plot_compare('with_baselines', "Drawdown_rate",
                                os.path.join(out_dd_base, "with_baselines", f"{t}.png"),
                                fill_below_zero=True, title_text="Drawdown", y_label="Drawdown rate (%)")
                    plot_compare('with_ablations', "Drawdown_rate",
                                os.path.join(out_dd_base, "with_ablations", f"{t}.png"),
                                fill_below_zero=True, title_text="Drawdown", y_label="Drawdown rate (%)")

                    # (5) 누적 수익률 — 미실현 포함
                    plot_compare('with_baselines', "Cumulative_Return_Including_Unrealized",
                                os.path.join(out_crcU_base, "with_baselines", f"{t}.png"),
                                fill_below_zero=False, title_text="Cumulative return", y_label="Cumulative return (%)")
                    plot_compare('with_ablations', "Cumulative_Return_Including_Unrealized",
                                os.path.join(out_crcU_base, "with_ablations", f"{t}.png"),
                                fill_below_zero=False, title_text="Cumulative return", y_label="Cumulative return (%)")

                    # (6) 드로우다운 — 미실현 포함
                    plot_compare('with_baselines', "Drawdown_rate_unrealized",
                                os.path.join(out_ddU_base, "with_baselines", f"{t}.png"),
                                fill_below_zero=True, title_text="Drawdown", y_label="Drawdown rate (%)")
                    plot_compare('with_ablations', "Drawdown_rate_unrealized",
                                os.path.join(out_ddU_base, "with_ablations", f"{t}.png"),
                                fill_below_zero=True, title_text="Drawdown", y_label="Drawdown rate (%)")
