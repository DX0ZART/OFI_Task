import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA

class generate_OFI_features():
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.levels = list(range(10))

    def preprocess_data(self):
        # Delete rows with action == "T"
        self.data = self.data[self.data['action'] != 'T']
        # Check duplicates
        self.data = self.data.drop_duplicates(subset=['ts_recv', 'symbol', 'action'], keep='last')
        
        ## Sort by time
        self.data = self.data.sort_values(by=['ts_recv'], ascending=True)
        
        ## Convert columns to numeric
        for lvl in self.levels:
            self.data[f'bid_px_{lvl:02d}'] = pd.to_numeric(self.data[f'bid_px_{lvl:02d}'], errors='coerce')
            self.data[f'bid_sz_{lvl:02d}'] = pd.to_numeric(self.data[f'bid_sz_{lvl:02d}'], errors='coerce')
            self.data[f'ask_px_{lvl:02d}'] = pd.to_numeric(self.data[f'ask_px_{lvl:02d}'], errors='coerce')
            self.data[f'ask_sz_{lvl:02d}'] = pd.to_numeric(self.data[f'ask_sz_{lvl:02d}'], errors='coerce')
        ## Drop NA
        self.data = self.data.dropna().reset_index(drop=True)

        self.cal_OF()

    def cal_OF(self):
        """
            Calculates OF(m, ) (i,n) 
        """
        for lvl in self.levels:
            px_bid = self.data[f'bid_px_{lvl:02d}']
            sz_bid = self.data[f'bid_sz_{lvl:02d}']
            px_ask = self.data[f'ask_px_{lvl:02d}']
            sz_ask = self.data[f'ask_sz_{lvl:02d}']

            px_bid_prev = px_bid.shift()
            sz_bid_prev = sz_bid.shift()
            px_ask_prev = px_ask.shift()
            sz_ask_prev = sz_ask.shift()

            # Bid Order Flow
            bid_of = np.where(
                px_bid > px_bid_prev, sz_bid,
                np.where(px_bid < px_bid_prev, -sz_bid, sz_bid - sz_bid_prev)
            )

            # Ask Order Flow
            ask_of = np.where(
                px_ask > px_ask_prev, -sz_ask,
                np.where(px_ask < px_ask_prev, sz_ask, sz_ask - sz_ask_prev)
            )

            self.data[f'OF_bid_level_{lvl}'] = bid_of
            self.data[f'OF_ask_level_{lvl}'] = ask_of

    def cal_Best_Level_OFI(self):
        """ 
            calculates the accumulative OFIs at the best bid/ask side 
            during a given time interval
            @Level 1
        """
        of_bid = self.data['OF_bid_level_0']
        of_ask = self.data['OF_ask_level_0']

        rolling_ofi = (of_bid - of_ask).rolling(window=self.h, min_periods=1).sum()
        self.data[f'OFI_Best_Level_{self.h}'] = rolling_ofi

        # Drop the first window of the rolling sum
        self.data = self.data.dropna().reset_index(drop=True)

    def cal_Multi_Level_OFI(self):    
        """ use the average size to scale OFIs at the corresponding levels 
        """
        for lvl in self.levels:
            ofi_raw = (self.data[f'OF_bid_level_{lvl}'] - self.data[f'OF_ask_level_{lvl}']).rolling(window=self.h, min_periods=1).sum()

            bid_sz = self.data[f'bid_sz_{lvl:02d}']
            ask_sz = self.data[f'ask_sz_{lvl:02d}']
            Q_M = ((bid_sz + ask_sz) / 2).rolling(window=self.h, min_periods=1).mean() / 10 # M =10

            self.data[f'OFI_Multi_Level_{lvl}'] = ofi_raw / Q_M.replace(0, np.nan) 

    def cal_Integrated_Level_OFI(self):
        """ there exist strong correlations between multi-level OFIs, 
            and that the first principal component can explain over 
            89% of the total variance among multi-level OFIs
        """
        ofi_cols = [f'OFI_Multi_Level_{i}' for i in self.levels]
        X = self.data[ofi_cols].dropna().values

        pca = PCA(n_components=1)
        PC1_proj = pca.fit_transform(X) 
        w1 = pca.components_[0]  # shape: (10,)
        w1_l1 = np.linalg.norm(w1, ord=1)
        w1_normalized = w1 / w1_l1

        full_X = self.data[ofi_cols].fillna(0).values
        integrated_ofi = np.dot(full_X, w1_normalized)

        self.data['OFI_Integrated_Level'] = integrated_ofi
        self.w1 = w1_normalized

    def cal_log_ret(self):
        """ calculate the log return of the asset""" 
        self.data['mid_price'] = (self.data['bid_px_00'] + self.data['ask_px_00']) / 2
        self.data[f'log_return_{self.h}'] = np.log(self.data['mid_price'] / self.data['mid_price'].shift(self.h))
        self.data = self.data.dropna().reset_index(drop=True)

    def cal_Cross_Best_OFI(self):
        ofi_df = self.data.pivot(index='ts_recv', columns='symbol', values=f'OFI_Best_Level_{self.h}')
        ret_df = self.data.pivot(index='ts_recv', columns='symbol', values=f'log_return_{self.h}')

        pred_df = pd.DataFrame(index=ofi_df.index, columns=ofi_df.columns)

        for ts in ofi_df.index:
            X = ofi_df.loc[ts]
            y = ret_df.loc[ts]

            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit()

            X_all = sm.add_constant(X.fillna(0))
            y_pred = model.predict(X_all)
            pred_df.loc[ts] = y_pred
            self.cross_best_df = pred_df

        return pred_df
    
    def cal_Cross_Integrated_OFI(self):
        ofi_df = self.data.pivot(index='ts_recv', columns='symbol', values=f'OFI_Integrated_Level')
        ret_df = self.data.pivot(index='ts_recv', columns='symbol', values=f'log_return_{self.h}')

        pred_df = pd.DataFrame(index=ofi_df.index, columns=ofi_df.columns)

        for ts in ofi_df.index:
            X = ofi_df.loc[ts]
            y = ret_df.loc[ts]

            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit()

            X_all = sm.add_constant(X.fillna(0))
            y_pred = model.predict(X_all)
            pred_df.loc[ts] = y_pred
            self.cross_integrated_df = pred_df

        return pred_df


    def generate_features(self, h = 1):
        """ Generate 3 dataframes:
            1. self.data: contains the OFI features
            2. self.cross_best_df: contains the cross-sectional prediction of the best level OFI
            3. self.cross_integrated_df: contains the cross-sectional prediction of the integrated level OFI
        """
        self.h = h
        self.preprocess_data()
        self.cal_Best_Level_OFI()
        self.cal_Multi_Level_OFI()
        self.cal_Integrated_Level_OFI()
        self.cal_log_ret()
        self.cal_Cross_Best_OFI()
        self.cal_Cross_Integrated_OFI()

if __name__ == "__main__":

    data = pd.read_csv("first_25000_rows.csv")
    generator = generate_OFI_features(data)
    generator.generate_features(h=5)
    
    print(generator.data.head())
    print(generator.cross_best_df.head())
    print(generator.cross_integrated_df.head())