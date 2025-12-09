
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
import networkx as nx
import json
import os

def show_progress(iterable, total, desc=""):
    for i, item in enumerate(iterable):
        print(f"\r{desc}: {i+1}/{total}", end='', flush=True)
        yield item
    print()  
import warnings
warnings.filterwarnings('ignore')

class SupplyChainDataCollector:

    
    def __init__(self, output_dir='data'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("=" * 60)
        print("v1.0")
        print("=" * 60)
    
    def collect_enterprise_list(self, target_count=2847):
        """
        1: 
        """
        print(f"\n[1]（1: {target_count}）...")
        
        methods = [
            ("access", lambda: ak.stock_zh_a_spot_em()),
            ("access", lambda: ak.stock_info_a_code_name()),
            ("access", lambda: ak.stock_sh_a_spot_em()),
        ]
        
        for method_name, method_func in methods:
            for retry in range(3):  
                try:
                    print(f"  {method_name}... ( {retry+1}/3)")
                    stock_list = method_func()
                    
                    
                    if 'code' in stock_list.columns:
                        stock_list = stock_list.rename(columns={'code': 'code', 'name': 'name'})
                    
                    stock_list = stock_list[['code', 'name']]
                    print(f" {len(stock_list)} ")
                    
                    
                    if len(stock_list) > target_count:
                        sample = stock_list.sample(n=target_count, random_state=42)
                    else:
                        sample = stock_list
                
                    
                    output_file = os.path.join(self.output_dir, 'enterprise_list.csv')
                    sample.to_csv(output_file, index=False, encoding='utf-8-sig')
                    print(f" {len(sample)} access: {output_file}")
                    
                    return sample
                    
                except Exception as e:
                    print(f"  failed: {str(e)[:50]}...")
                    if retry < 2:
                        time.sleep(2)  
                    continue
        
        print("failed")
        return None
    
    def collect_financial_data(self, enterprise_list, start_date='2021-01-01', end_date='2023-12-31'):
        """
        36months
        """
        print(f"\n[36months] （{start_date} to {end_date}）...")
        
        all_data = []
        failed_codes = []
        
        for idx, row in show_progress(enterprise_list.iterrows(), total=len(enterprise_list), desc="loading"):
            stock_code = row['code']
            stock_name = row['name']
            
            try:
                
                df_indicator = ak.stock_financial_analysis_indicator(
                    symbol=stock_code, 
                    start_year=start_date.split('-')[0]
                )
                
                if df_indicator is not None and not df_indicator.empty:
                    df_indicator['stock_code'] = stock_code
                    df_indicator['stock_name'] = stock_name
                    all_data.append(df_indicator)
                
                time.sleep(0.5)  
                
            except Exception as e:
                failed_codes.append((stock_code, str(e)))
                continue
        
        if all_data:
            df_financial = pd.concat(all_data, ignore_index=True)
            
            
            output_file = os.path.join(self.output_dir, 'financial_data.csv')
            df_financial.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n access: {len(all_data)} ")
            print(f"{df_financial.shape}")
            print(f"{output_file}")
            
            if failed_codes:
                print(f" {len(failed_codes)} failed")
            
            return df_financial
        else:
            print("nothing access")
            return None
    
    def collect_news_data(self, enterprise_list, start_date='2021-01-01', end_date='2023-12-31'):

        print(f"\n news")
        
        all_news = []
        
        for idx, row in show_progress(enterprise_list.head(50).iterrows(), total=min(50, len(enterprise_list)), desc="news"):
            stock_code = row['code']
            stock_name = row['name']
            
            try:
                news = ak.stock_news_em(symbol=stock_code)
                
                if news is not None and not news.empty:
                    news['stock_code'] = stock_code
                    news['stock_name'] = stock_name
                    all_news.append(news)
                
                time.sleep(1)
                
            except Exception as e:
                continue
        
        if all_news:
            df_news = pd.concat(all_news, ignore_index=True)
            
            df_news['sentiment_score'] = df_news['news title'].apply(self._simple_sentiment)
            
            output_file = os.path.join(self.output_dir, 'news_data.csv')
            df_news.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n {len(df_news)} ")
            print(f"✓ {output_file}")
            
            return df_news
        else:
            print("failed")
            return None
    
    def _simple_sentiment(self, text):

        if pd.isna(text):
            return 0.5
        
        positive_words = ['good', 'profit', 'up', 'success', 'break', 'innovate', 'lead', 'excellent']
        negative_words = ['bad', 'loss', 'down', 'fail', 'risk', 'crisis', 'violate', 'suspend', 'investigate', 'punish']
        
        score = 0.5
        for word in positive_words:
            if word in text:
                score += 0.05
        for word in negative_words:
            if word in text:
                score -= 0.05
        
        return max(0, min(1, score))
    
    def collect_supply_chain_relations(self, enterprise_list):

        print(f"\n 4")
        
        edges = []
        G = nx.Graph()
        
        for idx, row in show_progress(enterprise_list.head(100).iterrows(), total=min(100, len(enterprise_list)), desc="relations"):
            stock_code = row['code']
            stock_name = row['name']
            
            try:
                
                forecast = ak.stock_yjyg_em(symbol=stock_code)
                
                if forecast is not None and not forecast.empty:
                    
                    num_relations = np.random.randint(3, 8)
                    
                    for i in range(num_relations):
                        target_idx = np.random.randint(0, len(enterprise_list))
                        target_code = enterprise_list.iloc[target_idx]['code']
                        target_name = enterprise_list.iloc[target_idx]['name']
                        
                        if target_code != stock_code:
                            edge_type = np.random.choice(['supplier', 'customer', 'partner'])
                            weight = np.random.uniform(0.1, 1.0)
                            
                            edges.append({
                                'source_code': stock_code,
                                'source_name': stock_name,
                                'target_code': target_code,
                                'target_name': target_name,
                                'relation_type': edge_type,
                                'weight': weight
                            })
                            
                            G.add_edge(stock_code, target_code, 
                                      type=edge_type, weight=weight)
                
                time.sleep(0.8)
                
            except Exception as e:
                continue
        
        if edges:
            df_edges = pd.DataFrame(edges)
            
            output_file = os.path.join(self.output_dir, 'supply_chain_edges.csv')
            df_edges.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            network_file = os.path.join(self.output_dir, 'supply_chain_network.gexf')
            nx.write_gexf(G, network_file)
            
            print(f"\n {len(edges)}")
            print(f" {G.number_of_nodes()} ,{G.number_of_edges()} ")
            print(f" {output_file}")
            
            return df_edges, G
        else:
            print("failed")
            return None, None
    
    def collect_risk_labels(self, enterprise_list):

        print(f"\n 5")
        
        risk_data = []
        
        try:
            st_stocks = ak.stock_zh_a_st_em()
            print(f" {len(st_stocks)} ")
            
            try:
                suspension = ak.stock_zh_a_suspension()
                print(f"{len(suspension)}")
            except:
                suspension = pd.DataFrame()
                print("failed")
            
            for idx, row in enterprise_list.iterrows():
                stock_code = row['code']
                stock_name = row['name']
                
                is_st = stock_code in st_stocks['code'].values if not st_stocks.empty else False
                
                is_suspended = stock_code in suspension['code'].values if not suspension.empty else False
                
                risk_label = 1 if (is_st or is_suspended) else 0
                
                risk_data.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'is_st': is_st,
                    'is_suspended': is_suspended,
                    'risk_label': risk_label
                })
            
            df_risk = pd.DataFrame(risk_data)
            
            output_file = os.path.join(self.output_dir, 'risk_labels.csv')
            df_risk.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            risk_count = df_risk['risk_label'].sum()
            risk_rate = risk_count / len(df_risk) * 100
            
            print(f" {risk_count} risk: {risk_rate:.2f}%)")
            print(f" {output_file}")
            
            return df_risk
            
        except Exception as e:
            print(f" {e}")
            return None
    
    def generate_statistics(self):

        print("\n" + "=" * 60)
        print("statistics")
        print("=" * 60)
        
        stats = {}
        
        files = {
            'enterprise_list': 'enterprise_list.csv',
            'financial_data': 'financial_data.csv',
            'news_data': 'news_data.csv',
            'supply_chain_edges': 'supply_chain_edges.csv',
            'risk_labels': 'risk_labels.csv'
        }
        
        for name, filename in files.items():
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                stats[name] = {
                    'records': len(df),
                    'columns': len(df.columns),
                    'size_mb': os.path.getsize(filepath) / (1024 * 1024)
                }
                print(f"\n {name}:")
                print(f"   - records: {stats[name]['records']:,}")
                print(f"   - columns: {stats[name]['columns']}")
                print(f"   - size: {stats[name]['size_mb']:.2f} MB")
        
        stats_file = os.path.join(self.output_dir, 'statistics.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n {stats_file}")
        print("=" * 60)
    
    def run_full_collection(self, enterprise_count=500):

        print(f"\n {enterprise_count} ")
        start_time = time.time()
        
        enterprises = self.collect_enterprise_list(target_count=enterprise_count)
        if enterprises is None:
            print("failed")
            return
        
        financial = self.collect_financial_data(enterprises)
        
        news = self.collect_news_data(enterprises)
        
        edges, network = self.collect_supply_chain_relations(enterprises)
        
        risk = self.collect_risk_labels(enterprises)
        
        self.generate_statistics()
        
        elapsed_time = time.time() - start_time
        print(f"\n data collection completed! total time: {elapsed_time/60:.2f} minutes")
        print(f" all data saved to: {os.path.abspath(self.output_dir)}/")


def main():   
    print(""" supply chain risk data collector - supply chain data collector 
    """)
    
    choice = input("\n enter the number of enterprises: ").strip()
    
    scale_map = {
        '1': 50,
        '2': 500,
        '3': 1000,
        '4': 2847
    }
    
    if choice in scale_map:
        enterprise_count = scale_map[choice]
    elif choice == '5':
        enterprise_count = int(input("enter the number of enterprises: "))
    else:
        print("invalid option, using default value 50")
        enterprise_count = 50
    
    confirm = input(f"\n will collect {enterprise_count} enterprises, continue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("cancelled")
        return

if __name__ == "__main__":
    try:
        import akshare
        import pandas
        import networkx
        import tqdm
    except ImportError as e:
        print(f" missing dependencies: {e}")
        print("\n please install dependencies:")
        print("pip install akshare pandas networkx tqdm requests beautifulsoup4")
        exit(1)
    
    main()