
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

class MultiSourceEncoder(nn.Module):
    """MultiSourceEncoder"""
    def __init__(self, financial_dim, news_dim, hidden_dim):
        super().__init__()
        
        # financial data encoder (CNN + LSTM)
        self.financial_conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.financial_lstm = nn.LSTM(financial_dim, hidden_dim, batch_first=True)
        
        # news sentiment encoder (MLP)
        self.news_encoder = nn.Sequential(
            nn.Linear(news_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # attention fusion
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, financial, news):
        # financial data encoder
        h_fin, _ = self.financial_lstm(financial.unsqueeze(1))
        h_fin = h_fin[:, -1, :]
        
        # news data encoder
        h_news = self.news_encoder(news)
        
        # attention fusion
        h_stack = torch.stack([h_fin, h_news], dim=1)  # [batch, 2, hidden]
        attn_weights = F.softmax(self.attention(h_stack).squeeze(-1), dim=1)  # [batch, 2]
        h_fused = (h_stack * attn_weights.unsqueeze(-1)).sum(dim=1)  # [batch, hidden]
        
        return h_fused, attn_weights

class GraphConvLayer(nn.Module):
    """GraphConvLayer"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, adj):
        # simplified graph convolution: H' = σ(AHW)
        # adj: [batch, batch] adjacency matrix
        # x: [batch, in_dim] feature matrix
        support = self.linear(x)
        output = torch.mm(adj, support)
        return F.relu(output)

class GNN_MSEF(nn.Module):
    """GNN-MSEF full model"""
    def __init__(self, financial_dim, news_dim, hidden_dim=128, num_gnn_layers=3):
        super().__init__()
        
        # multi-source data encoder
        self.encoder = MultiSourceEncoder(financial_dim, news_dim, hidden_dim)
        
        # graph neural network layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_gnn_layers)
        ])
        
        # temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # risk prediction head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # binary classification
        )
        
    def forward(self, financial, news, adj):
        # multi-source data fusion
        h_fused, attn_weights = self.encoder(financial, news)
        
        # graph neural network
        h_graph = h_fused
        for gnn in self.gnn_layers:
            h_graph = gnn(h_graph, adj)
        
        # feature concatenation
        h_final = torch.cat([h_fused, h_graph], dim=1)
        
        # risk prediction
        logits = self.classifier(h_final)
        
        return logits, attn_weights

def prepare_data(data_dir='data'):
    """prepare training data"""
    print("loading data...")
    
    # 1. load enterprise list
    df_ent = pd.read_csv(f'{data_dir}/enterprise_list.csv')
    print(f"enterprise count: {len(df_ent)}")
    
    # 2. load financial data
    df_fin = pd.read_csv(f'{data_dir}/financial_data.csv')
    
    # 3. load news data
    df_news = pd.read_csv(f'{data_dir}/news_data.csv')
    
    # 4. load supply chain network
    G = nx.read_gexf(f'{data_dir}/supply_chain_network.gexf')
    
    # 5. load risk labels
    df_risk = pd.read_csv(f'{data_dir}/risk_labels.csv')
    
    print("processing features...")
    
    # aggregate features for each enterprise
    features = []
    labels = []
    valid_codes = []
    
    for code in df_ent['code']:
        # financial features (latest data)
        fin_data = df_fin[df_fin['stock_code'] == code]
        if len(fin_data) == 0:
            continue
            
        # select key financial indicators
        fin_cols = ['diluted EPS (yuan)', 'ROE (%)', 'debt ratio (%)', 
                    'gross profit margin (%)', 'current ratio', 'quick ratio']
        fin_features = []
        for col in fin_cols:
            if col in fin_data.columns:
                val = fin_data[col].iloc[-1]  # latest data
                fin_features.append(val if pd.notna(val) else 0)
            else:
                fin_features.append(0)
        
        # news sentiment features
        news_data = df_news[df_news['stock_code'] == code]
        if len(news_data) > 0:
            news_sentiment = news_data['sentiment_score'].mean()
            news_count = len(news_data)
        else:
            news_sentiment = 0.5
            news_count = 0
        
        news_features = [news_sentiment, news_count]
        
        # merge features
        all_features = fin_features + news_features
        features.append(all_features)
        
        # label
        risk = df_risk[df_risk['stock_code'] == code]['risk_label'].iloc[0]
        labels.append(risk)
        valid_codes.append(code)
    
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    # standardization
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # build adjacency matrix
    n = len(valid_codes)
    code_to_idx = {code: i for i, code in enumerate(valid_codes)}
    adj_matrix = np.eye(n, dtype=np.float32)  # self-loop
    
    for edge in G.edges():
        if edge[0] in code_to_idx and edge[1] in code_to_idx:
            i, j = code_to_idx[edge[0]], code_to_idx[edge[1]]
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
    
    # normalize adjacency matrix
    D = np.diag(np.sum(adj_matrix, axis=1) ** -0.5)
    adj_norm = D @ adj_matrix @ D
    
    print(f"feature dimension: {features.shape}")
    print(f"label distribution: normal={np.sum(labels==0)}, risk={np.sum(labels==1)}")
    
    return features, labels, adj_norm, valid_codes

def train_model(epochs=100, lr=0.001):
    """train GNN-MSEF model"""
    print("="*60)
    print("start training GNN-MSEF model")
    print("="*60)
    
    # prepare data
    features, labels, adj_matrix, codes = prepare_data()
    
    # split training/test set
    indices = np.arange(len(features))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    
    # convert to tensor
    X = torch.FloatTensor(features)
    y = torch.LongTensor(labels)
    adj = torch.FloatTensor(adj_matrix)
    
    # split features
    financial = X[:, :6]  # first 6 financial features
    news = X[:, 6:]       # last 2 news features
    
    # model
    model = GNN_MSEF(financial_dim=6, news_dim=2, hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # weighted loss function (handle class imbalance)
    class_counts = np.bincount(labels)
    class_weights = torch.FloatTensor([1.0, max(class_counts) / class_counts[1]])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # train
    best_f1 = 0
    best_epoch = 0
    history = {'train_loss': [], 'test_acc': [], 'test_f1': []}
    
    # save initial model
    torch.save(model.state_dict(), 'gnn_msef_best.pth')
    
    for epoch in range(epochs):
        model.train()
        
        # forward propagation
        logits, attn = model(financial, news, adj)
        
        # calculate loss
        loss = criterion(logits[train_idx], y[train_idx])
        
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # evaluate
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = model(financial, news, adj)
                preds = logits.argmax(dim=1)
                
                # test set metrics
                test_preds = preds[test_idx].numpy()
                test_labels = y[test_idx].numpy()
                
                acc = accuracy_score(test_labels, test_preds)
                precision = precision_score(test_labels, test_preds, zero_division=0)
                recall = recall_score(test_labels, test_preds, zero_division=0)
                f1 = f1_score(test_labels, test_preds, zero_division=0)
                
                history['train_loss'].append(loss.item())
                history['test_acc'].append(acc)
                history['test_f1'].append(f1)
                
                print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | "
                      f"Acc: {acc:.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), 'gnn_msef_best.pth')
                    print(f"  → save best model (F1={f1:.4f})")
    
    print(f"\nbest model: Epoch {best_epoch}, F1={best_f1:.4f}")
    
    # final evaluation
    print("\n" + "="*60)
    print("final evaluation results")
    print("="*60)
    
    model.load_state_dict(torch.load('gnn_msef_best.pth'))
    model.eval()
    
    with torch.no_grad():
        logits, attn_weights = model(financial, news, adj)
        preds = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)[:, 1]
        
        # test set metrics
        test_preds = preds[test_idx].numpy()
        test_labels = y[test_idx].numpy()
        test_probs = probs[test_idx].numpy()
        
        acc = accuracy_score(test_labels, test_preds)
        precision = precision_score(test_labels, test_preds)
        recall = recall_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds)
        auc = roc_auc_score(test_labels, test_probs)
        
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
    
    # save results
    results = {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'history': history,
        'test_predictions': test_preds.tolist(),
        'test_labels': test_labels.tolist(),
        'attention_weights': attn_weights[test_idx].numpy().tolist()
    }
    
    import json
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n model saved: gnn_msef_best.pth")
    print(" results saved: experiment_results.json")
    
    return model, results, history

if __name__ == '__main__':
    # train model
    model, results, history = train_model(epochs=100, lr=0.001)

