
import sys
import os

def check_dependencies():
    """check dependencies"""
    print("checking dependencies...")
    required = ['torch', 'sklearn', 'networkx', 'matplotlib', 'seaborn']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f" missing dependencies: {', '.join(missing)}")
        print(f"\n please install: pip install {' '.join(missing)}")
        return False
    
    print("✓ dependencies checked")
    return True

def check_data():
    print("\n checking data files...")
    required_files = [
        'data/enterprise_list.csv',
        'data/financial_data.csv',
        'data/news_data.csv',
        'data/supply_chain_network.gexf',
        'data/risk_labels.csv'
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print(f" missing data files:")  
        for f in missing:
            print(f"  - {f}")
        return False
    
    print(" data files complete")
    return True

def run_training():
    """run model training"""
    print("\n" + "="*60)
    print("step 1: train GNN-MSEF model")
    print("="*60)
    
    from gnn_msef_model import train_model
    model, results, history = train_model(epochs=100, lr=0.001)
    
    return results

def run_visualization():
    """generate visualizations"""
    print("\n" + "="*60)
    print("step 2: generate visualizations")
    print("="*60)
    
    from generate_visualizations import generate_all_visualizations
    generate_all_visualizations()

def print_summary(results):
    """print experiment summary"""
    print("\n" + "="*60)
    print("experiment summary")
    print("="*60)
    
    print(f"\n model performance:")
    print(f"  - Accuracy:  {results['accuracy']:.4f}")
    print(f"  - Precision: {results['precision']:.4f}")
    print(f"  - Recall:    {results['recall']:.4f}")
    print(f"  - F1-Score:  {results['f1_score']:.4f}")
    print(f"  - AUC-ROC:   {results['auc_roc']:.4f}")
    
    print(f"\n generated files:")
    print(f"  - gnn_msef_best.pth (model weights)")
    print(f"  - experiment_results.json (detailed results)")
    print(f"  - figure2_early_warning.pdf/png")
    print(f"  - figure3_feature_importance.pdf/png")
    print(f"  - figure4_network_visualization.pdf/png")
    print(f"  - figure5_attention_heatmap.pdf/png")
    print(f"  - table1_performance_comparison.csv")
    
    print(f"\n next step:")
    print(f"  1. check generated charts")
    print(f"  2. insert charts into paper")
    print(f"  3. reference data in table1_performance_comparison.csv")
    print(f"  4. write result analysis based on experiment_results.json")

def main():
    
    # check environment
    if not check_dependencies():
        return
    
    if not check_data():
        print("\n please ensure data collection has been run: python data_collector.py")
        return
    
    # confirm run
    print("\n experiment will include:")
    print("  1. train GNN-MSEF model (~2-5 minutes)")
    print("  2. generate visualizations (~1 minute)")
    
    response = input("\n whether to start experiment? (y/n): ").strip().lower()
    if response != 'y':
        print("cancelled")
        return
    
    try:
        # run experiment
        results = run_training()
        run_visualization()
        print_summary(results)
        
        print("\n" + "="*60)
        print(" experiment completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n experiment error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

