import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from pathlib import Path


plt.rcParams['figure.figsize'] = [12, 8]

def parse_filename(filename):
    """Extract model and GPU count from filename"""
    match = re.match(r'(\w+)(\d+)_(\d+)gpu', filename.replace('.txt', ''))
    if match:
        model = match.group(1) + match.group(2) 
        gpu_count = int(match.group(3))
        return model, gpu_count
    return None, None

def read_and_process_file(filepath):
    """Read and process a single data file"""
    filename = Path(filepath).name
    model, gpu_count = parse_filename(filename)
    
    # handling for DP file
    if 'ddp' in filename.lower():
        model = 'res152_ddp'
        gpu_count = 2
    
    if model is None:
        return None
    
    data = []
    current_config = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# m=') or line.startswith('# DDP') or line.startswith('#  m='):
     
                parts = line.split(',')
                if 'm=' in parts[0]:
                    m_value = int(parts[0].split('=')[1])
                else:
                    m_value = 0  # dp has not microbatching
                
                if 'global_batch=' in parts[1]:
                    global_batch = int(parts[1].split('=')[1])
                elif 'global_batch=' in line:
                    global_batch = int(line.split('global_batch=')[1].split(',')[0])
                else:
               
                    for part in parts:
                        if 'batch=' in part:
                            global_batch = int(part.split('=')[1])
                            break
                    else:
                        global_batch = 256  
                
                current_config = {'m': m_value, 'global_batch': global_batch}
            elif line and not line.startswith('#') and not line.startswith('epoch'):
                parts = line.split(',')
                if len(parts) == 5:
                    try:
                        epoch_data = {
                            'model': model,
                            'gpu_count': gpu_count,
                            'm': current_config.get('m'),
                            'global_batch': current_config.get('global_batch'),
                            'epoch': int(parts[0]),
                            'train_loss': float(parts[1]),
                            'train_acc': float(parts[2]),
                            'throughput': float(parts[3]),
                            'epoch_time': float(parts[4])
                        }
                        data.append(epoch_data)
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse line: {line}")
                        continue
    
    return data

def create_throughput_vs_batch_plot(all_data):
    """Create throughput vs batch size plot without res152"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # filter out none values and exclude res152
    valid_data = [d for d in all_data if d.get('global_batch') is not None and d.get('m') is not None 
                 and 'res152' not in d['model']]
    
    # get average throughput for each configuration
    config_data = {}
    for entry in valid_data:
        key = (entry['model'], entry['gpu_count'], entry['m'], entry['global_batch'])
        if key not in config_data:
            config_data[key] = []
        config_data[key].append(entry['throughput'])
    
    # batch sizes for proper ordering
    batch_sizes = [128, 256, 512, 1024, 2048]
    
    # plot for m=4
    colors = {'res18': 'blue', 'res34': 'green', 'res50': 'red'}
    markers = {2: 'o', 4: 's'}
    linestyles = {'res18': '-', 'res34': '--', 'res50': '-.'}
    
    plotted_labels = set()
    
    for model in ['res18', 'res34', 'res50']:
        for gpu_count in [2, 4]:
            throughputs_by_batch = []
            valid_batches = []
            for batch in batch_sizes:
                key = (model, gpu_count, 4, batch)
                if key in config_data:
                    avg_throughput = np.mean(config_data[key])
                    throughputs_by_batch.append(avg_throughput)
                    valid_batches.append(batch)
            
            if throughputs_by_batch:
                color = colors.get(model, 'black')
                marker = markers.get(gpu_count, 'o')
                linestyle = linestyles.get(model, '-')
                label = f"{model} {gpu_count}GPU"
                
                if label not in plotted_labels:
                    ax1.plot(valid_batches, throughputs_by_batch, marker=marker, color=color, 
                            markersize=8, label=label, linestyle=linestyle, linewidth=2)
                    plotted_labels.add(label)
                else:
                    ax1.plot(valid_batches, throughputs_by_batch, marker=marker, color=color, 
                            markersize=8, linestyle=linestyle, linewidth=2)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Global Batch Size', fontsize=12)
    ax1.set_ylabel('Throughput (images/sec)', fontsize=12)
    ax1.set_title('Throughput vs Batch Size (m=4)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_xticks(batch_sizes)
    ax1.set_xticklabels(['128', '256', '512', '1024', '2048'])
    
    # plot for m=8
    plotted_labels = set()
    for model in ['res18', 'res34', 'res50']:
        for gpu_count in [2, 4]:
            throughputs_by_batch = []
            valid_batches = []
            for batch in batch_sizes:
                key = (model, gpu_count, 8, batch)
                if key in config_data:
                    avg_throughput = np.mean(config_data[key])
                    throughputs_by_batch.append(avg_throughput)
                    valid_batches.append(batch)
            
            if throughputs_by_batch:
                color = colors.get(model, 'black')
                marker = markers.get(gpu_count, 'o')
                linestyle = linestyles.get(model, '-')
                label = f"{model} {gpu_count}GPU"
                
                if label not in plotted_labels:
                    ax2.plot(valid_batches, throughputs_by_batch, marker=marker, color=color, 
                            markersize=8, label=label, linestyle=linestyle, linewidth=2)
                    plotted_labels.add(label)
                else:
                    ax2.plot(valid_batches, throughputs_by_batch, marker=marker, color=color, 
                            markersize=8, linestyle=linestyle, linewidth=2)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Global Batch Size', fontsize=12)
    ax2.set_ylabel('Throughput (images/sec)', fontsize=12)
    ax2.set_title('Throughput vs Batch Size (m=8)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_xticks(batch_sizes)
    ax2.set_xticklabels(['128', '256', '512', '1024', '2048'])
    
    plt.tight_layout()
    plt.savefig('throughput_vs_batch.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_accuracy_progression_plot(all_data):
    """Create accuracy progression plots with DDP vs Pipeline comparison as 4th graph"""
    # filter out None values
    valid_data = [d for d in all_data if d.get('global_batch') is not None and d.get('m') is not None]
    
    # search best configuration for each model 
    best_configs = {}
    
    for entry in valid_data:
        target_epoch = 80 if entry['model'] in ['res152', 'res152_ddp'] and entry['epoch'] == 80 else 15
        if entry['epoch'] == target_epoch:
            key = (entry['model'], entry['gpu_count'], entry['m'], entry['global_batch'])
            if key not in best_configs or entry['train_acc'] > best_configs[key]['accuracy']:
                best_configs[key] = {
                    'accuracy': entry['train_acc'],
                    'config': key
                }
    
    top_configs = {}
    for config_info in best_configs.values():
        model = config_info['config'][0]
        if model != 'res152_ddp':  # exclude DP from first 3 plots
            if model not in top_configs or config_info['accuracy'] > top_configs[model]['accuracy']:
                top_configs[model] = config_info
    
    # accuracy progression for top configurations
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(top_configs) + 2))
    
    # plot first 3 models only
    plot_idx = 0
    for idx, (model, config_info) in enumerate(top_configs.items()):
        if plot_idx >= 3:  
            break
            
        ax = axes[plot_idx]
        target_config = config_info['config']
        
        model_epochs = [entry for entry in valid_data 
                       if (entry['model'], entry['gpu_count'], entry['m'], entry['global_batch']) == target_config]
        
        epochs = sorted(list(set([e['epoch'] for e in model_epochs])))
        accuracies = [next(e['train_acc'] for e in model_epochs if e['epoch'] == epoch) for epoch in epochs]
        
        label = f"{model} (Batch: {target_config[3]})"
        
        ax.plot(epochs, accuracies, marker='o', linewidth=2, markersize=4, 
                color=colors[idx], label=label)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Training Accuracy', fontsize=10)
        ax.set_title(f'{model} - Best Config\n{target_config[1]} GPU, m={target_config[2]}, batch={target_config[3]}', 
                   fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plot_idx += 1
    
  
    ax = axes[3]
    
   
    res152_ppl_epochs = [entry for entry in valid_data 
                        if entry['model'] == 'res152' and entry['gpu_count'] == 2 
                        and entry['m'] == 4 and entry['global_batch'] == 256]
    

    res152_ddp_epochs = [entry for entry in valid_data 
                       if entry['model'] == 'res152_ddp' and entry['gpu_count'] == 2 
                       and entry['global_batch'] == 256]
    
    if res152_ppl_epochs and res152_ddp_epochs:
        # PP data
        ppl_epochs = sorted(list(set([e['epoch'] for e in res152_ppl_epochs])))
        ppl_acc = [next(e['train_acc'] for e in res152_ppl_epochs if e['epoch'] == epoch) 
                  for epoch in ppl_epochs]
        
        # DP data  
        ddp_epochs = sorted(list(set([e['epoch'] for e in res152_ddp_epochs])))
        ddp_acc = [next(e['train_acc'] for e in res152_ddp_epochs if e['epoch'] == epoch) 
                 for epoch in ddp_epochs]
        
        ax.plot(ppl_epochs, ppl_acc, marker='o', linewidth=2, markersize=4, 
               color='red', label='res152_ppl (m=4, batch=256)')
        ax.plot(ddp_epochs, ddp_acc, marker='s', linewidth=2, markersize=4, 
               color='blue', label='res152_ddp (batch=256)')
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Training Accuracy', fontsize=10)
        ax.set_title('ResNet152: DDP vs Pipeline Parallelism\n2 GPU, Batch=256', 
                   fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('accuracy_progression.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_scaling_efficiency_plot(all_data):
    """Create scaling efficiency plot without res152"""

    valid_data = [d for d in all_data if d.get('global_batch') is not None and d.get('m') is not None 
                 and 'res152' not in d['model']]
    
    efficiency_data = {}
    
    config_groups = {}
    for entry in valid_data:
        key = (entry['model'], entry['m'], entry['global_batch'])
        if key not in config_groups:
            config_groups[key] = {'2gpu': [], '4gpu': []}
        
        if entry['gpu_count'] == 2:
            config_groups[key]['2gpu'].append(entry['throughput'])
        elif entry['gpu_count'] == 4:
            config_groups[key]['4gpu'].append(entry['throughput'])
    
    #  calculate efficiency
    for config, throughputs in config_groups.items():
        if throughputs['2gpu'] and throughputs['4gpu']:
            avg_2gpu = np.mean(throughputs['2gpu'])
            avg_4gpu = np.mean(throughputs['4gpu'])
            efficiency = avg_4gpu / avg_2gpu if avg_2gpu > 0 else 0
            efficiency_data[config] = efficiency
    
    valid_efficiency_data = {k: v for k, v in efficiency_data.items() if k[2] is not None}
    
    if not valid_efficiency_data:
        print("No valid efficiency data to plot")
        return
    
    # plot efficiency
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = sorted(list(set([k[0] for k in valid_efficiency_data.keys()])))
    batch_sizes = sorted(list(set([k[2] for k in valid_efficiency_data.keys()])))
    
    x = np.arange(len(batch_sizes))
    width = 0.8 / len(models) 
    
    for i, model in enumerate(models):
        efficiencies = []
        for batch in batch_sizes:
            key = (model, 4, batch)
            efficiencies.append(valid_efficiency_data.get(key, 0))
        
        ax.bar(x + i*width, efficiencies, width, label=model, alpha=0.8)
    
    ax.set_xlabel('Global Batch Size', fontsize=12)
    ax.set_ylabel('Scaling Efficiency (4GPU / 2GPU)', fontsize=12)
    ax.set_title('Scaling Efficiency: 4GPU vs 2GPU Throughput (m=4)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models)-1)/2)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Ideal Scaling (2x)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('scaling_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_summary_table(all_data):
    """Create a summary table of best performances"""
    # Filter out None values
    valid_data = [d for d in all_data if d.get('global_batch') is not None and d.get('m') is not None]
    
    summary_data = []
    
    for entry in valid_data:

        target_epoch = 80 if entry['model'] in ['res152', 'res152_ddp'] and entry['epoch'] == 80 else 15
        if entry['epoch'] == target_epoch:
            summary_data.append(entry)
    
    df_summary = pd.DataFrame(summary_data)
    
    # group by configuration and get best accuracy
    best_per_model = df_summary.loc[df_summary.groupby(['model', 'gpu_count'])['train_acc'].idxmax()]
    
    print("Best Configurations per Model:")
    print("=" * 80)
    for _, row in best_per_model.iterrows():
        if row['model'] == 'res152_ddp':
            m_info = "DDP"
        else:
            m_info = f"m={row['m']}"
        print(f"{row['model']:10} | {row['gpu_count']} GPU | {m_info:4} | "
              f"Batch={row['global_batch']:4} | Acc: {row['train_acc']:.3f} | "
              f"Throughput: {row['throughput']:7.1f} img/s")
    
    return best_per_model

def main():
    files = [
        'res50_2gpu.txt',
        'res34_4gpu.txt', 
        'res18_2gpu.txt',
        'res18_4gpu.txt',
        'res152_2gpu.txt',
        'res34_2gpu.txt',
        'res152_4gpu.txt',
        'res50_4gpu.txt',
        'res152_2gpu_ddp.txt' 
    ]
    

    all_data = []
    for file in files:
        if os.path.exists(file):
            print(f"Processing {file}...")
            data = read_and_process_file(file)
            if data:
                all_data.extend(data)
                print(f"  Found {len(data)} data points")
        else:
            print(f"Warning: {file} not found")
    
    if not all_data:
        print("No data found! Please check file names and paths.")
        return
    
    print(f"Processed {len(all_data)} total data points")
    

    valid_data = [d for d in all_data if d.get('global_batch') is not None and d.get('m') is not None]
    print(f"Valid data points: {len(valid_data)}")
    
    if not valid_data:
        print("No valid data to plot!")
        return
    
   
    create_throughput_vs_batch_plot(valid_data)
    create_accuracy_progression_plot(valid_data)
    create_scaling_efficiency_plot(valid_data)
    

    best_configs = create_performance_summary_table(valid_data)
    

    print("\n" + "="*80)
    print("Key Insights:")
    print("1. Check 'throughput_vs_batch.png' for performance scaling (res18, res34, res50 only)")
    print("2. Check 'accuracy_progression.png' for convergence behavior with DDP vs Pipeline comparison") 
    print("3. Check 'scaling_efficiency.png' for GPU utilization efficiency (res18, res34, res50 only)")
    print("4. See summary table above for best configurations")

if __name__ == "__main__":
    main()
