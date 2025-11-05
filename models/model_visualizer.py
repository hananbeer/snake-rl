import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Optional, List
import time

class ModelVisualizer:
    def __init__(self, model: nn.Module, update_interval: float = 0.5):
        """
        Initialize the model visualizer.
        
        Args:
            model: The neural network model to visualize
            update_interval: Minimum time between updates (seconds)
        """
        self.model = model
        self.update_interval = update_interval
        self.last_update_time = 0
        self.fig = None
        self.axes = None
        self.weight_history = {'linear1': [], 'linear2': []}
        self.update_count = 0
        self._setup_plots()
    
    def _setup_plots(self):
        """Set up the matplotlib figure and subplots."""
        try:
            plt.ion()  # Turn on interactive mode
        except Exception:
            pass  # If interactive mode fails, continue anyway
        
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.suptitle('Neural Network Model Visualization', fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = gridspec.GridSpec(2, 4, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Weight heatmaps
        self.ax_linear1_weights = self.fig.add_subplot(gs[0, 0])
        self.ax_linear2_weights = self.fig.add_subplot(gs[0, 1])
        
        # Layer statistics
        self.ax_stats = self.fig.add_subplot(gs[0, 2:])
        
        # Weight statistics over time
        self.ax_weight_stats = self.fig.add_subplot(gs[1, :])
        
        self._plot_weight_history()
        plt.show(block=False)
        plt.pause(0.1)
    
    def _get_layer_weights(self, layer_name: str):
        """Extract weights from a specific layer."""
        if hasattr(self.model, layer_name):
            layer = getattr(self.model, layer_name)
            if isinstance(layer, nn.Linear):
                return layer.weight.data.cpu().numpy()
        return None
    
    def _plot_weight_heatmap(self, weights: np.ndarray, ax, title: str, max_size: int = 100):
        """Plot weight matrix as a heatmap."""
        ax.clear()
        
        # Downsample if too large for visualization
        if weights.shape[0] > max_size or weights.shape[1] > max_size:
            # Take a sample
            row_indices = np.linspace(0, weights.shape[0] - 1, min(max_size, weights.shape[0]), dtype=int)
            col_indices = np.linspace(0, weights.shape[1] - 1, min(max_size, weights.shape[1]), dtype=int)
            weights = weights[np.ix_(row_indices, col_indices)]
        
        im = ax.imshow(weights, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        ax.set_title(f'{title}\nShape: {weights.shape}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Input Neurons')
        ax.set_ylabel('Output Neurons')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_statistics(self, stats: dict):
        """Plot layer statistics table."""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # Create table
        table_data = []
        headers = ['Layer', 'Mean', 'Std', 'Min', 'Max', 'Params']
        
        for layer_name, layer_stats in stats.items():
            table_data.append([
                layer_name,
                f'{layer_stats["mean"]:.4f}',
                f'{layer_stats["std"]:.4f}',
                f'{layer_stats["min"]:.4f}',
                f'{layer_stats["max"]:.4f}',
                f'{layer_stats["params"]:,}'
            ])
        
        table = self.ax_stats.table(cellText=table_data, colLabels=headers,
                                   cellLoc='center', loc='center',
                                   colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        self.ax_stats.set_title('Layer Statistics', fontsize=12, fontweight='bold', pad=10)
    
    def _compute_statistics(self) -> dict:
        """Compute statistics for all layers."""
        stats = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.data.cpu().numpy()
                bias = module.bias.data.cpu().numpy() if module.bias is not None else None
                
                all_params = weights.flatten()
                if bias is not None:
                    all_params = np.concatenate([all_params, bias.flatten()])
                
                stats[name] = {
                    'mean': float(all_params.mean()),
                    'std': float(all_params.std()),
                    'min': float(all_params.min()),
                    'max': float(all_params.max()),
                    'params': int(all_params.size)
                }
        
        return stats
    
    def update(self, force: bool = False):
        """
        Update the visualization with current model weights.
        
        Args:
            force: Force update even if update_interval hasn't passed
        """
        current_time = time.time()
        if not force and (current_time - self.last_update_time) < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # Get weights
        linear1_weights = self._get_layer_weights('linear1')
        linear2_weights = self._get_layer_weights('linear2')
        
        # Plot weight heatmaps
        if linear1_weights is not None:
            self._plot_weight_heatmap(linear1_weights, self.ax_linear1_weights, 'Linear1 Weights')
        
        if linear2_weights is not None:
            self._plot_weight_heatmap(linear2_weights, self.ax_linear2_weights, 'Linear2 Weights')
        
        # Compute and plot statistics
        stats = self._compute_statistics()
        self._plot_statistics(stats)
        
        # Update weight history
        if linear1_weights is not None:
            self.weight_history['linear1'].append({
                'mean': float(linear1_weights.mean()),
                'std': float(linear1_weights.std()),
                'min': float(linear1_weights.min()),
                'max': float(linear1_weights.max())
            })
        if linear2_weights is not None:
            self.weight_history['linear2'].append({
                'mean': float(linear2_weights.mean()),
                'std': float(linear2_weights.std()),
                'min': float(linear2_weights.min()),
                'max': float(linear2_weights.max())
            })
        
        # Keep only last 100 updates for history
        for key in self.weight_history:
            if len(self.weight_history[key]) > 100:
                self.weight_history[key] = self.weight_history[key][-100:]
        
        # Plot weight history
        self._plot_weight_history()
        
        self.update_count += 1
        
        # Refresh
        plt.draw()
        plt.pause(0.01)
    
    def _plot_weight_history(self):
        """Plot weight statistics over time."""
        self.ax_weight_stats.clear()
        
        if len(self.weight_history['linear1']) == 0:
            self.ax_weight_stats.text(0.5, 0.5, 'No history yet', 
                                     ha='center', va='center', 
                                     transform=self.ax_weight_stats.transAxes,
                                     fontsize=12)
            self.ax_weight_stats.set_title('Weight Statistics Over Time', 
                                          fontsize=12, fontweight='bold')
            return
        
        x = range(len(self.weight_history['linear1']))
        
        # Plot mean and std for both layers
        linear1_means = [h['mean'] for h in self.weight_history['linear1']]
        linear1_stds = [h['std'] for h in self.weight_history['linear1']]
        linear2_means = [h['mean'] for h in self.weight_history['linear2']]
        linear2_stds = [h['std'] for h in self.weight_history['linear2']]
        
        self.ax_weight_stats.plot(x, linear1_means, label='Linear1 Mean', 
                                  color='blue', linewidth=2)
        self.ax_weight_stats.fill_between(x, 
                                         [m - s for m, s in zip(linear1_means, linear1_stds)],
                                         [m + s for m, s in zip(linear1_means, linear1_stds)],
                                         alpha=0.2, color='blue', label='Linear1 ±Std')
        
        self.ax_weight_stats.plot(x, linear2_means, label='Linear2 Mean', 
                                  color='red', linewidth=2)
        self.ax_weight_stats.fill_between(x,
                                         [m - s for m, s in zip(linear2_means, linear2_stds)],
                                         [m + s for m, s in zip(linear2_means, linear2_stds)],
                                         alpha=0.2, color='red', label='Linear2 ±Std')
        
        self.ax_weight_stats.set_xlabel('Update Number', fontsize=10)
        self.ax_weight_stats.set_ylabel('Weight Value', fontsize=10)
        self.ax_weight_stats.set_title('Weight Statistics Over Time', 
                                      fontsize=12, fontweight='bold')
        self.ax_weight_stats.legend(loc='best', fontsize=9)
        self.ax_weight_stats.grid(True, alpha=0.3)
    
    def close(self):
        """Close the visualization window."""
        plt.close(self.fig)
        plt.ioff()

