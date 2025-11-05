import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Dict, List, Union, Optional
import time


class SignalVisualizer:
    def __init__(self, update_interval: float = 0.1, max_history: int = 1000):
        """
        Initialize the signal visualizer.
        
        Args:
            update_interval: Minimum time between updates (seconds)
            max_history: Maximum number of historical values to keep per signal
        """
        self.update_interval = update_interval
        self.max_history = max_history
        self.last_update_time = 0
        self.fig = None
        self.axes = None
        
        # Storage for different signal types
        self.scalar_signals: Dict[str, List[float]] = {}
        self.vector_signals: Dict[str, List[np.ndarray]] = {}
        self.grid_signals: Dict[str, List[np.ndarray]] = {}
        
        self.update_count = 0
        self._setup_plots()
    
    def _setup_plots(self):
        """Set up the matplotlib figure and subplots."""
        try:
            plt.ion()  # Turn on interactive mode
        except Exception:
            pass  # If interactive mode fails, continue anyway
        
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Game Signals Visualization', fontsize=16, fontweight='bold')
        
        # Create grid layout - we'll dynamically adjust based on signal types
        # Start with a flexible layout
        self.gs = gridspec.GridSpec(3, 3, figure=self.fig, hspace=0.4, wspace=0.3)
        
        # We'll create axes dynamically based on registered signals
        self.axes_dict: Dict[str, plt.Axes] = {}
        self.signal_layout: Dict[str, tuple] = {}  # Maps signal name to (row, col) position
        
        plt.show(block=False)
        plt.pause(0.1)
    
    def register_scalar_signal(self, name: str, initial_value: Optional[float] = None):
        """
        Register a scalar signal for visualization.
        
        Args:
            name: Unique name for the signal
            initial_value: Optional initial value
        """
        if name not in self.scalar_signals:
            self.scalar_signals[name] = []
            if initial_value is not None:
                self.scalar_signals[name].append(initial_value)
            self._create_axis_for_signal(name, 'scalar')
    
    def register_vector_signal(self, name: str, size: Optional[int] = None, initial_value: Optional[np.ndarray] = None):
        """
        Register a vector signal for visualization.
        
        Args:
            name: Unique name for the signal
            size: Expected size of the vector (for initialization)
            initial_value: Optional initial value
        """
        if name not in self.vector_signals:
            self.vector_signals[name] = []
            if initial_value is not None:
                self.vector_signals[name].append(initial_value.copy())
            self._create_axis_for_signal(name, 'vector')
    
    def register_grid_signal(self, name: str, shape: Optional[tuple] = None, initial_value: Optional[np.ndarray] = None):
        """
        Register a grid signal for visualization.
        
        Args:
            name: Unique name for the signal
            shape: Expected shape of the grid (height, width) for initialization
            initial_value: Optional initial value
        """
        if name not in self.grid_signals:
            self.grid_signals[name] = []
            if initial_value is not None:
                self.grid_signals[name].append(initial_value.copy())
            self._create_axis_for_signal(name, 'grid')
    
    def _create_axis_for_signal(self, name: str, signal_type: str):
        """Create an axis for a signal and assign it a position in the grid."""
        total_signals = len(self.scalar_signals) + len(self.vector_signals) + len(self.grid_signals)
        
        # Calculate grid dimensions
        n_signals = total_signals
        if n_signals == 0:
            n_signals = 1
        
        n_cols = min(3, int(np.ceil(np.sqrt(n_signals))))
        n_rows = int(np.ceil(n_signals / n_cols))
        
        # Recreate grid if needed
        if n_rows > 3 or n_cols > 3:
            self.gs = gridspec.GridSpec(n_rows, n_cols, figure=self.fig, hspace=0.4, wspace=0.3)
        
        # Assign position
        idx = len(self.axes_dict)
        row = idx // n_cols
        col = idx % n_cols
        
        self.signal_layout[name] = (row, col, signal_type)
        
        # Create axis
        ax = self.fig.add_subplot(self.gs[row, col])
        self.axes_dict[name] = ax
    
    def update_scalar(self, name: str, value: float):
        """
        Update a scalar signal value.
        
        Args:
            name: Signal name
            value: New scalar value
        """
        if name not in self.scalar_signals:
            self.register_scalar_signal(name, value)
            return
        
        self.scalar_signals[name].append(float(value))
        if len(self.scalar_signals[name]) > self.max_history:
            self.scalar_signals[name] = self.scalar_signals[name][-self.max_history:]
    
    def update_vector(self, name: str, value: Union[List, np.ndarray]):
        """
        Update a vector signal value.
        
        Args:
            name: Signal name
            value: New vector value (list or numpy array)
        """
        if name not in self.vector_signals:
            arr = np.array(value) if not isinstance(value, np.ndarray) else value
            self.register_vector_signal(name, size=len(arr), initial_value=arr)
            return
        
        arr = np.array(value) if not isinstance(value, np.ndarray) else value
        self.vector_signals[name].append(arr.copy())
        if len(self.vector_signals[name]) > self.max_history:
            self.vector_signals[name] = self.vector_signals[name][-self.max_history:]
    
    def update_grid(self, name: str, value: np.ndarray):
        """
        Update a grid signal value.
        
        Args:
            name: Signal name
            value: New grid value (2D numpy array)
        """
        if name not in self.grid_signals:
            self.register_grid_signal(name, shape=value.shape, initial_value=value)
            return
        
        self.grid_signals[name].append(value.copy())
        if len(self.grid_signals[name]) > self.max_history:
            self.grid_signals[name] = self.grid_signals[name][-self.max_history:]
    
    def _plot_scalar_signal(self, name: str, ax: plt.Axes):
        """Plot a scalar signal as a line graph over time."""
        ax.clear()
        
        if name not in self.scalar_signals or len(self.scalar_signals[name]) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title(name, fontsize=10, fontweight='bold')
            return
        
        values = self.scalar_signals[name]
        x = range(len(values))
        
        ax.plot(x, values, linewidth=2, color='blue')
        ax.set_title(f'{name}\nCurrent: {values[-1]:.4f}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_vector_signal(self, name: str, ax: plt.Axes):
        """Plot a vector signal as a bar chart."""
        ax.clear()
        
        if name not in self.vector_signals or len(self.vector_signals[name]) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title(name, fontsize=10, fontweight='bold')
            return
        
        latest = self.vector_signals[name][-1]
        
        # Convert boolean arrays to int for better visualization
        if latest.dtype == bool:
            latest = latest.astype(int)
        
        n = len(latest)
        x = range(n)
        
        colors = plt.cm.viridis(np.linspace(0, 1, n))
        bars = ax.bar(x, latest, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_title(f'{name}\nSize: {n}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Index', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars if space permits
        if n <= 20:
            for i, (bar, val) in enumerate(zip(bars, latest)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}' if isinstance(val, float) else f'{int(val)}',
                       ha='center', va='bottom', fontsize=7)
    
    def _plot_grid_signal(self, name: str, ax: plt.Axes):
        """Plot a grid signal as a heatmap."""
        ax.clear()
        
        if name not in self.grid_signals or len(self.grid_signals[name]) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title(name, fontsize=10, fontweight='bold')
            return
        
        latest = self.grid_signals[name][-1]
        
        # Handle 2D arrays
        if latest.ndim == 2:
            im = ax.imshow(latest, aspect='auto', cmap='viridis', interpolation='nearest')
            ax.set_title(f'{name}\nShape: {latest.shape}', fontsize=10, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            # If not 2D, try to reshape or show as 1D
            ax.text(0.5, 0.5, f'Invalid shape: {latest.shape}', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title(name, fontsize=10, fontweight='bold')
    
    def update(self, force: bool = False):
        """
        Update the visualization with current signal values.
        
        Args:
            force: Force update even if update_interval hasn't passed
        """
        current_time = time.time()
        if not force and (current_time - self.last_update_time) < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # Plot all registered signals
        for name, ax in self.axes_dict.items():
            if name in self.scalar_signals:
                self._plot_scalar_signal(name, ax)
            elif name in self.vector_signals:
                self._plot_vector_signal(name, ax)
            elif name in self.grid_signals:
                self._plot_grid_signal(name, ax)
        
        self.update_count += 1
        
        # Refresh
        plt.draw()
        plt.pause(0.01)
    
    def close(self):
        """Close the visualization window."""
        if self.fig is not None:
            plt.close(self.fig)
        plt.ioff()

