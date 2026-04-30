"""
Professional plotting tools for academic publications
符合 Nature/Science 风格的专业作图工具

Features:
- Publication-ready quality (300 DPI)
- Colorblind-friendly palettes
- Consistent styling across all plots
- LaTeX-style math rendering
- Optimized for both print and digital
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import rcParams
from typing import List, Tuple, Optional, Union, Dict
import seaborn as sns
import pandas as pd

# ============================================================================
# Global Style Configuration
# ============================================================================

def set_publication_style(font_size=12, font_family='sans-serif', use_latex=False):
    """
    设置出版级别的全局绘图风格

    Parameters:
    -----------
    font_size : int
        基础字体大小（默认 10pt，适合双栏论文）
    font_family : str
        字体家族 ('sans-serif', 'serif', 'monospace')
    use_latex : bool
        是否使用 LaTeX 渲染（需要安装 LaTeX）
    """
    # Color palette - colorblind-friendly
    colors = {
        'blue': '#0173B2',      # IBM Blue
        'orange': '#DE8F05',    # IBM Orange
        'green': '#029E73',     # IBM Green
        'red': '#CC3311',       # IBM Red
        'purple': '#949494',    # IBM Purple
        'brown': '#ECE133',     # IBM Yellow
        'pink': '#56B4E9',      # Sky Blue
        'gray': '#999999',      # Medium Gray
    }

    # Set seaborn style as base
    sns.set_style("whitegrid", {
        'axes.edgecolor': '0.2',
        'grid.color': '0.9',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
    })

    # Matplotlib rcParams
    rcParams['figure.dpi'] = 100  # Screen display
    rcParams['savefig.dpi'] = 300  # Print quality
    rcParams['figure.figsize'] = (6, 4)  # Default size

    # Font settings
    rcParams['font.size'] = font_size
    rcParams['font.family'] = font_family
    rcParams['axes.labelsize'] = font_size + 1
    rcParams['axes.titlesize'] = font_size + 2
    rcParams['xtick.labelsize'] = font_size - 1
    rcParams['ytick.labelsize'] = font_size - 1
    rcParams['legend.fontsize'] = font_size - 1

    # LaTeX rendering (optional)
    if use_latex:
        rcParams['text.usetex'] = True
        rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    # Line and marker settings
    rcParams['lines.linewidth'] = 1.5
    rcParams['lines.markersize'] = 6
    rcParams['patch.linewidth'] = 0.5

    # Axes settings
    rcParams['axes.linewidth'] = 0.8
    rcParams['axes.labelpad'] = 4
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False

    # Grid settings
    rcParams['grid.alpha'] = 0.3

    # Legend settings
    rcParams['legend.frameon'] = True
    rcParams['legend.framealpha'] = 0.9
    rcParams['legend.fancybox'] = False
    rcParams['legend.edgecolor'] = '0.8'

    # Save settings
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.05
    rcParams['savefig.transparent'] = False

    return colors

# Default colors
COLORS = set_publication_style()

# ============================================================================
# Color Palettes
# ============================================================================

def get_color_palette(n_colors=5, palette='colorblind'):
    """
    获取配色方案

    Parameters:
    -----------
    n_colors : int
        需要的颜色数量
    palette : str
        配色方案: 'colorblind' (默认), 'vibrant', 'muted', 'pastel'

    Returns:
    --------
    list : 颜色列表
    """
    palettes = {
        'colorblind': ['#0173B2', '#DE8F05', '#029E73', '#CC3311', '#949494', '#ECE133'],
        'vibrant': ['#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311', '#009988'],
        'muted': ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255'],
        'pastel': ['#BBCCEE', '#CCEEFF', '#CCDDAA', '#EEEEBB', '#FFCCCC', '#DDDDDD'],
        'nature': ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4'],
    }

    if palette not in palettes:
        palette = 'colorblind'

    colors = palettes[palette]

    # Repeat if needed
    while len(colors) < n_colors:
        colors.extend(colors)

    return colors[:n_colors]

# ============================================================================
# Enhanced Stacked Bar Plot (方差分解图)
# ============================================================================

def create_variance_decomposition_plot(
    x_vals_1: List[float],
    y_vals_1: List[float],
    x_vals_2: List[float],
    y_vals_2: List[float],
    titles: List[str],
    xticklabels: List[List[str]],
    xlabel: str = "Sample Configuration",
    ylabel: str = "Variance",
    figsize: Tuple[float, float] = (10, 4),
    colors: Tuple[str, str] = None,
    save_path: Optional[Union[str, List[str]]] = None,
    show_values: bool = False,
    add_total_line: bool = True,
    legend_loc: Optional[str] = None
):
    """
    创建专业的方差分解堆叠柱状图

    Parameters:
    -----------
    x_vals_1, y_vals_1 : List[float]
        第一组数据的 rectifier 方差和总方差
    x_vals_2, y_vals_2 : List[float]
        第二组数据的 rectifier 方差和总方差
    titles : List[str]
        两个子图的标题
    xticklabels : List[List[str]]
        x 轴标签
    xlabel, ylabel : str
        轴标签
    figsize : Tuple[float, float]
        图形尺寸
    colors : Tuple[str, str]
        两部分的颜色 (rectifier, unlabeled)
    save_path : str or List[str]
        保存路径
    show_values : bool
        是否在柱子上显示数值
    add_total_line : bool
        是否添加总方差的线图
    legend_loc : str or None
        图例位置 ('best', 'upper left', 'upper right', etc.)
        如果为 None，自动选择最佳位置（默认）
    """
    if colors is None:
        colors = (COLORS['blue'], COLORS['orange'])

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for idx, (ax, x_vals, y_vals, title, xticks) in enumerate([
        (axes[0], x_vals_1, y_vals_1, titles[0], xticklabels[0]),
        (axes[1], x_vals_2, y_vals_2, titles[1], xticklabels[1])
    ]):
        x_positions = np.arange(len(x_vals))
        bar_width = 0.6

        # Calculate unlabeled contribution
        unlabeled_vals = [y - x for x, y in zip(x_vals, y_vals)]

        # Plot stacked bars
        bars1 = ax.bar(x_positions, x_vals, bar_width,
                      label='Rectifier Variance',
                      color=colors[0], alpha=0.8, edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x_positions, unlabeled_vals, bar_width,
                      bottom=x_vals, label='Unlabeled Variance',
                      color=colors[1], alpha=0.8, edgecolor='white', linewidth=0.5)

        # Add total variance line
        if add_total_line:
            ax.plot(x_positions, y_vals, 'o-', color='black',
                   linewidth=2, markersize=5, label='Total', zorder=10)

        # Add values on bars
        if show_values:
            for i, (x, y, u) in enumerate(zip(x_vals, y_vals, unlabeled_vals)):
                # Rectifier value
                ax.text(i, x/2, f'{x:.2f}', ha='center', va='center',
                       fontsize=8, color='white', fontweight='bold')
                # Unlabeled value
                if u > y * 0.1:  # Only show if large enough
                    ax.text(i, x + u/2, f'{u:.2f}', ha='center', va='center',
                           fontsize=8, color='white', fontweight='bold')

        # Styling
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(xticks, rotation=0)

        # Grid
        ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # Legend
        if idx == 0:
            loc = legend_loc if legend_loc is not None else 'best'
            ax.legend(loc=loc, frameon=True, framealpha=0.9,
                     edgecolor='0.8', fontsize=11)

    plt.tight_layout()

    # Save
    if save_path is not None:
        _save_figure(fig, save_path)

    return fig, axes

# ============================================================================
# Enhanced Multi-criteria Comparison Plot
# ============================================================================

def plot_method_comparison(
    dfs: List,
    methods: List[str],
    criteria: List[str],
    parameters: List[str],
    figsize: Tuple[float, float] = (7, 5),
    colors: Optional[List[str]] = None,
    save_path: Optional[Union[str, List[str]]] = None,
    log_scale: Optional[List[bool]] = None,
    baseline_method: Optional[str] = None,
    show_reduction: bool = False,
    legend_loc: Optional[str] = None,
    criterion_rename: Optional[Dict[str, str]] = None
):
    """
    创建多指标方法对比图（改进版）

    Parameters:
    -----------
    dfs : List
        DataFrame 列表，每个对应一个参数配置
    methods : List[str]
        方法名称列表
    criteria : List[str]
        评价指标列表
    parameters : List[str]
        参数名称列表（x轴）
    figsize : Tuple
        图形尺寸
    colors : List[str]
        方法对应的颜色
    save_path : str or List[str]
        保存路径
    log_scale : List[bool]
        每个指标是否使用对数坐标
    baseline_method : str
        基准方法（用于计算相对改进）
    show_reduction : bool
        是否显示相对于 baseline 的改进
    legend_loc : str or None
        图例位置 ('best', 'upper left', 'upper right', etc.)
        如果为 None，自动选择最佳位置（默认）
    """
    if colors is None:
        colors = get_color_palette(len(methods), 'colorblind')

    if log_scale is None:
        log_scale = [False] * len(criteria)

    # Store data
    data_dict = dict(zip(parameters, dfs))

    # Determine layout
    n_criteria = len(criteria)
    if n_criteria <= 3:
        nrows, ncols = 1, n_criteria
        figsize = (figsize[0] * n_criteria * 0.8, figsize[1])
    elif n_criteria == 4:
        nrows, ncols = 2, 2
        figsize = (figsize[0] * 2, figsize[1] * 2)
    elif n_criteria <= 6:
        nrows, ncols = 2, 3
        figsize = (figsize[0] * 1.5, figsize[1] * 1.2)
    else:
        nrows = int(np.ceil(n_criteria / 3))
        ncols = 3
        figsize = (figsize[0] * 1.5, figsize[1] * nrows * 0.6)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_criteria == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot each criterion
    for i, criterion in enumerate(criteria):
        ax = axes[i]
        x = np.arange(len(data_dict))
        bar_width = 0.8 / len(methods)

        # Calculate positions
        offsets = np.linspace(-(len(methods)-1)*bar_width/2,
                             (len(methods)-1)*bar_width/2,
                             len(methods))

        # Plot bars for each method
        for j, (method, offset, color) in enumerate(zip(methods, offsets, colors)):
            values = [df.loc[method, criterion] for df in data_dict.values()]

            # Calculate reduction if baseline specified
            if show_reduction and baseline_method and method != baseline_method:
                baseline_values = [df.loc[baseline_method, criterion]
                                  for df in data_dict.values()]
                values = [b/v if v > 0 else 0 for b, v in zip(baseline_values, values)]

            bars = ax.bar(x + offset, values, bar_width,
                         label=method, color=color, alpha=0.85,
                         edgecolor='white', linewidth=0.5)

            # Add value labels on top of bars (optional)
            # for k, (pos, val) in enumerate(zip(x + offset, values)):
            #     if val > 0:
            #         ax.text(pos, val, f'{val:.2f}', ha='center',
            #                va='bottom', fontsize=7, rotation=0)

        # Styling
        if criterion_rename and criterion in criterion_rename:
            criterion_title = criterion_rename[criterion]
        else:
            criterion_title = criterion.replace('_', ' ').replace('Est ', 'Est. ').replace('Err ', 'Error ').replace('Var', 'Variance').title()
        ax.set_title(criterion_title, fontsize=13, fontweight='bold', pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(data_dict.keys(), fontsize=11)

        if show_reduction and baseline_method:
            ax.set_ylabel(f'Reduction vs {baseline_method}', fontsize=12)
            ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        else:
            ax.set_ylabel(criterion_title, fontsize=12)

        # Log scale if needed
        if log_scale[i]:
            ax.set_yscale('log')

        # Grid
        ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # Legend (only first subplot)
        if i == 0:
            loc = legend_loc if legend_loc is not None else 'best'
            ax.legend(loc=loc, frameon=True, framealpha=0.9,
                     edgecolor='0.8', fontsize=11, ncol=1)

    # Hide unused subplots
    for i in range(n_criteria, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    # Save
    if save_path is not None:
        _save_figure(fig, save_path)

    return fig, axes

def plot_method_comparison_one_parameter(
    df: pd.DataFrame,          # 修改：只接受一个 DataFrame
    methods: List[str],
    criteria: List[str],
    parameter_name: str,       # 修改：只接受一个参数名称
    figsize: Tuple[float, float] = (5, 5), # 修改：默认尺寸改为单个子图的基础尺寸
    colors: Optional[List[str]] = None,
    save_path: Optional[Union[str, List[str]]] = None,
    log_scale: Optional[List[bool]] = None,
    baseline_method: Optional[str] = None,
    show_reduction: bool = False,
    legend_loc: Optional[str] = None,
    criterion_rename: Optional[Dict[str, str]] = None
):
    """
    创建多指标方法对比图（单参数、单行布局版）

    Parameters:
    -----------
    df : DataFrame
        包含数据的单一 DataFrame
    methods : List[str]
        方法名称列表
    criteria : List[str]
        评价指标列表
    parameter_name : str
        当前参数配置的名称（将显示在x轴）
    figsize : Tuple
        单个子图的基础尺寸 (宽, 高)，总宽度会自动乘以指标数量
    ... (其余参数保持不变)
    """
    
    # 1. 内部适配：将单个输入封装为列表/字典，以复用原有逻辑
    # 这样可以保持后续的绘图循环代码完全不变
    data_dict = {parameter_name: df}
    parameters = [parameter_name] 

    if colors is None:
        # 假设 get_color_palette 在外部定义，或者你可以替换为默认颜色列表
        try:
            colors = get_color_palette(len(methods), 'colorblind')
        except NameError:
            # 如果外部函数不存在，提供一个默认色板
            import matplotlib.cm as cm
            colors = [cm.tab10(i) for i in range(len(methods))]

    if log_scale is None:
        log_scale = [False] * len(criteria)

    # 2. 布局逻辑修改：强制所有子图在一行
    n_criteria = len(criteria)
    nrows = 1
    ncols = n_criteria
    
    # 自动调整总画布大小：宽度 = 单图宽度 * 指标数量
    total_figsize = (figsize[0] * ncols, figsize[1])

    fig, axes = plt.subplots(nrows, ncols, figsize=total_figsize)
    
    # 确保 axes 始终是列表/数组形式，方便后续迭代
    if n_criteria == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 3. 绘图循环 (保持原有逻辑不变)
    for i, criterion in enumerate(criteria):
        ax = axes[i]
        x = np.arange(len(data_dict)) # 因为只有一个参数，这里 x=[0]
        bar_width = 0.8 / len(methods)

        # Calculate positions
        offsets = np.linspace(-(len(methods)-1)*bar_width/2,
                             (len(methods)-1)*bar_width/2,
                             len(methods))

        # Plot bars for each method
        for j, (method, offset, color) in enumerate(zip(methods, offsets, colors)):
            # 从字典中获取值
            values = [data_dict[p].loc[method, criterion] for p in parameters]

            # Calculate reduction if baseline specified
            if show_reduction and baseline_method and method != baseline_method:
                baseline_values = [data_dict[p].loc[baseline_method, criterion]
                                  for p in parameters]
                values = [b/v if v > 0 else 0 for b, v in zip(baseline_values, values)]

            bars = ax.bar(x + offset, values, bar_width,
                         label=method, color=color, alpha=0.85,
                         edgecolor='white', linewidth=0.5)

        # Styling
        if criterion_rename and criterion in criterion_rename:
            criterion_title = criterion_rename[criterion]
        else:
            criterion_title = criterion.replace('_', ' ').replace('Est ', 'Est. ').replace('Err ', 'Error ').replace('Var', 'Variance').title()
        ax.set_title(criterion_title, fontsize=13, fontweight='bold', pad=8)

        # 设置 X 轴
        ax.set_xticks(x)
        ax.set_xticklabels(parameters, fontsize=11) # 显示参数名称

        if show_reduction and baseline_method:
            ax.set_ylabel(f'Reduction vs {baseline_method}', fontsize=12)
            ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        else:
            ax.set_ylabel(criterion_title, fontsize=12)

        # Log scale if needed
        if log_scale[i]:
            ax.set_yscale('log')

        # Grid
        ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # Legend (only first subplot)
        if i == 0:
            loc = legend_loc if legend_loc is not None else 'best'
            ax.legend(loc=loc, frameon=True, framealpha=0.9,
                     edgecolor='0.8', fontsize=11, ncol=1)

    plt.tight_layout()

    # Save
    if save_path is not None:
        # 这里假设 _save_figure 是外部定义的辅助函数
        try:
            _save_figure(fig, save_path)
        except NameError:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes

# ============================================================================
# New: Line Plot for Trend Visualization
# ============================================================================

def plot_performance_trends(
    data_dict: Dict[str, Dict[str, List[float]]],
    x_values: List,
    xlabel: str = "Sample Size (n)",
    ylabel: str = "Variance",
    title: str = "Performance Trends",
    figsize: Tuple[float, float] = (8, 5),
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    log_scale_x: bool = False,
    log_scale_y: bool = False,
    add_ci: bool = False,
    ci_data: Optional[Dict] = None,
    save_path: Optional[Union[str, List[str]]] = None,
    legend_loc: Optional[str] = None
):
    """
    创建性能趋势线图

    Parameters:
    -----------
    data_dict : Dict[str, Dict[str, List[float]]]
        数据字典，格式: {method_name: {metric_name: [values]}}
    x_values : List
        x 轴数值
    xlabel, ylabel, title : str
        标签和标题
    figsize : Tuple
        图形尺寸
    colors : List[str]
        颜色列表
    markers : List[str]
        标记样式列表
    log_scale_x, log_scale_y : bool
        是否使用对数坐标
    add_ci : bool
        是否添加置信区间
    ci_data : Dict
        置信区间数据
    save_path : str or List[str]
        保存路径
    legend_loc : str or None
        图例位置 ('best', 'upper left', 'upper right', etc.)
        如果为 None，自动选择最佳位置（默认）
    """
    if colors is None:
        colors = get_color_palette(len(data_dict), 'colorblind')

    if markers is None:
        markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']

    fig, ax = plt.subplots(figsize=figsize)

    for i, (method, color, marker) in enumerate(zip(data_dict.keys(), colors, markers)):
        # Assume single metric for simplicity, or take first metric
        metric_data = list(data_dict[method].values())[0]

        ax.plot(x_values, metric_data, marker=marker, color=color,
               linewidth=2, markersize=7, label=method, alpha=0.85)

        # Add confidence intervals if provided
        if add_ci and ci_data and method in ci_data:
            lower = ci_data[method]['lower']
            upper = ci_data[method]['upper']
            ax.fill_between(x_values, lower, upper, color=color, alpha=0.15)

    # Styling
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    if log_scale_x:
        ax.set_xscale('log')
    if log_scale_y:
        ax.set_yscale('log')

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    loc = legend_loc if legend_loc is not None else 'best'
    ax.legend(loc=loc, frameon=True, framealpha=0.9,
             edgecolor='0.8', fontsize=11)

    plt.tight_layout()

    # Save
    if save_path is not None:
        _save_figure(fig, save_path)

    return fig, ax

# ============================================================================
# New: Heatmap for Multi-dimensional Comparison
# ============================================================================

def plot_performance_heatmap(
    data_matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    xlabel: str = "Configuration",
    ylabel: str = "Method",
    title: str = "Performance Heatmap",
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = 'RdYlGn_r',
    annot: bool = True,
    fmt: str = '.2f',
    save_path: Optional[Union[str, List[str]]] = None
):
    """
    创建性能热力图

    Parameters:
    -----------
    data_matrix : np.ndarray
        数据矩阵（行=方法，列=配置）
    row_labels : List[str]
        行标签（方法名）
    col_labels : List[str]
        列标签（配置名）
    xlabel, ylabel, title : str
        标签和标题
    figsize : Tuple
        图形尺寸
    cmap : str
        颜色映射
    annot : bool
        是否显示数值
    fmt : str
        数值格式
    save_path : str or List[str]
        保存路径
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(data_matrix, cmap=cmap, aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Value', rotation=270, labelpad=15, fontsize=10)

    # Add annotations
    if annot:
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, format(data_matrix[i, j], fmt),
                             ha="center", va="center", color="black", fontsize=9)

    # Labels
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()

    # Save
    if save_path is not None:
        _save_figure(fig, save_path)

    return fig, ax

# ============================================================================
# Utility Functions
# ============================================================================

def _save_figure(fig, save_path):
    """保存图形到文件"""
    if isinstance(save_path, str):
        save_path = [save_path]

    for path in save_path:
        fig.savefig(path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f'✓ Saved: {path}')

def add_subplot_labels(axes, labels=None, loc='upper left', fontsize=12):
    """
    为子图添加标签 (a), (b), (c) 等

    Parameters:
    -----------
    axes : array-like
        子图数组
    labels : List[str]
        标签列表（默认 a, b, c, ...）
    loc : str
        标签位置
    fontsize : int
        字体大小
    """
    if labels is None:
        labels = [f'({chr(97+i)})' for i in range(len(axes))]

    for ax, label in zip(axes, labels):
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
               fontsize=fontsize, fontweight='bold', va='top', ha='left')

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Variance decomposition
    print("Example 1: Variance Decomposition Plot")
    x1 = [1.0, 0.8, 0.6, 0.5]
    y1 = [1.2, 1.0, 0.8, 0.7]
    x2 = [1.0, 1.0, 1.0, 1.0]
    y2 = [1.5, 1.2, 1.0, 0.9]

    fig, axes = create_variance_decomposition_plot(
        x1, y1, x2, y2,
        titles=["Fixed n, Varying N", "Fixed N/n, Varying n"],
        xticklabels=[["200,5k", "200,10k", "200,15k", "200,20k"],
                    ["200,5k", "400,10k", "600,15k", "800,20k"]],
        show_values=False,
        add_total_line=True
    )
    plt.show()

    print("\n✓ Professional plotting tools loaded successfully!")
    print("  Use set_publication_style() to apply global settings")
    print("  Available functions:")
    print("    - create_variance_decomposition_plot()")
    print("    - plot_method_comparison()")
    print("    - plot_performance_trends()")
    print("    - plot_performance_heatmap()")
