import matplotlib.pyplot as plt
import numpy as np

def create_two_subplots(x_vals_1, y_vals_1, x_vals_2, y_vals_2, titles, labels, xticklabels, figsize=(8, 3), save_path=None):
    """
    创建两个平行的子图，绘制堆叠柱状图，分别展示两组数据的下半部分和总高度。

    参数:
    - x_vals_1, y_vals_1: 第一组数据的下半部分和总高度
    - x_vals_2, y_vals_2: 第二组数据的下半部分和总高度
    - titles: 包含两个子图标题的列表
    - labels: 包含 x 轴和 y 轴标签的元组 (xlabel, ylabel)
    - figsize: 图形的尺寸
    """
    def draw_bar_subplot(ax, x_vals, y_vals, bar_width=0.4, title="Stacked Bar Chart", xticklabels=None):
        """
        绘制堆叠柱状图，包含下半部分和上半部分。

        参数:
        - ax: matplotlib 子图对象
        - x_vals: 下半部分的高度列表 (xi)
        - y_vals: 总高度列表 (yi)
        - bar_width: 柱子的宽度
        - title: 子图的标题
        """
        x_positions = np.arange(len(x_vals))  # 柱的 x 位置

        # 绘制下半部分
        ax.bar(
            x_positions,
            x_vals,
            color='skyblue',
            label='Variance of Rectifier',
            width=bar_width
        )

        # 绘制上半部分
        ax.bar(
            x_positions,
            [y - x for x, y in zip(x_vals, y_vals)],  # 上半部分高度
            bottom=x_vals,  # 从下半部分开始叠加
            color='orange',
            label='Variance of Unlabeled Data',
            width=bar_width
        )

        # 添加标题和标签
        ax.set_title(title)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.legend()

    # 创建两个平行子图
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 绘制子图
    draw_bar_subplot(axes[0], x_vals_1, y_vals_1, title=titles[0], xticklabels=xticklabels[0])
    draw_bar_subplot(axes[1], x_vals_2, y_vals_2, title=titles[1], xticklabels=xticklabels[1])

    # 调整布局并显示图形
    plt.tight_layout()
    if save_path is not None:
        if type(save_path) == str:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        elif type(save_path) == list:
            for path in save_path:
                plt.savefig(path, bbox_inches='tight', dpi=300)
        else:
            raise ValueError(f"save_path must be a string or a list of strings, but got {type(save_path)}")
    print(f'✓ Saved: {save_path}')
    plt.show()



def plot_subplots(dfs, methods, criteria, parameters, figsize=(8, 6), save_path=None):
    """
    绘制四个子图，每个子图对应一个 criterion，柱状图的颜色表示一种方法。

    参数：
    - dfs: list of Pandas DataFrame，包含方法（行）和指标（列）的数据。
    - methods: list，方法名称列表（行索引）。
    - criteria: list，指标名称列表（列名称）。
    - parameters: list，参数名称列表（如 ['Parameter 1', 'Parameter 2', 'Parameter 3']）。
    - figsize: tuple，图形大小。
    """
    # 将数据存储为字典
    dfs = dict(zip(parameters, dfs))

    if len(criteria) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))  # 创建 1x2 的子图布局
    elif len(criteria) == 3:
        fig, axes = plt.subplots(1, 3, figsize=(figsize[0]*3, figsize[1]))  # 创建 1x3 的子图布局
    elif len(criteria) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(figsize[0]*2, figsize[1]*2))  # 创建 2x2 的子图布局
    elif len(criteria) <= 6:
        fig, axes = plt.subplots(2, 3, figsize=(figsize[0]*3, figsize[1]*2))  # 创建 2x3 的子图布局
    else:
        raise ValueError(f"Number of criteria must be less than 7, but got {len(criteria)}")
    axes = axes.flatten()  # 将子图展平，方便索引

    # 遍历每个 criterion
    for i, criterion in enumerate(criteria):
        ax = axes[i]
        x = np.arange(len(dfs))  # 每个参数的位置
        bar_width = 0.15  # 每个柱子的宽度
        offsets = np.linspace(-bar_width * 2, bar_width * 2, len(methods))  # 不同方法的偏移量

        # 绘制每种方法的柱状图
        for offset, method in zip(offsets, methods):
            # 获取该方法在所有参数下的当前 criterion 的值
            method_values = [df.loc[method, criterion] for df in dfs.values()]
            ax.bar(x + offset, method_values, bar_width, label=method)

        # 设置子图标题和标签
        ax.set_title(f'Criterion: {criterion}')
        ax.set_xticks(x)
        ax.set_xticklabels(dfs.keys())  # 参数名称作为 x 轴标签
        ax.set_ylabel(criterion)

        # 添加图例到第一个子图
        if i == 0:
            ax.legend()

    # 调整布局并显示
    plt.tight_layout()
    
    if save_path is not None:
        if type(save_path) == str:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        elif type(save_path) == list:
            for path in save_path:
                plt.savefig(path, bbox_inches='tight', dpi=300)
        else:
            raise ValueError(f"save_path must be a string or a list of strings, but got {type(save_path)}")
    print(f'✓ Saved: {save_path}')
    plt.show()
    
# # 示例数据
# x_vals_1 = [1, 2, 3]  # 第一组数据的下半部分高度
# y_vals_1 = [4, 5, 6]  # 第一组数据的总高度

# x_vals_2 = [2, 1, 4]  # 第二组数据的下半部分高度
# y_vals_2 = [5, 3, 7]  # 第二组数据的总高度

# # 调用函数绘制两个子图
# create_two_subplots(
#     x_vals_1, y_vals_1,
#     x_vals_2, y_vals_2,
#     titles=["Fixed n, Varying N", "Fixed N/n, Varying n"],
#     labels=("Value of samples (n, N)", "Total Variance of PPI")
# )
