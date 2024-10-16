import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class CSVDataVisualizer:
    def __init__(self, file_path):
        # 初始化时读取 CSV 文件
        self.file_path = file_path
        self.data = None

    def load_data(self):
        # 加载 CSV 文件
        try:
            self.data = pd.read_csv(self.file_path)
            print("CSV 文件已成功加载。")
        except FileNotFoundError:
            print(f"未找到文件: {self.file_path}")

    def plot_distribution(self, column_name=None, show_percentage=False):
        # 如果指定列名，绘制该列数据分布，否则绘制所有数值列的分布
        if self.data is None:
            print("请先加载数据！")
            return

        if column_name:
            if column_name in self.data.columns:
                plt.figure(figsize=(10, 6))

                # 绘制直方图并计算频数
                counts, bins, patches = plt.hist(self.data[column_name], bins=15, edgecolor='black', alpha=0.7)

                if show_percentage:
                    # 计算百分比
                    total = len(self.data[column_name])
                    percentages = (counts / total) * 100

                    # 在每个柱形条上显示百分比，只显示大于0的百分比
                    for count, percentage, patch in zip(counts, percentages, patches):
                        height = patch.get_height()
                        if percentage > 0:  # 仅显示大于0的百分比
                            plt.text(patch.get_x() + patch.get_width() / 2, height + 1, f'{percentage:.2f}%',
                                     ha='center', va='bottom', fontsize=10)

                    plt.ylabel('nums')
                else:
                    plt.ylabel('频数')

                plt.title(f'{column_name}  distribution')
                plt.xlabel(column_name)
                plt.show()
            else:
                print(f"列 {column_name} 不存在于数据中。")
        else:
            # 绘制所有数值列的分布
            num_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
            if len(num_columns) == 0:
                print("没有找到数值列可以绘制分布图。")
                return

            self.data[num_columns].hist(bins=15, figsize=(15, 10), layout=(len(num_columns) // 3 + 1, 3))

            plt.tight_layout()
            plt.show()


# 使用示例
train_csv_dir = './COMP90086_2024_Project_train/train.csv'
visualizer = CSVDataVisualizer(train_csv_dir)
visualizer.load_data()
visualizer.plot_distribution(column_name="stable_height", show_percentage=True)  # 显示所有数值列的比例分布
