import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DataAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("文件搜索与数据分析")
        self.root.geometry("1200x800")
        self.root.option_add("*Font", "SimSun 12")
        # 默认参数
        self.sigma_multiplier = 3  # 三西格玛倍数
        self.lof_n_neighbors = 20  # LOF 邻居数
        self.dbscan_eps = 0.5  # DBSCAN eps
        self.dbscan_min_samples = 5  # DBSCAN min_samples
        self.autoencoder_epochs = 50  # Autoencoder 训练轮数
        self.autoencoder_batch_size = 32  # Autoencoder 批量大小

        # 数据相关
        self.selected_file_path = None
        self.detection_methods = []
        self.anomaly_results = {}
        self.anomaly_count = None  # 记录每个数据点的异常次数
        self.data = None  # 存储当前加载的数据
        self.selected_data_point = None  # 记录选中的数据点

        # GUI 组件
        self.create_widgets()

    def create_widgets(self):
        # 左侧面板（文件选择和结果显示）
        left_frame = tk.Frame(self.root)
        left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # 文件选择部分
        self.path_label = tk.Label(left_frame, text="搜索路径:")
        self.path_label.pack(pady=5)

        self.path_button = tk.Button(left_frame, text="选择文件夹", command=self.choose_folder)
        self.path_button.pack(pady=5)

        self.keyword_label = tk.Label(left_frame, text="关键词:")
        self.keyword_label.pack(pady=5)

        self.keyword_entry = tk.Entry(left_frame, width=50)
        self.keyword_entry.pack(pady=5)

        self.search_button = tk.Button(left_frame, text="搜索", command=self.search_files)
        self.search_button.pack(pady=10)

        # 结果展示区域
        self.result_frame = tk.Frame(left_frame)
        self.result_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.result_frame)
        self.v_scroll = tk.Scrollbar(self.result_frame, orient="vertical", command=self.canvas.yview)
        self.h_scroll = tk.Scrollbar(self.result_frame, orient="horizontal", command=self.canvas.xview)

        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.v_scroll.pack(side="right", fill="y")
        self.h_scroll.pack(side="bottom", fill="x")

        self.data_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.data_frame, anchor="nw")

        # 右侧控制面板
        right_frame = tk.Frame(self.root)
        right_frame.pack(side="right", fill="y", padx=10, pady=10)

        # 一维数据检测方法
        one_dim_frame = tk.LabelFrame(right_frame, text="一维数据检测方法")
        one_dim_frame.pack(pady=10, fill="x")

        self.method_vars = {
            "3sigma": tk.BooleanVar(),
            "zscore": tk.BooleanVar(),
            "iqr": tk.BooleanVar(),
            "lof": tk.BooleanVar(),
            "dbscan": tk.BooleanVar(),
            "autoencoder": tk.BooleanVar()
        }
        one_dim_methods = [
            ("三西格玛检测", "3sigma"),
            ("Z-Score检测", "zscore"),
            ("IQR检测", "iqr")
        ]

        for i, (text, key) in enumerate(one_dim_methods):
            frame = tk.Frame(one_dim_frame)
            frame.pack(pady=2)
            tk.Checkbutton(frame, text=text, variable=self.method_vars[key]).pack(side=tk.LEFT)
            tk.Button(frame, text="设置参数", command=lambda m=key: self.open_parameter_window(m)).pack(side=tk.LEFT)

        # 二维数据检测方法
        two_dim_frame = tk.LabelFrame(right_frame, text="二维数据检测方法")
        two_dim_frame.pack(pady=10, fill="x")
        two_dim_methods = [
            ("LOF算法", "lof"),
            ("DBSCAN算法", "dbscan"),
            ("Autoencoder检测", "autoencoder")
        ]

        for i, (text, key) in enumerate(two_dim_methods):
            frame = tk.Frame(two_dim_frame)
            frame.pack(pady=2)
            tk.Checkbutton(frame, text=text, variable=self.method_vars[key]).pack(side=tk.LEFT)
            tk.Button(frame, text="设置参数", command=lambda m=key: self.open_parameter_window(m)).pack(side=tk.LEFT)

        # 检测按钮
        self.detect_btn = tk.Button(right_frame, text="开始检测", command=self.run_detection)
        self.detect_btn.pack(pady=10)

        # 选中数据点信息显示区域
        self.selected_info_label = tk.Label(right_frame, text="选中数据点信息:", font=("Arial", 12, "bold"))
        self.selected_info_label.pack(pady=10)

        self.selected_info_text = tk.Text(right_frame, height=10, width=40, wrap="word")
        self.selected_info_text.pack(pady=5)

    def open_parameter_window(self, method):

        param_window = tk.Toplevel(self.root)
        param_window.title(f"{method} 参数设置")

        if method == "3sigma":
            tk.Label(param_window, text="三西格玛倍数:").grid(row=0, column=0, padx=10, pady=10)
            sigma_entry = tk.Entry(param_window)
            sigma_entry.grid(row=0, column=1, padx=10, pady=10)
            sigma_entry.insert(0, str(self.sigma_multiplier))

            tk.Button(
                param_window,
                text="保存",
                command=lambda: self.save_parameters(method, sigma_entry.get()),
            ).grid(row=1, column=0, columnspan=2, pady=10)

        elif method == "lof":
            tk.Label(param_window, text="邻居数 (n_neighbors):").grid(row=0, column=0, padx=10, pady=10)
            n_neighbors_entry = tk.Entry(param_window)
            n_neighbors_entry.grid(row=0, column=1, padx=10, pady=10)
            n_neighbors_entry.insert(0, str(self.lof_n_neighbors))

            tk.Button(
                param_window,
                text="保存",
                command=lambda: self.save_parameters(method, n_neighbors_entry.get()),
            ).grid(row=1, column=0, columnspan=2, pady=10)

        elif method == "dbscan":
            tk.Label(param_window, text="eps:").grid(row=0, column=0, padx=10, pady=10)
            eps_entry = tk.Entry(param_window)
            eps_entry.grid(row=0, column=1, padx=10, pady=10)
            eps_entry.insert(0, str(self.dbscan_eps))

            tk.Label(param_window, text="min_samples:").grid(row=1, column=0, padx=10, pady=10)
            min_samples_entry = tk.Entry(param_window)
            min_samples_entry.grid(row=1, column=1, padx=10, pady=10)
            min_samples_entry.insert(0, str(self.dbscan_min_samples))

            tk.Button(
                param_window,
                text="保存",
                command=lambda: self.save_parameters(method, eps_entry.get(), min_samples_entry.get()),
            ).grid(row=2, column=0, columnspan=2, pady=10)

        elif method == "autoencoder":
            tk.Label(param_window, text="训练轮数 (epochs):").grid(row=0, column=0, padx=10, pady=10)
            epochs_entry = tk.Entry(param_window)
            epochs_entry.grid(row=0, column=1, padx=10, pady=10)
            epochs_entry.insert(0, str(self.autoencoder_epochs))

            tk.Label(param_window, text="批量大小 (batch_size):").grid(row=1, column=0, padx=10, pady=10)
            batch_size_entry = tk.Entry(param_window)
            batch_size_entry.grid(row=1, column=1, padx=10, pady=10)
            batch_size_entry.insert(0, str(self.autoencoder_batch_size))

            tk.Button(
                param_window,
                text="保存",
                command=lambda: self.save_parameters(method, epochs_entry.get(), batch_size_entry.get()),
            ).grid(row=2, column=0, columnspan=2, pady=10)

    def save_parameters(self, method, *args):

        try:
            if method == "3sigma":
                self.sigma_multiplier = float(args[0])
            elif method == "lof":
                self.lof_n_neighbors = int(args[0])
            elif method == "dbscan":
                self.dbscan_eps = float(args[0])
                self.dbscan_min_samples = int(args[1])
            elif method == "autoencoder":
                self.autoencoder_epochs = int(args[0])
                self.autoencoder_batch_size = int(args[1])
            messagebox.showinfo("成功", "参数已保存！")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的参数值！")

    def choose_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.search_path = path
            self.path_label.config(text=f"当前路径: {path}")

    def search_files(self):
        if not hasattr(self, 'search_path') or not self.search_path:
            return

        keyword = self.keyword_entry.get().lower()
        supported_exts = ['.txt', '.csv', '.xls', '.xlsx', '.xlsm']


        for widget in self.data_frame.winfo_children():
            widget.destroy()

        files_found = []
        for root, dirs, files in os.walk(self.search_path):
            for file in files:
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in supported_exts):
                    if keyword in file_lower:
                        full_path = os.path.join(root, file)
                        files_found.append(full_path)


        self.display_files(files_found)

    def display_files(self, files):

        for widget in self.data_frame.winfo_children():
            widget.destroy()


        for idx, file in enumerate(files):
            btn = tk.Button(
                self.data_frame,
                text=file,
                command=lambda f=file: self.open_file(f),
                width=100,
                anchor='w'
            )
            btn.grid(row=idx, column=0, sticky='ew', padx=5, pady=2)

        # 更新滚动区域
        self.data_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def open_file(self, file_path):
        """打开文件并加载数据，支持 .txt (Tab分隔), .csv, .xls, .xlsx, .xlsm 文件类型"""
        try:
            # 根据文件类型选择读取方式
            if file_path.endswith('.txt'):
                # 读取 .txt 文件，明确指定 Tab 分隔符，并确保科学计数法被正确解析
                self.data = pd.read_csv(file_path, sep='\t', engine='python', header=0)
            elif file_path.endswith('.csv'):
                # 读取 .csv 文件
                self.data = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx', '.xlsm')):
                # 读取 Excel 文件
                self.data = pd.read_excel(file_path)
            else:
                return

            # 数据清洗：确保数值列被正确解析为浮点数
            for col in self.data.columns:
                try:
                    # 尝试将列转换为数值类型
                    self.data[col] = pd.to_numeric(self.data[col], errors='raise')
                except ValueError:
                    # 如果转换失败，保留原始数据（可能是非数值列）
                    pass

            # 删除全为 NaN 的列
            self.data.dropna(axis=1, how='all', inplace=True)

            # 初始化异常次数记录
            self.anomaly_count = pd.DataFrame(
                np.zeros(self.data.shape, dtype=int),
                columns=self.data.columns,
                index=self.data.index
            )

            # 保存选中的文件路径
            self.selected_file_path = file_path

            # 显示清洗后的数据
            self.display_data(self.data)

        except Exception as e:
            pass

    def display_data(self, data):
        for widget in self.data_frame.winfo_children():
            widget.destroy()
        transposed_data = data.transpose()
        row_count = len(transposed_data)
        col_count = len(transposed_data.columns)
        for col in range(col_count):
            col_label = tk.Label(self.data_frame, text=transposed_data.columns[col], font=("Arial", 12, "bold"))
            col_label.grid(row=0, column=col + 1, padx=5, pady=5)
        for row in range(row_count):
            row_label = tk.Label(self.data_frame, text=transposed_data.index[row], font=("Arial", 12, "bold"))
            row_label.grid(row=row + 1, column=0, padx=5, pady=5)
            current_col = transposed_data.iloc[row, :]
            min_col = current_col.min()
            max_col = current_col.max()
            for col in range(col_count):
                cell_value = transposed_data.iloc[row, col]
                if max_col != min_col:  # 避免除以零
                    normalized_value = (cell_value - min_col) / (max_col - min_col)
                else:
                    normalized_value = 0  # 如果最大值等于最小值，归一化为 0

                # 格式化为科学计数法
                formatted_value = f"{cell_value:.2e}"

                # 动态计算颜色
                if self.data is not None:
                    # 基础颜色（根据数据点的值）
                    base_color = self.get_color_based_on_value(normalized_value)

                    # 异常检测检查
                    if hasattr(self, 'anomaly_count') and self.anomaly_count is not None:
                        anomaly_level = self.anomaly_count.iloc[col, row]
                        if anomaly_level > 0:
                            # 如果有异常，显示异常颜色
                            final_color = self.get_color_based_on_anomaly_level(anomaly_level)
                        else:
                            # 如果无异常，显示基础颜色
                            final_color = base_color
                    else:
                        # 如果未启用异常检测，显示基础颜色
                        final_color = base_color
                else:
                    final_color = "white"  # 默认颜色

                # 创建数据点标签，并绑定点击事件
                cell_label = tk.Label(
                    self.data_frame,
                    text=formatted_value,
                    font=("Arial", 12),
                    bg=final_color
                )
                cell_label.grid(row=row + 1, column=col + 1, padx=5, pady=5)

                # 绑定点击事件
                cell_label.bind("<Button-1>", lambda e, r=row, c=col: self.select_data_point(r, c))

        # 更新滚动区域
        self.data_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def select_data_point(self, row, col):
        self.selected_data_point = (row, col)
        self.show_selected_data_point_info()

    def show_selected_data_point_info(self):
        if self.selected_data_point is None:
            return

        row, col = self.selected_data_point
        info = f"选中数据点: 行 {row + 1}, 列 {col + 1}\n\n"

        for method, result in self.anomaly_results.items():
            if isinstance(result, pd.DataFrame):
                if result.iloc[col, row]:
                    info += f"- {method} 检测到异常\n"
            elif isinstance(result, np.ndarray):
                if result.ndim == 1:
                    if result[col]:
                        info += f"- {method} 检测到异常\n"
                else:
                    if result[col, row]:
                        info += f"- {method} 检测到异常\n"
            else:
                if result[col]:
                    info += f"- {method} 检测到异常\n"
        if info == f"选中数据点: 行 {row + 1}, 列 {col + 1}\n\n":
            info += "- 未检测到异常\n"

        # 更新显示
        self.selected_info_text.config(state="normal")
        self.selected_info_text.delete(1.0, tk.END)
        self.selected_info_text.insert(tk.END, info)
        self.selected_info_text.config(state="disabled")

    def get_color_based_on_value(self, normalized_value):
        """根据归一化的值返回对应的颜色"""
        # 蓝色强度固定为 255
        blue_intensity = 255  # B: 255
        # 计算红色和绿色强度（0-255 区间）
        red_green_intensity = int(255 * (1 - normalized_value))  # RG: 255-0
        # 返回颜色值
        return f"#{red_green_intensity:02X}{red_green_intensity:02X}{blue_intensity:02X}"

    def get_color_based_on_anomaly_level(self, anomaly_level):
        """根据异常次数返回对应的颜色"""
        if anomaly_level == 0:
            return "#FFFFFF"  # 无异常，白色
        elif anomaly_level == 1:
            return "#FFCCCC"  # 浅红
        elif anomaly_level == 2:
            return "#FF9999"  # 稍深红
        elif anomaly_level == 3:
            return "#FF6666"  # 更深红
        elif anomaly_level == 4:
            return "#FF3333"  # 深红
        elif anomaly_level == 5:
            return "#FF0000"  # 红色
        else:
            return "#CC0000"  # 最深红（超过5次）

    def run_detection(self):
        if not self.selected_file_path:
            return

        if any([self.method_vars[key].get() for key in self.method_vars]):
            self.anomaly_count = pd.DataFrame(
                np.zeros(self.data.shape, dtype=int),
                columns=self.data.columns,
                index=self.data.index
            )
        else:
            self.anomaly_count = None

        numeric_data = self.data.select_dtypes(include=[np.number])

        self.anomaly_results = {}

        if self.method_vars["3sigma"].get():
            self.three_sigma_detection(numeric_data)
        if self.method_vars["zscore"].get():
            self.zscore_detection(numeric_data)
        if self.method_vars["iqr"].get():
            self.iqr_detection(numeric_data)

        if self.method_vars["lof"].get() or self.method_vars["dbscan"].get() or self.method_vars["autoencoder"].get():
            two_dim_data = numeric_data

            if self.method_vars["lof"].get():
                self.lof_detection(two_dim_data)
            if self.method_vars["dbscan"].get():
                self.dbscan_detection(two_dim_data)
            if self.method_vars["autoencoder"].get():
                self.autoencoder_detection(two_dim_data)
        self.display_data(self.data)
        self.show_anomaly_report()

    def three_sigma_detection(self, data):
        mean = data.mean()
        std = data.std()
        anomalies = (data < (mean - self.sigma_multiplier * std)) | (data > (mean + self.sigma_multiplier * std))
        self.anomaly_results["3sigma"] = anomalies
        self.anomaly_count += anomalies.astype(int)

    def zscore_detection(self, data):
        z_scores = (data - data.mean()) / data.std()
        anomalies = np.abs(z_scores) > self.sigma_multiplier
        self.anomaly_results["zscore"] = anomalies
        self.anomaly_count += anomalies.astype(int)

    def iqr_detection(self, data):
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        anomalies = (data < lower) | (data > upper)
        self.anomaly_results["iqr"] = anomalies
        self.anomaly_count += anomalies.astype(int)

    def lof_detection(self, data):
        if data.empty:
            return

        # 确保数据是二维数组
        if len(data.shape) == 1:
            data = data.values.reshape(-1, 1)
        else:
            data = data.values
        scaler = StandardScaler()
        X = scaler.fit_transform(data)
        lof = LocalOutlierFactor(n_neighbors=self.lof_n_neighbors)  # 参数可根据数据调整
        labels = lof.fit_predict(X)
        anomalies = labels == -1
        anomalies_2d = np.tile(anomalies, (data.shape[1], 1)).T
        anomalies_df = pd.DataFrame(anomalies_2d, columns=self.data.columns, index=self.data.index)
        self.anomaly_results["lof"] = anomalies_df
        self.anomaly_count += anomalies_df.astype(int)

    def dbscan_detection(self, data):
        if data.empty:
            return
        if len(data.shape) == 1:
            data = data.values.reshape(-1, 1)
        else:
            data = data.values
        scaler = StandardScaler()
        X = scaler.fit_transform(data)
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)  # 参数可根据数据调整
        labels = dbscan.fit_predict(X)
        anomalies = labels == -1
        anomalies_2d = np.tile(anomalies, (data.shape[1], 1)).T
        anomalies_df = pd.DataFrame(anomalies_2d, columns=self.data.columns, index=self.data.index)
        self.anomaly_results["dbscan"] = anomalies_df
        self.anomaly_count += anomalies_df.astype(int)

    def autoencoder_detection(self, data):

        if data.empty:
            return
        if len(data.shape) == 1:
            data = data.values.reshape(-1, 1)
        else:
            data = data.values
        scaler = StandardScaler()
        X = scaler.fit_transform(data)
        # 构建 Autoencoder 模型
        input_dim = X.shape[1]
        encoding_dim = max(1, input_dim // 2)  # 编码层维度为输入维度的一半

        model = Sequential([
            Dense(encoding_dim, activation="relu", input_shape=(input_dim,)),
            Dense(input_dim, activation="linear")
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

        # 训练模型
        model.fit(X, X, epochs=self.autoencoder_epochs, batch_size=self.autoencoder_batch_size, verbose=0)

        # 计算重构误差
        reconstructed = model.predict(X)
        mse = np.mean(np.power(X - reconstructed, 2), axis=1)

        # 标记异常点（重构误差大于阈值的点为异常点）
        threshold = np.percentile(mse, 95)  # 取 95% 分位数作为阈值
        anomalies = mse > threshold

        # 将结果转换为与数据形状一致的二维数组
        anomalies_2d = np.tile(anomalies, (data.shape[1], 1)).T

        # 将结果转换为 DataFrame，并保留原始数据的列名和索引
        anomalies_df = pd.DataFrame(anomalies_2d, columns=self.data.columns, index=self.data.index)
        self.anomaly_results["autoencoder"] = anomalies_df

        # 更新异常次数
        self.anomaly_count += anomalies_df.astype(int)

    def show_anomaly_report(self):
        report = "异常检测结果:\n\n"
        for method, result in self.anomaly_results.items():
            if isinstance(result, pd.DataFrame):
                anomaly_count = result.sum().sum()
            else:
                anomaly_count = result.sum()
            report += f"- {method}: 检测到 {anomaly_count} 个异常点\n"

        # 弹出窗口
        report_window = tk.Toplevel(self.root)
        report_window.title("异常检测报告")
        report_window.geometry("400x300")

        report_label = tk.Label(report_window, text=report, font=("Arial", 12), justify="left")
        report_label.pack(padx=10, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.mainloop()