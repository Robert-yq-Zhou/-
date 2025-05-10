import tkinter as tk
from tkinter import filedialog, messagebox
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
class DataAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("文件搜索与数据分析")
        self.root.geometry("1200x800")

        # 初始化属性
        self.selected_file_path = None
        self.current_figure = None
        self.search_path = ""
        self.log_x_var = tk.BooleanVar(value=False)
        self.log_y_var = tk.BooleanVar(value=False)
        # 创建界面组件
        self.create_widgets()

    def create_widgets(self):
        # 左侧面板
        left_panel = tk.Frame(self.root, width=400)
        left_panel.pack(side="left", fill="both", expand=False)

        # 路径选择部分
        self.path_label = tk.Label(left_panel, text="搜索路径: 未选择")
        self.path_label.pack(pady=10)

        self.path_button = tk.Button(left_panel, text="选择文件夹",
                                     command=self.choose_folder)
        self.path_button.pack(pady=10)

        # 关键词搜索
        self.keyword_label = tk.Label(left_panel, text="关键词:")
        self.keyword_label.pack(pady=10)

        self.keyword_entry = tk.Entry(left_panel, width=30)
        self.keyword_entry.pack(pady=10)

        # 搜索按钮
        self.search_button = tk.Button(left_panel, text="搜索文件",
                                       command=self.search_files)
        self.search_button.pack(pady=10)

        # 文件列表容器
        self.file_list_frame = tk.Frame(left_panel)
        self.file_list_frame.pack(fill="both", expand=True)

        # 滚动条和画布
        self.file_canvas = tk.Canvas(self.file_list_frame)
        self.file_canvas.pack(side="left", fill="both", expand=True)

        self.v_scroll = tk.Scrollbar(self.file_list_frame,
                                     orient="vertical",
                                     command=self.file_canvas.yview)
        self.v_scroll.pack(side="right", fill="y")

        self.file_canvas.configure(yscrollcommand=self.v_scroll.set)
        self.file_container = tk.Frame(self.file_canvas)
        self.file_canvas.create_window((0, 0), window=self.file_container, anchor="nw")

        # 右侧面板
        right_panel = tk.Frame(self.root)
        right_panel.pack(side="right", fill="both", expand=True)

        # 控制面板
        control_frame = tk.Frame(right_panel)
        control_frame.pack(pady=20, fill="x")
        tk.Label(control_frame, text="对数轴:").grid(row=0, column=7, padx=5)
        tk.Checkbutton(control_frame, text="lg(X)", variable=self.log_x_var).grid(row=0, column=8, padx=2)
        tk.Checkbutton(control_frame, text="lg(Y)", variable=self.log_y_var).grid(row=0, column=9, padx=2)

        # 轴选择组件
        self.x_axis_var = tk.StringVar()
        self.y_axis_var = tk.StringVar()
        self.fit_type_var = tk.StringVar(value="一次拟合")

        tk.Label(control_frame, text="X轴:").grid(row=0, column=0, padx=5)
        self.x_axis_menu = tk.OptionMenu(control_frame, self.x_axis_var, "")
        self.x_axis_menu.grid(row=0, column=1, padx=5)

        tk.Label(control_frame, text="Y轴:").grid(row=0, column=2, padx=5)
        self.y_axis_menu = tk.OptionMenu(control_frame, self.y_axis_var, "")
        self.y_axis_menu.grid(row=0, column=3, padx=5)

        # 拟合类型选择
        tk.Label(control_frame, text="拟合类型:").grid(row=0, column=4, padx=5)
        self.fit_type_menu = tk.OptionMenu(control_frame, self.fit_type_var,
                                           "一次拟合", "二次拟合", "指数拟合", "不绘制拟合曲线")
        self.fit_type_menu.grid(row=0, column=5, padx=5)

        # 拟合按钮
        self.fit_button = tk.Button(control_frame, text="执行拟合",
                                    command=self.fit_and_plot)
        self.fit_button.grid(row=0, column=6, padx=10)

        # 图表显示区域
        self.chart_frame = tk.Frame(right_panel)
        self.chart_frame.pack(fill="both", expand=True)

    def choose_folder(self):
        """选择文件夹方法"""
        selected_path = filedialog.askdirectory(title="选择文件夹")
        if selected_path:
            self.search_path = selected_path
            self.path_label.config(text=f"当前路径: {self.search_path}")

    def search_files(self):
        """执行文件搜索"""
        if not self.search_path:
            messagebox.showwarning("警告", "请先选择文件夹")
            return

        keyword = self.keyword_entry.get().strip().lower()
        supported_exts = ['.txt', '.csv', '.xls', '.xlsx']
        found_files = []

        try:
            for root_dir, _, filenames in os.walk(self.search_path):
                for filename in filenames:
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext in supported_exts:
                        if keyword in filename.lower():
                            found_files.append(os.path.join(root_dir, filename))

            self.display_files(found_files)

            if not found_files:
                messagebox.showinfo("提示", "未找到匹配文件")
        except Exception as e:
            messagebox.showerror("错误", f"搜索失败: {str(e)}")

    def display_files(self, files):
        """显示搜索结果"""
        # 清空现有文件列表
        for widget in self.file_container.winfo_children():
            widget.destroy()

        # 显示新文件列表
        for idx, file_path in enumerate(files):
            btn = tk.Button(
                self.file_container,
                text=os.path.basename(file_path),
                width=40,
                command=lambda fp=file_path: self.open_file(fp)
            )
            btn.pack(pady=2, padx=5, fill="x")

        # 更新滚动区域
        self.file_container.update_idletasks()
        self.file_canvas.config(scrollregion=self.file_canvas.bbox("all"))

    def open_file(self, file_path):
        """打开并处理文件"""
        try:
            self.selected_file_path = file_path
            data = self.read_file_data(file_path)
            self.update_axis_options(data)
        except Exception as e:
            messagebox.showerror("文件错误", f"无法读取文件: {str(e)}")

    def read_file_data(self, file_path):
        """读取文件数据"""
        try:
            if file_path.endswith(('.xls', '.xlsx')):
                return pd.read_excel(file_path)
            else:
                # 自动检测分隔符
                with open(file_path, 'r') as f:
                    first_line = f.readline()

                if ';' in first_line:
                    sep = ';'
                elif ',' in first_line:
                    sep = ','
                else:
                    sep = '\t'

                return pd.read_csv(file_path, sep=sep, engine='python')
        except Exception as e:
            raise ValueError(f"文件读取失败: {str(e)}")

    def update_axis_options(self, data):
        """更新轴选择选项"""
        columns = data.columns.tolist()

        # 更新X轴菜单
        x_menu = self.x_axis_menu["menu"]
        x_menu.delete(0, "end")
        for col in columns:
            x_menu.add_command(
                label=col,
                command=lambda v=col: self.x_axis_var.set(v)
            )

        # 更新Y轴菜单
        y_menu = self.y_axis_menu["menu"]
        y_menu.delete(0, "end")
        for col in columns:
            y_menu.add_command(
                label=col,
                command=lambda v=col: self.y_axis_var.set(v)
            )

        # 设置默认值
        if columns:
            self.x_axis_var.set(columns[0])
            self.y_axis_var.set(columns[1] if len(columns) > 1 else columns[0])

    def fit_and_plot(self):
        """执行拟合和绘图"""
        log_x = self.log_x_var.get()
        log_y = self.log_y_var.get()
        if not self.selected_file_path:
            messagebox.showwarning("警告", "请先选择文件")
            return

        try:
            x_col = self.x_axis_var.get()
            y_col = self.y_axis_var.get()
            fit_type = self.fit_type_var.get()

            data = self.read_file_data(self.selected_file_path)

            if x_col not in data.columns or y_col not in data.columns:
                raise ValueError("选择的列不存在")

            X = data[x_col].values.reshape(-1, 1)
            y = data[y_col].values
            if log_x and np.any(data[self.x_axis_var.get()] <= 0):
                raise ValueError("X轴数据需全为正数才能取对数")
            if log_y and np.any(data[self.y_axis_var.get()] <= 0):
                raise ValueError("Y轴数据需全为正数才能取对数")
            # 执行拟合
            if fit_type == "一次拟合":
                y_pred, equation = self.linear_fit(X, y)
            elif fit_type == "二次拟合":
                y_pred, equation = self.quadratic_fit(X, y)
            elif fit_type == "指数拟合":
                y_pred, equation = self.exponential_fit(X, y)
            elif fit_type == "不绘制拟合曲线":
                y_pred = None
                equation = None
            else:
                raise ValueError("未知的拟合类型")
            # 显示图表
            self.create_chart_figure(data, x_col, y_col, y_pred, equation, log_x, log_y)

        except Exception as e:
            messagebox.showerror("拟合错误", str(e))

    def linear_fit(self, X, y):
        """一次线性拟合"""
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        equation = f"y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}"
        return y_pred, equation

    def quadratic_fit(self, X, y):
        """二次多项式拟合"""
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)

        # 系数排序为 [intercept, x, x^2]
        equation = (f"y = {model.coef_[2]:.4f}x² + "
                    f"{model.coef_[1]:.4f}x + "
                    f"{model.intercept_:.4f}")
        return y_pred, equation

    def exponential_fit(self, X, y):
        """指数拟合 y = a*exp(bx)"""
        if np.any(y <= 0):
            raise ValueError("指数拟合需要所有y值大于0")

        y_log = np.log(y)
        model = LinearRegression()
        model.fit(X, y_log)

        a = np.exp(model.intercept_)
        b = model.coef_[0]
        y_pred = a * np.exp(b * X.flatten())

        equation = f"y = {a:.4f}e^({b:.4f}x)"
        return y_pred, equation

    def create_chart_figure(self, data, x_col, y_col, y_pred, equation, log_x, log_y):
        """创建并显示图表"""
        # 清除旧图表
        if self.current_figure:
            self.current_figure.get_tk_widget().destroy()

        # 创建新图表
        fig = plt.Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        #对数处理
        plot_x = np.log(data[x_col]) if log_x else data[x_col]
        plot_y = np.log(data[y_col]) if log_y else data[y_col]
        # 绘制原始数据点
        ax.scatter(plot_x, plot_y, color='blue', label='原始数据', alpha=0.7)

        # 绘制拟合曲线（如果有）
        if y_pred is not None:
            sorted_idx = np.argsort(data[x_col])
            if log_x:
                plot_x_fit = np.log(data[x_col].values[sorted_idx])
            else:
                plot_x_fit = data[x_col].values[sorted_idx]

            ax.plot(plot_x_fit, y_pred[sorted_idx], color='red', linewidth=2, label='拟合曲线')
            title = f"{y_col} vs {x_col}\n{equation}"
        else:
            title = f"{y_col} vs {x_col}"

        # 添加图表元素
        x_label = f"lg({x_col})" if log_x else x_col
        y_label = f"lg({y_col})" if log_y else y_col
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        # 如果是对数坐标，添加网格线样式
        if log_x or log_y:
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
        # 格式化坐标轴
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.tight_layout()

        # 嵌入到界面
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.current_figure = canvas


if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.mainloop()