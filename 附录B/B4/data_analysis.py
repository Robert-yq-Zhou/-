import os
import traceback
import threading
import numpy as np
import pandas as pd
from kivy.uix.screenmanager import Screen
from kivy.uix.recycleview import RecycleView
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.label import Label
from kivy.properties import (
    StringProperty, ObjectProperty, ListProperty, NumericProperty
)
from kivy.uix.popup import Popup
from plyer import filechooser
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import chardet

class DataAnalysisScreen(Screen):
    data = None
    current_file = StringProperty("")
    status_text = StringProperty("准备就绪")
    data_columns = NumericProperty(0)
    anomaly_results = {}
    scaler = StandardScaler()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_lock = threading.Lock()
        self.anomaly_count = None

    def open_file_dialog(self):
        filechooser.open_file(
            title="选择数据文件",
            filters=[("支持格式", "*.csv;*.xls;*.xlsx;*.txt")],
            on_selection=self.handle_file_selected
        )

    def handle_file_selected(self, selection):
        if not selection:
            return
        try:
            self.current_file = selection[0]
            self._load_data()
            self._update_ui_after_loading()
            self.status_text = f"成功加载文件: {os.path.basename(self.current_file)}"
        except Exception as e:
            self.status_text = f"[color=ff0000]加载失败: {str(e)}[/color]"
            self._clear_data()

    def _load_data(self):
        if self.current_file.endswith('.txt'):
            self._load_txt_with_auto_detect()
        elif self.current_file.endswith(('.xls', '.xlsx')):
            self.data = pd.read_excel(self.current_file)
        else:
            self.data = pd.read_csv(self.current_file)
        self._clean_data()

    def _load_txt_with_auto_detect(self):
        with open(self.current_file, 'rb') as f:
            rawdata = f.read(10000)
            encoding = chardet.detect(rawdata)['encoding']

        with open(self.current_file, 'r', encoding=encoding) as f:
            first_line = f.readline()

        sep = '\t'  # 强制使用制表符分隔

        # 使用检测到的分隔符加载数据
        self.data = pd.read_csv(
            self.current_file,
            sep=sep,  # 固定使用tab
            engine='c',  # 可以切换回更快的C引擎
            encoding=encoding,
            header=None if '\t' not in first_line else 'infer'  # 容错处理
        )

    def _clean_data(self):
        """增强版数据清洗方法"""
        # 第一步：删除全为空的列
        self.data.dropna(axis=1, how='all', inplace=True)

        # 第二步：删除未命名列（处理残留列名）
        unnamed_cols = [col for col in self.data.columns if 'unnamed' in str(col).lower()]
        if unnamed_cols:
            print(f"发现未命名列: {unnamed_cols}，正在删除...")
            self.data.drop(unnamed_cols, axis=1, inplace=True)

        # 第三步：转换数值类型（增强容错性）
        for col in self.data.columns:
            try:
                # 尝试转换为数值类型
                converted = pd.to_numeric(self.data[col], errors='coerce')

                # 保留原始格式检测
                if (converted.dtype == float) and (converted % 1 == 0).all():
                    # 如果是整数型的浮点数，转换为int
                    self.data[col] = converted.astype(int)
                else:
                    self.data[col] = converted

            except Exception as e:
                print(f"列 [{col}] 无法转换为数值类型，保持原始格式: {str(e)}")
                continue

        # 第四步：再次清理空列（防止转换产生新空列）
        self.data.dropna(axis=1, how='all', inplace=True)

    def _clear_data(self):
        """重置所有数据相关状态"""
        self.data = None
        self.anomaly_count = None
        self.current_file = ""
        self.data_columns = 0
        self.anomaly_results.clear()

        # 清除界面显示
        if hasattr(self.ids, 'data_view'):
            self.ids.data_view.data = []
        self.status_text = "准备就绪"

    def handle_file_selected(self, selection):
        if not selection:
            return
        try:
            self.current_file = selection[0]
            self._load_data()
            self._update_ui_after_loading()
            self.status_text = f"成功加载文件: {os.path.basename(self.current_file)}"
        except Exception as e:
            self.status_text = f"[color=ff0000]加载失败: {str(e)}[/color]"
            self._clear_data()  # 调用正确的清理方法
            traceback.print_exc()

    def _update_ui_after_loading(self):
        self.data_columns = len(self.data.columns)
        self._init_anomaly_count()
        self.update_display()

    def _init_anomaly_count(self):
        self.anomaly_count = pd.DataFrame(
            np.zeros(self.data.shape, dtype=int),
            columns=self.data.columns,
            index=self.data.index
        )

    def update_display(self):
        try:
            formatted = self._format_data()
            self.ids.data_view.data = formatted
        except Exception as e:
            self._show_error(f"显示错误: {str(e)}")

    def _format_data(self):
        formatted = []
        original_data = self.data  # 使用原始数据，不再转置

        # ===== 列头部分 =====
        # 左上角空白单元格
        formatted.append({
            'text': "",
            'bg_color': [0.9, 0.9, 0.9, 1],
            'is_header': True,
            'col': -1,
            'row': -1
        })

        # 列标题（原始数据的列名）
        for col_idx, col_name in enumerate(original_data.columns):
            formatted.append({
                'text': str(col_name),
                'bg_color': [0.9, 0.9, 0.9, 1],
                'is_header': True,
                'col': col_idx,
                'row': -1  # 列头行固定为-1
            })

        # ===== 数据行部分 =====
        for row_idx, (index, row) in enumerate(original_data.iterrows()):
            # 行标题（原始数据的索引）
            formatted.append({
                'text': str(index),
                'bg_color': [0.9, 0.9, 0.9, 1],
                'is_header': True,
                'col': -1,  # 行头列固定为-1
                'row': row_idx
            })

            # 数据单元格
            for col_idx, value in enumerate(row):
                bg_color = self._get_cell_color(col_idx, row_idx, value)
                formatted.append({
                    'text': f"{value:.4f}" if isinstance(value, float) else str(value),
                    'bg_color': bg_color,
                    'is_header': False,
                    'col': col_idx,
                    'row': row_idx
                })

        return formatted

    def _get_cell_color(self, col_idx, row_idx, value):
        original_row = row_idx
        original_col = col_idx

        if original_row >= len(self.data) or original_col >= len(self.data.columns):
            return [1,1,1,1]

        col_name = self.data.columns[original_col]
        if col_name not in self.data.select_dtypes(include=[np.number]):
            return [1,1,1,1]

        min_val = self.data[col_name].min()
        max_val = self.data[col_name].max()
        norm_val = (value - min_val) / (max_val - min_val) if max_val != min_val else 0
        base_color = [1 - norm_val, 1 - norm_val, 1, 1]

        if self.anomaly_count is not None:
            anomaly_level = self.anomaly_count.iloc[original_row, original_col]
            if anomaly_level > 0:
                red_intensity = 1.0 - (0.2 * min(anomaly_level, 5))
                base_color = [red_intensity, red_intensity*0.5, red_intensity*0.5, 1]

        return base_color

    def run_detection(self):
        if self.data is None:
            self._show_error("请先加载数据文件")
            return

        self.ids.detect_btn.disabled = True
        self.status_text = "正在检测异常..."
        self.anomaly_count.iloc[:, :] = 0
        self.anomaly_results.clear()

        threading.Thread(target=self._threaded_detection, daemon=True).start()

    def _threaded_detection(self):
        try:
            with self.data_lock:
                numeric_data = self.data.select_dtypes(include=[np.number])
                scaled_data = pd.DataFrame(
                    self.scaler.fit_transform(numeric_data),
                    columns=numeric_data.columns,
                    index=numeric_data.index
                )

                if self.ids.cb_3sigma.state == 'down':
                    self._3sigma_detection(scaled_data)
                if self.ids.cb_lof.state == 'down':
                    self._lof_detection(scaled_data)
                if self.ids.cb_dbscan.state == 'down':
                    self._dbscan_detection(scaled_data)
                if self.ids.cb_autoencoder.state == 'down':
                    self._autoencoder_detection(scaled_data)
                if self.ids.cb_isoforest.state == 'down':
                    self._isoforest_detection(scaled_data)
                if self.ids.cb_iqr.state == 'down':
                    self._iqr_detection(numeric_data)

            Clock.schedule_once(lambda dt: self._update_ui())
        except Exception as e:
            Clock.schedule_once(lambda dt: self._show_error(str(e)))

    # 异常检测方法实现
    def _3sigma_detection(self, scaled_data):
        anomalies = pd.DataFrame(False, index=scaled_data.index, columns=scaled_data.columns)
        for col in scaled_data.columns:
            mean = scaled_data[col].mean()
            std = scaled_data[col].std()
            anomalies[col] = (scaled_data[col] < (mean - 3*std)) | (scaled_data[col] > (mean + 3*std))
        self.anomaly_count += anomalies.astype(int)
        self.anomaly_results['3-Sigma'] = anomalies.sum().sum()

    def _lof_detection(self, scaled_data):
        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        preds = clf.fit_predict(scaled_data)
        anomalies = pd.Series(preds == -1, index=scaled_data.index)
        self.anomaly_count.loc[anomalies] += 1
        self.anomaly_results['LOF'] = anomalies.sum()

    def _dbscan_detection(self, scaled_data):
        dbscan = DBSCAN(eps=0.5, min_samples=10)
        clusters = dbscan.fit_predict(scaled_data)
        anomalies = pd.Series(clusters == -1, index=scaled_data.index)
        self.anomaly_count.loc[anomalies] += 1
        self.anomaly_results['DBSCAN'] = anomalies.sum()

    def _iqr_detection(self, data):
        anomalies = pd.DataFrame(False, index=data.index, columns=data.columns)
        for col in data.columns:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            anomalies[col] = (data[col] < (q1 - 1.5*iqr)) | (data[col] > (q3 + 1.5*iqr))
        self.anomaly_count += anomalies.astype(int)
        self.anomaly_results['IQR'] = anomalies.sum().sum()

    def _isoforest_detection(self, scaled_data):
        clf = IsolationForest(contamination=0.1)
        preds = clf.fit_predict(scaled_data)
        anomalies = pd.Series(preds == -1, index=scaled_data.index)
        self.anomaly_count.loc[anomalies] += 1
        self.anomaly_results['Isolation Forest'] = anomalies.sum()

    def _autoencoder_detection(self, scaled_data):
        model = Sequential([
            Dense(8, activation='relu', input_shape=(scaled_data.shape[1],)),
            Dense(4, activation='relu'),
            Dense(8, activation='relu'),
            Dense(scaled_data.shape[1], activation='linear')
        ])
        model.compile(optimizer=Adam(0.001), loss='mse')
        model.fit(scaled_data, scaled_data, epochs=20, batch_size=32, verbose=0)
        reconstructions = model.predict(scaled_data)
        mse = np.mean(np.power(scaled_data - reconstructions, 2), axis=1)
        threshold = np.percentile(mse, 95)
        anomalies = pd.Series(mse > threshold, index=scaled_data.index)
        self.anomaly_count.loc[anomalies] += 1
        self.anomaly_results['AutoEncoder'] = anomalies.sum()

    def export_data(self):
        if self.data is None:
            return
        try:
            filechooser.save_file(
                title="保存检测结果",
                filters=[("CSV文件", "*.csv")],
                on_selection=self._handle_export_selected
            )
        except Exception as e:
            self.status_text = f"导出失败: {str(e)}"

    def _handle_export_selected(self, selection):
        if not selection:
            return
        try:
            output_data = self.data.copy()
            output_data["异常次数"] = self.anomaly_count.sum(axis=1)
            output_data.to_csv(selection[0], index=False)
            self.status_text = f"结果已保存到: {os.path.basename(selection[0])}"
        except Exception as e:
            self.status_text = f"[color=ff0000]导出错误: {str(e)}[/color]"

    def _update_ui(self):
        self.ids.detect_btn.disabled = False
        self.status_text = f"检测完成，共发现{sum(self.anomaly_results.values())}个异常点"
        self.update_display()

    def _show_error(self, message):
        self.ids.detect_btn.disabled = False
        self.status_text = f"[color=ff0000]{message}[/color]"

class DataRecycleView(RecycleView):
    pass

class DataLabel(Label):
    col = ObjectProperty(-1)
    row = ObjectProperty(-1)
    bg_color = ListProperty([1, 1, 1, 1])
    is_header = ObjectProperty(False)