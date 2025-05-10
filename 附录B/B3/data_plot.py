from kivy.uix.screenmanager import Screen
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.uix.label import Label
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from plyer import filechooser
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import chardet
import os


class DataPlotScreen(Screen):
    data = None

    def open_file_dialog(self):
        filechooser.open_file(
            title="Select Data File",
            filters=[("All Supported Formats", "*.csv;*.xls;*.xlsx;*.txt")],
            on_selection=self.load_data
        )

    def load_data(self, selection):
        if not selection:
            return

        try:
            file_path = selection[0]
            self._clear_plot()
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(file_path)
            elif file_path.endswith('.txt'):
                self._load_txt_file(file_path)
            self._update_spinners()
            self._show_success("File loaded successfully!")
        except Exception as e:
            self._show_error(f"Load failed: {str(e)}")

    def _load_txt_file(self, file_path):
        with open(file_path, 'rb') as f:
            rawdata = f.read(10000)
            encoding = chardet.detect(rawdata)['encoding']
        with open(file_path, 'r', encoding=encoding) as f:
            first_line = f.readline()
        sep = '\t' if '\t' in first_line else ',' if ',' in first_line else ';' if ';' in first_line else '\s+'
        self.data = pd.read_csv(file_path, sep=sep, engine='python', encoding=encoding)

    def _update_spinners(self):
        if self.data is not None:
            columns = self.data.columns.tolist()
            self.ids.x_axis.values = columns
            self.ids.y_axis.values = columns
            self.ids.x_axis.text = columns[0] if columns else "Select X-axis"
            self.ids.y_axis.text = columns[1] if len(columns) > 1 else "Select Y-axis"

    def update_plot(self):
        if self.data is None or not self._validate_axes():
            return

        x_col = self.ids.x_axis.text
        y_col = self.ids.y_axis.text
        fit_type = self.ids.fit_type.text

        fig, ax = plt.subplots()
        ax.scatter(self.data[x_col], self.data[y_col], label='Data Points')

        if fit_type != '不拟合':
            x = self.data[x_col].values.reshape(-1, 1)
            y = self.data[y_col].values
            self._add_fit_curve(ax, x, y, fit_type)

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        ax.grid(True)
        self._display_figure(fig)

    def _validate_axes(self):
        return (
                self.ids.x_axis.text != "Select X-axis" and
                self.ids.y_axis.text != "Select Y-axis" and
                self.ids.x_axis.text in self.data.columns and
                self.ids.y_axis.text in self.data.columns
        )

    def _add_fit_curve(self, ax, x, y, fit_type):
        try:
            if fit_type == '线性拟合':
                model = LinearRegression()
                model.fit(x, y)
                y_pred = model.predict(x)
                ax.plot(x, y_pred, color='red', label='Linear Fit')
            elif fit_type == '二次拟合':
                poly = PolynomialFeatures(degree=2)
                x_poly = poly.fit_transform(x)
                model = LinearRegression()
                model.fit(x_poly, y)
                y_pred = model.predict(x_poly)
                sorted_idx = np.argsort(x.flatten())
                ax.plot(x[sorted_idx], y_pred[sorted_idx], color='green', label='Quadratic Fit')
            elif fit_type == '指数拟合':
                if np.any(y <= 0):
                    raise ValueError("Exponential fit requires positive Y values!")
                log_y = np.log(y)
                model = LinearRegression()
                model.fit(x, log_y)
                y_pred = np.exp(model.predict(x))
                ax.plot(x, y_pred, color='purple', label='Exponential Fit')
        except Exception as e:
            self._show_error(f"Fit error: {str(e)}")

    def _display_figure(self, fig):
        self._clear_plot()
        canvas = FigureCanvasKivyAgg(fig)

        if hasattr(canvas, '_idle_draw_id'):
            canvas.mpl_disconnect(canvas._idle_draw_id)
        canvas._idle_draw_id = None

        canvas.bind(
            on_touch_down=self._handle_touch,
            on_touch_move=self._handle_touch,
            on_touch_up=self._handle_touch
        )

        self.ids.plot_area.add_widget(canvas)
        plt.close(fig)

    def _handle_touch(self, widget, touch):
        if widget.collide_point(*touch.pos):
            print(f"Canvas touched at: {touch.pos}")
            return True
        return False

    def _clear_plot(self):
        for child in self.ids.plot_area.children[:]:
            if isinstance(child, FigureCanvasKivyAgg):
                child.unbind(
                    on_touch_down=self._handle_touch,
                    on_touch_move=self._handle_touch,
                    on_touch_up=self._handle_touch
                )
        self.ids.plot_area.clear_widgets()

    def _show_success(self, message):
        self._clear_plot()
        self.ids.plot_area.add_widget(Label(
            text=message,
            color=(0, 1, 0, 1),
            font_size=20,
            halign='center'
        ))

    def _show_error(self, message):
        self._clear_plot()
        self.ids.plot_area.add_widget(Label(
            text=message,
            color=(1, 0, 0, 1),
            font_size=20,
            halign='center'
        ))