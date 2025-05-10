import os
from kivy.uix.screenmanager import Screen
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from kivy.lang import Builder
from kivy.core.text import LabelBase

current_dir = os.path.dirname(os.path.abspath(__file__))  # src目录的绝对路径
font_path = os.path.abspath('fonts/SourceHanSerifSC-Regular.otf')
LabelBase.register(
    name='SourceHanSerif',
    fn_regular=font_path
)
# 一：主屏幕
class MainScreen(Screen):
    pass  # 加载前定义，防报错

# 二：KV文件
Builder.load_file('main.kv')          # 加载MainScreen
Builder.load_file('file_search.kv')   # 加载FileSearchScreen
Builder.load_file('data_plot.kv')     # 加载DataPlotScreen
Builder.load_file('data_analysis.kv') # 加载DataAnalysisScreen

# 三：导入屏幕
from file_search import FileSearchScreen
from data_plot import DataPlotScreen
from data_analysis import DataAnalysisScreen

# 四：应用主类
class MainApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(FileSearchScreen(name='file_search'))
        sm.add_widget(DataPlotScreen(name='data_plot'))
        sm.add_widget(DataAnalysisScreen(name='data_analysis'))
        sm.current = 'main'
        return sm

if __name__ == '__main__':
    MainApp().run()