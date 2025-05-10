from kivy.uix.screenmanager import Screen
from kivy.uix.button import Button
from kivy.clock import Clock
from threading import Thread
import os


class FileSearchScreen(Screen):
    selected_path = None

    def open_folder_dialog(self):
        from plyer import filechooser
        filechooser.choose_dir(
            title="选择文件夹",
            on_selection=self.handle_folder_selected
        )

    def handle_folder_selected(self, selection):
        if selection:
            self.selected_path = selection[0]
            self.ids.path_label.text = f"所选文件夹：{os.path.basename(self.selected_path)}"
            self.ids.results_list.clear_widgets()

    def perform_search(self):
        keyword = self.ids.keyword_input.text.strip().lower()
        self.ids.results_list.clear_widgets()

        if not self.selected_path:
            self.ids.status_label.text = "错误：未选择文件夹！"
            return

        if not keyword:
            self.ids.status_label.text = "错误：请输入关键词！"
            return

        self.ids.search_btn.disabled = True
        self.ids.status_label.text = "搜索中..."

        Thread(
            target=self._threaded_search,
            args=(self.selected_path, keyword),
            daemon=True
        ).start()

    def _threaded_search(self, path, keyword):
        found = []
        try:
            for root, _, files in os.walk(path):
                for f in files:
                    if keyword in f.lower():
                        found.append(os.path.join(root, f))
        except Exception as e:
            error_msg = f"错误：{str(e)}"
        else:
            error_msg = ""

        Clock.schedule_once(lambda dt: self._update_ui(found, error_msg))

    def _update_ui(self, found_files, error_msg):
        self.ids.search_btn.disabled = False
        self.ids.status_label.text = error_msg or f"找到 {len(found_files)} 个文件"

        for path in found_files:
            self.ids.results_list.add_widget(
                Button(
                    text=os.path.basename(path),
                    size_hint_y=None,
                    height=40,
                    font_name='fonts/SourceHanSerifSC-Regular.otf'
                )
            )

    def reset_search(self):
        self.selected_path = None
        self.ids.path_label.text = "未选择文件夹"
        self.ids.keyword_input.text = ""
        self.ids.results_list.clear_widgets()
        self.ids.status_label.text = "准备就绪"