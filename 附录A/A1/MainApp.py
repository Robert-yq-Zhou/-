import tkinter as tk
from tkinter import messagebox
import subprocess
import threading


class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("控制面板")
        self.root.geometry("800x600")
        self.root.attributes("-topmost", True)  # 初始置顶
        self.root.option_add("*Font", "SimSun 12")
        # 监听ESC键退出
        self.root.bind("<Escape>", self.quit_program)

        self.create_widgets()

        self.allow_topmost = True

    def create_widgets(self):
        # 主容器使用Frame实现弹性布局
        container = tk.Frame(self.root)
        container.pack(expand=True, fill="both", padx=20, pady=20)

        # 按钮布局网格
        button_grid = tk.Frame(container)
        button_grid.pack(expand=True)

        # 功能按钮
        buttons = [
            ("定时截屏", self.start_screenshot),
            ("搜索文件", self.search_files),
            ("查看文件并绘图", self.open_file_search),
            ("数据分析", self.open_data_analysis)
        ]

        for idx, (text, cmd) in enumerate(buttons):
            btn = tk.Button(button_grid, text=text, width=20, height=2,
                            command=lambda c=cmd: self.safe_launch(c))
            btn.grid(row=idx // 2, column=idx % 2, padx=15, pady=15)

        # 退出按钮
        exit_btn = tk.Button(container, text="退出", command=self.quit_program)
        exit_btn.pack(side="bottom", pady=10)

    def safe_launch(self, command):
        try:
            command()
        except Exception as e:
            messagebox.showerror("错误", f"启动失败: {str(e)}")

    def start_screenshot(self):
        threading.Thread(target=self._run_screenshot, daemon=True).start()

    def _run_screenshot(self):
        try:
            from screenshot import screenshot_timer
            screenshot_timer()
            self.root.after(0, lambda: messagebox.showinfo("提示", "截图服务已启动"))
        except ImportError:
            self.root.after(0, lambda: messagebox.showerror("错误", "找不到截图模块"))

    def search_files(self):
        self._launch_subprocess("file_search.py")

    def open_file_search(self):
        self._launch_subprocess("file_plot.py")

    def open_data_analysis(self):
        self._launch_subprocess("data_analysis.py")

    def _launch_subprocess(self, script_name):

        def callback():
            self.root.attributes("-topmost", False)
            proc = subprocess.Popen(["python", script_name])
            threading.Thread(target=self._monitor_process, args=(proc,), daemon=True).start()

        self.root.after(100, callback)

    def _monitor_process(self, process):

        process.wait()
        self.root.after(0, lambda: self.root.attributes("-topmost", True))

    def quit_program(self, event=None):
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)


    # 窗口层级最终调整
    def final_adjust():
        root.attributes("-topmost", True)
        root.lift()  # 确保窗口位于最前


    root.after(500, final_adjust)

    root.mainloop()