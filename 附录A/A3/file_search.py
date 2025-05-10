import os
import tkinter as tk
from tkinter import filedialog, messagebox


class FileSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("文件搜索工具")
        self.root.geometry("600x400")
        self.root.option_add("*Font", "SimSun 12")
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="请输入关键词：").grid(row=0, column=0, padx=10, pady=10)
        self.keyword_entry = tk.Entry(self.root, width=40)
        self.keyword_entry.grid(row=0, column=1, padx=10, pady=10)
        self.select_path_button = tk.Button(self.root, text="选择搜索路径", command=self.select_search_path)
        self.select_path_button.grid(row=1, column=0, padx=10, pady=10)
        self.path_label = tk.Label(self.root, text="尚未选择路径")
        self.path_label.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        search_button = tk.Button(self.root, text="开始搜索", command=self.search_files)
        search_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.results_listbox = tk.Listbox(self.root, width=70, height=10)
        self.results_listbox.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.results_listbox.yview)
        scrollbar.grid(row=3, column=2, sticky="ns")

        self.results_listbox.config(yscrollcommand=scrollbar.set)

    def select_search_path(self):

        folder_path = filedialog.askdirectory(title="选择搜索路径")
        if folder_path:  # 如果选择了文件夹
            self.path_label.config(text=folder_path)  # 更新显示路径

    def search_files(self):

        keyword = self.keyword_entry.get()
        search_path = self.path_label.cget("text")

        if not os.path.isdir(search_path):
            messagebox.showerror("错误", "路径无效或未选择")
            return

        if not keyword:
            messagebox.showerror("错误", "请输入关键词")
            return

        self.results_listbox.delete(0, tk.END)

        found_files = []
        for root_dir, dirs, files in os.walk(search_path):
            for file_name in files:
                if keyword.lower() in file_name.lower():  # 忽略大小写搜索
                    found_files.append(os.path.join(root_dir, file_name))

        if not found_files:
            messagebox.showinfo("提示", "没有找到符合条件的文件")
        else:
            for file in found_files:
                self.results_listbox.insert(tk.END, file)

def start_file_search():
    root = tk.Tk()
    app = FileSearchApp(root)
    root.mainloop()


if __name__ == "__main__":
    start_file_search()
