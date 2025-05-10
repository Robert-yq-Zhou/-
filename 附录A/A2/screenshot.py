import pyautogui
import time
from datetime import datetime
import os

# 设置或创建（如果不存在）保存截图的文件夹
save_folder = "screenshots"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


# 定义截图并保存的函数
def take_screenshot():
    # 获取当前时间并格式化为字符串（用于命名截图文件）
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    screenshot = pyautogui.screenshot()

    # 设置保存路径
    screenshot_path = os.path.join(save_folder, f"{current_time}.png")

    # 保存截图
    screenshot.save(screenshot_path)
    print(f"截图已保存: {screenshot_path}")


# 定义一个定时截图的函数
def screenshot_timer():
    while True:
        take_screenshot()
        time.sleep(300)  # 每隔5分钟截图一次


# 调用定时截图的功能
if __name__ == "__main__":
    screenshot_timer()
