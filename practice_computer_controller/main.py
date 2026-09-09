import pyautogui
import time

from mpl_toolkits.mplot3d.proj3d import world_transformation

# 移动鼠标
pyautogui.moveTo(1284, 768, duration=0)

# 点击
pyautogui.click()

# 输入文字
pyautogui.write("hello world", interval=0.1)

# 按键
pyautogui.press("enter")

time.sleep(2)
