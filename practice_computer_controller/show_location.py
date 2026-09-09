import pyautogui
import time

while True:
    x, y = pyautogui.position()

    my_screen = pyautogui.size()

    print(f"显示屏大小：{my_screen}")

    print(f"鼠标位置: x={x}, y={y}")

    time.sleep(0.1)