import ctypes
import time

user32 = ctypes.windll.user32

VK_W = 0x57
VK_A = 0x41
VK_S = 0x53
VK_D = 0x44
KEYEVENTF_KEYUP = 0x0002


def key_hold(vk):
    """按住某键（只按不抬）—— 游戏场景用它保持移动"""
    user32.keybd_event(vk, 0, 0, 0)


def key_release(vk):
    """松开某键"""
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)


def key_is_down(vk):
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def repeat_key(vk, duration, interval=0.03):
    """持续发送按下事件（模拟键盘自动重复）—— 记事本会出现一串字符"""
    end = time.time() + duration
    while time.time() < end:
        user32.keybd_event(vk, 0, 0, 0)      # 重复发送"按下"
        time.sleep(interval)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)  # 最后释放
    print(f"  已发送 {vk} {duration:.1f} 秒（持续重复）")


if __name__ == "__main__":
    print("== keybd_event 测试 ==")
    time.sleep(5)

    # 1. 持续发送 W 3 秒 -> 记事本出现一串 wwww
    print("持续发送 W 3 秒...")
    repeat_key(VK_W, 10.0)

    time.sleep(1)

    # 2. WASD 四个同时持续发送 3 秒 -> 记事本出现 wws s a a dd
    print("WASD 同时持续发送 3 秒...")
    end = time.time() + 10.0
    while time.time() < end:
        for vk in [VK_W, VK_A, VK_S, VK_D]:
            user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.03)
    # 全部释放
    for vk in [VK_W, VK_A, VK_S, VK_D]:
        user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
    print("释放全部")

    print("== 测试完成，看记事本有没有出现字符 ==")