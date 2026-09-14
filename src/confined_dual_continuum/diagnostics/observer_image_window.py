"""录像进程主线程的相机窗口；复用 RGB 帧，不额外渲染或依赖 Pillow。"""
import tkinter as tk
from tkinter import ttk


class ObserverImageWindow:
    def __init__(self, width, height):
        self.root = tk.Tk()
        self.root.title('从臂相机观测 — observer_camera')
        self.root.geometry(f'{width+24}x{height+64}')
        self.text = tk.StringVar(value='等待相机图像')
        ttk.Label(self.root, textvariable=self.text, padding=8).pack(fill='x')
        self.label = ttk.Label(self.root)
        self.label.pack(padx=8, pady=4)
        self.photo = None
        self.root.protocol('WM_DELETE_WINDOW', self.close)

    def poll(self):
        if self.root is not None:
            self.root.update_idletasks()
            self.root.update()

    def show(self, rgb, time_s):
        if self.root is None:
            return
        height, width = rgb.shape[:2]
        ppm = f'P6\n{width} {height}\n255\n'.encode('ascii')+rgb.tobytes()
        self.photo = tk.PhotoImage(master=self.root, data=ppm, format='PPM')
        self.label.configure(image=self.photo)
        self.text.set(f'仿真时间 {time_s:.3f} s ｜ 从臂原始相机图像')
        self.poll()

    def close(self):
        if self.root is not None:
            self.root.destroy()
            self.root = None
