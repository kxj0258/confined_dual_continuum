"""Tk 主线程误差窗口，仿真工作线程仅发布数值，不调用 Tk。"""
import tkinter as tk
from tkinter import ttk
import numpy as np
from confined_dual_continuum.diagnostics.executor_records import COLUMNS
from confined_dual_continuum.diagnostics.executor_error_plot import expanded_limit


class ErrorMonitor:
    def __init__(self, records, stop_event, *, refresh_hz=20):
        self.records, self.stop_event = records, stop_event
        self.limit_mm = 10.
        self.root = tk.Tk()
        self.root.title('双臂跟踪监视 — mm')
        self.root.geometry('960x760')
        self.interval_ms = max(1, round(1000/refresh_hz))
        self.text = tk.StringVar(value='准备运行')
        ttk.Label(self.root, textvariable=self.text, padding=12, font=('Consolas', 11)).pack(fill='x')
        self.canvas = tk.Canvas(self.root, background='white', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=12, pady=6)
        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill='x')
        ttk.Button(controls, text='停止仿真并保存', command=stop_event.set).pack(side='left')
        ttk.Label(controls, text='X 红  Y 绿  Z 蓝  总误差 黑；显示最近 20 秒仿真时间，完整数据保存为 CSV').pack(side='left', padx=12)
        self.error_lines = [self.canvas.create_line(0, 0, 1, 1, fill=c, width=2)
                      for c in ('#d64b42', '#2d955f', '#397cc1', '#202124')]
        self.gap_line = self.canvas.create_line(0, 0, 1, 1, fill='#7b3fb5', width=2)
        self.error_axis = self.canvas.create_line(0, 0, 1, 1, fill='#777777')
        self.gap_axis = self.canvas.create_line(0, 0, 1, 1, fill='#777777')
        self.trigger_line = self.canvas.create_line(0, 0, 1, 1, fill='#d97706', dash=(5, 3))
        self.safe_line = self.canvas.create_line(0, 0, 1, 1, fill='#d64b42', dash=(5, 3))
        self.collision_line = self.canvas.create_line(0, 0, 1, 1, fill='#202124', dash=(2, 3))
        self.error_title = self.canvas.create_text(0, 0, text='主臂末端误差',
                                                   fill='#202124', font=('Segoe UI', 11, 'bold'))
        self.gap_title = self.canvas.create_text(0, 0, text='主从中心线最近距离',
                                                 fill='#202124', font=('Segoe UI', 11, 'bold'))
        self.gap_labels = [self.canvas.create_text(0, 0, text='', anchor='e',
                                                   fill=color, font=('Segoe UI', 9, 'bold'))
                           for color in ('#d97706', '#d64b42', '#202124')]
        self.gap_ticks = [self.canvas.create_text(0, 0, text='', anchor='e', fill='#666666')
                          for _ in range(5)]
        self.error_time_labels = [self.canvas.create_text(0, 0, text='', fill='#666666')
                                  for _ in range(2)]
        self.gap_time_labels = [self.canvas.create_text(0, 0, text='', fill='#666666')
                                for _ in range(2)]
        self.labels = [self.canvas.create_text(0, 0, text='', anchor=a, fill='#555555')
                       for a in ('w', 'w', 'w', 'e', 'w', 'w')]
        self.root.protocol('WM_DELETE_WINDOW', self.close)

    def close(self):
        self.stop_event.set()
        self.root.destroy()

    def tick(self):
        rows, info = self.records.snapshot()
        if not rows:
            self.text.set(info['status'])
        if rows:
            data = np.asarray(rows)
            latest = data[-1]
            error_mm = latest[COLUMNS.index('error_norm_m')]*1000
            self.text.set(f"{info['status']}\n仿真 {latest[0]:.3f} s  |  实时倍率 {info['realtime_factor']:.2f}×  |  当前误差 {error_mm:.3f} mm\n"
                          f"RMSE {info['rmse_m']*1000:.3f} mm  |  最大误差 {info['maximum_error_m']*1000:.3f} mm  |  饱和比例 {info['saturation_fraction']:.1%}")
            data = data[data[:, 0] >= latest[0]-20.]
            if len(data) > 1500:
                data = data[np.linspace(0, len(data)-1, 1500).astype(int)]
            w, h = max(self.canvas.winfo_width(), 200), max(self.canvas.winfo_height(), 300)
            left, right = 65., w-20.
            top, middle, bottom = 25., h/2., h-35.
            columns = [COLUMNS.index(name) for name in ('error_x_m', 'error_y_m', 'error_z_m', 'error_norm_m')]
            values = data[:, columns]*1000
            # 用全程最大范数扩展，避免滚动窗口和降采样漏掉峰值后缩回。
            self.limit_mm = expanded_limit(info['maximum_error_m']*1000, self.limit_mm)
            low, high = -self.limit_mm, self.limit_mm
            t0, t1 = float(data[0, 0]), max(float(data[-1, 0]), float(data[0, 0])+.02)
            x = left+(data[:, 0]-t0)/(t1-t0)*(right-left)
            for i, line in enumerate(self.error_lines):
                y = middle-(values[:, i]-low)/(high-low)*(middle-top-20)
                coordinates = np.column_stack((x, y))
                if len(coordinates) == 1:
                    coordinates = np.repeat(coordinates, 2, axis=0)
                self.canvas.coords(line, *coordinates.ravel().tolist())
            self.canvas.coords(self.error_axis, left, top, left, middle-20, right, middle-20)
            self.canvas.coords(self.gap_axis, left, middle+20, left, bottom, right, bottom)
            gap_rows = self.records.interarm_snapshot()
            if gap_rows:
                gap_data = np.asarray(gap_rows)
                gap_data = gap_data[gap_data[:, 0] >= t0]
                if len(gap_data):
                    gx = left+(gap_data[:, 0]-t0)/(t1-t0)*(right-left)
                    centerline_mm = gap_data[:, 1]*1000. + 12.5
                    gy = bottom-(centerline_mm/20.)*(bottom-middle-20)
                    coords = np.column_stack((gx, gy))
                    if len(coords) == 1:
                        coords = np.repeat(coords, 2, axis=0)
                    self.canvas.coords(self.gap_line, *coords.ravel().tolist())
            for value, line in ((18.5, self.trigger_line), (15.5, self.safe_line),
                                (12.5, self.collision_line)):
                y = bottom-(value/20.)*(bottom-middle-20)
                self.canvas.coords(line, left, y, right, y)
            self.canvas.coords(self.error_title, left+6, top+2)
            self.canvas.coords(self.gap_title, left+6, middle+24)
            for label, value, line in zip(self.gap_labels, (18.5, 15.5, 12.5),
                                          (self.trigger_line, self.safe_line, self.collision_line)):
                line_coords = self.canvas.coords(line)
                self.canvas.coords(label, right-4, line_coords[1]-7)
                self.canvas.itemconfigure(label, text=f'{value:.1f} mm')
            for label, value in zip(self.gap_ticks, (0, 5, 10, 15, 20)):
                y = bottom-(value/20.)*(bottom-middle-20)
                self.canvas.coords(label, left-8, y)
                self.canvas.itemconfigure(label, text=f'{value} mm')
            for labels, y in ((self.error_time_labels, middle-4),
                              (self.gap_time_labels, bottom+18)):
                self.canvas.coords(labels[0], left, y)
                self.canvas.coords(labels[1], right, y)
                self.canvas.itemconfigure(labels[0], text=f'{t0:.2f} s')
                self.canvas.itemconfigure(labels[1], text=f'{t1:.2f} s')
            positions = ((4, top, f'{high:.2f}'), (4, middle-20, f'{low:.2f}'),
                         (left, middle+2, '末端误差 / mm'), (left, bottom+18, '最近中心线距离 / mm'),
                         (left, bottom+35, f'{t0:.2f} s'), (right, bottom+35, f'{t1:.2f} s'))
            for label, (xpos, ypos, text) in zip(self.labels, positions):
                self.canvas.coords(label, xpos, ypos)
                self.canvas.itemconfigure(label, text=text)
        self.root.after(self.interval_ms, self.tick)

    def run(self, start_worker):
        self.root.after(0, start_worker)
        self.root.after(self.interval_ms, self.tick)
        self.root.mainloop()
