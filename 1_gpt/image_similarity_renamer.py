import os
import csv
import math
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps
from scipy.fftpack import dct
from scipy.ndimage import sobel

SUPPORTED = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tif', '.tiff'}


def phash(img, hash_size=8, highfreq=4):
    # 32x32 DCT, keep low-frequency 8x8 coefficients.
    size = hash_size * highfreq
    gray = ImageOps.exif_transpose(img).convert('L').resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    d = dct(dct(arr, axis=0, norm='ortho'), axis=1, norm='ortho')[:hash_size, :hash_size]
    med = np.median(d[1:, :])
    bits = d > med
    return bits.flatten()


def dhash(img, size=16):
    gray = ImageOps.exif_transpose(img).convert('L').resize((size + 1, size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    return (arr[:, 1:] > arr[:, :-1]).flatten()


def visual_feature(img):
    # Preserve the whole screenshot with letterboxing; useful for tall phone screenshots.
    src = ImageOps.exif_transpose(img).convert('RGB')
    w, h = src.size
    ratio = min(w, h) / max(w, h) if max(w, h) else 1.0
    thumb = ImageOps.contain(src, (48, 64), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (48, 64), (0, 0, 0))
    canvas.paste(thumb, ((48 - thumb.width) // 2, (64 - thumb.height) // 2))
    gray_raw = np.asarray(canvas.convert('L'), dtype=np.float32) / 255.0
    # Normalized luminance and edge structure make the comparison robust to small brightness changes.
    gray = (gray_raw - gray_raw.mean()) / (gray_raw.std() + 1e-6)
    gx = sobel(gray_raw, axis=1)
    gy = sobel(gray_raw, axis=0)
    edge = np.hypot(gx, gy)
    edge = edge / (edge.max() + 1e-6)
    # Small color histogram catches major UI/color layout changes.
    hsv = np.asarray(canvas.convert('HSV'), dtype=np.float32)
    hist_h, _ = np.histogram(hsv[..., 0], bins=12, range=(0, 255), density=True)
    hist_s, _ = np.histogram(hsv[..., 1], bins=8, range=(0, 255), density=True)
    hist_v, _ = np.histogram(hsv[..., 2], bins=8, range=(0, 255), density=True)
    ph = phash(src)
    dh = dhash(src)
    return {
        'p': ph,
        'd': dh,
        'g': gray.flatten(),
        'raw': gray_raw.flatten(),
        'edge': edge.flatten(),
        'hist': np.concatenate([hist_h, hist_s, hist_v]),
        'ratio': ratio,
        'size': src.size,
    }


def hamming(a, b):
    return np.mean(a != b)


def cosine_distance(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / den)


def similarity(f1, f2):
    # 0..1, higher = more visually similar.
    p = hamming(f1['p'], f2['p'])
    d = hamming(f1['d'], f2['d'])
    g = min(2.0, cosine_distance(f1['g'], f2['g'])) / 2.0
    raw = min(1.0, float(np.mean(np.abs(f1['raw'] - f2['raw']))))
    edge = min(1.0, float(np.mean(np.abs(f1['edge'] - f2['edge']))))
    hist = min(2.0, cosine_distance(f1['hist'], f2['hist'])) / 2.0
    ratio = min(1.0, abs(math.log((f1['ratio'] + 1e-6) / (f2['ratio'] + 1e-6))) / 1.5)
    # For UI screenshots, pHash/dHash dominate while the thumbnail catches broad layout changes.
    dist = 0.28 * p + 0.18 * d + 0.22 * g + 0.16 * raw + 0.10 * edge + 0.04 * hist + 0.02 * ratio
    return max(0.0, min(1.0, 1.0 - dist))


def list_images(folder, recursive=False):
    root = Path(folder)
    it = root.rglob('*') if recursive else root.iterdir()
    return sorted([p for p in it if p.is_file() and p.suffix.lower() in SUPPORTED], key=lambda p: p.name.lower())


def make_groups(items, threshold, progress=None):
    n = len(items)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    total = n * (n - 1) // 2
    done = 0
    # Two cheap filters prevent comparing every image in folders with mixed aspect ratios.
    for i in range(n):
        fi = items[i][1]
        for j in range(i + 1, n):
            fj = items[j][1]
            # Phone screenshots generally have very similar aspect ratios; allow a generous range.
            if abs(math.log((fi['ratio'] + 1e-6) / (fj['ratio'] + 1e-6))) > 1.2:
                done += 1
                continue
            if similarity(fi, fj) >= threshold:
                union(i, j)
            done += 1
            if progress and (done % max(1, total // 100) == 0 or done == total):
                progress(done / max(1, total))

    groups = defaultdict(list)
    for i, (path, feat) in enumerate(items):
        groups[find(i)].append((path, feat))

    # Sort groups by their first original filename, then sort members by similarity
    # to the group's representative. This makes adjacent files naturally related.
    result = []
    for members in groups.values():
        rep = members[0]
        ordered = sorted(members, key=lambda x: (-similarity(rep[1], x[1]), x[0].name.lower()))
        result.append(ordered)
    result.sort(key=lambda g: g[0][0].name.lower())
    return result


def build_plan(groups, prefix='Group'):
    rows = []
    group_num = 1
    for group in groups:
        for idx, (path, feat) in enumerate(group, 1):
            new_name = f'{prefix}-{group_num:03d}-{idx:02d}{path.suffix.lower()}'
            rows.append((path, new_name, group_num, idx, len(group)))
        group_num += 1
    return rows


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('图片相似度分组与重命名')
        self.geometry('980x700')
        self.minsize(850, 600)
        self.folder = ''
        self.groups = []
        self.plan = []
        self.q = queue.Queue()
        self._build()
        self.after(100, self._poll)

    def _build(self):
        pad = {'padx': 8, 'pady': 6}
        top = ttk.Frame(self)
        top.pack(fill='x', **pad)
        ttk.Label(top, text='图片文件夹：').pack(side='left')
        self.folder_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.folder_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(top, text='选择', command=self.choose).pack(side='left')

        opts = ttk.Frame(self)
        opts.pack(fill='x', **pad)
        ttk.Label(opts, text='相似度阈值：').pack(side='left')
        self.threshold_var = tk.DoubleVar(value=0.82)
        self.threshold_scale = ttk.Scale(opts, from_=0.65, to=0.97, variable=self.threshold_var, orient='horizontal', length=240)
        self.threshold_scale.pack(side='left', padx=5)
        self.threshold_label = ttk.Label(opts, text='82%')
        self.threshold_label.pack(side='left')
        self.threshold_scale.bind('<Motion>', lambda e: self.threshold_label.config(text=f'{self.threshold_var.get()*100:.0f}%'))
        self.threshold_scale.bind('<ButtonRelease-1>', lambda e: self.threshold_label.config(text=f'{self.threshold_var.get()*100:.0f}%'))
        self.recursive_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text='包含子文件夹', variable=self.recursive_var).pack(side='left', padx=12)
        ttk.Label(opts, text='前缀：').pack(side='left')
        self.prefix_var = tk.StringVar(value='Group')
        ttk.Entry(opts, textvariable=self.prefix_var, width=12).pack(side='left', padx=5)
        ttk.Button(opts, text='开始分析', command=self.start_scan).pack(side='right')

        self.progress = ttk.Progressbar(self, mode='determinate')
        self.progress.pack(fill='x', padx=8, pady=3)
        self.status = ttk.Label(self, text='请选择图片文件夹。')
        self.status.pack(fill='x', padx=8, pady=3)

        mid = ttk.Frame(self)
        mid.pack(fill='both', expand=True, padx=8, pady=6)
        self.tree = ttk.Treeview(mid, columns=('group','old','new','count'), show='headings')
        for col, text, width in [('group','分组','90'), ('old','原文件名','320'), ('new','新文件名','320'), ('count','组内数量','90')]:
            self.tree.heading(col, text=text)
            self.tree.column(col, width=int(width), anchor='w')
        ys = ttk.Scrollbar(mid, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=ys.set)
        self.tree.pack(side='left', fill='both', expand=True)
        ys.pack(side='right', fill='y')

        bottom = ttk.Frame(self)
        bottom.pack(fill='x', padx=8, pady=8)
        ttk.Button(bottom, text='导出 CSV 预览', command=self.export_csv).pack(side='left')
        ttk.Button(bottom, text='执行重命名', command=self.execute).pack(side='right')
        ttk.Label(bottom, text='提示：执行前建议先导出 CSV；程序采用临时文件名过渡，避免重名覆盖。').pack(side='left', padx=15)

    def choose(self):
        folder = filedialog.askdirectory(title='选择图片文件夹')
        if folder:
            self.folder = folder
            self.folder_var.set(folder)

    def start_scan(self):
        folder = self.folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror('错误', '请选择有效的图片文件夹。')
            return
        self.folder = folder
        self.tree.delete(*self.tree.get_children())
        self.progress['value'] = 0
        self.status.config(text='正在读取图片……')
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self):
        try:
            paths = list_images(self.folder, self.recursive_var.get())
            if not paths:
                self.q.put(('error', '没有找到支持的图片文件。'))
                return
            items = []
            for i, p in enumerate(paths):
                try:
                    with Image.open(p) as im:
                        im.load()
                        feat = visual_feature(im)
                    items.append((p, feat))
                except Exception:
                    continue
                self.q.put(('status', (i + 1) / len(paths), f'读取图片 {i+1}/{len(paths)}'))
            if not items:
                self.q.put(('error', '图片无法读取。'))
                return
            threshold = float(self.threshold_var.get())
            self.q.put(('status', 0, f'开始计算相似度，共 {len(items)} 张……'))
            groups = make_groups(items, threshold, lambda x: self.q.put(('progress', x, f'计算相似度 {x*100:.0f}%')))
            self.groups = groups
            self.plan = build_plan(groups, self.prefix_var.get().strip() or 'Group')
            self.q.put(('done', len(items), len(groups)))
        except Exception as e:
            self.q.put(('error', repr(e)))

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == 'status':
                    self.progress['value'] = msg[1] * 100
                    self.status.config(text=msg[2])
                elif kind == 'progress':
                    self.progress['value'] = msg[1] * 100
                    self.status.config(text=msg[2])
                elif kind == 'done':
                    self.progress['value'] = 100
                    self.render_plan()
                    self.status.config(text=f'完成：{msg[1]} 张图片，分成 {msg[2]} 组。')
                elif kind == 'error':
                    messagebox.showerror('分析失败', msg[1])
                    self.status.config(text='分析失败。')
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def render_plan(self):
        self.tree.delete(*self.tree.get_children())
        for old, new, g, idx, count in self.plan:
            self.tree.insert('', 'end', values=(f'Group-{g:03d}', old.name, new, count))

    def export_csv(self):
        if not self.plan:
            messagebox.showwarning('提示', '请先分析图片。')
            return
        out = filedialog.asksaveasfilename(title='保存重命名预览', defaultextension='.csv', filetypes=[('CSV', '*.csv')], initialfile='rename_preview.csv')
        if not out:
            return
        with open(out, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['group', 'old_path', 'new_name', 'group_size'])
            for old, new, g, idx, count in self.plan:
                w.writerow([f'Group-{g:03d}', str(old), new, count])
        self.status.config(text=f'已导出：{out}')

    def execute(self):
        if not self.plan:
            messagebox.showwarning('提示', '请先分析图片。')
            return
        answer = messagebox.askyesno('确认重命名', '这将修改文件名，但不会删除图片。\n\n是否继续？')
        if not answer:
            return
        try:
            # Two-phase rename: first move all originals to unique temporary names.
            temp = []
            for i, (old, new, *_rest) in enumerate(self.plan):
                t = old.with_name(f'.__imgsim_tmp_{i:06d}__{old.name}')
                os.replace(old, t)
                temp.append((t, old.with_name(new)))
            for t, dest in temp:
                if dest.exists():
                    raise FileExistsError(f'目标文件已经存在：{dest}')
                os.replace(t, dest)
            self.status.config(text=f'重命名完成：{len(self.plan)} 张图片。')
            messagebox.showinfo('完成', f'已重命名 {len(self.plan)} 张图片。')
            self.start_scan()
        except Exception as e:
            messagebox.showerror('重命名失败', f'{e}\n\n部分文件可能已经改名；请检查文件夹和 CSV 预览。')


if __name__ == '__main__':
    App().mainloop()
