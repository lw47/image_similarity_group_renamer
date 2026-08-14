import os
import subprocess
import csv
import math
import threading
import queue
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
from PIL import Image, ImageOps, ImageTk, ImageFilter
from scipy.fftpack import dct
from scipy.ndimage import sobel

SUPPORTED = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tif', '.tiff'}
SCRIPT_DIR = Path(__file__).resolve().parent
LAST_FOLDER_FILE = SCRIPT_DIR / 'last_folder.txt'


REQUIREMENTS_FILE = SCRIPT_DIR / 'requirements.txt'
REQUIREMENTS_CONTENT = """numpy
Pillow
scipy
"""


def ensure_requirements_file():
    """Generate requirements.txt beside the script on first launch if missing."""
    try:
        if not REQUIREMENTS_FILE.exists():
            REQUIREMENTS_FILE.write_text(REQUIREMENTS_CONTENT, encoding='utf-8')
            log(f'首次启动：已生成 requirements.txt: {REQUIREMENTS_FILE}')
        else:
            log(f'已检测 requirements.txt: {REQUIREMENTS_FILE}')
    except Exception:
        log('生成/读取 requirements.txt 失败:')
        traceback.print_exc()


def log(msg):
    print(f'[{datetime.now():%H:%M:%S}] {msg}', flush=True)


def phash(img, hash_size=8, highfreq=4):
    size = hash_size * highfreq
    gray = ImageOps.exif_transpose(img).convert('L').resize(
        (size, size), Image.Resampling.LANCZOS
    )
    arr = np.asarray(gray, dtype=np.float32)
    d = dct(dct(arr, axis=0, norm='ortho'), axis=1, norm='ortho')[:hash_size, :hash_size]
    med = np.median(d[1:, :])
    return (d > med).flatten()


def dhash(img, size=16):
    gray = ImageOps.exif_transpose(img).convert('L').resize(
        (size + 1, size), Image.Resampling.LANCZOS
    )
    arr = np.asarray(gray, dtype=np.float32)
    return (arr[:, 1:] > arr[:, :-1]).flatten()


def visual_feature(img):
    src = ImageOps.exif_transpose(img).convert('RGB')
    w, h = src.size
    ratio = min(w, h) / max(w, h) if max(w, h) else 1.0
    thumb = ImageOps.contain(src, (48, 64), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (48, 64), (0, 0, 0))
    canvas.paste(thumb, ((48 - thumb.width) // 2, (64 - thumb.height) // 2))
    gray_raw = np.asarray(canvas.convert('L'), dtype=np.float32) / 255.0
    gray = (gray_raw - gray_raw.mean()) / (gray_raw.std() + 1e-6)
    gx = sobel(gray_raw, axis=1)
    gy = sobel(gray_raw, axis=0)
    edge = np.hypot(gx, gy)
    edge = edge / (edge.max() + 1e-6)
    hsv = np.asarray(canvas.convert('HSV'), dtype=np.float32)
    hist_h, _ = np.histogram(hsv[..., 0], bins=12, range=(0, 255), density=True)
    hist_s, _ = np.histogram(hsv[..., 1], bins=8, range=(0, 255), density=True)
    hist_v, _ = np.histogram(hsv[..., 2], bins=8, range=(0, 255), density=True)
    return {
        'p': phash(src),
        'd': dhash(src),
        'g': gray.flatten(),
        'raw': gray_raw.flatten(),
        'edge': edge.flatten(),
        'hist': np.concatenate([hist_h, hist_s, hist_v]),
        'ratio': ratio,
        'size': src.size,
    }


def hamming(a, b):
    return float(np.mean(a != b))


def cosine_distance(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / den)


def similarity(f1, f2):
    p = hamming(f1['p'], f2['p'])
    d = hamming(f1['d'], f2['d'])
    g = min(2.0, cosine_distance(f1['g'], f2['g'])) / 2.0
    raw = min(1.0, float(np.mean(np.abs(f1['raw'] - f2['raw']))))
    edge = min(1.0, float(np.mean(np.abs(f1['edge'] - f2['edge']))))
    hist = min(2.0, cosine_distance(f1['hist'], f2['hist'])) / 2.0
    ratio = min(1.0, abs(math.log((f1['ratio'] + 1e-6) / (f2['ratio'] + 1e-6))) / 1.5)
    dist = 0.28 * p + 0.18 * d + 0.22 * g + 0.16 * raw + 0.10 * edge + 0.04 * hist + 0.02 * ratio
    return max(0.0, min(1.0, 1.0 - dist))


def list_images(folder, recursive=False):
    root = Path(folder)
    it = root.rglob('*') if recursive else root.iterdir()
    return sorted(
        [p for p in it if p.is_file() and p.suffix.lower() in SUPPORTED],
        key=lambda p: p.name.lower()
    )


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
    for i in range(n):
        fi = items[i][1]
        for j in range(i + 1, n):
            fj = items[j][1]
            if abs(math.log((fi['ratio'] + 1e-6) / (fj['ratio'] + 1e-6))) <= 1.2:
                if similarity(fi, fj) >= threshold:
                    union(i, j)
            done += 1
            if progress and (done == total or done % max(1, total // 100) == 0):
                progress(done / max(1, total))

    groups = defaultdict(list)
    for i, (path, feat) in enumerate(items):
        groups[find(i)].append((path, feat))

    result = []
    for members in groups.values():
        rep = members[0]
        ordered = sorted(
            members,
            key=lambda x: (-similarity(rep[1], x[1]), x[0].name.lower())
        )
        result.append(ordered)

    result.sort(key=lambda g: g[0][0].name.lower())
    return result


def build_plan(groups, prefix='Group'):
    rows = []
    for group_num, group in enumerate(groups, 1):
        for idx, (path, _feat) in enumerate(group, 1):
            new_name = f'{prefix}-{group_num:03d}-{idx:02d}{path.suffix.lower()}'
            rows.append((path, new_name, group_num, idx, len(group)))
    return rows


class App(tk.Tk):
    def _load_last_folder(self):
        try:
            if LAST_FOLDER_FILE.exists():
                folder = LAST_FOLDER_FILE.read_text(encoding='utf-8').strip()
                if folder and os.path.isdir(folder):
                    self.folder = folder
                    self.folder_var.set(folder)
                    log(f'恢复上次文件夹: {folder}')
                elif folder:
                    log(f'上次文件夹不存在，未恢复: {folder}')
        except Exception:
            log('读取上次文件夹失败:')
            traceback.print_exc()

    def _save_last_folder(self, folder):
        try:
            folder = str(Path(folder).resolve())
            LAST_FOLDER_FILE.write_text(folder, encoding='utf-8')
            log(f'已保存上次文件夹: {folder}')
        except Exception:
            log('保存上次文件夹失败:')
            traceback.print_exc()

    def __init__(self):
        super().__init__()
        ensure_requirements_file()
        self.title('图片相似度分组与重命名')
        self.geometry('1200x760')
        self.minsize(1000, 650)
        self.folder = ''
        self.groups = []
        self.plan = []
        self.q = queue.Queue()
        self.row_paths = {}
        self.preview_photo = None
        self.last_preview_path = None
        self._build()
        self._load_last_folder()
        self.after(100, self._poll)
        log('程序启动')

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
        self.threshold_var = tk.IntVar(value=82)
        # 先创建并挂载标签，再创建 Scale 的初始值，避免 ttk.Scale.set() 在
        # 标签尚未创建时触发 command 回调。
        self.threshold_label = ttk.Label(opts, text='82%', width=5)
        self.threshold_label.pack(side='left')
        self.threshold_scale = ttk.Scale(
            opts, from_=65, to=97, orient='horizontal', length=240,
            command=self.on_threshold_move
        )
        self.threshold_scale.pack(side='left', padx=5, before=self.threshold_label)
        self.threshold_scale.set(82)
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

        paned = ttk.Panedwindow(self, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=8, pady=6)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=4)
        paned.add(right, weight=2)

        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill='both', expand=True)
        self.tree = ttk.Treeview(
            tree_frame,
            columns=('group', 'old', 'new', 'count', 'resolution', 'filesize'),
            show='headings',
            selectmode='browse'
        )
        for col, text, width in [
            ('group', '分组', 90),
            ('old', '原文件名', 173),
            ('new', '新文件名', 173),
            ('count', '组内序号', 90),
            ('resolution', '尺寸', 120),
            ('filesize', '文件大小(KB)', 100),
        ]:
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor='w')
        ys = ttk.Scrollbar(tree_frame, orient='vertical', command=self.tree.yview)
        xs = ttk.Scrollbar(tree_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.tree.grid(row=0, column=0, sticky='nsew')
        ys.grid(row=0, column=1, sticky='ns')
        xs.grid(row=1, column=0, sticky='ew')
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        # 高对比度：分组标题深色、图片行浅色，并设置字体。
        self.tree.tag_configure('separator_blank', background='#D8D8D8', foreground='#D8D8D8')
        self.tree.tag_configure('image', background='#FFFFFF', foreground='#111111')
        self.tree.tag_configure('image_selected', background='#CFE8FF', foreground='#000000')
        self.tree.bind('<Button-1>', self.on_tree_click, add='+')
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select, add='+')
        self.tree.bind('<KeyPress-Left>', self.on_tree_key, add='+')
        self.tree.bind('<KeyPress-Right>', self.on_tree_key, add='+')
        self.tree.bind('<KeyRelease-Left>', lambda e: 'break')
        self.tree.bind('<KeyRelease-Right>', lambda e: 'break')
        self.tree.bind('<Double-Button-1>', self.on_tree_double_click, add='+')
        self.tree.bind('<Button-3>', self.on_tree_right_click, add='+')

        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(label='用系统默认程序打开这张图片', command=self.context_open_image)
        self.context_menu.add_command(label='在文件资源管理器中定位文件', command=self.context_reveal_in_explorer)
        self.context_menu.add_command(label='复制完整路径', command=self.context_copy_full_path)
        self.context_menu.add_command(label='复制文件名', command=self.context_copy_filename)
        self.context_menu.add_separator()
        self.context_menu.add_command(label='打开所在文件夹', command=self.context_open_folder)
        self.context_menu.add_separator()
        self.context_menu.add_command(label='生成传统 ASCII 画', command=self.context_ascii_art)
        self.context_menu.add_command(label='生成高清字符画', command=self.context_unicode_art)
        self.context_menu.add_command(label='生成彩色高清字符画', command=self.context_color_art)
        self.context_menu.add_separator()
        self.context_menu.add_command(
            label='永久删除（不进入回收站）',
            command=self.context_delete_permanently,
            foreground='#D00000',
            activeforeground='#FFFFFF',
            activebackground='#D00000',
            font=('TkDefaultFont', 10, 'bold')
        )

        self.preview_canvas = tk.Canvas(right, background='#202020', highlightthickness=1, highlightbackground='#555555')
        self.preview_canvas.pack(fill='both', expand=True, padx=0, pady=0)
        self.preview_canvas.bind('<Configure>', lambda _e: self.refresh_preview())

        bottom = ttk.Frame(self)
        bottom.pack(fill='x', padx=8, pady=8)
        ttk.Button(bottom, text='导出 CSV 预览', command=self.export_csv).pack(side='left')
        ttk.Button(bottom, text='执行重命名', command=self.execute).pack(side='right')
        ttk.Label(bottom, text='日志直接输出到当前 CMD / PowerShell。').pack(side='left', padx=15)

    def on_threshold_move(self, value):
        try:
            percent = int(round(float(value)))
        except (TypeError, ValueError):
            return
        percent = max(65, min(97, percent))
        self.threshold_var.set(percent)
        # 不在 command 回调里再次调用 Scale.set()，否则会造成递归回调。
        self.threshold_label.config(text=f'{percent}%')

    def choose(self):
        log('打开文件夹选择器')
        folder = filedialog.askdirectory(title='选择图片文件夹')
        if folder:
            self.folder = folder
            self.folder_var.set(folder)
            self._save_last_folder(folder)
            log(f'选择文件夹: {folder}')

    def start_scan(self):
        log('点击“开始分析”')
        folder = self.folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror('错误', '请选择有效的图片文件夹。')
            return
        self.folder = folder
        self._save_last_folder(folder)
        self.tree.delete(*self.tree.get_children())
        self.row_paths.clear()
        self.clear_preview()
        self.progress['value'] = 0
        self.status.config(text='正在读取图片……')
        threshold = self.threshold_var.get() / 100.0
        prefix = self.prefix_var.get().strip() or 'Group'
        log(f'扫描参数: folder={folder!r}, threshold={threshold:.2f}, recursive={self.recursive_var.get()}, prefix={prefix!r}')
        threading.Thread(target=self._scan_worker, args=(threshold, prefix), daemon=True).start()

    def _scan_worker(self, threshold, prefix):
        log('后台分析线程启动')
        try:
            paths = list_images(self.folder, self.recursive_var.get())
            log(f'找到 {len(paths)} 张支持的图片')
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
                except Exception as e:
                    log(f'跳过无法读取的图片: {p} ; {e}')
                self.q.put(('status', (i + 1) / len(paths), f'读取图片 {i + 1}/{len(paths)}'))

            if not items:
                self.q.put(('error', '图片无法读取。'))
                return

            self.q.put(('status', 0, f'开始计算相似度，共 {len(items)} 张……'))
            groups = make_groups(items, threshold, lambda x: self.q.put(('progress', x, f'计算相似度 {x * 100:.0f}%')))
            self.groups = groups
            self.plan = build_plan(groups, prefix)
            log(f'重命名前缀已锁定: {prefix!r}')
            log(f'分析完成: {len(items)} 张图片, {len(groups)} 组')
            for gi, group in enumerate(groups, 1):
                log(f'  Group-{gi:03d}: {len(group)} 张')
            self.q.put(('done', len(items), len(groups)))
        except Exception:
            log('分析线程异常:')
            traceback.print_exc()
            self.q.put(('error', traceback.format_exc()))

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
                    log(f'ERROR: {msg[1]}')
                    messagebox.showerror('分析失败', msg[1])
                    self.status.config(text='分析失败。')
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def render_plan(self):
        log(f'开始渲染结果列表，共 {len(self.plan)} 条')
        self.tree.delete(*self.tree.get_children())
        self.row_paths.clear()
        current_group = None

        for old, new, g, idx, count in self.plan:
            if current_group is not None and g != current_group:
                # 分组标题是不可选中的特殊行。
                self.tree.insert(
                    '', 'end',
                    values=('', '', '', '', '', ''),
                    tags=('separator_blank',)
                )

            try:
                with Image.open(old) as _im:
                    _im = ImageOps.exif_transpose(_im)
                    resolution = f'{_im.width} × {_im.height}'
            except Exception:
                resolution = '-'
            try:
                filesize_kb = old.stat().st_size / 1024
                filesize = f'{filesize_kb:.2f}'
            except OSError:
                filesize = '-'

            iid = self.tree.insert(
                '', 'end',
                values=(f'Group-{g:03d}', old.name, new, f'{idx}/{count}', resolution, filesize),
                tags=('image',)
            )
            self.row_paths[iid] = old
            current_group = g

        log(f'结果列表渲染完成：{len(self.tree.get_children())} 行（含分组标题行）')
        # 默认选中第一张图片，便于直接使用键盘左右键。
        first = self._image_row_ids()
        if first:
            self._select_image_row(first[0], preview=True)
            self.after_idle(self.tree.focus_set)

    def _image_row_ids(self):
        return [iid for iid in self.tree.get_children('') if iid in self.row_paths]

    def _select_image_row(self, iid, preview=True):
        if iid not in self.row_paths:
            return False
        self.tree.selection_set(iid)
        self.tree.focus(iid)
        self.tree.focus_set()
        self.tree.see(iid)
        if preview:
            path = self.row_paths[iid]
            log(f'切换图片: {iid} -> {path}')
            self.show_preview(path)
        return True

    def on_tree_click(self, event):
        row = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        log(f'Tree 点击: row={row!r}, column={col!r}')
        # 分组标题行完全不可选中。
        if row and row not in self.row_paths:
            log('点击分组标题，不允许选择')
            self.tree.selection_remove(self.tree.selection())
            return 'break'
        if row in self.row_paths:
            self.tree.focus_set()
            self._select_image_row(row, preview=True)
            return 'break'
        return None

    def on_tree_double_click(self, event):
        row = self.tree.identify_row(event.y)
        if row not in self.row_paths:
            return 'break'
        path = self.row_paths[row]
        log(f'双击打开系统图片工具: {path}')
        try:
            os.startfile(str(path))
            log('系统图片工具调用成功')
        except Exception:
            log('系统图片工具调用失败')
            traceback.print_exc()
        return 'break'

    def _context_selected_path(self):
        row = getattr(self, '_context_row', None)
        if row in self.row_paths:
            return Path(self.row_paths[row])
        return None

    def on_tree_right_click(self, event):
        row = self.tree.identify_row(event.y)
        log(f'Tree 右键: row={row!r}')
        if row not in self.row_paths:
            self._context_row = None
            try:
                self.context_menu.unpost()
            except tk.TclError:
                pass
            return 'break'

        self._context_row = row
        self._select_image_row(row, preview=True)
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()
        return 'break'

    def context_open_image(self):
        path = self._context_selected_path()
        if not path:
            return
        log(f'右键菜单：打开图片: {path}')
        try:
            os.startfile(str(path))
        except Exception:
            log('打开图片失败')
            traceback.print_exc()

    def context_reveal_in_explorer(self):
        path = self._context_selected_path()
        if not path:
            return
        log(f'右键菜单：定位文件: {path}')
        try:
            subprocess.Popen(['explorer', '/select,', str(path)])
        except Exception:
            log('定位文件失败')
            traceback.print_exc()

    def context_copy_full_path(self):
        path = self._context_selected_path()
        if not path:
            return
        value = str(path.resolve())
        self.clipboard_clear()
        self.clipboard_append(value)
        self.update()
        self.status.config(text='已复制完整路径。')
        log(f'右键菜单：复制完整路径: {value}')

    def context_copy_filename(self):
        path = self._context_selected_path()
        if not path:
            return
        value = path.name
        self.clipboard_clear()
        self.clipboard_append(value)
        self.update()
        self.status.config(text='已复制文件名。')
        log(f'右键菜单：复制文件名: {value}')

    def context_open_folder(self):
        path = self._context_selected_path()
        if not path:
            return
        log(f'右键菜单：打开所在文件夹: {path.parent}')
        try:
            os.startfile(str(path.parent))
        except Exception:
            log('打开所在文件夹失败')
            traceback.print_exc()

    def context_delete_permanently(self):
        path = self._context_selected_path()
        if not path:
            return
        log(f'右键菜单：永久删除: {path}')
        answer = messagebox.askyesno(
            '永久删除确认',
            f'确定要永久删除此文件吗？\n\n{path}\n\n此操作不会进入回收站，无法通过回收站恢复。',
            icon='warning'
        )
        if not answer:
            log('永久删除：用户取消')
            return
        try:
            path.unlink()
            log(f'永久删除成功: {path}')
            iid = getattr(self, '_context_row', None)
            if iid in self.row_paths:
                self.row_paths.pop(iid, None)
                try:
                    self.tree.delete(iid)
                except tk.TclError:
                    pass
            self._context_row = None
            self.status.config(text='已永久删除文件。')
            self.clear_preview()
        except Exception:
            log('永久删除失败')
            traceback.print_exc()
            messagebox.showerror('删除失败', f'无法永久删除文件：\n{path}')

    def _load_ascii_source(self, path, target_width=120, max_height=100):
        with Image.open(path) as original:
            im = ImageOps.exif_transpose(original).convert('RGB')
            src_w, src_h = im.size
            if src_w <= 0 or src_h <= 0:
                raise ValueError('无效图片尺寸')
            # Conservative photographic preprocessing.
            gray = ImageOps.autocontrast(im.convert('L'), cutoff=0.35)
            gray = gray.filter(ImageFilter.UnsharpMask(radius=0.8, percent=55, threshold=5))
            # Terminal glyphs are roughly twice as tall as they are wide.
            char_aspect = 0.50
            out_w = max(2, min(int(target_width), src_w))
            out_h = max(2, int(round(out_w * src_h / src_w * char_aspect)))
            if out_h > max_height:
                out_h = int(max_height)
                out_w = max(2, int(round(out_h * src_w / src_h / char_aspect)))
            rgb = im.resize((out_w, out_h), Image.Resampling.LANCZOS)
            gray = gray.resize((out_w, out_h), Image.Resampling.LANCZOS)
            arr = np.asarray(gray, dtype=np.float32) / 255.0
            # Mild gamma preserves photographic midtones.
            arr = np.power(np.clip(arr, 0.0, 1.0), 0.95)
            return rgb, arr, out_w, out_h

    @staticmethod
    def _floyd_levels(arr, levels):
        work = arr.copy().astype(np.float32)
        out = np.empty(work.shape, dtype=np.int32)
        den = max(1, levels - 1)
        h, w = work.shape
        for y in range(h):
            for x in range(w):
                old = float(np.clip(work[y, x], 0.0, 1.0))
                q = int(round(old * den))
                q = max(0, min(den, q))
                out[y, x] = q
                err = old - q / den
                if x + 1 < w:
                    work[y, x + 1] += err * 7 / 16
                if y + 1 < h:
                    if x:
                        work[y + 1, x - 1] += err * 3 / 16
                    work[y + 1, x] += err * 5 / 16
                    if x + 1 < w:
                        work[y + 1, x + 1] += err * 1 / 16
        return out

    def _print_ascii_header(self, title, width):
        bar = '=' * max(12, min(140, width))
        print('\n' + bar + f'\n {title}\n' + bar, flush=True)

    def context_ascii_art(self):
        """传统黑白 ASCII：使用标准 ASCII 字符 + Floyd-Steinberg。"""
        path = self._context_selected_path()
        if not path:
            return
        log(f'右键菜单：生成传统 ASCII 画: {path}')
        chars = " .`^,:;Il!i~+_-?][}{1)(|/\\tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
        try:
            _rgb, arr, out_w, out_h = self._load_ascii_source(path)
            idx = self._floyd_levels(arr, len(chars))
            self._print_ascii_header(f' ASCII ART: {path.name} ', out_w)
            for row in idx:
                print(''.join(chars[int(i)] for i in row), flush=True)
            print('=' * max(12, min(140, out_w)) + '\n', flush=True)
            self.status.config(text='传统 ASCII 画已输出到当前 CMD / PowerShell。')
            log(f'传统 ASCII 输出完成: {path}, size={out_w}x{out_h}, levels={len(chars)}')
        except Exception:
            log('传统 ASCII 画生成失败')
            traceback.print_exc()

    def context_unicode_art(self):
        """高清字符画：Unicode Braille，2x4 采样单元，保留更多空间细节。"""
        path = self._context_selected_path()
        if not path:
            return
        log(f'右键菜单：生成高清字符画: {path}')
        try:
            with Image.open(path) as original:
                im = ImageOps.exif_transpose(original).convert('L')
                src_w, src_h = im.size
                if src_w <= 0 or src_h <= 0:
                    raise ValueError('无效图片尺寸')
                max_cols = 120
                cell_w, cell_h = 2, 4
                cols = max(2, min(max_cols, src_w // cell_w if src_w >= cell_w else 2))
                rows = max(2, int(round((cols * cell_w) * src_h / src_w / cell_h)))
                rows = min(rows, 90)
                pix_w, pix_h = cols * cell_w, rows * cell_h
                im = im.resize((pix_w, pix_h), Image.Resampling.LANCZOS)
                arr = np.asarray(im, dtype=np.float32) / 255.0
                arr = np.power(np.clip(arr, 0.0, 1.0), 0.92)
                # Bright regions become background; dark regions activate braille dots.
                threshold = self._floyd_levels(1.0 - arr, 2)

            dot_bits = ((0,0,1),(0,1,2),(0,2,4),(0,3,64),(1,0,8),(1,1,16),(1,2,32),(1,3,128))
            self._print_ascii_header(f' HIGH-RES UNICODE: {path.name} ', cols)
            for cy in range(rows):
                line=[]
                for cx in range(cols):
                    mask=0
                    for dx,dy,bit in dot_bits:
                        if threshold[cy*4+dy, cx*2+dx] > 0:
                            mask |= bit
                    line.append(chr(0x2800 + mask) if mask else ' ')
                print(''.join(line).rstrip(), flush=True)
            print('=' * max(12, min(140, cols)) + '\n', flush=True)
            self.status.config(text='高清 Unicode 字符画已输出到当前 CMD / PowerShell。')
            log(f'高清字符画输出完成: {path}, cells={cols}x{rows}, glyph=Braille')
        except Exception:
            log('高清字符画生成失败')
            traceback.print_exc()

    @staticmethod
    def _enable_windows_vt():
        if os.name != 'nt':
            return
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass

    def context_color_art(self):
        """彩色高清字符画：ANSI 真彩 + 半块字符，每个字符携带上下两个颜色采样。"""
        path = self._context_selected_path()
        if not path:
            return
        log(f'右键菜单：生成彩色高清字符画: {path}')
        try:
            self._enable_windows_vt()
            with Image.open(path) as original:
                im = ImageOps.exif_transpose(original).convert('RGB')
                src_w, src_h = im.size
                if src_w <= 0 or src_h <= 0:
                    raise ValueError('无效图片尺寸')
                cols = max(2, min(100, src_w))
                # One '▀' represents two vertical color samples.
                rows = max(2, int(round(cols * src_h / src_w / 2)))
                rows = min(rows, 80)
                pix_w, pix_h = cols, rows * 2
                im = im.resize((pix_w, pix_h), Image.Resampling.LANCZOS)
                arr = np.asarray(im, dtype=np.uint8)

            self._print_ascii_header(f' TRUECOLOR SYMBOL ART: {path.name} ', cols)
            reset='\x1b[0m'
            for y in range(0, pix_h, 2):
                parts=[]
                for x in range(pix_w):
                    r1,g1,b1=map(int, arr[y,x])
                    if y+1 < pix_h:
                        r2,g2,b2=map(int, arr[y+1,x])
                    else:
                        r2,g2,b2=r1,g1,b1
                    parts.append(f'\x1b[38;2;{r1};{g1};{b1}m\x1b[48;2;{r2};{g2};{b2}m▀')
                print(''.join(parts)+reset, flush=True)
            print('\x1b[0m' + '=' * max(12, min(140, cols)) + '\n', flush=True)
            self.status.config(text='彩色高清字符画已输出到当前 CMD / PowerShell。')
            log(f'彩色高清字符画输出完成: {path}, cells={cols}x{rows}, ANSI=24bit')
        except Exception:
            log('彩色高清字符画生成失败')
            traceback.print_exc()

    def on_tree_select(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return
        row = sel[0]
        if row in self.row_paths:
            log(f'Tree 选择事件: {row} -> {self.row_paths[row]}')
            self.show_preview(self.row_paths[row])
        else:
            # 理论上不应发生；若发生立即清掉选择。
            self.tree.selection_remove(row)

    def on_tree_key(self, event):
        rows = self._image_row_ids()
        if not rows:
            return 'break'
        current = self.tree.focus()
        try:
            idx = rows.index(current)
        except ValueError:
            sel = self.tree.selection()
            try:
                idx = rows.index(sel[0])
            except (ValueError, IndexError):
                idx = 0 if event.keysym == 'Right' else len(rows) - 1
        if event.keysym == 'Right':
            idx = min(len(rows) - 1, idx + 1)
        elif event.keysym == 'Left':
            idx = max(0, idx - 1)
        self._select_image_row(rows[idx], preview=True)
        return 'break'

    def clear_preview(self):
        self.preview_canvas.delete('all')
        self.preview_photo = None
        self.last_preview_path = None

    def show_preview(self, path):
        path = Path(path)
        log(f'Preview request: {path}')
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                original_size = im.size
                self.last_preview_path = path
                self._preview_source = im.copy()
            self.refresh_preview()
            log(f'Preview loaded: {path}, size={original_size}')
        except Exception:
            log(f'Preview failed: {path}')
            traceback.print_exc()
            self.preview_canvas.delete('all')
            self.preview_photo = None

    def refresh_preview(self):
        if not hasattr(self, '_preview_source') or self._preview_source is None:
            return
        source = self._preview_source
        cw = max(10, self.preview_canvas.winfo_width())
        ch = max(10, self.preview_canvas.winfo_height())
        sw, sh = source.size
        scale = min(cw / sw, ch / sh)
        nw = max(1, int(sw * scale))
        nh = max(1, int(sh * scale))
        if (nw, nh) != source.size:
            preview = source.resize((nw, nh), Image.Resampling.LANCZOS)
        else:
            preview = source.copy()
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.preview_canvas.delete('all')
        self.preview_canvas.create_image(cw // 2, ch // 2, image=self.preview_photo, anchor='center')

    def export_csv(self):
        log('点击“导出 CSV 预览”')
        if not self.plan:
            messagebox.showwarning('提示', '请先分析图片。')
            return
        out = filedialog.asksaveasfilename(
            title='保存重命名预览',
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv')],
            initialfile='rename_preview.csv'
        )
        if not out:
            return
        with open(out, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['group', 'old_path', 'new_name', 'group_size'])
            for old, new, g, idx, count in self.plan:
                w.writerow([f'Group-{g:03d}', str(old), new, count])
        self.status.config(text=f'已导出：{out}')
        log(f'CSV 已导出: {out}')

    def execute(self):
        log('点击“执行重命名”')
        if not self.plan:
            messagebox.showwarning('提示', '请先分析图片。')
            return
        if not messagebox.askyesno('确认重命名', '这将修改文件名，但不会删除图片。\n\n是否继续？'):
            return
        try:
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
            log(f'重命名完成: {len(self.plan)} 张')
            messagebox.showinfo('完成', f'已重命名 {len(self.plan)} 张图片。')
        except Exception as e:
            log(f'重命名失败: {e}')
            traceback.print_exc()
            messagebox.showerror('重命名失败', f'{e}\n\n部分文件可能已经改名；请检查文件夹和 CSV 预览。')


if __name__ == '__main__':
    log('========== 图片相似度分组工具启动 ==========')
    try:
        app = App()
        app.mainloop()
    except Exception:
        log('程序启动/运行异常:')
        traceback.print_exc()
        raise
