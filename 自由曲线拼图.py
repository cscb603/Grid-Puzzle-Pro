# -*- coding: utf-8 -*-
import sys
import os
import re
import subprocess
import numpy as np
import warnings
import random
import time
import colorsys
import concurrent.futures
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QListWidget, QLabel,
    QMessageBox, QListWidgetItem, QProgressBar, QCheckBox, QComboBox
)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QSize, QMimeData
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance, ImageFilter
from PIL import ImageFile
import psutil

# 忽略PIL的EXIF警告
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------
# 智能作者名提取算法
# -----------------------
def extract_author_label(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    
    PREFIXES = [
        r'手机班摄影班',  # 长前缀优先
        r'手机班',
        r'影友'
    ]
    
    SEPARATORS = [
        r'作业',
        r'作品'
    ]
    
    prefix = None
    remaining_name = name
    for p in PREFIXES:
        pattern = rf'^{p}\s+'
        match = re.match(pattern, name)
        if match:
            prefix = p
            remaining_name = name[match.end():].strip()
            break
    
    separator = None
    author_part = remaining_name
    for s in SEPARATORS:
        pattern = rf'.*?({s})'
        match = re.search(pattern, remaining_name)
        if match:
            separator = s
            author_part = remaining_name[:match.start()].strip()
            break
    
    if not author_part:
        parts = remaining_name.split()
        if parts:
            author_part = parts[0]
        else:
            return "未知作者 作品"
    
    if re.match(r'^[A-Za-z]+\.[^ ]+', author_part):
        author = author_part
    else:
        author = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\.\-_]', '', author_part)
    
    if prefix:
        return f"{prefix} {author} 作品"
    else:
        return f"{author} 作品"

def draw_author_tag(img, text, column_width):
    if img is None:
        return None
    
    im = img.copy()
    
    font_size = max(14, column_width // 22)
    corner_radius = 40
    padding_x = max(20, font_size // 2)
    padding_y = max(12, font_size // 3)
    bottom_margin = 15
    opacity = 0.15
    
    try:
        font = ImageFont.truetype("msyh.ttc", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("msjh.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
    
    temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_img)
    
    try:
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_ascent = bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = temp_draw.textsize(text, font=font)
        text_ascent = text_h
    
    bg_w = text_w + padding_x * 2
    bg_h = text_h + padding_y * 2
    
    img_width, img_height = im.size
    x = (img_width - bg_w) // 2
    y = img_height - bg_h - bottom_margin
    y = max(0, y)
    
    temp_layer = Image.new('RGBA', im.size, (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(temp_layer)
    
    bg_color = (240, 240, 240)
    layer_draw.rounded_rectangle(
        [x, y, x + bg_w, y + bg_h],
        radius=corner_radius,
        fill=(*bg_color, int(255 * opacity))
    )
    
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    
    im = Image.alpha_composite(im, temp_layer)
    
    final_draw = ImageDraw.Draw(im)
    text_color = (50, 50, 50)
    
    text_x = x + padding_x
    text_y = y + (bg_h - text_ascent) // 2 + text_ascent - (text_h // 2)
    
    if not 'bbox' in locals():
        text_y = y + (bg_h - text_h) // 2
    
    final_draw.text(
        (text_x, text_y),
        text,
        fill=text_color,
        font=font
    )
    
    if im.mode == 'RGBA':
        im = im.convert('RGB')
    
    return im

# 取消图片曲线边界与自由分栏相关函数，简化为稳定的直线列瀑布流

class MemoryManager:
    @staticmethod
    def check_available_memory(threshold_mb=1500):
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)
        return available_mb >= threshold_mb, available_mb

class ImageProcessor:
    _cv_face_cascade = None
    _cv_available = False
    @staticmethod
    def process_image(path, column_width, rounded_corners, crop_3_4=False, resize_for_columns=True, max_side=None):
        try:
            with Image.open(path) as img:
                img = ImageProcessor._correct_orientation(img)
                if max_side:
                    img = ImageProcessor._downscale_if_large(img, max_side)
                if crop_3_4:
                    img = ImageProcessor._smart_crop_to_3_4(img)
                img = ImageProcessor._crop_extreme_ratio(img)
                if resize_for_columns:
                    img = ImageProcessor._resize_to_column(img, column_width)
                
                if rounded_corners:
                    radius = max(2, int(column_width * 0.10))
                    img = ImageProcessor._add_rounded_corners(img, radius)
                return img.copy()
        except Exception as e:
            print(f"处理图片 {path} 失败: {str(e)}")
            return None
    
    @staticmethod
    def _correct_orientation(img):
        try:
            exif = img._getexif() if hasattr(img, '_getexif') else None
            orientation = exif.get(0x0112, 1) if exif else 1
            if orientation > 1:
                rotation = {3: 180, 6: 270, 8: 90}.get(orientation, 0)
                return img.rotate(rotation, expand=True)
        except Exception:
            pass
        return img
    
    @staticmethod
    def _crop_extreme_ratio(img):
        width, height = img.size
        ratio = width / height
        if ratio > 2:
            new_width = int(height * 2)
            left = (width - new_width) // 2
            return img.crop((left, 0, left + new_width, height))
        elif ratio < 0.5:
            new_height = int(width * 2)
            top = (height - new_height) // 2
            return img.crop((0, top, width, top + new_height))
        return img
    
    @staticmethod
    def _downscale_if_large(img, max_side):
        try:
            w, h = img.size
            ms = int(max_side)
            if ms <= 0:
                return img
            m = max(w, h)
            if m > ms:
                scale = ms / float(m)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        except Exception:
            pass
        return img
    
    @staticmethod
    def _resize_to_column(img, column_width):
        scale = column_width / img.width
        new_size = (column_width, max(1, int(img.height * scale)))
        return img.resize(new_size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def _add_rounded_corners(img, radius):
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0, img.size[0], img.size[1]), radius=radius, fill=255)
        result = ImageOps.fit(img, mask.size, centering=(0.5, 0.5))
        result.putalpha(mask)
        return result
    
    @staticmethod
    def _crop_to_3_4(img):
        width, height = img.size
        target_width = int(height * 3 / 4)
        target_height = height
        if target_width > width:
            target_width = width
            target_height = int(width * 4 / 3)
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        return img.crop((left, top, right, bottom))

    @staticmethod
    def _smart_crop_to_3_4(img):
        try:
            import cv2
            if not ImageProcessor._cv_available:
                try:
                    face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    ImageProcessor._cv_face_cascade = cv2.CascadeClassifier(face_xml)
                    ImageProcessor._cv_available = True
                except Exception:
                    ImageProcessor._cv_face_cascade = None
                    ImageProcessor._cv_available = False
            w, h = img.size
            scale = 1.0
            max_side = max(w, h)
            if max_side > 360:
                scale = 360.0 / max_side
            small = img.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.Resampling.LANCZOS)
            arr = np.array(small.convert('RGB'))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            faces = []
            if ImageProcessor._cv_face_cascade is not None:
                faces = ImageProcessor._cv_face_cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=5, minSize=(30,30)
                )
            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda r: r[2]*r[3])
                cx = (x + fw/2) / scale
                cy = (y + fh/2) / scale
            else:
                edges = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
                m = np.maximum(edges, 0)
                s = m.sum()
                if s <= 0:
                    cx, cy = w*0.5, h*0.5
                else:
                    ys, xs = np.indices(m.shape)
                    cx = (xs*m).sum()/s/scale
                    cy = (ys*m).sum()/s/scale
        except Exception:
            w, h = img.size
            cx, cy = w*0.5, h*0.5
        aspect = 3/4
        w0, h0 = img.size
        tw = int(h0*aspect)
        th = h0
        if tw > w0:
            tw = w0
            th = int(w0/aspect)
        left = int(max(0, min(cx - tw/2, w0 - tw)))
        top = int(max(0, min(cy - th/2, h0 - th)))
        right = left + tw
        bottom = top + th
        return img.crop((left, top, right, bottom))

class WaterfallGeneratorPro:
    def __init__(self):
        self.column = 3
        self.spacing = 15
        self.max_width = 3000
        self.quality = 90
        self.column_width = (self.max_width - (self.column - 1) * self.spacing) // self.column
        self.parent_window = None
        # 布局样式：columns（直线列瀑布流）、rows（智能行组合1-3列）
        self.layout_style = 'columns'
        # 排序策略：balanced | landscape_first | portrait_first | random
        self.sort_strategy = 'balanced'
        self._last_opened_dir = None
        self._last_opened_ts = 0.0
        self._last_image_path = None
        self._last_image_ts = 0.0
        self._color_cache = {}
        self.input_max_side = 3600

    def _normalize_path(self, p):
        try:
            return os.path.normpath(os.path.abspath(os.path.expanduser(str(p))))
        except Exception:
            return str(p)
    
    def generate_waterfall(self, selected, parent_window, crop_bottom, rounded_corners, 
                          crop_3_4=False, align_bottom=False, tag_enabled=True):
        self.parent_window = parent_window
        valid_pairs = []
        
        try:
            seen = set()
            unique_selected = []
            for path in selected:
                norm_path = os.path.normpath(path).lower()
                if norm_path not in seen:
                    unique_selected.append(path)
                    seen.add(norm_path)
            
            if len(selected) != len(unique_selected):
                print(f"已自动剔除 {len(selected) - len(unique_selected)} 张重复图片")
            
            total_count = len(unique_selected)
            if total_count < self.column:
                raise ValueError(f"去重后图片不足，至少需要{self.column}张图片")
            
            ok, avail = MemoryManager.check_available_memory(threshold_mb=500)
            if not ok:
                raise MemoryError(f"内存不足（剩余约 {int(avail)} MB），处理中断")

            workers = max(2, min(4, (os.cpu_count() or 4)))
            completed = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futures = []
                for path in unique_selected:
                    resize_flag = (self.layout_style != 'rows')
                    futures.append(ex.submit(
                        ImageProcessor.process_image,
                        path, self.column_width, rounded_corners,
                        crop_3_4, resize_for_columns=resize_flag, max_side=self.input_max_side
                    ))
                for fut, path in zip(concurrent.futures.as_completed(futures), unique_selected):
                    img = fut.result()
                    if img:
                        valid_pairs.append((path, img))
                    completed += 1
                    if parent_window:
                        parent_window.progress_bar.setValue(int((completed / total_count) * 100))
                        QApplication.processEvents()
            
            self._validate_images(valid_pairs)
            
            if self.layout_style == 'rows':
                est_height = self._estimate_rows_height(valid_pairs)
                canvas = Image.new('RGBA', (self.max_width, est_height), (250, 250, 250, 255))
            else:
                temp_col_heights = self._calculate_column_heights_basic(valid_pairs)
                canvas = self._create_canvas(temp_col_heights)
            
            effective_tag = tag_enabled and (self.layout_style != 'rows')
            final_max_height = self._paste_images(canvas, valid_pairs, align_bottom, effective_tag)
            
            if self.layout_style != 'rows' and (crop_bottom or align_bottom):
                canvas = canvas.crop((0, 0, self.max_width, final_max_height))
            
            png_path = self._generate_output_path(unique_selected)
            print(f"准备生成文件: {png_path}")
            
            final_path = self._save_and_optimize(canvas, png_path)
            
            if not os.path.exists(final_path):
                raise Exception(f"生成的文件不存在: {final_path}")
            
            self._show_success_message(final_path)
            self.open_image(final_path)
            self.open_file_in_explorer(final_path)
            
        except MemoryError as e:
            self._show_message("内存不足", str(e), QMessageBox.Warning)
        except ValueError as e:
            self._show_message("错误", str(e), QMessageBox.Critical)
        except Exception as e:
            self._show_message("错误", f"生成失败: {str(e)}", QMessageBox.Critical)
            print(f"错误详情: {str(e)}")
        finally:
            for _, img in valid_pairs:
                if img:
                    img.close()
            if parent_window:
                parent_window.progress_bar.setValue(0)
                parent_window.progress_bar.setVisible(False)

    def set_columns(self, n):
        n = max(1, int(n))
        self.column = n
        self.column_width = (self.max_width - (self.column - 1) * self.spacing) // self.column

    def set_layout_style(self, style):
        self.layout_style = style if style in ('columns', 'rows') else 'columns'

    def set_sort_strategy(self, strategy):
        self.sort_strategy = strategy if strategy in (
            'balanced', 'landscape_first', 'portrait_first', 'random',
            'hue_gradient', 'value_transition', 'contrast_balance'
        ) else 'balanced'

    # 模板斜切已移除

    # 移除模式切换相关方法
    
    def _validate_images(self, valid_pairs):
        if not valid_pairs:
            raise ValueError("没有有效图片可供处理")
        if len(valid_pairs) < self.column:
            raise ValueError(f"至少需要{self.column}张图片")
    
    def _calculate_column_heights_basic(self, valid_pairs):
        col_heights = np.zeros(self.column, dtype=int)
        for _, img in valid_pairs:
            col_idx = np.argmin(col_heights)
            col_heights[col_idx] += img.height + self.spacing
        return col_heights
    
    def _create_canvas(self, col_heights):
        total_height = max(col_heights) if len(col_heights) else 0
        return Image.new('RGBA', (self.max_width, total_height + 500), (250, 250, 250, 255))
    
    def _paste_images(self, canvas, valid_pairs, align_bottom, tag_enabled):
        if self.layout_style == 'rows':
            return self._paste_images_rows(canvas, valid_pairs, tag_enabled)
        # 模板斜切模式已移除

        # 默认直线列瀑布流
        sorted_pairs = self._apply_sort_to_pairs(valid_pairs)
        col_groups = [[] for _ in range(self.column)]
        sim_col_heights = np.zeros(self.column, dtype=int)

        for path, img in sorted_pairs:
            col_idx = np.argmin(sim_col_heights)
            col_groups[col_idx].append((path, img))
            sim_col_heights[col_idx] += img.height + self.spacing

        max_height = int(max(sim_col_heights)) if len(sim_col_heights) else 0

        final_col_heights = []

        for col_idx, items in enumerate(col_groups):
            if not items:
                final_col_heights.append(0)
                continue

            x = col_idx * (self.column_width + self.spacing)
            y = 0.0
            current_spacing = self.spacing

            if align_bottom and len(items) > 1:
                target_bottom = max_height - self.spacing
                images_total_height = sum(img.height for _, img in items)
                total_gap_needed = target_bottom - images_total_height
                if total_gap_needed > 0:
                    current_spacing = total_gap_needed / (len(items) - 1)

            for path, img in items:
                if tag_enabled:
                    author_label = extract_author_label(path)
                    img_with_tag = draw_author_tag(img, author_label, self.column_width)
                    if img_with_tag:
                        img = img_with_tag

                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    canvas.paste(img, (x, int(y)), img)
                else:
                    canvas.paste(img, (x, int(y)))

                y += img.height + current_spacing

            final_col_heights.append(int(y - current_spacing))

        return max(final_col_heights) if final_col_heights else 0

    # 不再使用固定分配函数，统一走排序分组逻辑

    def _estimate_rows_height(self, valid_pairs):
        ordered = self._order_items_for_rows(valid_pairs)

        # rows building
        rows = []
        i = 0
        n = len(ordered)
        while i < n:
            rem = n - i
            if rem == 4:
                rows.append(ordered[i:i+2])
                i += 2
                rows.append(ordered[i:i+2])
                i += 2
                continue
            if rem == 3:
                rows.append(ordered[i:i+3])
                i += 3
                continue
            if rem == 2:
                rows.append(ordered[i:i+2])
                i += 2
                continue
            window = ordered[i:i+3]
            has_portrait = any(it['cat'] == 'P' for it in window)
            if has_portrait:
                rows.append(ordered[i:i+2])
                i += 2
            else:
                rows.append(ordered[i:i+3])
                i += 3

        # estimate height sum
        y = 0
        for row in rows:
            g = len(row)
            total_ratio = sum(it['img'].width / it['img'].height if it['img'].height else 1.0 for it in row)
            available_width = self.max_width - (g - 1) * self.spacing
            row_h = max(1, int(available_width / max(1e-6, total_ratio)))
            y += row_h + self.spacing
        return max(0, y - self.spacing)

    def _paste_images_rows(self, canvas, valid_pairs, tag_enabled):
        ordered = self._order_items_for_rows(valid_pairs)

        rows = []
        i = 0
        n = len(ordered)
        while i < n:
            rem = n - i
            if rem == 4:
                rows.append(ordered[i:i+2])
                i += 2
                rows.append(ordered[i:i+2])
                i += 2
                continue
            if rem == 3:
                rows.append(ordered[i:i+3])
                i += 3
                continue
            if rem == 2:
                rows.append(ordered[i:i+2])
                i += 2
                continue
            window = ordered[i:i+3]
            has_portrait = any(it['cat'] == 'P' for it in window)
            if has_portrait:
                rows.append(ordered[i:i+2])
                i += 2
            else:
                rows.append(ordered[i:i+3])
                i += 3

        y = 0
        final_height = 0
        for row in rows:
            g = len(row)
            # 计算该行的目标高度，使得按原比例缩放后恰好填满整行宽度
            total_ratio = sum(it['img'].width / it['img'].height if it['img'].height else 1.0 for it in row)
            available_width = self.max_width - (g - 1) * self.spacing
            row_h = max(1, int(available_width / max(1e-6, total_ratio)))

            # 按比例计算每张图的宽度，最后一张补齐精度误差
            widths = []
            accum = 0
            for idx, it in enumerate(row):
                r = it['img'].width / it['img'].height if it['img'].height else 1.0
                w = int(r * row_h) if idx < g - 1 else max(1, available_width - accum)
                widths.append(w)
                accum += w

            x = 0
            for i, it in enumerate(row):
                path, img = it['path'], it['img']
                w = widths[i]
                im = img.resize((w, row_h), Image.Resampling.LANCZOS)
                if tag_enabled:
                    author_label = extract_author_label(path)
                    im2 = draw_author_tag(im, author_label, w)
                    im = im2 if im2 else im
                canvas.paste(im, (x, y))
                x += w + self.spacing
            y += row_h + self.spacing
            final_height = y
        return max(0, final_height - self.spacing)

    def _get_color_feats(self, pth, im):
        norm_p = self._normalize_path(pth)
        try:
            st = os.stat(norm_p)
            key = (norm_p, int(st.st_mtime), st.st_size)
        except Exception:
            key = (norm_p, im.size)
        if key in self._color_cache:
            return self._color_cache[key]
        try:
            import cv2
            w, h = im.size
            max_dim = max(w, h)
            scale = 96.0 / max_dim if max_dim > 96 else 1.0
            small = im.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.Resampling.LANCZOS)
            arr = np.array(small.convert('RGB'))
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            H = hsv[:,:,0].astype(np.float32) / 179.0
            S = hsv[:,:,1].astype(np.float32) / 255.0
            V = hsv[:,:,2].astype(np.float32) / 255.0
            h_mean = float(H.mean()); s_mean = float(S.mean()); v_mean = float(V.mean())
        except Exception:
            w, h = im.size
            max_dim = max(w, h)
            scale = 96.0 / max_dim if max_dim > 96 else 1.0
            small = im.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.Resampling.LANCZOS)
            arr = np.array(small.convert('RGB'))
            r = arr[:,:,0].astype(np.float32)/255.0
            g = arr[:,:,1].astype(np.float32)/255.0
            b = arr[:,:,2].astype(np.float32)/255.0
            r_m = float(r.mean()); g_m = float(g.mean()); b_m = float(b.mean())
            h_mean, s_mean, v_mean = colorsys.rgb_to_hsv(r_m, g_m, b_m)
        self._color_cache[key] = (h_mean, s_mean, v_mean)
        return h_mean, s_mean, v_mean

    def _apply_sort_to_pairs(self, valid_pairs):
        if self.sort_strategy not in ('hue_gradient', 'value_transition', 'contrast_balance'):
            return valid_pairs
        enriched = []
        for path, img in valid_pairs:
            h, s, v = self._get_color_feats(path, img)
            enriched.append({'path': self._normalize_path(path), 'img': img, 'h': h, 's': s, 'v': v})
        if self.sort_strategy == 'hue_gradient':
            ordered = sorted(enriched, key=lambda x: x['h'])
        elif self.sort_strategy == 'value_transition':
            ordered = sorted(enriched, key=lambda x: x['v'])
        else:
            warm = [e for e in enriched if (e['h'] < 0.11 or e['h'] > 0.61)]
            cool = [e for e in enriched if (e['h'] >= 0.11 and e['h'] <= 0.61)]
            warm.sort(key=lambda x: x['h']); cool.sort(key=lambda x: x['h'])
            i = j = 0; toggle = True
            ordered = []
            while i < len(warm) or j < len(cool):
                if toggle and i < len(warm): ordered.append(warm[i]); i += 1
                elif j < len(cool): ordered.append(cool[j]); j += 1
                toggle = not toggle
        return [(e['path'], e['img']) for e in ordered]

    def _order_items_for_rows(self, valid_pairs):
        items = []
        for path, img in valid_pairs:
            r = img.width / img.height if img.height else 1.0
            cat = 'L' if r >= 1.2 else ('P' if r <= 0.8 else 'S')
            items.append({'path': self._normalize_path(path), 'img': img, 'ratio': r, 'cat': cat})
        if self.sort_strategy in ('hue_gradient', 'value_transition', 'contrast_balance'):
            enriched = []
            for it in items:
                h_val, s_val, v_val = self._get_color_feats(it['path'], it['img'])
                it2 = dict(it); it2['h'] = h_val; it2['s'] = s_val; it2['v'] = v_val
                enriched.append(it2)
            if self.sort_strategy == 'hue_gradient':
                return sorted(enriched, key=lambda x: x['h'])
            if self.sort_strategy == 'value_transition':
                return sorted(enriched, key=lambda x: x['v'])
            warm = [e for e in enriched if (e['h'] < 0.11 or e['h'] > 0.61)]
            cool = [e for e in enriched if (e['h'] >= 0.11 and e['h'] <= 0.61)]
            warm.sort(key=lambda x: x['h']); cool.sort(key=lambda x: x['h'])
            i = j = 0; toggle = True; ordered = []
            while i < len(warm) or j < len(cool):
                if toggle and i < len(warm): ordered.append(warm[i]); i += 1
                elif j < len(cool): ordered.append(cool[j]); j += 1
                toggle = not toggle
            return ordered
        prev = None; repeat = 0; consecutive_limit = 2; bias = None
        if self.sort_strategy == 'landscape_first': bias = 'L'
        elif self.sort_strategy == 'portrait_first': bias = 'P'
        elif self.sort_strategy == 'random': bias = 'R'
        pools = {'L': [it for it in items if it['cat'] == 'L'],
                 'P': [it for it in items if it['cat'] == 'P'],
                 'S': [it for it in items if it['cat'] == 'S']}
        import random as _rnd
        ordered = []
        while pools['L'] or pools['P'] or pools['S']:
            if bias == 'R':
                keys = ['L', 'P', 'S']; _rnd.shuffle(keys); keys = sorted(keys, key=lambda k: -len(pools[k]))
            else:
                keys = sorted(['L', 'P', 'S'], key=lambda k: -len(pools[k]))
                if bias in ('L', 'P'): keys = [bias] + [k for k in keys if k != bias]
            placed = False
            for k in keys:
                if pools[k] and (prev != k or repeat < consecutive_limit):
                    ordered.append(pools[k].pop(0))
                    if prev == k: repeat += 1
                    else: prev = k; repeat = 1
                    placed = True; break
            if not placed:
                for k in keys:
                    if pools[k]: ordered.append(pools[k].pop(0)); prev = k; repeat = 1; break
        return ordered

    # 模板斜切方法已移除

    # 移除自由分栏/异形拼图生成方法
    
    def _generate_output_path(self, selected):
        dir_path = os.path.dirname(self._normalize_path(selected[0]))
        base = "智能拼图"
        # 扫描现有文件，找到下一个可用序号
        existing = []
        try:
            for f in os.listdir(dir_path):
                if not (f.endswith('.jpg') or f.endswith('.png')):
                    continue
                if f.startswith(base):
                    existing.append(f)
        except Exception:
            pass
        nums = [0]
        for f in existing:
            name, _ = os.path.splitext(f)
            if name == base:
                nums.append(0)
            elif name.startswith(base + '_'):
                suf = name[len(base)+1:]
                try:
                    nums.append(int(suf))
                except Exception:
                    pass
        next_n = 0
        if any(f.startswith(base) for f in existing):
            next_n = max(nums) + 1
        # 返回PNG占位路径，后续直接生成JPG
        if next_n == 0:
            return os.path.join(dir_path, base + '.png')
        else:
            return os.path.join(dir_path, f"{base}_{next_n}.png")
    
    def _save_and_optimize(self, canvas, png_path):
        try:
            # 直接输出为JPG以提升速度
            jpg_path = os.path.splitext(png_path)[0] + '.jpg'
            img = canvas
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            img.save(jpg_path, format='JPEG', quality=self.quality, optimize=False, subsampling=2)
            print(f"JPG文件已保存: {jpg_path}")
            return jpg_path
        except Exception as e:
            print(f"JPG保存失败，尝试PNG: {str(e)}")
            try:
                canvas.save(png_path, format='PNG', quality=self.quality, optimize=False)
                print(f"PNG文件已保存: {png_path}")
                return png_path
            except Exception as e2:
                print(f"保存优化图片失败: {str(e2)}")
                raise e2
    
    def _show_success_message(self, file_path):
        dir_path = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        file_size = self._get_file_size(file_path)
        
        message = f"""智能拼图生成成功！
        
📁 文件位置:
{dir_path}

📄 文件名: {file_name}
📊 文件大小: {file_size}

图片已自动打开查看。
文件夹也已打开并选中文件。"""
        
        self._show_message("生成成功", message, QMessageBox.Information)
    
    def _get_file_size(self, file_path):
        try:
            size = os.path.getsize(file_path)
            if size < 1024:
                return f"{size} B"
            elif size < 1024 * 1024:
                return f"{size/1024:.1f} KB"
            else:
                return f"{size/(1024*1024):.1f} MB"
        except:
            return "未知"
    
    def open_image(self, file_path):
        try:
            now = time.time()
            if self._last_image_path == file_path and (now - self._last_image_ts) < 20:
                print(f"跳过重复打开图片预览: {file_path}")
                return
            if os.name == 'nt':
                os.startfile(file_path)
            elif os.name == 'posix':
                if sys.platform == 'darwin':
                    subprocess.run(['open', file_path], check=True)
                else:
                    subprocess.run(['xdg-open', file_path], check=True)
            print(f"已尝试打开图片: {file_path}")
            self._last_image_path = file_path
            self._last_image_ts = now
        except Exception as e:
            print(f"打开图片失败: {str(e)}")
            self._show_message("提示", f"无法自动打开图片，请手动在文件夹中查看。", QMessageBox.Information)
    
    def open_file_in_explorer(self, file_path):
        try:
            file_path = os.path.normpath(file_path)
            dir_path = os.path.dirname(file_path)
            now = time.time()
            if self._last_opened_dir == dir_path:
                print(f"跳过重复打开文件夹: {dir_path}")
                return
            if os.name == 'nt':
                cmd = f'explorer /select,"{file_path}"'
                subprocess.Popen(cmd, shell=True)
            elif os.name == 'posix':
                if sys.platform == 'darwin':
                    subprocess.run(['open', '-R', file_path], check=True)
                else:
                    subprocess.run(['xdg-open', dir_path], check=True)
            self._last_opened_dir = dir_path
            self._last_opened_ts = now
            print(f"已打开文件夹并选中文件: {file_path}")
        except Exception as e:
            print(f"打开文件资源管理器失败: {str(e)}")
            self._show_message("提示", f"已生成文件：{file_path}\n但无法自动打开文件夹。", QMessageBox.Information)
    
    def _show_message(self, title, text, icon):
        if self.parent_window:
            msg_box = QMessageBox(self.parent_window)
            msg_box.setWindowTitle(title)
            msg_box.setText(text)
            msg_box.setIcon(icon)
            msg_box.exec_()
        else:
            print(f"{title}: {text}")

class ImageSelectionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.generator = WaterfallGeneratorPro()
        self.image_paths = []
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('星TAP拼图工具 v3.1 (智能版)')
        self.setGeometry(100, 100, 800, 600)
        self.setAcceptDrops(True)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)
        
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        button_info = [
            ("选择图片", self.select_images),
            ("追加图片", self.add_images),
            ("随机排序", self.random_sort),
            ("生成拼图", self.generate_waterfall),
            ("清空列表", self.clear_image_list),
            ("删除选中项", self.delete_selected_items)
        ]
        
        for text, callback in button_info:
            btn = QPushButton(text, self)
            btn.clicked.connect(callback)
            btn.setFixedHeight(32)
            btn.setMinimumWidth(100)
            btn_layout.addWidget(btn)
        
        layout.addLayout(btn_layout)

        # --- 控制区：列数 + 布局 ---
        control_top = QHBoxLayout()
        control_top.setSpacing(12)

        column_layout = QHBoxLayout()
        column_layout.setSpacing(6)
        lbl_cols = QLabel("列数:")
        column_layout.addWidget(lbl_cols)
        btn_col1 = QPushButton("1列")
        btn_col2 = QPushButton("2列")
        btn_col3 = QPushButton("3列")
        for b, n in [(btn_col1, 1), (btn_col2, 2), (btn_col3, 3)]:
            b.setFixedHeight(28)
            b.setMinimumWidth(68)
            b.clicked.connect(lambda _, nn=n: self.set_columns_ui(nn))
            column_layout.addWidget(b)
        column_layout.addStretch(1)

        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(6)
        mode_layout.addWidget(QLabel("布局:"))
        self.mode_columns = QPushButton("直线列")
        self.mode_columns.setFixedHeight(28)
        self.mode_columns.setMinimumWidth(88)
        self.mode_columns.clicked.connect(lambda: self.set_layout_style_ui('columns'))
        mode_layout.addWidget(self.mode_columns)
        self.mode_rows = QPushButton("智能组合(每行1-3列)")
        self.mode_rows.setFixedHeight(28)
        self.mode_rows.setMinimumWidth(160)
        self.mode_rows.clicked.connect(lambda: self.set_layout_style_ui('rows'))
        mode_layout.addWidget(self.mode_rows)
        mode_layout.addStretch(1)

        control_top.addLayout(column_layout)
        control_top.addLayout(mode_layout)
        layout.addLayout(control_top)

        # 排序策略（小白友好，紧凑下拉）：
        sort_layout = QHBoxLayout()
        sort_layout.setSpacing(6)
        sort_layout.addWidget(QLabel("排序:"))
        self.sort_combo = QComboBox()
        for label in ["均衡交错(推荐)", "横图优先", "竖图优先", "随机混排", "色彩渐变", "明暗过渡", "对比平衡"]:
            self.sort_combo.addItem(label)
        self.sort_combo.setCurrentIndex(0)
        self.sort_combo.setFixedHeight(28)
        self.sort_combo.setMinimumWidth(180)
        self.sort_combo.currentIndexChanged.connect(self.apply_sort_strategy_ui)
        sort_layout.addWidget(self.sort_combo)
        sort_layout.addStretch(1)
        layout.addLayout(sort_layout)
        
        options_layout = QHBoxLayout()
        
        self.crop_bottom_checkbox = QCheckBox("裁剪底边空白", self)
        self.crop_bottom_checkbox.setChecked(True)
        options_layout.addWidget(self.crop_bottom_checkbox)
        
        self.align_bottom_checkbox = QCheckBox("对齐底边(自动拉伸间距)", self)
        self.align_bottom_checkbox.setChecked(False)
        options_layout.addWidget(self.align_bottom_checkbox)
        
        self.rounded_corners_checkbox = QCheckBox("圆角图片", self)
        self.rounded_corners_checkbox.setChecked(False)
        options_layout.addWidget(self.rounded_corners_checkbox)
        
        self.crop_3_4_checkbox = QCheckBox("裁剪成3:4比例", self)
        options_layout.addWidget(self.crop_3_4_checkbox)
        
        self.author_tag_checkbox = QCheckBox("作者标签", self)
        self.author_tag_checkbox.setChecked(False)
        options_layout.addWidget(self.author_tag_checkbox)
        
        # 移除曲线相关/异形拼图选项
        
        options_layout.addStretch(1)
        layout.addLayout(options_layout)
        
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.image_list = QListWidget(self)
        self.image_list.setIconSize(QSize(100, 100))
        self.image_list.setDragDropMode(QListWidget.InternalMove)
        layout.addWidget(self.image_list)
        
        self.image_count_label = QLabel(self)
        self.update_image_count_label()
        layout.addWidget(self.image_count_label)
        
        dark_stylesheet = """
        QWidget {
            background-color: #2E2E2E;
            color: white;
        }
        QPushButton {
            background-color: #444444;
            border: 1px solid #666666;
            padding: 5px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #555555;
        }
        QListWidget {
            background-color: #333333;
            border: 1px solid #666666;
        }
        QProgressBar {
            background-color: #333333;
            border: 1px solid #666666;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #007ACC;
        }
        QCheckBox {
            color: white;
            padding: 5px;
        }
        QCheckBox:hover {
            background-color: #3E3E3E;
        }
        """
        self.setStyleSheet(dark_stylesheet)
        self.setLayout(layout)
    
    # 已移除曲线强度设置

    def set_columns_ui(self, n):
        self.generator.set_columns(n)
        self.update_image_count_label()

    def set_layout_style_ui(self, style):
        self.generator.set_layout_style(style)
        is_rows = (style == 'rows')
        self.author_tag_checkbox.setEnabled(not is_rows)
        self.align_bottom_checkbox.setEnabled(not is_rows)
        self.crop_bottom_checkbox.setEnabled(not is_rows)
        if is_rows:
            self.author_tag_checkbox.setChecked(False)
        self.update_image_count_label()

    def apply_sort_strategy_ui(self):
        idx = self.sort_combo.currentIndex()
        mapping = {0: 'balanced', 1: 'landscape_first', 2: 'portrait_first', 3: 'random', 4: 'hue_gradient', 5: 'value_transition', 6: 'contrast_balance'}
        self.generator.set_sort_strategy(mapping.get(idx, 'balanced'))
        self.update_image_count_label()

    # 移除模式切换回调
    
    def select_images(self):
        self._handle_image_selection(True)
    
    def add_images(self):
        self._handle_image_selection(False)
    
    def _handle_image_selection(self, replace):
        files, _ = QFileDialog.getOpenFileNames(self, '选择图片', '', '图片文件 (*.jpg *.jpeg *.png *.bmp)')
        if files:
            if replace:
                self.image_paths = files
            else:
                current_set = set(os.path.normpath(p).lower() for p in self.image_paths)
                new_files = [f for f in files if os.path.normpath(f).lower() not in current_set]
                self.image_paths.extend(new_files)
            
            self.update_image_list(self.image_paths, True)
            self.update_image_count_label()
    
    def update_image_list(self, new_paths, replace):
        if replace:
            self.image_list.clear()
        
        for path in new_paths:
            try:
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    item = QListWidgetItem(QIcon(scaled_pixmap), os.path.basename(path))
                    self.image_list.addItem(item)
            except Exception as e:
                print(f"加载图片缩略图失败: {str(e)}")
    
    def update_image_count_label(self):
        count = len(self.image_paths)
        mode_map = {'columns': '直线列', 'rows': '智能组合'}
        mode_text = mode_map.get(getattr(self.generator, 'layout_style', 'columns'), '直线列')
        sort_map = {
            'balanced': '均衡交错',
            'landscape_first': '横图优先',
            'portrait_first': '竖图优先',
            'random': '随机混排',
            'hue_gradient': '色彩渐变',
            'value_transition': '明暗过渡',
            'contrast_balance': '对比平衡'
        }
        sort_text = sort_map.get(getattr(self.generator, 'sort_strategy', 'balanced'), '均衡交错')
        self.image_count_label.setText(
            f"已选择 {count} 张图片 | 布局: {mode_text} | 排序: {sort_text} | 当前列数: {self.generator.column}"
        )
    
    def random_sort(self):
        import random
        random.shuffle(self.image_paths)
        self.update_image_list(self.image_paths, True)
    
    def generate_waterfall(self):
        if len(self.image_paths) < self.generator.column:
            QMessageBox.warning(self, "警告", f"至少需要选择 {self.generator.column} 张图片")
            return
        
        self.update_image_count_label()
        self.progress_bar.setVisible(True)
        QApplication.processEvents()
        
        self.generator.generate_waterfall(
            self.image_paths,
            self,
            crop_bottom=self.crop_bottom_checkbox.isChecked(),
            rounded_corners=self.rounded_corners_checkbox.isChecked(),
            crop_3_4=self.crop_3_4_checkbox.isChecked(),
            align_bottom=self.align_bottom_checkbox.isChecked(),
            tag_enabled=self.author_tag_checkbox.isChecked()
        )
    
    def clear_image_list(self):
        self.image_paths = []
        self.image_list.clear()
        self.update_image_count_label()
    
    def delete_selected_items(self):
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            return
        
        selected_indices = [self.image_list.row(item) for item in selected_items]
        selected_indices.sort(reverse=True)
        
        for index in selected_indices:
            del self.image_paths[index]
        
        self.update_image_list(self.image_paths, True)
        self.update_image_count_label()
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            files = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if image_files:
                current_set = set(os.path.normpath(p).lower() for p in self.image_paths)
                new_files = [f for f in image_files if os.path.normpath(f).lower() not in current_set]
                self.image_paths.extend(new_files)
                self.update_image_list(self.image_paths, True)
                self.update_image_count_label()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageSelectionWindow()
    window.show()
    sys.exit(app.exec_())
