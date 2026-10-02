"""BEV polygon editor; configuration storage is usable without a display."""
from __future__ import annotations

import base64
import copy
import math
import os
from pathlib import Path
import shutil
import tempfile

import yaml

# Keys match the polygons consumed by tracking and spatial aggregation.
KINDS = {
    '동선 / Route': ('routes', ('start_gate', 'end_gate'), ('시작 구역 / Start', '종료 구역 / End')),
    '공간 / Region': ('regions', ('polygon',), ('공간 구역 / Region',)),
    '시선 / Gaze': ('targets', ('source_polygon', 'target_polygon'), ('출발 구역 / Source', '대상 구역 / Target')),
}


def validate_polygon(points):
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError('구역은 꼭짓점 3개 이상으로 그려야 합니다.')
    vertices = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError('좌표는 [x, y] 형식이어야 합니다.')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in point):
            raise ValueError('좌표는 0~1 범위의 유효한 정규화 좌표여야 합니다.')
        vertices.append(tuple(point))
    if vertices[0] == vertices[-1]:
        vertices.pop()
    if len(set(vertices)) != len(vertices) or len(vertices) < 3:
        raise ValueError('중복된 꼭짓점은 사용할 수 없습니다.')
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(vertices, vertices[1:] + vertices[:1]))
    if abs(area) < 1e-10:
        raise ValueError('구역의 면적이 0입니다. 꼭짓점을 다시 지정하세요.')

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c):
        return abs(cross(a, b, c)) < 1e-12 and min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])

    edges = list(zip(vertices, vertices[1:] + vertices[:1]))
    for i, (a, b) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            if j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            c, d = edges[j]
            intersect = cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0
            if intersect or any((on_segment(a, b, c), on_segment(a, b, d), on_segment(c, d, a), on_segment(c, d, b))):
                raise ValueError('구역의 선이 서로 교차합니다. 꼭짓점을 순서대로 지정하세요.')


class SpatialConfig:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.original = self.path.read_bytes()
        cfg = yaml.safe_load(self.original) or {}
        if not isinstance(cfg, dict):
            raise ValueError('YAML 설정은 객체 형식이어야 합니다.')
        self.spatial = copy.deepcopy(cfg.get('spatial') or {})
        if not isinstance(self.spatial, dict):
            raise ValueError('spatial은 객체 형식이어야 합니다.')
        self.spatial.setdefault('routes', [])
        self.spatial.setdefault('regions', [])
        self.spatial.setdefault('gaze', {})
        if not isinstance(self.spatial['gaze'], dict):
            raise ValueError('spatial.gaze는 객체 형식이어야 합니다.')
        self.spatial['gaze'].setdefault('targets', [])
        self.saved = copy.deepcopy(self.spatial)

    def items(self, kind):
        key = KINDS[kind][0]
        items = self.spatial['gaze']['targets'] if key == 'targets' else self.spatial[key]
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ValueError(f'{key}는 구역 객체의 목록이어야 합니다.')
        return items

    @property
    def dirty(self):
        return self.spatial != self.saved

    def save(self):
        for kind, (_, keys, _) in KINDS.items():
            names = set()
            for item in self.items(kind):
                name = item.get('name', '')
                if not isinstance(name, str) or not name.strip() or name.strip() in names:
                    raise ValueError(f'{kind}: 비어 있거나 중복된 이름이 있습니다.')
                names.add(name.strip())
                if KINDS[kind][0] == 'routes' and not isinstance(item.get('bidirectional', False), bool):
                    raise ValueError(f'{name}: bidirectional은 true/false여야 합니다.')
                for key in keys:
                    try:
                        validate_polygon(item.get(key))
                    except ValueError as exc:
                        raise ValueError(f'{name} / {key}: {exc}') from exc
        if self.path.read_bytes() != self.original:
            raise RuntimeError('GUI를 연 뒤 YAML 파일이 변경되었습니다. 닫고 다시 열어 주세요.')
        cfg = yaml.safe_load(self.original) or {}
        cfg['spatial'] = self.spatial
        rendered = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        backup = self.path.with_name(self.path.name + '.spatial.bak')
        if not backup.exists():
            shutil.copy2(self.path, backup)
        fd, temporary = tempfile.mkstemp(prefix=self.path.name + '.', suffix='.partial', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                stream.write(rendered)
                stream.flush()
                os.fsync(stream.fileno())
            shutil.copymode(self.path, temporary)
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        self.original = self.path.read_bytes()
        self.saved = copy.deepcopy(self.spatial)
        print(f'Spatial configuration saved: {self.path}', flush=True)
        print('Routes: %d | Regions: %d | Gaze targets: %d' % tuple(len(self.items(k)) for k in KINDS), flush=True)
        return backup


class SpatialEditor:
    def __init__(self, root, model, image):
        import tkinter as tk
        from tkinter import ttk, messagebox
        self.tk, self.ttk, self.messagebox = tk, ttk, messagebox
        self.root, self.model, self.image = root, model, image
        self.kind = next(iter(KINDS))
        self.index = None
        self.draft = None
        self.draft_key = None
        self.transform = (0, 0, 1, 1)
        self.loading = False
        root.title('Offline MCMOT — BEV spatial editor')
        root.geometry('1250x820')
        root.minsize(900, 700)
        root.protocol('WM_DELETE_WINDOW', self.close)
        body = ttk.Frame(root, padding=8)
        body.pack(fill='both', expand=True)
        side = ttk.Frame(body, width=330)
        side.pack(side='left', fill='y', padx=(0, 10))
        ttk.Label(side, text='동선 · 공간 · 시선 구역 설정', font=('', 13, 'bold')).pack(anchor='w', pady=6)
        self.kind_var = tk.StringVar(value=self.kind)
        combo = ttk.Combobox(side, textvariable=self.kind_var, values=list(KINDS), state='readonly', width=30)
        combo.pack(fill='x', pady=5)
        combo.bind('<<ComboboxSelected>>', self.change_kind)
        self.listbox = tk.Listbox(side, height=6, exportselection=False, width=35)
        self.listbox.pack(fill='both', expand=True)
        self.listbox.bind('<<ListboxSelect>>', self.select)
        buttons = ttk.Frame(side)
        buttons.pack(fill='x', pady=6)
        ttk.Button(buttons, text='추가 / Add', command=self.add).pack(side='left')
        ttk.Button(buttons, text='삭제 / Delete', command=self.delete).pack(side='left', padx=4)
        ttk.Button(buttons, text='↑', width=3, command=lambda: self.move(-1)).pack(side='left')
        ttk.Button(buttons, text='↓', width=3, command=lambda: self.move(1)).pack(side='left')
        ttk.Label(side, text='이름 / Name').pack(anchor='w')
        self.name = tk.StringVar()
        ttk.Entry(side, textvariable=self.name).pack(fill='x', pady=3)
        self.bidirectional = tk.BooleanVar()
        self.bidirectional_check = ttk.Checkbutton(side, text='양방향 동선 / Bidirectional', variable=self.bidirectional)
        self.bidirectional_check.pack(anchor='w', pady=4)
        self.name.trace_add('write', self.metadata_changed)
        self.bidirectional.trace_add('write', self.metadata_changed)
        ttk.Label(side, text='그릴 구역 / Polygon').pack(anchor='w', pady=(8, 0))
        self.role = tk.StringVar()
        self.roles = ttk.Combobox(side, textvariable=self.role, state='readonly')
        self.roles.pack(fill='x', pady=3)
        ttk.Button(side, text='그리기 / 다시 그리기', command=self.start_draw).pack(fill='x', pady=3)
        row = ttk.Frame(side)
        row.pack(fill='x', pady=3)
        ttk.Button(row, text='점 취소 / Undo', command=self.undo).pack(side='left')
        ttk.Button(row, text='확정 / Finish', command=self.finish).pack(side='left', padx=4)
        ttk.Button(side, text='그리기 취소 / Cancel drawing', command=self.cancel_draw).pack(fill='x', pady=3)
        ttk.Separator(side).pack(fill='x', pady=10)
        ttk.Label(side, text='유형 선택 → 추가 → 이름 입력\n구역 선택 → 그리기 → 3점 이상 → 확정\n시작/종료 또는 출발/대상을 모두 지정합니다.\n시선 출발 = 사람 위치, 대상 = 물체 위치\n↑/↓로 시선 검사 순서를 조정합니다.\n저장 후 all을 실행하세요.', justify='left', wraplength=310).pack(anchor='w')
        ttk.Button(side, text='저장 / Save', command=self.save).pack(fill='x', pady=(10, 3))
        ttk.Button(side, text='저장 후 종료 / Save & close', command=self.save_close).pack(fill='x', pady=3)
        self.canvas = tk.Canvas(body, bg='#20242b', highlightthickness=0)
        self.canvas.pack(side='right', fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda event: self.redraw())
        self.canvas.bind('<Button-1>', self.click)
        self.canvas.bind('<Button-3>', lambda event: self.undo())
        root.bind('<Escape>', lambda event: self.cancel_draw())
        self.status = tk.StringVar(value=f'BEV loaded | {model.path.name}')
        ttk.Label(root, textvariable=self.status, padding=8).pack(fill='x')
        self.update_roles()
        self.refresh(0 if self.model.items(self.kind) else None)

    def update_roles(self):
        self.roles['values'] = KINDS[self.kind][2]
        self.role.set(KINDS[self.kind][2][0])
        self.bidirectional_check.configure(state='normal' if KINDS[self.kind][0] == 'routes' else 'disabled')

    def refresh(self, index, metadata=True):
        self.loading = True
        self.listbox.delete(0, 'end')
        keys = KINDS[self.kind][1]
        for item in self.model.items(self.kind):
            complete = all(len(item.get(key, [])) >= 3 for key in keys)
            self.listbox.insert('end', ('✓ ' if complete else '✗ ') + (item.get('name') or '(이름 없음)'))
        self.index = index
        if index is not None:
            self.listbox.selection_set(index)
            self.listbox.see(index)
            if metadata:
                item = self.model.items(self.kind)[index]
                self.name.set(item.get('name', ''))
                self.bidirectional.set(item.get('bidirectional', False))
        elif metadata:
            self.name.set('')
            self.bidirectional.set(False)
        self.loading = False
        self.redraw()

    def metadata_changed(self, *args):
        if self.loading or self.index is None:
            return
        item = self.model.items(self.kind)[self.index]
        item['name'] = self.name.get()
        if KINDS[self.kind][0] == 'routes':
            item['bidirectional'] = self.bidirectional.get()
        self.refresh(self.index, metadata=False)

    def discard_draft(self):
        if self.draft and not self.messagebox.askyesno('그리기 취소', '확정하지 않은 꼭짓점을 취소할까요?'):
            return False
        self.draft = self.draft_key = None
        return True

    def change_kind(self, event=None):
        if not self.discard_draft():
            self.kind_var.set(self.kind)
            return
        self.kind = self.kind_var.get()
        self.update_roles()
        self.refresh(0 if self.model.items(self.kind) else None)

    def select(self, event=None):
        if self.loading:
            return
        selection = self.listbox.curselection()
        if not selection or selection[0] == self.index:
            return
        if not self.discard_draft():
            self.refresh(self.index)
            return
        self.refresh(selection[0])

    def add(self):
        if not self.discard_draft():
            return
        items = self.model.items(self.kind)
        prefix = {'routes': '동선', 'regions': '공간', 'targets': '시선'}[KINDS[self.kind][0]]
        number = 1
        names = {item.get('name') for item in items}
        while f'{prefix} {number}' in names:
            number += 1
        item = {'name': f'{prefix} {number}'}
        if KINDS[self.kind][0] == 'routes':
            item['bidirectional'] = False
        for key in KINDS[self.kind][1]:
            item[key] = []
        items.append(item)
        self.refresh(len(items) - 1)

    def delete(self):
        if self.index is None:
            return
        if not self.messagebox.askyesno('삭제', f'{self.name.get()} 구역을 삭제할까요?'):
            return
        self.draft = self.draft_key = None
        items = self.model.items(self.kind)
        items.pop(self.index)
        self.refresh(min(self.index, len(items) - 1) if items else None)

    def move(self, direction):
        if self.index is None or not self.discard_draft():
            return
        items = self.model.items(self.kind)
        target = self.index + direction
        if 0 <= target < len(items):
            items[self.index], items[target] = items[target], items[self.index]
            self.refresh(target)

    def start_draw(self):
        if self.index is None:
            self.messagebox.showinfo('구역 선택', '먼저 구역을 추가하거나 목록에서 선택하세요.')
            return
        if not self.discard_draft():
            return
        self.draft = []
        self.draft_key = KINDS[self.kind][1][KINDS[self.kind][2].index(self.role.get())]
        self.status.set(f'Drawing: {self.name.get()} / {self.role.get()} — click 3+ vertices, then Finish')
        self.redraw()

    def click(self, event):
        if self.draft is None:
            return
        x, y, width, height = self.transform
        if x <= event.x <= x + width and y <= event.y <= y + height:
            self.draft.append([round((event.x - x) / width, 7), round((event.y - y) / height, 7)])
            self.status.set(f'Drawing {self.draft_key}: {len(self.draft)} vertices | Finish to apply')
            self.redraw()

    def undo(self):
        if self.draft:
            self.draft.pop()
            self.redraw()

    def finish(self):
        if self.draft is None:
            return
        try:
            validate_polygon(self.draft)
        except ValueError as exc:
            self.messagebox.showerror('구역 오류', str(exc))
            return
        self.model.items(self.kind)[self.index][self.draft_key] = self.draft
        self.draft = self.draft_key = None
        self.status.set('Polygon applied — Save to write YAML')
        self.refresh(self.index)

    def cancel_draw(self):
        if self.discard_draft():
            self.status.set('Drawing cancelled; existing polygon retained')
            self.redraw()

    def redraw(self):
        import cv2
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 2 or height < 2:
            return
        ih, iw = self.image.shape[:2]
        scale = min(width / iw, height / ih)
        dw, dh = max(1, round(iw * scale)), max(1, round(ih * scale))
        x, y = (width - dw) / 2, (height - dh) / 2
        transform = (x, y, dw, dh)
        if transform != self.transform or not hasattr(self, 'photo'):
            resized = cv2.resize(self.image, (dw, dh), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode('.png', resized)
            if not ok:
                raise RuntimeError('BEV image could not be displayed')
            self.photo = self.tk.PhotoImage(data=base64.b64encode(encoded.tobytes()))
        self.transform = transform
        self.canvas.delete('all')
        self.canvas.create_image(x, y, anchor='nw', image=self.photo)
        colors = ('#33e1a0', '#ffb74d')
        for i, item in enumerate(self.model.items(self.kind)):
            centers = []
            for j, key in enumerate(KINDS[self.kind][1]):
                points = item.get(key) or []
                if len(points) < 3:
                    centers.append(None)
                    continue
                coords = [(x + px * dw, y + py * dh) for px, py in points]
                flat = [v for point in coords for v in point]
                selected = i == self.index
                self.canvas.create_polygon(*flat, fill=colors[j], stipple='gray25' if selected else 'gray12', outline=colors[j], width=3 if selected else 1)
                center = (sum(px for px, _ in coords) / len(coords), sum(py for _, py in coords) / len(coords))
                centers.append(center)
                if selected:
                    self.canvas.create_text(*center, text=f'{item.get("name", "")}\n{KINDS[self.kind][2][j]}', fill='white', font=('', 11, 'bold'))
                    for px, py in coords:
                        self.canvas.create_oval(px - 3, py - 3, px + 3, py + 3, fill=colors[j], outline='white')
            if i == self.index and len(centers) == 2 and all(centers):
                arrow = 'both' if item.get('bidirectional') else 'last'
                self.canvas.create_line(*centers[0], *centers[1], fill='#ff6b81', width=3, arrow=arrow)
        if self.draft:
            coords = [(x + px * dw, y + py * dh) for px, py in self.draft]
            if len(coords) > 1:
                self.canvas.create_line(*[v for p in coords for v in p], fill='#ffffff', width=3)
            for number, (px, py) in enumerate(coords, 1):
                self.canvas.create_oval(px - 4, py - 4, px + 4, py + 4, fill='white', outline='#111111')
                self.canvas.create_text(px + 10, py - 10, text=str(number), fill='white')

    def save(self):
        if self.draft is not None:
            self.messagebox.showerror('그리기 진행 중', '확정 또는 그리기 취소 후 저장하세요.')
            return False
        try:
            backup = self.model.save()
        except (ValueError, RuntimeError, OSError) as exc:
            self.messagebox.showerror('저장 실패', str(exc))
            return False
        self.status.set(f'Saved: {self.model.path.name} | Original backup: {backup.name}')
        return True

    def save_close(self):
        if self.save():
            self.root.destroy()

    def close(self):
        if self.model.dirty or self.draft is not None:
            answer = self.messagebox.askyesnocancel('종료', '변경한 구역을 YAML에 저장할까요?')
            if answer is None or (answer and not self.save()):
                return
        self.root.destroy()


def launch(config_path, image_path):
    import cv2
    try:
        import tkinter as tk
    except ImportError as exc:
        raise RuntimeError('Spatial GUI requires Python with Tk support (tkinter).') from exc
    model = SpatialConfig(config_path)
    for kind in KINDS:
        model.items(kind)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'BEV image could not be read: {image_path}')
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError('Spatial GUI requires a desktop/X11 DISPLAY connection.') from exc
    SpatialEditor(root, model, image)
    print(f'Spatial GUI opened: {image_path}', flush=True)
    print('Draw polygons, then Save & close. YAML spatial settings feed the next all/run.', flush=True)
    root.mainloop()
