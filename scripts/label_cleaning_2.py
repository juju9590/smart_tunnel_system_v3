import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import copy

class LabelEditor:
    def __init__(self, dataset_path):
        self.image_dir = os.path.join(dataset_path, "images")
        self.label_dir = os.path.join(dataset_path, "labels")

        if not os.path.exists(self.image_dir) or not os.path.exists(self.label_dir):
            messagebox.showerror("오류", f"경로를 확인해주세요:\n{self.image_dir}\n{self.label_dir}")
            self.initialized = False
            return

        self.image_list = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))
        ])

        if not self.image_list:
            messagebox.showerror("오류", "이미지 폴더 내에 이미지 파일이 없습니다.")
            self.initialized = False
            return

        root = tk.Tk()
        self.screen_w = root.winfo_screenwidth()
        self.screen_h = root.winfo_screenheight()
        root.destroy()

        self.current_idx = 0
        self.boxes = []
        self.history = []
        self.redo_history = []
        self.selected_box_idx = -1
        self.drag_mode = None
        self.pending_box = None
        self.start_x = 0
        self.start_y = 0
        self.canvas_shape = (0, 0, 0)
        self.initialized = True 

        self.conf_threshold = 0.25

        self.class_colors = [
            (255, 50, 50), (50, 255, 50), (50, 50, 255), (0, 255, 255),
            (255, 0, 255), (255, 255, 0), (0, 165, 255),
            (128, 0, 128), (0, 128, 0), (128, 128, 128)
        ]

    def load_labels(self):
        img_name = self.image_list[self.current_idx]
        file_base_name = os.path.splitext(img_name)[0]
        path = os.path.join(self.label_dir, file_base_name + ".txt")
        
        boxes = []
        has_conf = False
        has_no_conf = False
        
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 6:  # class x y w h conf
                        cls, x, y, w, h, conf = map(float, parts)
                        boxes.append([int(cls), x, y, w, h, conf])
                        has_conf = True
                    elif len(parts) == 5:  # 기존 5열
                        cls, x, y, w, h = map(float, parts)
                        boxes.append([int(cls), x, y, w, h, 1.0])
                        has_no_conf = True
        
        # ★ 디버깅: 첫 이미지에서 포맷 경고
        if self.current_idx == 0:
            if has_no_conf and not has_conf:
                print("경고: txt 파일이 5열(class x y w h) 입니다. confidence 필터가 동작하지 않습니다.")
                print("→ 자동 라벨링 코드를 6열로 저장하도록 수정하세요.")
            elif has_conf:
                print(f"확인: {len([b for b in boxes if b[5] < 1.0])}개 박스가 실제 confidence 값을 가집니다.")
        
        return boxes

    def save_labels(self):
        img_name = self.image_list[self.current_idx]
        path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")
        with open(path, "w") as f:
            for b in self.boxes:
                if len(b) == 6:
                    f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f} {b[5]:.6f}\n")
                else:
                    f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

    def change_image(self, step):
        self.current_idx = max(0, min(self.current_idx + step, len(self.image_list)-1))
        self.boxes = self.load_labels()
        self.selected_box_idx = -1
        self.pending_box = None
        self.history = [] 
        self.redo_history = []

    def save_history(self):
        self.history.append(copy.deepcopy(self.boxes))
        if len(self.history) > 50: 
            self.history.pop(0)
        self.redo_history.clear()

    def undo(self):
        if self.history:
            self.redo_history.append(copy.deepcopy(self.boxes))
            self.boxes = self.history.pop()
            self.selected_box_idx = -1
            self.save_labels()

    def redo(self):
        if self.redo_history:
            self.history.append(copy.deepcopy(self.boxes))
            if len(self.history) > 50:
                self.history.pop(0)
            self.boxes = self.redo_history.pop()
            self.selected_box_idx = -1
            self.save_labels()

    def move_box(self, dx, dy):
        h, w, _ = self.canvas_shape
        if w == 0 or h == 0: 
            return
        step_size = 2
        norm_dx = (dx * step_size) / (w * 0.75) 
        norm_dy = (dy * step_size) / h
        if self.pending_box:
            self.pending_box[0] += norm_dx
            self.pending_box[1] += norm_dy
        elif self.selected_box_idx != -1:
            self.save_history()
            self.boxes[self.selected_box_idx][1] += norm_dx
            self.boxes[self.selected_box_idx][2] += norm_dy
            self.save_labels()

    def mouse_events(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            wheel_delta = np.int16(flags >> 16) if flags > 100000 else flags
            self.change_image(-1 if wheel_delta > 0 else 1)
            return

        h_f, w_f, _ = self.canvas_shape
        w_img = int(w_f * 0.75)
        if x > w_img: 
            return 
        nx, ny = x / w_img, y / h_f

        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_x, self.start_y = nx, ny
            self.selected_box_idx = -1
            corner_th = 0.03
            edge_th = 0.01
            for i in reversed(range(len(self.boxes))):
                cls, bx, by, bw, bh = self.boxes[i][:5]
                x1, y1, x2, y2 = bx - bw/2, by - bh/2, bx + bw/2, by + bh/2
                near_corner = (abs(nx - x2) < corner_th and abs(ny - y2) < corner_th)
                if near_corner:
                    self.save_history()
                    self.drag_mode = "resize"
                    self.selected_box_idx = i
                    return
                near_left   = (abs(nx - x1) < edge_th and y1 <= ny <= y2)
                near_right  = (abs(nx - x2) < edge_th and y1 <= ny <= y2)
                near_top    = (abs(ny - y1) < edge_th and x1 <= nx <= x2)
                near_bottom = (abs(ny - y2) < edge_th and x1 <= nx <= x2)
                near_edge = near_left or near_right or near_top or near_bottom
                if near_edge:
                    self.save_history()
                    self.drag_mode = "move"
                    self.selected_box_idx = i
                    return
            self.drag_mode = "draw"

        elif event == cv2.EVENT_RBUTTONDOWN:
            for i in reversed(range(len(self.boxes))):
                cls, bx, by, bw, bh = self.boxes[i][:5]
                x1, y1, x2, y2 = bx - bw/2, by - bh/2, bx + bw/2, by + bh/2
                if x1 <= nx <= x2 and y1 <= ny <= y2:
                    self.save_history()
                    self.boxes.pop(i)
                    self.selected_box_idx = -1
                    self.save_labels()
                    break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_mode == "move" and self.selected_box_idx != -1:
                dx, dy = nx - self.start_x, ny - self.start_y
                self.boxes[self.selected_box_idx][1] += dx
                self.boxes[self.selected_box_idx][2] += dy
                self.start_x, self.start_y = nx, ny
            elif self.drag_mode == "resize" and self.selected_box_idx != -1:
                bx, by = self.boxes[self.selected_box_idx][1], self.boxes[self.selected_box_idx][2]
                self.boxes[self.selected_box_idx][3] = max(0.01, abs(nx - bx) * 2)
                self.boxes[self.selected_box_idx][4] = max(0.01, abs(ny - by) * 2)
            self.save_labels()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag_mode == "draw":
                w, h = abs(self.start_x - nx), abs(self.start_y - ny)
                if w > 0.01 and h > 0.01:
                    self.pending_box = [(self.start_x + nx) / 2, (self.start_y + ny) / 2, w, h, 1.0]
            self.drag_mode = None

    def draw_ui(self, canvas, w_img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        margin_x = w_img + 20
        h_canvas = self.canvas_shape[0]
        
        cv2.putText(canvas, f"CONF: {self.conf_threshold:.2f}", (margin_x, 40), font, 0.8, (0, 255, 255), 2)
        cv2.putText(canvas, f"FILE: {self.current_idx + 1}/{len(self.image_list)}", (margin_x, 70), font, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, f"NAME: {self.image_list[self.current_idx]}", (margin_x, 100), font, 0.4, (200, 200, 200), 1)
        cv2.line(canvas, (margin_x, 120), (self.canvas_shape[1]-20, 120), (100, 100, 100), 1)

        y_pos = 150
        cv2.putText(canvas, "LABELS:", (margin_x, y_pos), font, 0.5, (0, 255, 255), 1)
        y_pos += 30

        low_conf_count = 0
        real_conf_count = 0
        for i, box in enumerate(self.boxes):
            if y_pos > h_canvas - 200: 
                break
            cls, nx, ny, nw, nh = box[:5]
            conf = box[5] if len(box) > 5 else 1.0
            
            if conf < 0.99:  # 실제로 conf 값이 있는 경우
                real_conf_count += 1
            
            is_selected = (i == self.selected_box_idx)
            is_low_conf = conf < self.conf_threshold
            if is_low_conf: low_conf_count += 1
            
            if is_low_conf:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0) if is_selected else (180, 180, 180)
            
            prefix = ">>" if is_selected else "  "
            coord_txt = f"{prefix}[{i}] C:{int(cls)} {conf:.2f}"
            
            cv2.putText(canvas, coord_txt, (margin_x, y_pos), font, 0.45, color, 1 if not is_selected else 2)
            y_pos += 25

        y_guide = h_canvas - 180
        cv2.putText(canvas, f"LOW: {low_conf_count} / REAL_CONF: {real_conf_count}", (margin_x, y_guide), font, 0.6, (0, 0, 255), 2)
        guides = [
            "A/D: Prev/Next",
            "': +100 Jump",
            "0-9: Set Cls",
            "R-Click: Del",
            "Arrow Keys: Move",
            "Z / Ctrl+Z: Undo",
            "Y / Ctrl+Y: Redo",
            "Q: Quit"
        ]
        for i, txt in enumerate(guides):
            cv2.putText(canvas, txt, (margin_x, y_guide + 25 + (i*22)), font, 0.4, (150, 150, 150), 1)

    def run(self):
        if not self.initialized: 
            return
        cv2.namedWindow("Label Editor", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Label Editor", self.mouse_events)

        self.boxes = self.load_labels()
        while True:
            img_path = os.path.join(self.image_dir, self.image_list[self.current_idx])
            raw_full = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if raw_full is None: 
                print(f"이미지를 불러올 수 없습니다: {img_path}")
                break

            orig_h, orig_w = raw_full.shape[:2]
            max_w, max_h = int(self.screen_w * 0.7), int(self.screen_h * 0.8)
            scale = min(max_w / orig_w, max_h / orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            raw = cv2.resize(raw_full, (new_w, new_h), interpolation=cv2.INTER_AREA)

            h, w = raw.shape[:2]
            canvas_w = int(w / 0.75)
            canvas = np.zeros((h, canvas_w, 3), dtype=np.uint8)
            canvas[:] = (45, 45, 45)
            canvas[:, :w] = raw
            self.canvas_shape = canvas.shape

            for i, box in enumerate(self.boxes):
                cls, nx, ny, nw, nh = box[:5]
                conf = box[5] if len(box) > 5 else 1.0
                
                x1, y1 = int((nx - nw/2) * w), int((ny - nh/2) * h)
                x2, y2 = int((nx + nw/2) * w), int((ny + nh/2) * h)
                
                is_low_conf = conf < self.conf_threshold
                
                if is_low_conf:
                    color = (0, 0, 255)
                    thickness = 1
                else:
                    color = self.class_colors[int(cls) % 10]
                    thickness = 2
                
                if i == self.selected_box_idx:
                    cv2.rectangle(canvas, (x1-2, y1-2), (x2+2, y2+2), (255, 255, 255), 1)
                
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(canvas, f"{conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if self.pending_box:
                px, py, pw, ph = self.pending_box[:4]
                cv2.rectangle(
                    canvas,
                    (int((px-pw/2)*w), int((py-ph/2)*h)), 
                    (int((px+pw/2)*w), int((py+ph/2)*h)),
                    (255, 255, 255), 1
                )

            self.draw_ui(canvas, w)
            cv2.imshow("Label Editor", canvas)
            
            full_key = cv2.waitKeyEx(30)
            if full_key == -1: 
                continue 
            
            key = full_key & 0xFF 

            if key == 26:  # Ctrl+Z
                self.undo()
                continue
            if key == 25:  # Ctrl+Y
                self.redo()
                continue

            if key == ord('q') or key == 27: 
                break
            elif key == ord('c'):
                self.conf_threshold = max(0.0, self.conf_threshold - 0.05)
                print(f"Threshold: {self.conf_threshold:.2f}")  # ★ 터미널 출력
            elif key == ord('v'):
                self.conf_threshold = min(1.0, self.conf_threshold + 0.05)
                print(f"Threshold: {self.conf_threshold:.2f}")  # ★ 터미널 출력
            elif key == ord('f'):
                self.save_history()
                before = len(self.boxes)
                self.boxes = [b for b in self.boxes if (b[5] if len(b) > 5 else 1.0) >= self.conf_threshold]
                after = len(self.boxes)
                print(f"삭제: {before - after}개")
                self.save_labels()
            elif ord('0') <= key <= ord('9'):
                if self.pending_box:
                    self.save_history()
                    self.boxes.append([key - ord('0')] + self.pending_box)
                    self.pending_box = None
                    self.save_labels()
                elif self.selected_box_idx != -1:
                    self.boxes[self.selected_box_idx][0] = key - ord('0')
                    self.save_labels()
            elif key == ord('a'): 
                self.change_image(-1)
            elif key == ord('d'): 
                self.change_image(1)
            elif key == ord("'"):
                self.change_image(100)
            elif key == ord('z'):
                self.undo()
            elif key == ord('y'):
                self.redo()
            elif key == ord("'"):
                self.change_image(100)    # +100장 (이미 있음)
            elif key == ord(";"):
                self.change_image(-100)   # -100장 (추가)
            elif key == ord("."):
                self.change_image(10)     # +10장 (추가)
            elif key == ord(","):
                self.change_image(-10)    # -10장 (추가)
            elif full_key in [2490368, 65362]:
                self.move_box(0, -1)
            elif full_key in [2621440, 65364]:
                self.move_box(0, 1)
            elif full_key in [2424832, 65361]:
                self.move_box(-1, 0)
            elif full_key in [2555904, 65363]:
                self.move_box(1, 0)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("1. 폴더 선택 창을 띄웁니다.")
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True) 
    path = filedialog.askdirectory(title="images/labels 폴더가 포함된 상위 폴더 선택")
    
    if path:
        print(f"2. 선택된 경로: {path}")
        editor = LabelEditor(path)
        if editor.initialized:
            editor.run()
    else:
        print("오류: 폴더 선택이 취소되었습니다.")