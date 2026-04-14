import os
import random
import string
import time
import itertools
from multiprocessing import Pool, cpu_count
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# 參數
num_images = 50000
output_dir = "synthetic"
font_path = "SpicyRice-Regular.ttf"

train_ratio = 0.8
val_ratio = 0.1

image_width = 120
image_height = 100

bg_color = (2, 108, 223)
text_color = (255, 255, 255, 255)
margin = 2

# 字體大小
font_size_range = (40, 58)
intra_word_size_variation = (-5, 5)

# 旋轉角度
angle_range = (-12, 12)

# y 軸偏移
y_offset_range = (-6, 6)

# 字元間距
kerning_range = (-3, 2)

# 字元縮放
scale_x_range = (0.92, 1.04)
scale_y_range = (0.95, 1.08)

# shear
shear_x_range = (-0.05, 0.05)

# 字元水平抖動
char_x_jitter_range = (-2, 2)

# 收窄範圍並往左移
global_shift_x_range = (-5, 0)
global_shift_y_range = (-2, 5)

# overlap
overlap_mode_probs = {
    "normal": 0.55,
    "overlap": 0.35,
    "hard_overlap": 0.10,
}
overlap_extra_shift = {
    "normal": (0, 1),
    "overlap": (1, 2.5),
    "hard_overlap": (2.5, 4),
}

# 整串最後縮放
word_scale_range = (0.90, 0.98)

# 筆畫關閉加粗
stroke_width_range = (0, 0)

# 模糊
blur_prob = 0.15
blur_radius = 0.15

NUM_WORKERS = max(1, cpu_count() - 1)
BATCH_SIZE = 5000

# INIT 目錄
splits = ["train", "val", "test"]
for split in splits:
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

if not os.path.exists(font_path):
    print(f"錯誤 找不到字體檔案 '{font_path}'。")
    exit()

# 工具函數
def weighted_choice(prob_dict):
    r = random.random()
    acc = 0.0
    for key, prob in prob_dict.items():
        acc += prob
        if r <= acc:
            return key
    return list(prob_dict.keys())[-1]

def alpha_bbox(img_rgba):
    alpha = img_rgba.getchannel("A")
    return alpha.getbbox()

def crop_to_content(img_rgba):
    bbox = alpha_bbox(img_rgba)
    if bbox is None:
        return img_rgba
    return img_rgba.crop(bbox)

def safe_resize(img, new_w, new_h):
    new_w = max(1, int(round(new_w)))
    new_h = max(1, int(round(new_h)))
    return img.resize((new_w, new_h), Image.BICUBIC)

def apply_affine_rgba(img, shear_x=0.0):
    w, h = img.size
    shift = abs(shear_x) * h
    new_w = int(round(w + shift + 4))
    transformed = img.transform(
        (new_w, h),
        Image.AFFINE,
        (1, shear_x, 0, 0, 1, 0),
        resample=Image.BICUBIC,
    )
    return crop_to_content(transformed)

def maybe_offset_dot_for_i_j(char, img_rgba):
    if char not in ("i", "j"):
        return img_rgba
    dx = random.randint(-2, 2)
    dy = random.randint(-2, 2)
    canvas = Image.new("RGBA", (img_rgba.width + 8, img_rgba.height + 8), (255, 255, 255, 0))
    canvas.paste(img_rgba, (4 + dx, 4 + dy), img_rgba)
    return crop_to_content(canvas)

def sharpen_alpha(img_rgba, threshold=80):
    arr = np.array(img_rgba)
    alpha = arr[:, :, 3].astype(float)
    mask_strong = alpha > threshold
    mask_weak = (alpha > 0) & (alpha <= threshold)
    alpha[mask_strong] = 255
    alpha[mask_weak] = np.clip(alpha[mask_weak] * 1.8, 0, 255)
    arr[:, :, 3] = alpha.astype(np.uint8)
    return Image.fromarray(arr, "RGBA")

def render_single_char(char, fp, size_override=None):
    if size_override is not None:
        size = size_override
    else:
        size = random.randint(*font_size_range)
    size = max(30, min(size, 66))
    font = ImageFont.truetype(fp, size)
    temp_size = int(size * 2.5)
    char_canvas = Image.new("RGBA", (temp_size, temp_size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(char_canvas)
    sw = random.randint(*stroke_width_range)
    draw.text(
        (temp_size // 2, temp_size // 2),
        char, fill=text_color, font=font, anchor="mm",
        stroke_width=sw, stroke_fill=text_color,
    )
    char_img = crop_to_content(char_canvas)
    sx = random.uniform(*scale_x_range)
    sy = random.uniform(*scale_y_range)
    char_img = safe_resize(char_img, char_img.width * sx, char_img.height * sy)
    shear_x = random.uniform(*shear_x_range)
    char_img = apply_affine_rgba(char_img, shear_x=shear_x)
    angle = random.uniform(*angle_range)
    char_img = char_img.rotate(angle, resample=Image.BICUBIC, expand=True)
    char_img = crop_to_content(char_img)
    char_img = maybe_offset_dot_for_i_j(char, char_img)
    return crop_to_content(char_img)

def compose_word(text, fp):
    base_size = random.randint(*font_size_range)
    char_sizes = []
    for ch in text:
        variation = random.randint(*intra_word_size_variation)
        char_size = base_size + variation
        char_size = max(30, min(char_size, 66))
        char_sizes.append(char_size)
    letters_imgs = [render_single_char(ch, fp, size_override=sz) for ch, sz in zip(text, char_sizes)]
    mode = weighted_choice(overlap_mode_probs)
    overlap_lo, overlap_hi = overlap_extra_shift[mode]
    temp_word_width = 500
    temp_word_height = 250
    word_canvas = Image.new("RGBA", (temp_word_width, temp_word_height), (255, 255, 255, 0))
    current_x = random.randint(15, 30)
    for i, char_img in enumerate(letters_imgs):
        y_offset = random.randint(*y_offset_range)
        x_jitter = random.randint(*char_x_jitter_range)
        current_y = (temp_word_height - char_img.height) // 2 + y_offset
        paste_x = int(current_x + x_jitter)
        paste_x = max(0, paste_x)
        current_y = max(0, min(current_y, temp_word_height - char_img.height))
        word_canvas.alpha_composite(char_img, (paste_x, int(current_y)))
        if i < len(letters_imgs) - 1:
            base_kern = random.uniform(*kerning_range)
            extra_overlap = random.uniform(overlap_lo, overlap_hi)
            current_x += char_img.width + base_kern - extra_overlap
    word_img = crop_to_content(word_canvas)
    ws = random.uniform(*word_scale_range)
    word_img = safe_resize(word_img, word_img.width * ws, word_img.height * ws)
    word_img = crop_to_content(word_img)
    word_img = sharpen_alpha(word_img)
    return word_img, mode

def _generate_single_image(text, fp):
    final_word_img, mode = compose_word(text, fp)
    max_w = image_width - (margin * 2)
    max_h = image_height - (margin * 2)
    if final_word_img.width > max_w or final_word_img.height > max_h:
        ratio = min(max_w / final_word_img.width, max_h / final_word_img.height)
        ratio *= random.uniform(0.98, 1.0)
        final_word_img = safe_resize(
            final_word_img,
            final_word_img.width * ratio,
            final_word_img.height * ratio
        )
        final_word_img = crop_to_content(final_word_img)
    else:
        current_fill = final_word_img.width / max_w
        if current_fill < 0.64:
            target_fill = random.uniform(0.67, 0.76)
            scale_up = target_fill / current_fill
            final_word_img = safe_resize(
                final_word_img,
                final_word_img.width * scale_up,
                final_word_img.height * scale_up
            )
            final_word_img = crop_to_content(final_word_img)
            if final_word_img.width > max_w or final_word_img.height > max_h:
                ratio = min(max_w / final_word_img.width, max_h / final_word_img.height)
                final_word_img = safe_resize(
                    final_word_img,
                    final_word_img.width * ratio,
                    final_word_img.height * ratio
                )
                final_word_img = crop_to_content(final_word_img)

    final_img = Image.new("RGB", (image_width, image_height), color=bg_color)
    final_x = (image_width - final_word_img.width) // 2 + random.randint(*global_shift_x_range)
    final_y = (image_height - final_word_img.height) // 2 + random.randint(*global_shift_y_range)
    final_x = max(margin, min(final_x, image_width - final_word_img.width - margin))
    final_y = max(margin, min(final_y, image_height - final_word_img.height - margin))
    final_img.paste(final_word_img, (int(final_x), int(final_y)), final_word_img)
    if random.random() < blur_prob:
        final_img = final_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return final_img

def generate_and_save(args):
    text, split_dir, font_path_local = args
    filename = f"{text}.png"
    filepath = os.path.join(split_dir, filename)
    if os.path.exists(filepath):
        return (filename, text, True)
    try:
        img = _generate_single_image(text, font_path_local)
        img.save(filepath)
        return (filename, text, False)
    except Exception as e:
        print(f"生成 {text}.png 時發生錯誤: {e}")
        return None

if __name__ == "__main__":
    print(f"有 {cpu_count()} 個 CPU 核心 使用 {NUM_WORKERS} 個 Worker 進程\n")
    print("生成不重複的文字組合")
    all_combinations = ["".join(p) for p in itertools.product(string.ascii_lowercase, repeat=4)]
    selected_texts = random.sample(all_combinations, num_images)
    train_count = int(num_images * train_ratio)
    val_count = int(num_images * val_ratio)
    datasets = {
        "train": selected_texts[:train_count],
        "val": selected_texts[train_count:train_count + val_count],
        "test": selected_texts[train_count + val_count:],
    }
    start_time = time.time()
    global_count = 0
    for split_name, texts in datasets.items():
        split_dir = os.path.join(output_dir, split_name)
        label_file_path = os.path.join(split_dir, "labels.txt")
        existing_labels = set()
        if os.path.exists(label_file_path):
            with open(label_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if parts:
                        existing_labels.add(parts[0])
        print(f"\n開始生成 [{split_name}] 資料集 共 {len(texts)} 張")
        split_start = time.time()
        task_args = [(text, split_dir, font_path) for text in texts]
        label_buffer = []
        with Pool(processes=NUM_WORKERS) as pool:
            for result in pool.imap_unordered(generate_and_save, task_args, chunksize=64):
                if result is None:
                    continue
                filename, text, skipped = result
                if not skipped and filename not in existing_labels:
                    label_buffer.append(f"{filename}\t{text}\n")
                global_count += 1
                if global_count % BATCH_SIZE == 0:
                    if label_buffer:
                        with open(label_file_path, "a", encoding="utf-8") as lf:
                            lf.writelines(label_buffer)
                        label_buffer.clear()
                    elapsed = time.time() - start_time
                    speed = global_count / elapsed
                    eta = (num_images - global_count) / speed if speed > 0 else 0
                    print(
                        f"  進度 {global_count:>7,} / {num_images:,} "
                        f"({global_count / num_images * 100:.1f}%) | "
                        f"速度 {speed:.0f} 張/秒 | "
                        f"預估剩餘 {eta:.0f} 秒"
                    )
        if label_buffer:
            with open(label_file_path, "a", encoding="utf-8") as lf:
                lf.writelines(label_buffer)
        split_elapsed = time.time() - split_start
        print(f"  [{split_name}] 完成 耗時 {split_elapsed:.1f} 秒")
    total_time = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"全部完成 總耗時 {total_time:.1f} 秒 ({total_time / 60:.1f} 分鐘)")
    print(f"平均速度 {num_images / total_time:.0f} 張/秒")
    print(f"輸出目錄 {output_dir}/")
