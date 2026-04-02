from svgelements import SVG, Path, Group, Color
from PIL import Image, ImageDraw, ImageColor
from skimage import morphology, measure, color as skcolor
from skimage.filters import gaussian
import numpy as np
import os
import sys
import traceback

# ==== 结构元：兼容新旧版本 ====
try:
    FP3 = morphology.footprint_rectangle((3, 3))  # 新版
except AttributeError:
    FP3 = morphology.square(3)                    # 旧版回退

def log(msg):
    print(msg, file=sys.stdout, flush=True)

# ==== 颜色分割（Lab ΔE + 形态学清理）====
def extract_color_mask(img, target_rgb, de_thresh=12, blur_sigma=0.6, min_area=64):
    data = np.array(img).astype(np.float32) / 255.0
    lab = skcolor.rgb2lab(data)
    target_lab = skcolor.rgb2lab(np.array(target_rgb, dtype=np.float32)[None, None, :] / 255.0)
    de = np.linalg.norm(lab - target_lab, axis=-1)
    de_smooth = gaussian(de, sigma=blur_sigma, preserve_range=True)
    mask = de_smooth <= de_thresh
    mask = morphology.binary_closing(mask, FP3)
    mask = morphology.remove_small_holes(mask, area_threshold=min_area * 2)
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    return (mask.astype(np.uint8) * 255)

# ==== 轮廓 -> Path（简化）====
def contours_to_path(contours, color_hex, simplify_tol=1.2, round_to_int=True):
    path = Path(fill=color_hex, stroke="none")
    for c in contours:
        c_simple = measure.approximate_polygon(c, tolerance=simplify_tol)
        if len(c_simple) < 3:
            continue
        pts = [(float(x), float(y)) for y, x in c_simple]
        if round_to_int:
            pts = [(int(round(x)), int(round(y))) for x, y in pts]
        d = "M" + " L".join(f"{x},{y}" for x, y in pts) + " Z"
        path += Path(d)
    return path

def mask_to_svg_path(mask, color_hex, simplify_tol=1.2):
    m = (mask > 0).astype(np.uint8)
    m = morphology.binary_dilation(m, FP3).astype(np.uint8)
    contours = measure.find_contours(m, 0.5)
    return contours_to_path(contours, color_hex, simplify_tol=simplify_tol)

# ==== 颜色转 RGBA ====
from typing import Tuple
def _to_rgba_tuple(fill) -> Tuple[int, int, int, int]:
    if fill is None:
        return (0, 0, 0, 0)
    if isinstance(fill, Color):
        t = fill.tuple
        if len(t) == 4:
            r, g, b, a = t
        else:
            r, g, b = t
            a = 255
        a = int(round(a if a > 1 else a * 255))
        return (int(r), int(g), int(b), a)
    if isinstance(fill, str):
        rgb = ImageColor.getrgb(fill)
        return (rgb[0], rgb[1], rgb[2], 255)
    if isinstance(fill, (tuple, list)):
        if len(fill) == 3:
            return (int(fill[0]), int(fill[1]), int(fill[2]), 255)
        if len(fill) == 4:
            r, g, b, a = fill
            a = int(round(a if a > 1 else a * 255))
            return (int(r), int(g), int(b), a)
    return (0, 0, 0, 255)

# ==== 写出 SVG + PNG ====
def save_svg_and_png(svg_obj, output_svg, output_png, w, h, raster_scale=4):
    # 强制添加命名空间，防止 CairoSVG 报错
    if getattr(svg_obj, "xmlns", None) in (None, ""):
        try:
            svg_obj.values["xmlns"] = "http://www.w3.org/2000/svg"
            svg_obj.values["xmlns:xlink"] = "http://www.w3.org/1999/xlink"
        except Exception:
            pass

    svg_text = str(svg_obj)

    # 先写 SVG
    with open(output_svg, "w", encoding="utf-8") as f:
        f.write(svg_text)
    log(f"📝 SVG written -> {output_svg}")

    # 尝试用 CairoSVG
    try:
        import cairosvg
        log("🔧 Using CairoSVG to rasterize PNG ...")
        cairosvg.svg2png(bytestring=svg_text.encode("utf-8"),
                         write_to=output_png,
                         output_width=w,
                         output_height=h)
        log(f"🖼️ PNG written by CairoSVG -> {output_png}")
        return
    except Exception as e:
        log("⚠️ CairoSVG failed, falling back to Pillow rasterizer.")
        log("CairoSVG error:\n" + "".join(traceback.format_exception_only(type(e), e)).strip())

    # 备用：Pillow 超采样绘制（只支持本脚本生成的 M/L/Z 折线路径）
    scale = max(2, int(raster_scale))
    big_w, big_h = w * scale, h * scale
    img = Image.new("RGBA", (big_w, big_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")

    # 遍历 Group -> Path
    for elem in svg_obj:
        if isinstance(elem, Group):
            for p in elem:
                if not isinstance(p, Path):
                    continue
                if not getattr(p, "d", None):  # 跳过空路径
                    continue
                fill_rgba = _to_rgba_tuple(p.fill)
                d = p.d
                tokens = d.replace(",", " ").split()
                polys, curr = [], []
                i = 0
                while i < len(tokens):
                    t = tokens[i]
                    if t.upper() in ("M", "L", "Z"):
                        if t.upper() == "Z" and curr:
                            polys.append(curr)
                            curr = []
                        i += 1
                        continue
                    # 读取坐标
                    try:
                        x = float(tokens[i]); y = float(tokens[i+1]); i += 2
                        curr.append((x * scale, y * scale))
                    except Exception:
                        i += 1
                if curr:
                    polys.append(curr)
                for poly in polys:
                    if len(poly) >= 3:
                        draw.polygon(poly, fill=fill_rgba)

    img_small = img.resize((w, h), Image.LANCZOS)
    img_small.save(output_png)
    log(f"🖼️ PNG written by Pillow -> {output_png}")

# ==== 主流程 ====
def convert_to_vector(input_path, output_svg, output_png):
    img = Image.open(input_path).convert("RGB")

    # 颜色目标
    red = [212, 0, 0]   # #D40000
    black = [0, 0, 0]

    # 分割
    log("🔍 Segmenting colors ...")
    red_mask = extract_color_mask(img, red,  de_thresh=12, blur_sigma=0.6, min_area=80)
    blk_mask = extract_color_mask(img, black, de_thresh=10, blur_sigma=0.6, min_area=80)

    # 生成 SVG（加 xmlns）
    svg = SVG(width=img.width, height=img.height,
              viewBox=f"0 0 {img.width} {img.height}",
              xmlns="http://www.w3.org/2000/svg",
              **{"xmlns:xlink": "http://www.w3.org/1999/xlink"})
    group = Group()

    # 顺序：底色 -> 前景
    log("✏️ Tracing contours and building paths ...")
    group.append(mask_to_svg_path(blk_mask, "#000000", simplify_tol=1.5))
    group.append(mask_to_svg_path(red_mask, "#D40000", simplify_tol=1.2))

    svg.append(group)

    # 写出
    log("💾 Writing SVG & PNG ...")
    save_svg_and_png(svg, output_svg, output_png, img.width, img.height, raster_scale=4)

# ==== 入口 ====
if __name__ == "__main__":
    input_path = "/Users/shul/projects/picture-flag-han/input.png"
    out_dir = "/Users/shul/projects/picture-flag-han/out"
    os.makedirs(out_dir, exist_ok=True)
    output_svg = os.path.join(out_dir, "output.svg")
    output_png = os.path.join(out_dir, "output.png")

    log(f">>> 当前工作目录 = {os.getcwd()}")
    log(f">>> 输入文件存在 = {os.path.exists(input_path)}  ->  {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    convert_to_vector(input_path, output_svg, output_png)

    for p in (output_svg, output_png):
        if os.path.exists(p):
            log(f"✅ 已生成: {p} ({os.path.getsize(p)} bytes)")
        else:
            log(f"❌ 未找到输出文件: {p}")
