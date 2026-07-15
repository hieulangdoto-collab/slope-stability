"""
bishop_engine.py
================
Engine tinh on dinh mai doc: mat truot tron, dat nhieu lop song song voi
be mat, Bishop don gian hoa + Fellenius (doi chieu cheo), grid search de
tim mat truot nguy hiem nhat.

THAM SO HOA VONG TRON: thay vi ep tam+ban kinh roi do nguoc diem vao (cach
lam de mat truot bi keo dai bat hop ly ra sau dinh mai), ban nay chon
TRUC TIEP hai dau mut cung truot tren mat dat - diem vao (entry_x, phia
tren mai) va diem ra co dinh tai chan doc (L, 0) - cong voi do sau cung
(sagitta) o giua, roi dung vong tron di qua dung hai diem do. Cach nay
kiem soat duoc kich thuoc khoi truot va tranh mat truot "an" ra rat xa
phia sau dinh mai.

KHONG phai phan mem da kiem dinh cho thiet ke thuc te - dung de hoc tap /
prototype / doi chieu ket qua.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import math

GAMMA_W = 9.81  # kN/m3


@dataclass
class SoilLayer:
    thickness: Optional[float]  # None cho lop cuoi (khong gioi han)
    gamma: float                # kN/m3
    cohesion: float             # kPa
    phi_deg: float              # do


@dataclass
class LayerBound:
    depth_start: float
    depth_end: float  # co the la math.inf
    gamma: float
    cohesion: float
    phi_rad: float


@dataclass
class Slice:
    b: float
    alpha: float
    W: float
    u: float
    cohesion: float
    phi_rad: float
    xm: float
    slip_y: float
    ground_y: float


@dataclass
class CircleResult:
    xc: float
    yc: float
    R: float
    entry_x: float
    sagitta: float
    fs_bishop: float
    fs_fellenius: Optional[float]
    slices: List[Slice] = field(default_factory=list)


def layer_bounds_from(layers: List[SoilLayer]) -> List[LayerBound]:
    bounds = []
    depth = 0.0
    for i, ly in enumerate(layers):
        depth_start = depth
        is_last = i == len(layers) - 1
        depth_end = math.inf if is_last else depth + ly.thickness
        depth = depth if is_last else depth_end
        bounds.append(
            LayerBound(
                depth_start=depth_start,
                depth_end=depth_end,
                gamma=ly.gamma,
                cohesion=ly.cohesion,
                phi_rad=math.radians(ly.phi_deg),
            )
        )
    return bounds


def ground_y(x: float, H: float, L: float) -> float:
    if x <= 0:
        return H
    if x <= L:
        return H - x * (H / L)
    return 0.0


def layer_at_depth(bounds: List[LayerBound], depth: float) -> LayerBound:
    for ly in bounds:
        if depth < ly.depth_end:
            return ly
    return bounds[-1]


def column_weight_and_strength(bounds: List[LayerBound], total_depth: float, b: float):
    W = 0.0
    for ly in bounds:
        top = max(0.0, ly.depth_start)
        bot = min(total_depth, ly.depth_end)
        overlap = bot - top
        if overlap > 0:
            W += ly.gamma * overlap * b
    base = layer_at_depth(bounds, max(0.0, total_depth - 1e-6))
    return W, base.cohesion, base.phi_rad


def circle_from_entry_sagitta(entry_x, H, L, sagitta):
    """Dung vong tron di qua diem vao (tren mat dat) va chan doc (L,0),
    voi do sau cung (sagitta) da cho. Tra ve (xc, yc, R) hoac None neu
    hinh hoc khong hop le."""
    x1, y1 = entry_x, ground_y(entry_x, H, L)
    x2, y2 = L, 0.0
    dx, dy = x2 - x1, y2 - y1
    chord = math.hypot(dx, dy)
    if chord < 1e-6:
        return None
    a = chord / 2  # nua day cung
    s = sagitta
    if s <= 1e-6:
        return None
    R = (a * a + s * s) / (2 * s)
    if R < a - 1e-9:
        return None

    ux, uy = dx / chord, dy / chord
    # hai phuong vuong goc voi day cung; chon huong co thanh phan y lon
    # hon lam huong "len tren" (ve phia tam vong tron)
    p1x, p1y = -uy, ux
    p2x, p2y = uy, -ux
    up = (p1x, p1y) if p1y >= p2y else (p2x, p2y)

    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    xc = mx + (R - s) * up[0]
    yc = my + (R - s) * up[1]
    return xc, yc, R


def build_slices(entry_x, L, xc, yc, R, H, bounds, water_y, n_slices=24) -> Optional[List[Slice]]:
    b = (L - entry_x) / n_slices
    if b <= 0:
        return None
    slices = []
    for i in range(n_slices):
        xL = entry_x + i * b
        xR = xL + b
        xm = (xL + xR) / 2
        dx = xm - xc
        under_sqrt = R * R - dx * dx
        if under_sqrt < 0:
            return None
        slip_y = yc - math.sqrt(under_sqrt)
        gy = ground_y(xm, H, L)
        total_depth = gy - slip_y
        if total_depth <= 1e-9:
            return None
        sin_a = max(-0.999, min(0.999, dx / R))
        alpha = math.asin(sin_a)
        if abs(alpha) > math.radians(80):
            return None
        W, cohesion, phi_rad = column_weight_and_strength(bounds, total_depth, b)
        u = GAMMA_W * (water_y - slip_y) if (water_y is not None and slip_y < water_y) else 0.0
        slices.append(Slice(b, alpha, W, u, cohesion, phi_rad, xm, slip_y, gy))
    return slices


def bishop_fs(slices: List[Slice]) -> Optional[float]:
    FS = 1.5
    for _ in range(60):
        num = 0.0
        den = 0.0
        for s in slices:
            m_alpha = math.cos(s.alpha) + (math.sin(s.alpha) * math.tan(s.phi_rad)) / FS
            if abs(m_alpha) < 1e-4:
                m_alpha = 1e-4
            num += (s.cohesion * s.b + (s.W - s.u * s.b) * math.tan(s.phi_rad)) / m_alpha
            den += s.W * math.sin(s.alpha)
        if den <= 0:
            return None
        new_fs = num / den
        if not math.isfinite(new_fs) or new_fs <= 0:
            return None
        if abs(new_fs - FS) < 1e-5:
            return new_fs
        FS = new_fs
    return FS if 0 < FS < 20 else None


def fellenius_fs(slices: List[Slice]) -> Optional[float]:
    top = 0.0
    bottom = 0.0
    for s in slices:
        cos_a = math.cos(s.alpha)
        if abs(cos_a) < 1e-4:
            continue
        top += (s.cohesion * s.b) / cos_a + (s.W * cos_a - (s.u * s.b) / cos_a) * math.tan(s.phi_rad)
        bottom += s.W * math.sin(s.alpha)
    if bottom <= 0:
        return None
    fs = top / bottom
    return fs if 0 < fs < 20 else None


def eval_candidate(entry_x, sagitta, L, H, bounds, water_y, n_slices=24) -> Optional[CircleResult]:
    if L - entry_x < 0.3 * H:
        return None
    if sagitta < 0.12 * H:
        return None  # tranh mat truot rat mong sat mat dat (khong dai dien cho co che truot sau)
    circ = circle_from_entry_sagitta(entry_x, H, L, sagitta)
    if circ is None:
        return None
    xc, yc, R = circ
    slices = build_slices(entry_x, L, xc, yc, R, H, bounds, water_y, n_slices)
    if not slices:
        return None
    fs = bishop_fs(slices)
    if not fs:
        return None
    return CircleResult(xc, yc, R, entry_x, sagitta, fs, None, slices)


def grid_search(H, L, bounds, water_y=None) -> Optional[CircleResult]:
    def scan(entry_range, sag_range, n):
        best = None
        for i in range(n):
            entry_x = entry_range[0] + (entry_range[1] - entry_range[0]) * i / (n - 1)
            for j in range(n):
                sagitta = sag_range[0] + (sag_range[1] - sag_range[0]) * j / (n - 1)
                cand = eval_candidate(entry_x, sagitta, L, H, bounds, water_y)
                if cand and (best is None or cand.fs_bishop < best.fs_bishop):
                    best = cand
        return best

    coarse_entry = (-1.3 * H, 0.85 * L)
    coarse_sag = (0.15 * H, 1.3 * H)
    coarse = scan(coarse_entry, coarse_sag, 15)
    if not coarse:
        return None

    span_e = (coarse_entry[1] - coarse_entry[0]) * 0.22
    span_s = (coarse_sag[1] - coarse_sag[0]) * 0.22
    fine = scan(
        (coarse.entry_x - span_e, min(0.92 * L, coarse.entry_x + span_e)),
        (max(0.01 * H, coarse.sagitta - span_s), coarse.sagitta + span_s),
        15,
    )
    best = fine if (fine and fine.fs_bishop < coarse.fs_bishop) else coarse
    best.fs_fellenius = fellenius_fs(best.slices)
    return best


def analyze_slope(
    H: float,
    beta_deg: float,
    layers: List[SoilLayer],
    water_depth_below_crest: Optional[float] = None,
) -> dict:
    """Ham tien ich cap cao: nhan thong so ky thuat, tra ve dict ket qua."""
    L = H / math.tan(math.radians(beta_deg))
    bounds = layer_bounds_from(layers)
    water_y = (H - water_depth_below_crest) if water_depth_below_crest is not None else None

    result = grid_search(H, L, bounds, water_y)
    if result is None:
        return {"ok": False, "message": "Khong tim duoc mat truot hop le cho hinh hoc nay."}

    fs = result.fs_bishop
    if fs < 1.0:
        status = "KHONG ON DINH"
    elif fs < 1.3:
        status = "CAN XEM XET"
    else:
        status = "ON DINH"

    return {
        "ok": True,
        "fs_bishop": round(fs, 3),
        "fs_fellenius": round(result.fs_fellenius, 3) if result.fs_fellenius else None,
        "status": status,
        "circle": {"xc": result.xc, "yc": result.yc, "R": result.R, "entry_x": result.entry_x},
        "slope_length_L": round(L, 2),
        "n_slices": len(result.slices),
    }


if __name__ == "__main__":
    layers = [
        SoilLayer(thickness=6, gamma=18.5, cohesion=12, phi_deg=26),
        SoilLayer(thickness=None, gamma=19.5, cohesion=20, phi_deg=22),
    ]
    out = analyze_slope(H=12, beta_deg=32, layers=layers)
    print("Khong nuoc ngam:", out)

    out_wet = analyze_slope(H=12, beta_deg=32, layers=layers, water_depth_below_crest=6)
    print("Co nuoc ngam:   ", out_wet)
