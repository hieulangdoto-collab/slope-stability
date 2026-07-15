"""
streamlit_app.py
================
Giao dien web nhanh cho bishop_engine.py, dung Streamlit - phu hop khi ban
muon co mot web app THAT (co the deploy len Streamlit Community Cloud,
Render, hay noi bo cong ty) ma khong can tu viet frontend rieng.

Chay thu:
    pip install streamlit matplotlib --break-system-packages
    streamlit run streamlit_app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from bishop_engine import SoilLayer, layer_bounds_from, ground_y, grid_search

st.set_page_config(page_title="Phân tích ổn định mái dốc", layout="wide")

st.title("🏔️ Công cụ phân tích ổn định mái dốc")
st.caption("Bishop đơn giản hóa · đối chiếu chéo Fellenius · đất nhiều lớp · KHÔNG thay thế phần mềm đã kiểm định")

col_input, col_output = st.columns([1, 2], gap="large")

with col_input:
    st.subheader("Hình học")
    H = st.slider("Chiều cao mái dốc H (m)", 4.0, 40.0, 12.0, 0.5)
    beta_deg = st.slider("Góc mái dốc β (°)", 15, 70, 32)

    st.subheader("Lớp đất")
    n_layers = st.radio("Số lớp đất", [1, 2, 3], index=1, horizontal=True)
    layers = []
    for i in range(n_layers):
        with st.expander(f"Lớp {i + 1}", expanded=True):
            is_last = i == n_layers - 1
            if not is_last:
                thickness = st.number_input(f"Bề dày lớp {i + 1} (m)", 0.5, 30.0, 6.0, 0.5, key=f"th{i}")
            else:
                thickness = None
                st.caption("Lớp cuối: bề dày không giới hạn")
            gamma = st.number_input(f"Dung trọng γ lớp {i + 1} (kN/m³)", 14.0, 22.0, 18.5, 0.5, key=f"g{i}")
            cohesion = st.number_input(f"Lực dính c lớp {i + 1} (kPa)", 0.0, 100.0, 12.0, 1.0, key=f"c{i}")
            phi_deg = st.number_input(f"Góc ma sát φ lớp {i + 1} (°)", 0.0, 45.0, 26.0, 1.0, key=f"p{i}")
            layers.append(SoilLayer(thickness=thickness, gamma=gamma, cohesion=cohesion, phi_deg=phi_deg))

    st.subheader("Nước ngầm")
    water_on = st.checkbox("Có mực nước ngầm", value=False)
    water_depth = None
    if water_on:
        water_depth = st.slider("Độ sâu dưới đỉnh mái (m)", 0.0, H, min(6.0, H), 0.5)

    run = st.button("🔍 Phân tích ổn định", type="primary", use_container_width=True)

with col_output:
    if run:
        import math

        L = H / math.tan(math.radians(beta_deg))
        bounds = layer_bounds_from(layers)
        water_y = (H - water_depth) if water_depth is not None else None

        with st.spinner("Đang tìm mặt trượt nguy hiểm nhất…"):
            result = grid_search(H, L, bounds, water_y)

        if result is None:
            st.error("Không tìm được mặt trượt hợp lệ cho hình học này. Thử giảm góc dốc hoặc kiểm tra lại thông số.")
        else:
            fs = result.fs_bishop
            if fs < 1.0:
                status, color = "KHÔNG ỔN ĐỊNH", "#E4572E"
            elif fs < 1.3:
                status, color = "CẦN XEM XÉT", "#E0B03E"
            else:
                status, color = "ỔN ĐỊNH", "#6FBE8D"

            c1, c2, c3 = st.columns(3)
            c1.metric("FS — Bishop đơn giản hóa", f"{fs:.3f}")
            c2.metric("FS — Fellenius (đối chiếu)", f"{result.fs_fellenius:.3f}" if result.fs_fellenius else "—")
            diff = abs(fs - result.fs_fellenius) / result.fs_fellenius * 100 if result.fs_fellenius else None
            c3.metric("Chênh lệch 2 phương pháp", f"{diff:.1f}%" if diff is not None else "—")

            st.markdown(
                f"<div style='padding:10px 16px;border-radius:6px;border:2px solid {color};"
                f"display:inline-block;font-weight:700;color:{color};'>{status}</div>",
                unsafe_allow_html=True,
            )

            # ---- ve mat cat ----
            fig, ax = plt.subplots(figsize=(9, 5.5))
            top_extra = max(0.6 * H, 0.3 * L)
            bot_extra = max(0.6 * H, 0.3 * L)
            xs = np.linspace(-top_extra, L + bot_extra, 400)
            gys = [ground_y(x, H, L) for x in xs]
            ax.plot(xs, gys, color="black", linewidth=2)
            ax.fill_between(xs, gys, -0.3 * H, color="#C97B3D", alpha=0.35)

            xs_slip = np.linspace(result.entry_x, L, 200)
            ys_slip = [
                result.yc - math.sqrt(max(0, result.R**2 - (x - result.xc) ** 2)) for x in xs_slip
            ]
            ax.plot(xs_slip, ys_slip, color="#B23A2E", linewidth=2.5, label="Mặt trượt nguy hiểm nhất")
            ax.fill_between(xs_slip, ys_slip, [ground_y(x, H, L) for x in xs_slip], color=color, alpha=0.3)

            if water_y is not None:
                ax.axhline(water_y, color="#3477A8", linestyle="--", linewidth=1.5, label="Mực nước ngầm")

            ax.set_xlabel("x (m)")
            ax.set_ylabel("Cao độ (m)")
            ax.set_title(f"Mặt cắt ngang — FS = {fs:.2f} ({status})")
            ax.legend(loc="upper right")
            ax.set_aspect("equal")
            st.pyplot(fig)

            st.caption(
                "Công cụ minh họa: đất nhiều lớp song song bề mặt, một mặt trượt tròn, grid search tìm mặt trượt "
                "nguy hiểm nhất, giải theo Bishop đơn giản hóa và đối chiếu chéo với Fellenius. Không thay thế "
                "phần mềm đã kiểm định (SLOPE/W, Slide…) cho thiết kế thực tế."
            )
    else:
        st.info("Thiết lập thông số bên trái rồi bấm **Phân tích ổn định**.")
