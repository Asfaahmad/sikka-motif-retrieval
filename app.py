import glob, os, re, time
import numpy as np, cv2, streamlit as st, torch, torchvision

st.set_page_config(page_title="Temu Kembali Motif Sikka", page_icon="🧵", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.judul-model {font-size: 1.05rem; font-weight: 700; margin: 0.6rem 0 0.2rem 0;}
.info-model {color: #666; font-size: 0.85rem; margin-bottom: 0.4rem;}
.ok {color: #1a7f37; font-weight: 700;}
.no {color: #cf222e; font-weight: 700;}
.cap {font-size: 0.8rem; line-height: 1.25; margin-top: -0.4rem;}
</style>
""", unsafe_allow_html=True)

COND_ID = {"indoor_morning": "dalam–pagi", "indoor_noon": "dalam–siang", "indoor_night": "dalam–malam",
           "outdoor_morning": "luar–pagi", "outdoor_noon": "luar–siang", "outdoor_night": "luar–malam"}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def gray_clahe(rgb):
    g = CLAHE.apply(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    return np.repeat(g[..., None], 3, axis=2)


def hsv_hist(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()


def to_jpg(rgb, size=320):
    im = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = {"Model 1": "Histogram HSV 16×4×4 (256-d)",
          "Model 2": "ResNet-18 pralatih, RGB (512-d)",
          "Model 3": "Grayscale + CLAHE → ResNet-18 (512-d, usulan)"}
DEMO_KET = {"gelap": "Lebih gelap, seperti difoto malam", "kuning": "Lampu kuning",
            "crop": "Dipotong dan diputar 6 derajat", "biru": "Lampu biru dan sedikit gelap",
            "terang": "Terlalu terang dan ada noise kamera"}


def norm_cond(name):
    t = name.lower()
    io = "outdoor" if "outdoor" in t else "indoor"
    tm = next((k for k in ("morning", "noon", "night") if k in t), "noon")
    return f"{io}_{tm}"


@st.cache_resource(show_spinner="Menyiapkan model dan menghitung fitur 432 foto galeri (sekali saja, ±1 menit) ...")
def load():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    m = m.eval().to(dev)
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "*", "*.jpg")))
    idx = {"paths": paths, "kelas": [], "kondisi": [], "kunci": [], "thumbs": [], "models": MODELS, "demo": []}
    feats = {k: [] for k in MODELS}
    batch = []

    def flush():
        if not batch:
            return
        for name, imgs in (("Model 2", batch), ("Model 3", [gray_clahe(r) for r in batch])):
            x = np.stack([((cv2.resize(r, (224, 224), interpolation=cv2.INTER_AREA) / 255.0 - MEAN) / STD)
                          .transpose(2, 0, 1) for r in imgs]).astype(np.float32)
            with torch.no_grad():
                feats[name].append(m(torch.from_numpy(x).to(dev)).cpu().numpy())
        batch.clear()

    for p in paths:
        rgb = cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA)
        fn = os.path.basename(p)
        idx["kelas"].append(os.path.basename(os.path.dirname(p)))
        idx["kondisi"].append(norm_cond(fn))
        idx["kunci"].append(re.sub(r"\.rf\.[0-9a-f]+\.jpg$", "", fn))
        idx["thumbs"].append(to_jpg(rgb, 256))
        feats["Model 1"].append(hsv_hist(rgb)[None])
        batch.append(rgb)
        if len(batch) == 32:
            flush()
    flush()
    idx["feats"] = {}
    for k in MODELS:
        f = np.concatenate(feats[k]).astype(np.float32)
        idx["feats"][k] = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
    for p in sorted(glob.glob(os.path.join(ROOT, "demo", "*.jpg"))):
        fn = os.path.basename(p)
        motif = next((k for k in sorted(set(idx["kelas"]), key=len, reverse=True) if k in fn), None)
        jenis = os.path.splitext(fn)[0].split("_")[-1]
        idx["demo"].append({"nama": fn, "motif": motif, "ket": DEMO_KET.get(jenis, ""), "jpg": open(p, "rb").read()})
    return idx, m, dev


idx, net, DEV = load()


def resnet(rgb):
    x = ((cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
    with torch.no_grad():
        return net(torch.from_numpy(x[None].astype(np.float32)).to(DEV)).cpu().numpy()[0]


def feature(model, rgb):
    if model == "Model 1":
        f = hsv_hist(rgb)
    elif model == "Model 2":
        f = resnet(rgb)
    else:
        f = resnet(gray_clahe(rgb))
    return f / (np.linalg.norm(f) + 1e-12)


LIGHT = {
    "Asli (tanpa perubahan)": lambda f: f,
    "Gelap (×0,65)": lambda f: f * 0.65,
    "Terang (×1,35)": lambda f: f * 1.35,
    "Gamma 2,0 (bayangan pekat)": lambda f: 255.0 * (f / 255.0) ** 2.0,
    "Lampu kuning": lambda f: f * np.array([1.15, 1.0, 0.75], np.float32),
    "Lampu biru": lambda f: f * np.array([0.8, 1.0, 1.2], np.float32),
}


def apply_light(rgb, mode, gain):
    f = LIGHT[mode](rgb.astype(np.float32)) * gain
    return np.clip(f, 0, 255).astype(np.uint8)


def decode(b):
    bgr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Pengaturan")
sumber = st.sidebar.radio("Sumber foto kueri", ["Contoh foto demo", "Foto dari dataset", "Unggah foto sendiri"])
query, motif_benar, self_idx, ket = None, None, None, ""
if sumber == "Contoh foto demo":
    names = [d["nama"] for d in idx["demo"]]
    pilih = st.sidebar.selectbox("Pilih foto demo", names)
    d = idx["demo"][names.index(pilih)]
    query, motif_benar, ket = decode(d["jpg"]), d["motif"], d["ket"]
elif sumber == "Foto dari dataset":
    motifs = sorted(set(idx["kelas"]))
    mtf = st.sidebar.selectbox("Motif", motifs)
    pos = [i for i, k in enumerate(idx["kelas"]) if k == mtf]
    j = st.sidebar.selectbox("Foto", pos, format_func=lambda i: f"{idx['kunci'][i]} ({COND_ID[idx['kondisi'][i]]})")
    query = cv2.resize(cv2.cvtColor(cv2.imread(idx["paths"][j]), cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA)
    motif_benar, self_idx = mtf, j
    ket = f"Foto asli dataset, kondisi {COND_ID[idx['kondisi'][j]]}"
else:
    up = st.sidebar.file_uploader("Unggah foto kain (jpg/png)", type=["jpg", "jpeg", "png"])
    if up:
        query = decode(up.read())
        ket = "Foto unggahan pengguna (motif sebenarnya tidak diketahui)"

st.sidebar.subheader("Simulasi pencahayaan")
cahaya = st.sidebar.selectbox("Kondisi cahaya kueri", list(LIGHT))
gain = st.sidebar.slider("Kecerahan tambahan", 0.4, 1.6, 1.0, 0.05)

st.sidebar.subheader("Tampilan hasil")
mode = st.sidebar.radio("Mode", ["Bandingkan 3 model", "Satu model"])
model_pilih = list(MODELS) if mode == "Bandingkan 3 model" else [
    st.sidebar.selectbox("Model", list(MODELS), index=2, format_func=lambda m: f"{m}: {MODELS[m]}")]
k = st.sidebar.slider("Jumlah hasil teratas", 3, 10, 5)
if self_idx is not None:
    buang_diri = st.sidebar.checkbox("Kecualikan foto kueri itu sendiri dari hasil", value=True)
else:
    buang_diri = False

# ---------------------------------------------------------------- halaman utama
st.title("Temu Kembali Citra Motif Tenun Ikat Sikka")
st.caption("Pencarian motif berbasis konten pada 432 foto dataset Motif-Sikka v3 (Reja, 2025, CC BY 4.0) · "
           "kemiripan kosinus · Model 1 histogram HSV · Model 2 ResNet-18 · Model 3 grayscale + CLAHE → ResNet-18 (usulan)")

if query is None:
    st.info("Pilih foto demo atau foto dari dataset di panel kiri, atau unggah foto kain sendiri.")
    st.stop()

q = apply_light(query, cahaya, gain)
c1, c2, c3 = st.columns([1, 1, 2])
c1.image(to_jpg(query), caption="Foto kueri asli", width="stretch")
c2.image(to_jpg(q), caption=f"Setelah simulasi: {cahaya}, kecerahan ×{gain:.2f}".replace(".", ","), width="stretch")
with c3:
    st.markdown(f"**Keterangan:** {ket}")
    if motif_benar:
        st.markdown(f"**Motif sebenarnya:** `{motif_benar}`")
    st.markdown("Ubah *kondisi cahaya* di panel kiri untuk melihat model mana yang tetap stabil. "
                "✓ = motif hasil sama dengan motif kueri.")
    st.caption(f"Perangkat: {DEV.upper()}")

ringkas = []
for m in model_pilih:
    t0 = time.perf_counter()
    s = idx["feats"][m] @ feature(m, q)
    if buang_diri:
        s[self_idx] = -np.inf
    top = np.argsort(-s)[:k]
    dt = (time.perf_counter() - t0) * 1000
    benar = sum(idx["kelas"][j] == motif_benar for j in top) if motif_benar else None
    ringkas.append((m, benar, dt))
    st.markdown(f"<div class='judul-model'>{m}: {MODELS[m]}</div>", unsafe_allow_html=True)
    info = f"waktu pencarian {dt:.0f} ms".replace(".", ",")
    if benar is not None:
        info = f"motif benar {benar}/{k} · " + info
    st.markdown(f"<div class='info-model'>{info}</div>", unsafe_allow_html=True)
    cols = st.columns(k)
    for c, j in zip(cols, top):
        c.image(idx["thumbs"][j], width="stretch")
        tanda = ""
        if motif_benar:
            tanda = "<span class='ok'>✓</span> " if idx["kelas"][j] == motif_benar else "<span class='no'>✗</span> "
        c.markdown(f"<div class='cap'>{tanda}{idx['kelas'][j]}<br>{COND_ID[idx['kondisi'][j]]} · skor {s[j]:.3f}</div>",
                   unsafe_allow_html=True)

if motif_benar and len(ringkas) > 1:
    st.divider()
    st.subheader("Ringkasan perbandingan")
    st.table({"Model": [f"{m}: {MODELS[m]}" for m, _, _ in ringkas],
              f"Motif benar di {k} teratas": [f"{b}/{k}" for _, b, _ in ringkas],
              "Waktu pencarian (ms)": [f"{d:.0f}" for _, _, d in ringkas]})

with st.expander("Tentang aplikasi ini"):
    st.markdown(
        "Prototipe ini adalah bagian dari Final Project Temu Kembali Citra: *Evaluasi Ketahanan Temu Kembali Citra "
        "Motif Tenun Ikat Sikka terhadap Variasi Pencahayaan*. Setiap foto galeri dan kueri diubah menjadi vektor fitur, "
        "lalu diurutkan berdasarkan kemiripan kosinus. Foto demo adalah foto dataset yang dimodifikasi pencahayaannya "
        "atau sudutnya, sehingga motif sebenarnya diketahui.")
