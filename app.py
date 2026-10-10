"""
DoğrulaAI v1.0 - Yapay Zeka Destekli Sahte Haber Tespit Sistemi
Streamlit arayüzü.

Çalıştırma:
    pip install streamlit pillow
    streamlit run app.py

Opsiyonel paketler (kuruluysa otomatik kullanılır):
    pytesseract, newspaper3k, beautifulsoup4, requests
"""

import io
import time

import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Motor entegrasyonu
# ---------------------------------------------------------------------------
try:
    from engine import analyze_news
    ENGINE_READY = True
except ImportError:
    ENGINE_READY = False

    def analyze_news(text: str) -> dict:
        """engine.py hazır olana kadar kullanılan geçici (mock) analiz fonksiyonu."""
        lowered = text.lower()
        words = text.split()

        if len(words) < 15:
            return {
                "label": "Yetersiz Veri",
                "confidence": 0.0,
                "linguistic_flags": [],
                "top_keywords": [],
                "summary_reason": "Metin, güvenilir bir karar için çok kısa. Daha uzun bir metin girin.",
            }

        flags = []
        if text.count("!") >= 2:
            flags.append("Aşırı ünlem kullanımı")
        if any(w in lowered for w in ["şok", "inanılmaz", "bomba", "son dakika", "mucize"]):
            flags.append("Clickbait / sansasyonel ifade")
        if any(w in lowered for w in ["hemen paylaş", "silinmeden", "gizli tutuluyor", "saklıyor"]):
            flags.append("Paylaşım baskısı ve komplo dili")
        if sum(1 for w in words if len(w) > 3 and w.isupper()) >= 2:
            flags.append("Büyük harfle vurgulama")

        keywords = [k for k in ["şok", "mucize", "gizli", "paylaş", "iddia", "tüik", "açıkladı", "veri"]
                    if k in lowered][:6]

        if len(flags) >= 3:
            label, conf = "Sahte", 0.92
            reason = "Metinde birden fazla manipülatif dil kalıbı ve doğrulanamayan iddia tespit edildi."
        elif flags or "iddia" in lowered:
            label, conf = "Şüpheli", 0.66
            reason = "Metin kısmen sansasyonel veya kaynaksız ifadeler içeriyor; doğrulama önerilir."
        else:
            label, conf = "Gerçek", 0.88
            reason = "Metin nötr bir üslupla yazılmış ve belirgin bir manipülasyon belirtisi içermiyor."

        return {
            "label": label,
            "confidence": conf,
            "linguistic_flags": flags,
            "top_keywords": keywords,
            "summary_reason": reason + " (Geçici mock motor çıktısı.)",
        }


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
SAMPLES = {
    "real": (
        "Türkiye İstatistik Kurumu (TÜİK) tarafından açıklanan verilere göre yıllık enflasyon "
        "oranı geçen aya kıyasla 1,2 puan gerileyerek yüzde 48,6 seviyesine indi. Kurum, "
        "hesaplamada kullanılan sepet ve ağırlıkların yöntem belgesinde ayrıntılı biçimde "
        "yayımlandığını belirtti. Ekonomistler düşüşün gıda ve enerji fiyatlarındaki yavaşlamadan "
        "kaynaklandığını, yıl sonu beklentilerinin ise temkinli kaldığını ifade etti."
    ),
    "suspicious": (
        "Sosyal medyada dolaşan bilgilere göre bir gıda markasının bazı ürünlerinde zararlı madde "
        "bulunduğu iddia ediliyor. Kaynağı belirtilmeyen paylaşımlarda yetkililerin konuyu "
        "kamuoyundan sakladığı öne sürülüyor. Henüz resmi bir açıklama yapılmadı, haberin doğruluğu "
        "teyit edilemedi ancak uzmanlar tüketicilerin dikkatli olması gerektiğini söylüyor."
    ),
    "fake": (
        "ŞOK ŞOK ŞOK!!! Doktorların SAKLADIĞI gerçek ortaya çıktı! Mutfağınızdaki bu sıradan "
        "baharat 3 günde bütün kanserleri yok ediyor ama ilaç firmaları bunu sizden gizli tutuyor! "
        "Silinmeden önce HEMEN PAYLAŞ, herkes bilmeli! İnanılmaz mucize tedavi, kimse konuşmuyor!!!"
    ),
}

OCR_FALLBACK_TEXT = (
    "SON DAKİKA: Ünlü bir markanın depolarında yıllardır saklanan gizli belgeler ortaya çıktı! "
    "Yetkililer sessizliğini koruyor. Herkes bu gerçeği öğrenmeli, silinmeden hemen paylaşın!"
)

RESULT_STYLES = {
    "Gerçek": ("✅", st.success),
    "Şüpheli": ("⚠️", st.warning),
    "Sahte": ("🚨", st.error),
    "Yetersiz Veri": ("ℹ️", st.info),
}


# ---------------------------------------------------------------------------
# Durum yönetimi ve callback'ler
# ---------------------------------------------------------------------------
def init_state() -> None:
    for key in ("news_text", "ocr_text", "url_text", "ocr_source"):
        st.session_state.setdefault(key, "")


def set_sample(name: str) -> None:
    st.session_state["news_text"] = SAMPLES[name]


def clear_text(key: str) -> None:
    st.session_state[key] = ""


# ---------------------------------------------------------------------------
# İş mantığı yardımcıları
# ---------------------------------------------------------------------------
def extract_text_from_image(image: Image.Image) -> tuple[str, bool]:
    """Görselden metin çıkarır. (metin, gerçek_ocr_mu) döner; hata olursa simüle eder."""
    try:
        import pytesseract

        text = pytesseract.image_to_string(image, lang="tur+eng").strip()
        if text:
            return text, True
    except Exception:
        pass
    return OCR_FALLBACK_TEXT, False


def fetch_article_text(url: str) -> tuple[str, str]:
    """URL'den metin çekmeyi dener. (metin, hata_mesajı) döner; başarılıysa hata boştur."""
    if not url.startswith(("http://", "https://")):
        return "", "Lütfen http:// veya https:// ile başlayan geçerli bir bağlantı girin."

    try:
        from newspaper import Article

        article = Article(url, language="tr")
        article.download()
        article.parse()
        if article.text.strip():
            return f"{article.title}\n\n{article.text}".strip(), ""
    except Exception:
        pass

    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = "\n".join(p for p in paragraphs if len(p) > 40)
        if text:
            return text, ""
        return "", "Sayfada okunabilir haber metni bulunamadı. Metni kopyalayıp yapıştırabilirsiniz."
    except ImportError:
        return "", ("Web içerik çekme modülü henüz hazır değil "
                    "(newspaper3k / beautifulsoup4 kurulu değil). Metni aşağıya elle yapıştırabilirsiniz.")
    except Exception as exc:
        return "", f"Sayfa okunamadı ({type(exc).__name__}). Metni elle yapıştırmayı deneyin."


def run_analysis(text: str) -> tuple[dict, float]:
    """Motoru çalıştırır; sonucu ve ölçülen süreyi (ms) döndürür."""
    start = time.perf_counter()
    result = analyze_news(text)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


# ---------------------------------------------------------------------------
# Arayüz bileşenleri
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
            .block-container {max-width: 1100px; padding-top: 2rem;}
            .app-header {
                padding: 1.4rem 1.6rem; border-radius: 12px; margin-bottom: 1.2rem;
                background: #0f2a4a; color: #ffffff;
            }
            .app-header h1 {margin: 0; font-size: 2rem; color: #ffffff;}
            .app-header p {margin: .35rem 0 0; color: #cbd8e8; font-size: 1rem;}
            .kw-badge {
                display: inline-block; padding: .2rem .7rem; margin: 0 .35rem .35rem 0;
                border-radius: 999px; background: #e8eef7; color: #1d3a63;
                font-size: .85rem; font-weight: 500; border: 1px solid #c9d6ea;
            }
            .counter {color: #6b7280; font-size: .85rem; margin-top: -.4rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <h1>🔍 DoğrulaAI</h1>
            <p>Haber metnini, ekran görüntüsünü veya bağlantısını girin; sistem içeriğin güvenilirliğini,
            üslup kusurlarını ve karara etki eden anahtar kelimeleri analiz etsin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not ENGINE_READY:
        st.caption("⚙️ `engine.py` bulunamadı, geçici mock motor kullanılıyor.")


def render_sidebar() -> None:
    with st.sidebar:
        st.title("DoğrulaAI v1.0")
        st.caption("SRS Uyumlu Prototip")

        st.subheader("Metodoloji")
        st.markdown(
            "- TF-IDF N-gram Analitiği\n"
            "- Lojistik Regresyon\n"
            "- Kural Tabanlı Üslup Filtresi"
        )

        st.subheader("Proje Ekibi")
        st.markdown(
            "**Büşra:** NLP Pipeline & Karar Destek Motoru  \n"
            "**Yüsra:** UI/UX & Entegrasyon Mimarisi"
        )


def render_counter(text: str) -> None:
    """[FR-1.4] Karakter ve kelime sayacı."""
    chars = len(text)
    words = len(text.split())
    st.markdown(
        f'<div class="counter">{chars:,} karakter &nbsp;|&nbsp; {words:,} kelime</div>'.replace(",", "."),
        unsafe_allow_html=True,
    )


def render_result(result: dict, elapsed_ms: float) -> None:
    """[IR-1..3, FR-4.3..4.5] Analiz sonucunu gösterir."""
    label = result.get("label", "Yetersiz Veri")
    confidence = min(max(float(result.get("confidence", 0.0)), 0.0), 1.0)
    flags = result.get("linguistic_flags") or []
    keywords = result.get("top_keywords") or []
    reason = result.get("summary_reason", "Gerekçe bilgisi sağlanmadı.")
    duration = result.get("processing_time_ms", elapsed_ms)

    icon, show = RESULT_STYLES.get(label, RESULT_STYLES["Yetersiz Veri"])

    st.divider()
    st.subheader("Analiz Sonucu")

    show(f"{icon} **Sınıflandırma: {label}**")

    col_score, col_time = st.columns(2)
    with col_score:
        st.metric("Güven Skoru", f"%{confidence * 100:.1f}")
        st.progress(confidence)
    with col_time:
        st.metric("Analiz Süresi", f"{float(duration):.1f} ms")

    st.markdown("**Üslup ve Dil Kusurları**")
    if flags:
        for flag in flags:
            st.warning(flag, icon="⚠️")
    else:
        st.success("Üslup manipülasyonu veya clickbait kalıbına rastlanmadı.")

    st.markdown("**Kararı En Çok Etkileyen Anahtar Kelimeler**")
    if keywords:
        badges = "".join(f'<span class="kw-badge">{k}</span>' for k in keywords)
        st.markdown(badges, unsafe_allow_html=True)
    else:
        st.caption("Bu analiz için anahtar kelime üretilmedi.")

    st.markdown("**Model Karar Gerekçesi**")
    st.info(reason)


def analyze_section(text: str, key: str) -> None:
    """Analiz butonu, spinner ve sonuç gösterimi (tüm sekmelerde ortak)."""
    if not st.button("🔍 Haberi Analiz Et", type="primary", use_container_width=True, key=key):
        return

    if not text.strip():
        st.warning("Analiz için önce bir haber metni girin.")
        return

    with st.spinner("Analiz ediliyor..."):
        try:
            result, elapsed_ms = run_analysis(text)
        except Exception as exc:
            st.error(f"Analiz motoru hata verdi: {exc}")
            return

    render_result(result, elapsed_ms)


# ---------------------------------------------------------------------------
# Sekmeler
# ---------------------------------------------------------------------------
def render_text_tab() -> None:
    st.caption("Hızlı demo için hazır örneklerden birini seçebilir veya kendi metninizi yapıştırabilirsiniz.")

    c1, c2, c3 = st.columns(3)
    c1.button("📰 Gerçek Haber", on_click=set_sample, args=("real",), use_container_width=True)
    c2.button("⚠️ Şüpheli Haber", on_click=set_sample, args=("suspicious",), use_container_width=True)
    c3.button("🚨 Sahte Haber", on_click=set_sample, args=("fake",), use_container_width=True)

    text = st.text_area(
        "Haber metni",
        key="news_text",
        height=220,
        placeholder="Analiz edilecek haber metnini buraya yapıştırın...",
    )

    info_col, clear_col = st.columns([4, 1])
    with info_col:
        render_counter(text)
    with clear_col:
        st.button("🗑️ Temizle", on_click=clear_text, args=("news_text",), key="clear_news", use_container_width=True)

    analyze_section(text, key="analyze_text")


def render_ocr_tab() -> None:
    st.caption("Tweet, sosyal medya gönderisi veya haber sitesi ekran görüntüsü yükleyin.")

    uploaded = st.file_uploader("Görsel yükle", type=["jpg", "jpeg", "png"], key="ocr_file")
    if uploaded is None:
        st.info("Henüz görsel yüklenmedi. JPG, JPEG veya PNG formatında bir dosya seçin.")
        return

    try:
        image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
    except Exception:
        st.error("Görsel okunamadı. Lütfen geçerli bir JPG veya PNG dosyası yükleyin.")
        return

    # Aynı dosya için OCR'ı yeniden çalıştırma (kullanıcı düzenlemelerini korur)
    source_id = f"{uploaded.name}-{uploaded.size}"
    if st.session_state["ocr_source"] != source_id:
        with st.spinner("Görselden metin çıkarılıyor..."):
            text, real_ocr = extract_text_from_image(image)
        st.session_state["ocr_text"] = text
        st.session_state["ocr_source"] = source_id
        st.session_state["ocr_real"] = real_ocr

    preview_col, text_col = st.columns(2, gap="large")
    with preview_col:
        st.image(image, caption="Yüklenen görsel", use_container_width=True)

    with text_col:
        if st.session_state.get("ocr_real"):
            st.success("Görselden metin başarıyla çıkarıldı.")
        else:
            st.info("OCR modülü bu ortamda kullanılamıyor; örnek metinle simüle edildi. "
                    "Metni aşağıda düzenleyebilirsiniz.")
        text = st.text_area("Çıkarılan metin", key="ocr_text", height=200)
        render_counter(text)

    analyze_section(text, key="analyze_ocr")


def render_url_tab() -> None:
    st.caption("Haber sitesinin bağlantısını yapıştırın; metin otomatik çekilmeye çalışılır.")

    url = st.text_input("Haber linki", placeholder="https://ornek-haber-sitesi.com/haber/...", key="url_input")

    if st.button("📥 Metni Getir", key="fetch_url"):
        if not url.strip():
            st.warning("Lütfen bir haber bağlantısı girin.")
        else:
            with st.spinner("Sayfa okunuyor..."):
                text, error = fetch_article_text(url.strip())
            if error:
                st.warning(error)
            else:
                st.session_state["url_text"] = text
                st.success("Haber metni çekildi.")

    text = st.text_area(
        "Haber metni",
        key="url_text",
        height=200,
        placeholder="Çekilen metin burada görünür; dilerseniz elle de yapıştırabilirsiniz.",
    )
    render_counter(text)
    analyze_section(text, key="analyze_url")


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="DoğrulaAI - Sahte Haber Tespit Sistemi",
        page_icon="🔍",
        layout="wide",
    )
    init_state()
    inject_css()
    render_sidebar()
    render_header()

    tab_text, tab_ocr, tab_url = st.tabs(
        ["📝 Haber Metni Girişi", "📸 Ekran Görüntüsü (OCR)", "🔗 Haber Linki (URL)"]
    )
    with tab_text:
        render_text_tab()
    with tab_ocr:
        render_ocr_tab()
    with tab_url:
        render_url_tab()


if __name__ == "__main__":
    main()