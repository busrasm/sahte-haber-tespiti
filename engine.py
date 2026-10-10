import json
import re
import string
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MIN_WORD_COUNT = 10
UPPERCASE_LIMIT = 0.25
PUNCT_DENSITY_LIMIT = 0.10
REAL_UPPER = 60.0
REAL_LOWER = 40.0

STOP_WORDS = {
    "ve", "veya", "ile", "bir", "bu", "şu", "o", "da", "de", "ki", "mi", "mı",
    "mu", "mü", "için", "gibi", "ama", "fakat", "ancak", "çok", "daha", "en",
    "her", "ne", "ya", "ise", "olarak", "olan", "kadar", "sonra", "önce",
    "göre", "bunu", "bunun", "çünkü", "hiç", "ise", "ancak", "ise", "bütün",
    "tüm", "ise", "ben", "sen", "biz", "siz", "onlar", "şey", "ile", "den",
}

MANIPULATIVE_TERMS = [
    "şok", "flaş", "mucize", "yetkililer gizliyor", "gizliyor", "saklıyor",
    "inanamayacaksınız", "herkes bunu konuşuyor", "kesinlikle paylaş",
    "paylaş", "bomba iddia", "skandal", "uyanın", "medya göstermiyor",
    "silinmeden önce", "çok acil",
]

# (metin, etiket) -> 1: Gerçek, 0: Sahte
MOCK_DATA = [
    ("Merkez Bankası Para Politikası Kurulu, bugün yaptığı toplantıda politika faizini sabit tutma kararı aldı. Karar metni kurumun internet sitesinde yayımlandı.", 1),
    ("Türkiye İstatistik Kurumu tarafından açıklanan verilere göre yıllık enflasyon oranı geçen aya kıyasla geriledi.", 1),
    ("Milli Eğitim Bakanlığı, yeni eğitim öğretim yılına ilişkin ders programını ve takvimini resmi internet sitesinde duyurdu.", 1),
    ("Meteoroloji Genel Müdürlüğü, hafta sonu için Marmara ve Ege bölgelerinde sağanak yağış beklendiğini açıkladı.", 1),
    ("Sağlık Bakanlığı, kış aylarında artan grip vakalarına karşı vatandaşların aşı yaptırmasını tavsiye etti.", 1),
    ("Ankara'da düzenlenen uluslararası enerji zirvesinde yenilenebilir enerji yatırımları ele alındı. Toplantıya çok sayıda ülkeden yetkili katıldı.", 1),
    ("Süper Lig'de oynanan karşılaşmada ev sahibi takım rakibini iki bir yenerek puanını yükseltti.", 1),
    ("Karayolları Genel Müdürlüğü, bayram tatili öncesinde otoyollarda yoğunluğun artması nedeniyle sürücüleri uyardı.", 1),
    ("Yükseköğretim Kurulu, bu yıl üniversite programlarına yerleştirme sonuçlarının internet üzerinden ilan edileceğini bildirdi.", 1),
    ("Bilim insanları, yeni geliştirilen yöntemin ameliyat sonrası iyileşme süresini kısaltabileceğini hakemli bir dergide yayımlanan makalede açıkladı.", 1),
    ("İstanbul Büyükşehir Belediyesi, toplu taşıma hatlarında yaz dönemi sefer saatlerinde değişiklik yapıldığını duyurdu.", 1),
    ("Dışişleri Bakanlığı'ndan yapılan yazılı açıklamada, iki ülke arasındaki ticaret anlaşmasına ilişkin görüşmelerin önümüzdeki hafta devam edeceği belirtildi.", 1),
    ("Tarım ve Orman Bakanlığı, kuraklık nedeniyle bazı illerde buğday rekoltesinin geçen yıla göre düştüğünü açıkladı.", 1),
    ("Türkiye Büyük Millet Meclisi Genel Kurulu'nda görüşülen kanun teklifinin ilk bölümü oy çokluğuyla kabul edildi.", 1),
    ("Ulaştırma ve Altyapı Bakanlığı, yeni hızlı tren hattında test sürüşlerinin başladığını ve hattın gelecek yıl açılacağını söyledi.", 1),
    ("Borsa İstanbul'da BIST 100 endeksi günü bir önceki kapanışa göre yüzde bir yükselişle tamamladı.", 1),

    ("ŞOK! Yetkililer gizliyor: Bu bitkiyi çayına atan herkes bir haftada tüm hastalıklardan kurtuluyor, doktorlar bunu duyurmak istemiyor!", 0),
    ("FLAŞ FLAŞ! Gizli dünya hükümeti iklim değişikliği yalanıyla milyarlarca insanı kontrol altına almaya çalışıyor, uyanın artık!", 0),
    ("İnanamayacaksınız! Ünlü şarkıcının evinde bulunan gizemli kutu ortalığı karıştırdı, herkes bunu konuşuyor!!!", 0),
    ("Mucize tedavi bulundu: Sadece limon ve karbonatla kanser bir ayda bitiyor, ilaç şirketleri gerçeği saklıyor, paylaş herkes duysun!", 0),
    ("Kesinlikle paylaş! Aşıların içine çip yerleştirildiği ortaya çıktı, bunu yazan gazeteciler sessizce susturuldu!!!", 0),
    ("Bomba iddia: Ay'a hiç gidilmediği kanıtlandı, uzay ajansı yıllardır halkı kandırıyor, işte gizlenen o fotoğraflar!", 0),
    ("SKANDAL! Market raflarındaki bu ürünü yiyenler zehirleniyor ama medya sizlere gerçeği asla göstermeyecek!", 0),
    ("Bu haberi okuyan milyoner oluyor! Gizli yöntemle evde oturarak günde on bin lira kazanan emekli teyze bankaları şaşkına çevirdi!", 0),
    ("Çok acil! Yarın gece yarısı tüm internet kapatılacak, bankadaki paranızı hemen çekin, hükümet bunu sizden saklıyor!", 0),
    ("Mucize gerçekleşti! Yirmi yıldır yatalak olan adam sadece bu suyu içerek ayağa kalktı, doktorlar şoke oldu!", 0),
    ("Yetkililer gizliyor: Büyük deprem tarihi çoktan belli ama bilim insanları halkı korkutmamak için açıklamıyor, uyanın!!!", 0),
    ("Şok iddia: Telefonunuzdaki 5G sinyalleri beyninizi kontrol ediyor, bunu fark eden mühendis esrarengiz şekilde ortadan kayboldu!", 0),
    ("Dünyayı gizlice yöneten üç aile ortaya çıktı! Bu videoyu silinmeden önce izleyin, çünkü yakında yasaklanacak!", 0),
    ("İnanılmaz! Tek bir damla bu yağ saçları bir gecede uzatıyor, kuaförler bu sırrın duyulmasını hiç istemiyor!", 0),
    ("FLAŞ: Ünlü siyasetçinin ölümü saklanıyor, yerine dublörü çıkıyor, sosyal medya bunu konuşuyor, paylaşmadan geçme!", 0),
    ("Şok edici gerçek: Suya katılan gizli kimyasallar insanları uyuşturuyor, belediyeler biliyor ama kimse konuşmuyor!", 0),
]


class NLPProcessor:
    def __init__(self, stop_words=None):
        self.stop_words = set(stop_words or STOP_WORDS)
        self._punct_table = str.maketrans(
            {c: " " for c in string.punctuation + "’‘“”…–—«»"}
        )

    def lower_tr(self, text):
        # python'un lower()'ı İ ve I'yı Türkçe'ye göre çevirmiyor, önce biz hallediyoruz
        return text.replace("İ", "i").replace("I", "ı").lower()

    def clean_text(self, text):
        text = self.lower_tr(text).translate(self._punct_table)
        tokens = [t for t in text.split() if t not in self.stop_words]
        return " ".join(tokens)

    def build_vectorizer(self):
        return TfidfVectorizer(
            preprocessor=self.clean_text,
            ngram_range=(1, 1),
            sublinear_tf=True,
        )


class BaseMLModel:
    def __init__(self, processor):
        self.processor = processor
        self.pipeline = Pipeline([
            ("tfidf", processor.build_vectorizer()),
            ("clf", LogisticRegression(C=10.0, max_iter=1000)),
        ])
        self.is_trained = False

    def train(self, texts, labels):
        self.pipeline.fit(texts, labels)
        self.is_trained = True

    def predict_real_proba(self, text):
        clf = self.pipeline.named_steps["clf"]
        idx = list(clf.classes_).index(1)
        return float(self.pipeline.predict_proba([text])[0][idx])

    def top_keywords(self, text, n=5):
        vec = self.pipeline.named_steps["tfidf"]
        row = vec.transform([text])
        names = vec.get_feature_names_out()
        cols = row.nonzero()[1]
        # ağırlığı eşit çıkanlarda sıra kararlı kalsın diye kelimeye göre de sıralıyoruz
        ranked = sorted(cols, key=lambda c: (-row[0, c], names[c]))
        return [str(names[c]) for c in ranked[:n]]


class DetectionSystem:
    def __init__(self):
        self.processor = NLPProcessor()
        self.model = BaseMLModel(self.processor)
        texts = [t for t, _ in MOCK_DATA]
        labels = [y for _, y in MOCK_DATA]
        self.model.train(texts, labels)

    def check_linguistic_flags(self, text):
        flags = []
        words = text.split()

        letters = [c for c in text if c.isalpha()]
        if letters:
            ratio = sum(c.isupper() for c in letters) / len(letters)
            if ratio > UPPERCASE_LIMIT:
                flags.append(f"Aşırı büyük harf kullanımı (%{ratio * 100:.0f})")

        marks = sum(text.count(c) for c in "!?")
        if words and (marks / len(words) > PUNCT_DENSITY_LIMIT or re.search(r"[!?]{2,}", text)):
            flags.append("Abartılı ünlem/soru işareti kullanımı")

        lowered = self.processor.lower_tr(text)
        for term in MANIPULATIVE_TERMS:
            if term in lowered:
                flags.append(f"Manipülatif ifade: '{term}'")

        return flags

    def build_reason(self, label, p_real, flags):
        if label == "Gerçek":
            reason = f"Model, metnin haber diline uygun olduğunu değerlendirdi (gerçek olasılığı %{p_real:.1f})."
        elif label == "Sahte":
            reason = f"Model, metni sahte haber örüntülerine yakın buldu (gerçek olasılığı %{p_real:.1f})."
        else:
            reason = f"Model kararsız kaldı (gerçek olasılığı %{p_real:.1f}), haberin ek doğrulamayla kontrol edilmesi önerilir."
        if flags:
            reason += f" Üslup kontrolünde {len(flags)} işaret bulundu."
        return reason

    def analyze_news(self, text: str) -> dict:
        start = time.perf_counter()
        text = text if isinstance(text, str) else str(text or "")

        def elapsed():
            return round((time.perf_counter() - start) * 1000, 2)

        # FR-1.2 / FR-1.3: kısa metinde modeli hiç çalıştırmıyoruz
        if len(text.split()) < MIN_WORD_COUNT:
            return {
                "label": "Yetersiz Veri",
                "confidence": 0.0,
                "linguistic_flags": [],
                "summary_reason": "Analiz için yeterli veri yok. Metin en az 10 kelime içermelidir.",
                "execution_time_ms": elapsed(),
                "top_keywords": [],
            }

        p_real = self.model.predict_real_proba(text) * 100

        # FR-3.7: %40-%60 arası gri bölge
        if p_real > REAL_UPPER:
            label, confidence = "Gerçek", p_real
        elif p_real < REAL_LOWER:
            label, confidence = "Sahte", 100 - p_real
        else:
            label, confidence = "Şüpheli", max(p_real, 100 - p_real)

        flags = self.check_linguistic_flags(text)
        keywords = self.model.top_keywords(text, n=5)

        return {
            "label": label,
            "confidence": round(confidence, 1),
            "linguistic_flags": flags,
            "summary_reason": self.build_reason(label, p_real, flags),
            "execution_time_ms": elapsed(),
            "top_keywords": keywords,
        }


if __name__ == "__main__":
    system = DetectionSystem()

    ornekler = [
        "Enerji ve Tabii Kaynaklar Bakanlığı, doğal gaz depolama tesisindeki doluluk oranının kış öncesinde hedeflenen seviyeye ulaştığını açıkladı.",
        "ŞOK ŞOK! Yetkililer gizliyor, bu mucize damlayı içen herkes bir gecede şifa buluyor, doktorlar sessiz kalıyor!!! Paylaş herkes duysun!",
    ]

    for metin in ornekler:
        print(metin)
        print(json.dumps(system.analyze_news(metin), ensure_ascii=False, indent=2))
        print("-" * 60)