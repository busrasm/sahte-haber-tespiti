import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

__all__ = ["analyze_news"]


_VERI = [
    # --- Gerçek ---
    ("Türkiye İstatistik Kurumu eylül ayı enflasyon verilerini açıkladı. Yıllık tüketici fiyat endeksi yüzde 32,9 olarak gerçekleşti, aylık artış yüzde 2,1 oldu.", "Gerçek"),
    ("Merkez Bankası Para Politikası Kurulu bugün toplandı ve politika faizini sabit tutma kararı aldı. Karar metni kurumun internet sitesinde yayımlandı.", "Gerçek"),
    ("Sağlık Bakanlığı, sonbahar döneminde grip aşısı uygulamasının 15 Ekim'de başlayacağını duyurdu. Risk gruplarındaki vatandaşlar aile hekimlerine başvurabilecek.", "Gerçek"),
    ("Meteoroloji Genel Müdürlüğü, hafta sonu Marmara ve Ege bölgelerinde sağanak yağış beklendiğini açıkladı. Sıcaklıkların mevsim normallerinin altına inmesi öngörülüyor.", "Gerçek"),
    ("Milli Eğitim Bakanlığı 2026-2027 eğitim öğretim yılı takvimini yayımladı. Birinci dönem 21 Ocak'ta sona erecek, karne günü 23 Ocak olarak belirlendi.", "Gerçek"),
    ("Ulaştırma ve Altyapı Bakanlığı açıklamasına göre Ankara-İzmir hızlı tren hattında yapım çalışmaları planlanan takvimle sürüyor, bazı kesimlerde ise ihale süreci devam ediyor.", "Gerçek"),
    ("İstanbul Büyükşehir Belediyesi, metro hatlarında yarın bakım çalışması yapılacağını duyurdu. Seferler 02.00'den sonra sınırlı sayıda gerçekleştirilecek.", "Gerçek"),
    ("AFAD verilerine göre Malatya'da saat 14.22'de 4,1 büyüklüğünde bir deprem meydana geldi. Herhangi bir can kaybı ya da yaralanma bildirilmedi.", "Gerçek"),
    ("Dışişleri Bakanlığı'ndan yapılan yazılı açıklamada, iki ülke arasındaki ticaret anlaşmasının görüşmelerinin gelecek ay Ankara'da yeniden başlayacağı belirtildi.", "Gerçek"),
    ("Borsa İstanbul'da BIST 100 endeksi günü yüzde 0,8 yükselişle 9.842 puandan tamamladı. Günlük işlem hacmi 98 milyar lira oldu.", "Gerçek"),
    ("Yükseköğretim Kurulu, üniversite yerleştirme sonuçlarının açıklandığını duyurdu. Adaylar sonuçlarını ÖSYM'nin giriş sayfasından öğrenebilecek.", "Gerçek"),
    ("Karayolları Genel Müdürlüğü, bayram tatili öncesinde otoyol ve köprü geçişlerinin ücretsiz olmayacağını ancak ek bir trafik denetimi planlandığını açıkladı.", "Gerçek"),

    # --- Şüpheli ---
    ("Kulislerden edindiğimiz bilgiye göre bir bakanın görevden alınacağı konuşuluyor ama henüz resmi bir açıklama yok, ismi de söylemek istemiyorlar.", "Şüpheli"),
    ("Bir tanıdığım belediyede çalışıyor, önümüzdeki ay bütün otoparklara zam gelecekmiş diyor. Doğru mu bilmiyorum ama duyduğumu paylaşıyorum.", "Şüpheli"),
    ("WhatsApp gruplarında dolaşan mesaja göre yarından itibaren bazı bankalar hesaplardan ek kesinti yapacakmış, kimse net bir şey söylemiyor.", "Şüpheli"),
    ("İddiaya göre ünlü bir futbolcu sezon sonunda yurt dışına transfer olacak. Menajerine yakın bir kaynak dedi ama teyit edemedik.", "Şüpheli"),
    ("Duyduğuma göre okulların açılışı tekrar ertelenebilirmiş. Veli grubunda biri yazmış, kaynağı belli değil ama herkes konuşuyor.", "Şüpheli"),
    ("Sosyal medyada yayılan söylentiye göre ünlü bir dizi oyuncusu setten ayrıldı. Yapım ekibinden açıklama gelmedi, kaynaklar da isim vermiyor.", "Şüpheli"),
    ("Kulis: Parti içinde kritik bir istifa olabileceği konuşuluyor. Bunu bize yakın bir isim fısıldadı ama doğrulayan çıkmadı.", "Şüpheli"),
    ("Birkaç kişiden aynı şeyi duydum, akaryakıta bu hafta sonu büyük bir zam geliyormuş. Resmi bir yazı görmedim ama herkes aynı şeyi söylüyor.", "Şüpheli"),
    ("Twitter'da biri yazmış, yeni vergi paketinde emeklilere dokunulacakmış. Ekran görüntüsü yok, sadece tweet var, doğruluğu şüpheli.", "Şüpheli"),
    ("Güvenilir olduğunu söyleyen bir kaynağa göre şehrin en büyük alışveriş merkezi bir süre kapatılacakmış. Belediyeden ses yok.", "Şüpheli"),
    ("İddialara göre iki büyük şirket birleşme görüşmesi yapıyor. Piyasada konuşuluyor ama şirketlerden henüz bir açıklama gelmedi.", "Şüpheli"),
    ("Bir arkadaşım hastanede çalışıyor, yeni bir salgının başladığını söylüyor. Bakanlıktan açıklama yok, belki de yanlış anlaşılmıştır.", "Şüpheli"),

    # --- Sahte ---
    ("ŞOK ŞOK ŞOK!!! Hükümet bu gerçeği yıllardır halktan GİZLİYOR! Hemen bakın, silinmeden paylaşın!!!", "Sahte"),
    ("FLAŞ! Gökyüzünde görülen o ışıkların aslında uzaylı üssü olduğu ortaya çıktı, yetkililer gizliyor, gizlenen gerçek budur!!!", "Sahte"),
    ("Doktorlar bu mucize çayı SAKLIYOR! Sadece 3 günde bütün kanserler yok oluyor, ilaç şirketleri bunu duymanızı istemiyor!!!", "Sahte"),
    ("İNANILMAZ KOMPLO! Aşıların içine mikroçip yerleştirildiği belgelendi, dünyayı gizli bir güç yönetiyor, uyanın artık!!!", "Sahte"),
    ("Büyük skandal! Ünlü bir iş insanının yıllardır gizli bir örgütün başında olduğu ortaya çıktı, medya bu haberi yasakladı!!!", "Sahte"),
    ("Dünya sonu yaklaşıyor! NASA'nın gizlediği gezegen önümüzdeki hafta dünyaya çarpacak, hemen bakın ve herkese iletin!!!", "Sahte"),
    ("MUCİZE! Bu yağı sürenler 1 haftada 20 yaş gençleşiyor, dermatologlar bunun ortaya çıkmasından çok korkuyor!!! Paylaş paylaş!", "Sahte"),
    ("Şok iddia: Depremleri aslında gizli bir ülkenin makineleri tetikliyor, bilim insanları susturuluyor, gerçekleri artık herkes görsün!!!", "Sahte"),
    ("ACİL!!! Şebeke suyuna zehir karıştırıldı, yetkililer gizliyor, bu mesajı 10 kişiye atmazsan başına bela gelir!", "Sahte"),
    ("İnanılmaz ama gerçek: Bütün gıdaların içine hükümet gizlice kontrol maddesi katıyor, gizlenen gerçek sonunda ifşa oldu!!!", "Sahte"),
    ("Yok artık!!! Meşhur şarkıcı aslında 5 yıl önce öldü, yerine klonu çıkıyor, kimse bunu konuşmuyor, hemen bakın!!!", "Sahte"),
    ("FLAŞ HABER: Ay'a inişin tamamen sahte olduğuna dair kesin kanıt bulundu, yıllardır bizi kandırıyorlar, büyük skandal!!!", "Sahte"),
]

_MANIPULATIF = [
    "şok", "flaş", "inanılmaz", "gizlenen gerçek", "hemen bakın",
    "yetkililer gizliyor", "büyük skandal", "mucize", "paylaş",
    "silinmeden", "uyanın", "ifşa", "yasakladı", "bunu duymanızı istemiyor",
]


def _egit():
    df = pd.DataFrame(_VERI, columns=["metin", "etiket"])
    hat = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000)),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])
    hat.fit(df["metin"], df["etiket"])
    return hat


_model = _egit()


def _kucult(metin):
    # python'un lower'ı İ ve I'yı Türkçeye göre çevirmiyor
    return metin.replace("İ", "i").replace("I", "ı").lower()


def _uslup_kontrol(metin):
    bulgular = []

    harfler = [c for c in metin if c.isalpha()]
    if len(harfler) >= 10:
        oran = sum(1 for c in harfler if c.isupper()) / len(harfler)
        if oran > 0.25:
            bulgular.append(f"Aşırı büyük harf kullanımı (%{oran * 100:.0f})")

    isaret = len(re.findall(r"[!?]", metin))
    tekrar = re.search(r"[!?]{2,}", metin)
    yogunluk = isaret / max(len(metin), 1)
    if tekrar or isaret >= 4 or yogunluk > 0.02:
        bulgular.append("Aşırı ünlem/soru işareti kullanımı")

    kucuk = _kucult(metin)
    bulunan = [k for k in _MANIPULATIF if k in kucuk]
    if bulunan:
        bulgular.append("Manipülatif/tıklama tuzağı ifadeler: " + ", ".join(bulunan))

    return bulgular


def _gerekce(label, guven, bulgular):
    if label == "Gerçek":
        ilk = f"Model, metnin tarafsız ve kaynağı belli bir haber diline sahip olduğunu değerlendirdi (güven %{guven})."
    elif label == "Şüpheli":
        ilk = f"Metin, kaynağı belirsiz iddialar ve kulis dili içerdiği için şüpheli olarak sınıflandırıldı (güven %{guven})."
    else:
        ilk = f"Metin, sansasyonel ve doğrulanamaz iddialar içerdiği için sahte haber olarak sınıflandırıldı (güven %{guven})."

    if bulgular:
        return ilk + f" Ayrıca üslup analizinde {len(bulgular)} sorun tespit edildi, bu da metnin güvenilirliğini düşürüyor."
    return ilk + " Üslup analizinde belirgin bir sorun bulunmadı."


def analyze_news(text: str) -> dict:
    metin = (text or "").strip()
    if not metin:
        return {
            "label": "Şüpheli",
            "confidence": 0.0,
            "linguistic_flags": [],
            "summary_reason": "Analiz edilecek bir metin verilmedi.",
        }

    olasiliklar = _model.predict_proba([metin])[0]
    en_iyi = olasiliklar.argmax()
    label = _model.classes_[en_iyi]
    guven = round(float(olasiliklar[en_iyi]) * 100, 1)

    bulgular = _uslup_kontrol(metin)

    return {
        "label": label,
        "confidence": guven,
        "linguistic_flags": bulgular,
        "summary_reason": _gerekce(label, guven, bulgular),
    }


if __name__ == '__main__':
    ornekler = [
        "Sağlık Bakanlığı yeni sezon için aşı takvimini açıkladı, uygulama gelecek hafta başlayacak.",
        "Duyduğuma göre bu hafta sonu bankalarda bir şeyler olacakmış, kaynağım yok ama herkes konuşuyor.",
        "FLAŞ!!! Yetkililer GİZLİYOR, hemen bakın, inanılmaz bir mucize ortaya çıktı!!!",
    ]
    for o in ornekler:
        sonuc = analyze_news(o)
        print("Metin     :", o)
        print("Etiket    :", sonuc["label"], f"(%{sonuc['confidence']})")
        print("Bulgular  :", sonuc["linguistic_flags"])
        print("Gerekçe   :", sonuc["summary_reason"])
        print("-" * 60)