# Görsel tabanlı ürün tanıma — rapor / tez metni için notlar

Bu dosya Word/LaTeX raporuna **kopyalanabilir metin** ve **hangi kod satırlarına bakılacağı** bilgisini verir. Güncel satır numaraları projede değişirse `streamlit_app.py`, `vision.py`, `store.py` içinde arama yapın.

---

## 1. Sistem özeti (giriş bölümü için)

- **Amaç:** Kullanıcı bir ürün fotoğrafı yükler; sistem **YOLO** (veya eğitim yoksa demo mod) ile görüntüden sınıf/etiket çıkarır, **OCR** ile metin okur, veritabanındaki ürünlerle **eşleştirir**. Sepete ekleme ve **simüle ödeme** Streamlit üzerinden yapılır.
- **Teknoloji:** Python, Streamlit, SQLite (`store.py`), Ultralytics YOLO, EasyOCR/Tesseract, isteğe bağlı HF NER / spaCy (`vision.py`), komut satırı araçları (`scripts/tools.py`).

---

## 2. Arayüz işlevleri (yazı halinde)

Uygulama **sol menüden** dört sayfaya ayrılır (`streamlit_app.py` satır ~534–549).

### 2.1 Giriş (`page_giris`, ~111–158)

| Öğe | İşlev |
|-----|--------|
| Giriş formu | E-posta + şifre ile `store.login_user`; başarılıysa oturum açılır, sayfa yenilenir. |
| Kayıt formu | `store.register_user` ile müşteri hesabı (admin değil). |
| Demo tablosu | Örnek yönetici / müşteri hesapları bilgisi. |

### 2.2 Mağaza (`page_magaza`, ~160–321)

| Öğe | İşlev |
|-----|--------|
| Dijital fiş | Son simüle ödemenin metin fişi; indirme (.txt), kapatma. |
| Sidebar sepet | Sepet satırları, +/−, ara toplam, “Ödemeyi tamamla” → `store.simulate_checkout`, balon animasyonu. |
| Fotoğraf yükleme | `analyze_product_image` ile analiz; YOLO/OCR metrikleri, uyarılar (model yoksa), okunan metin. |
| Otomatik sepet | Tanınan ürün ve stok uygunsa sepete ekleme (imza ile çift ekleme engeli). |
| Alışveriş geçmişi | `store.list_user_orders` ile son siparişler. |

### 2.3 Yönetim (`page_yonetim`, ~324–438, yalnızca `admin`)

| Sekme | İşlev |
|--------|--------|
| Özet | Tarama sayısı, simüle sipariş, ciro, düşük stok. |
| Ürün yönetimi | Tablo; stok +/−, silme; yeni ürün formu (`insert_product`). |
| Tarama günlüğü | `scan_list_recent`. |
| Ödeme kayıtları | `admin_list_by_kind("payment_simulation")`, JSON detay. |
| Kontrol | Ürün arama; yönetici için test görseli analizi. |

### 2.4 Veri klasörleri (`page_veri_klasorleri`, ~441–527, yalnızca `admin`)

| Sekme | İşlev |
|--------|--------|
| Özet | Ham veri ve manifest yollarının açıklaması. |
| Çok açılı (SKU) | SKU + görüntü kaydı, `append_manifest`. |
| QR | `decode_qr_from_pil`. |
| Düz görüntü / fiş | Ürün veya fiş görüntüsü kaydı; fişte OCR. |
| Manifest | CSV manifest tablosu ve dosya var mı kontrolü. |

### 2.5 Genel

- **`setup_chrome`:** `assets/custom.css` ve `assets/saat_widget.html` ile stil + saat bileşeni.
- **Oturum:** `auth`, `require_login`, sidebar’da çıkış.

---

## 3. Önemli kod dosyaları ve raporda gösterilecek parçalar

Aşağıdaki **satır aralıkları** mevcut sürüme göredir; raporda “Kaynak: `dosya`, satır x–y” diye yazın.

### 3.1 `streamlit_app.py`

| Konu | Satır (yaklaşık) | Raporda ne anlatılır |
|------|------------------|----------------------|
| Sayfa yönlendirme | 530–549 | `set_page_config`, `init_db`, sidebar `radio`, `if/elif` ile sayfa çağrıları. |
| Oturum / sepet | 27–94 | `_AUTH`, sepet anahtarları, `cart_get`, `setup_chrome`. |
| Mağaza akışı | 160–321 | Dosya anahtarı, `analyze_product_image`, checkout. |

**Kısa alıntı örneği (yönlendirme):**

```python
# streamlit_app.py — uygulama girişi
st.set_page_config(page_title="Görsel ürün tanıma", page_icon="🛒", layout="wide")
setup_chrome()
store.init_db()
with st.sidebar:
    sayfa = st.radio("Sayfa", ["Giriş", "Mağaza", "Yönetim", "Veri klasörleri"], ...)
if sayfa == "Giriş":
    page_giris()
elif sayfa == "Mağaza":
    page_magaza()
# ...
```

### 3.2 `vision.py`

| Konu | Satır (yaklaşık) | Raporda ne anlatılır |
|------|------------------|----------------------|
| YOLO + demo | 246–269 | `yolov8_product.pt` varsa eğitimli model; yoksa hash ile demo etiket. |
| Eşleştirme zinciri | 346–379 | OCR/SKU/ipucu/TF-IDF sırası; eğitimli modda rastgele yok. |
| Ana analiz | 382–426 | `analyze_product_image`: OCR, NLP, Mongo (isteğe bağlı), `log_scan`. |

**Kısa alıntı örneği (analiz özeti):**

```python
# vision.py — analiz boru hattı (özet)
y_label, y_conf, y_back = _yolo_detect(np_rgb)
ocr_txt, ocr_cf, o_back = _ocr_easy(pil_image)
matched = _match_product(y_label, ocr_txt, yolo_backend=y_back, ner_hints=ner)
# ...
store.log_scan(...)  # isteğe bağlı
```

### 3.3 `store.py`

| Konu | Satır (yaklaşık) | Raporda ne anlatılır |
|------|------------------|----------------------|
| Şema | 17–76 | Tablolar: `users`, `products`, `orders`, `order_items`, `admin_logs`, `scan_logs`. |
| Proje yolları | 88–118 | `ProjectPaths` — `data/`, `models/`, `experiments/`. |
| Simüle ödeme | 443–505 | Stok kontrolü, sipariş satırları, stok düşme, fiş dict. |
| Fiş metni | 418–440 | `format_receipt_text` — metin fiş formatı. |

**Kısa alıntı örneği (checkout özeti):**

```python
# store.py — simulate_checkout (mantık özeti)
# Sepetteki her kalem için stok kontrolü → order + order_items → stok UPDATE
# → admin_logs'a payment_simulation → receipt dict dönüşü
```

### 3.4 `scripts/tools.py`

| Konu | Satır (yaklaşık) | Raporda ne anlatılır |
|------|------------------|----------------------|
| `build` | 126–173 | `data/raw/products` → `dataset_yolo_cls/train|val`. |
| `train` | 176–208 | YOLO cls eğitimi, `best.pt` → `models/exports/yolov8_product.pt`. |
| `eval-yolo`, `cv-search`, `prototype` | 211+ | Doğrulama, sklearn, prototip (tez metninde ilgili bölüme). |

### 3.5 `test_store.py`

| Konu | Satır | Raporda ne anlatılır |
|------|-------|----------------------|
| Geçici DB | ~16–31 | `monkeypatch` ile izole test. |
| Testler | 34–46 | Şifre roundtrip; kayıt + ürün ekleme. |

### 3.6 `assets/` (arayüz, kod değil)

| Dosya | Raporda |
|-------|---------|
| `custom.css` | Genel düzen, tipografi (1 ekran görüntüsü + 2–3 cümle yeter). |
| `saat_widget.html` | Sidebar’da saat gösterimi. |

---

## 4. Rapor bölümlerine hızlı eşleme

| Tez bölümü | Öncelikli kaynak |
|-------------|------------------|
| Giriş / literatür | Metin (bu dosya §1). |
| Sistem mimarisi | Şekil: Streamlit → `vision` + `store`; şema `store._SCHEMA`. |
| Görüntü işleme ve eşleştirme | `vision.py` §3.2. |
| Veritabanı ve iş kuralları | `store.py` §3.3. |
| Arayüz | §2 + ekran görüntüleri (Giriş, Mağaza, Yönetim). |
| Eğitim / deney | `tools.py` + `experiments/` çıktıları. |
| Sonuç | Demo sınırları (simüle ödeme, model opsiyonel). |

---

## 5. Telif / alıntı notu

Raporda kod gösterirken: dosya adı, anlamlı satır aralığı ve mümkünse **kısaltılmış** blok (tam dosya yerine). Tam kaynak zip veya e-posta ekinde verilir.
