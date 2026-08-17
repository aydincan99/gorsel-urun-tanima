# Görsel Tabanlı Ürün Tanıma Sistemi

YOLO ve OCR tabanlı hibrit ürün tanıma platformu. Kullanıcı bir ürün fotoğrafı yüklediğinde sistem görsel sınıflandırma, metin çıkarımı ve veritabanı eşleştirmesi yapar; tanınan ürün stokta ise otomatik sepete eklenir. Sepet yönetimi, simüle ödeme ve dijital fiş üretimi Streamlit arayüzü üzerinden sunulur.

**Teknolojiler:** Python · Streamlit · YOLO (Ultralytics) · EasyOCR · SQLite · scikit-learn · Hugging Face Transformers

---

## Ekip

Bu proje **ortak geliştirme** olarak yürütülmüştür.

| Geliştirici | Katkı alanı |
|---|---|
| **Aydın Candemiır** | Streamlit arayüzü, uygulama akışı, entegrasyon, dağıtım |
| **Yiğit Duyu** | YOLO/OCR pipeline, NLP eşleştirme, model eğitimi ve değerlendirme |
| **Kutay Necdet Şen** | SQLite veritabanı, iş mantığı, yetkilendirme, veri yönetimi |

---

## Özellikler

- Ürün fotoğrafından otomatik tanıma (YOLO + OCR + metin eşleştirme)
- OCR yazım hatalarına dayanıklı fuzzy eşleştirme
- Sepet yönetimi, stok kontrolü ve simüle ödeme
- Dijital fiş üretimi ve indirme
- Rol tabanlı yönetim paneli (admin / müşteri)
- Veri seti ve manifest yönetimi
- Windows için tek tıkla kurulum (`calistir.bat`)

---

## Gereksinimler

- **İşletim sistemi:** Windows (önerilir)
- **Python:** 3.10 veya üzeri
- **Ağ:** İlk kurulumda internet bağlantısı (bağımlılık indirimi)

Projeyi Türkçe karakter içermeyen bir dizine klonlamanız önerilir (ör. `C:\projeler\gorsel-urun-tanima`).

---

## Kurulum ve Çalıştırma

### GitHub'dan klonlama

```bash
git clone https://github.com/aydincan99/gorsel-urun-tanima.git
cd gorsel-urun-tanima
```

### Yöntem 1 — Otomatik başlatma (Windows)

1. `calistir.bat` dosyasına çift tıklayın.
2. İlk çalıştırmada sanal ortam (`.venv`) oluşturulur ve bağımlılıklar kurulur.
3. Uygulama tarayıcıda açılır; açılmazsa: **http://localhost:8501**

### Yöntem 2 — Manuel kurulum (PowerShell / terminal)

```powershell
cd gorsel-urun-tanima
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

PowerShell execution policy nedeniyle `Activate.ps1` çalışmazsa doğrudan `.venv\Scripts\python.exe` kullanın.

---

## Demo hesaplar

| Rol | E-posta | Şifre |
|---|---|---|
| Yönetici | `admin@demo.local` | `Admin123!` |
| Müşteri | `user@demo.local` | `User123!` |

İlk açılışta SQLite veritabanı (`data/processed/app.db`) otomatik oluşturulur.

---

## Proje yapısı

| Dosya / klasör | Açıklama |
|---|---|
| `streamlit_app.py` | Streamlit arayüzü (giriş, mağaza, yönetim, veri klasörleri) |
| `vision.py` | YOLO, OCR, NLP, QR okuma ve ürün eşleştirme |
| `store.py` | SQLite veritabanı, kimlik doğrulama, sipariş ve stok işlemleri |
| `scripts/tools.py` | Veri seti oluşturma, model eğitimi, YOLO değerlendirme |
| `calistir.bat` | Windows ortamında otomatik kurulum ve başlatma |
| `assets/` | CSS ve HTML bileşenleri |
| `data/` | Ham görseller, manifest ve işlenmiş veritabanı |
| `models/exports/` | Eğitilmiş YOLO model dosyası (`yolov8_product.pt`) |
| `RAPOR.md` | Teknik dokümantasyon ve rapor rehberi |

---

## Model eğitimi (isteğe bağlı)

```powershell
.\.venv\Scripts\python.exe scripts/tools.py build
.\.venv\Scripts\python.exe scripts/tools.py train --epochs 40
```

Eğitilmiş model mevcut değilse uygulama demo modunda çalışmaya devam eder.

---

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest test_store.py test_vision_match.py -q
```

---

## Lisans

Bu depo eğitim ve portfolyo amaçlıdır. Ticari kullanım için geliştirici ekiple iletişime geçin.
