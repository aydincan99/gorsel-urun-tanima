# Görsel tabanlı ürün tanıma (bitirme projesi)

# modüller proje **kökünde** (`store.py`, `vision.py`). Arayüz Türkçe; `assets/` altında CSS ve HTML ayrı dosyalardır.

**Gereksinim:** Windows’ta **Python 3.10 veya üzeri** önerilir. İlk çalıştırmada **internet** gerekir 
---

## Hoca / değerlendirici için çalıştırma

Projeyi **Türkçe karakter içermeyen** bir klasöre çıkarın (ör. `C:\projeler\gorsel-urun-tanima`), sonra:

### Yöntem 1 — Çift tık (Windows)

1. `calistir.bat` dosyasına çift tıklayın.  
2. İlk seferde sanal ortam ve paketler kurulur; bitince tarayıcıda uygulama açılır.  
3. Açılmazsa tarayıcıda: **http://localhost:8501**

### Yöntem 2 — PowerShell veya VS Code terminali

```text
cd C:\projeler\gorsel-urun-tanima
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

PowerShell’de `Activate.ps1` çalışmazsa sorun değil; yukarıdaki gibi **doğrudan `.venv\Scripts\python.exe`** kullanın.

## Demo hesaplar

| Rol      | E-posta           | Şifre      |
|----------|-------------------|------------|
| Yönetici | `admin@demo.local`| `Admin123!`|
| Müşteri  | `user@demo.local` | `User123!` |

İlk açılışta veritabanı (`data/processed/app.db`) yoksa uygulama oluşturur.

---

Tez / rapor metni için: **`RAPOR.md`** — arayüz işlevleri, hangi dosyada hangi kod parçasının anlatılacağı ve satır aralıkları.

## Kaynak dosyalar

| Dosya | Rol |
|--------|-----|
| `calistir.bat` | Windows: ilk kurulum + Streamlit (cift tik) |
| `streamlit_app.py` | Streamlit arayuzu (tum sayfalar) |
| `store.py` | Yollar + SQLite (sema icinde) |
| `vision.py` | YOLO, OCR, manifest, NLP, QR, eslestirme |
| `assets/custom.css` | Stiller |
| `assets/saat_widget.html` | Saat bileşeni |
| `scripts/tools.py` | `build`, `train`, `eval-yolo`, … |
| `test_store.py` | Kısa `pytest` |
| `RAPOR.md` | Rapora kod + arayüz aktarımı (rehber) |

---

## Araçlar (model / veri seti)

```text
.\.venv\Scripts\python.exe scripts/tools.py build
.\.venv\Scripts\python.exe scripts/tools.py train --epochs 40
```