"""UI tarafi - streamlit run streamlit_app.py deyince aciliyo."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
import json
from pathlib import Path

import pandas as pd
from PIL import Image

import vision
import store
from store import ProjectPaths, format_tr_display
from vision import analyze_product_image, read_text_ocr
from hardening import configure_logging, public_error, sanitize_text, validate_upload

_AUTH = "oturum"  # session_state keyleri — isimleri degistirme sonra bozulur
_SEPET = "sepet"
_SEPET_FLASH = "sepet_mesaj"
_STOK_UYARI = "Stokta yeterli ürün yok."
_SON_FIS = "son_dijital_fis"
_SEPET_IMZA = "_sepet_otomatik_imza"  # ayni fotoda 2 kere auto-ekleme icin
_YUKLENEN_ANAHTAR = "_yuklenen_dosya_kimligi"


def auth() -> dict | None:
    return st.session_state.get(_AUTH)


def set_auth(user: dict | None) -> None:
    if user is None:
        st.session_state.pop(_AUTH, None)
    else:
        st.session_state[_AUTH] = user


def require_login(*, roller: set[str] | None = None) -> dict:
    u = auth()
    if not u:
        st.warning("Önce giriş yapın (soldan **Giriş** sayfasına dönün).")
        st.stop()
    if roller and u.get("role") not in roller:
        st.error("Bu sayfa için yetkiniz yok.")
        st.stop()
    return u


def cart_get() -> list[dict]:
    if _SEPET not in st.session_state:
        st.session_state[_SEPET] = []
    return st.session_state[_SEPET]


def cart_clear() -> None:
    st.session_state[_SEPET] = []


def cart_flash_mesaj() -> str | None:
    return st.session_state.pop(_SEPET_FLASH, None)


def cart_mesaj_ayarla(msg: str) -> None:
    st.session_state[_SEPET_FLASH] = msg


def cikis_butonu() -> None:
    if st.sidebar.button("Çıkış"):
        set_auth(None)
        cart_clear()
        st.rerun()


def _assets_dir() -> Path:
    return ProjectPaths.default().root / "assets"


def setup_chrome() -> None:
    p = _assets_dir() / "custom.css"
    if p.is_file():
        st.markdown(f"<style>\n{p.read_text(encoding='utf-8')}\n</style>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<style>.main .block-container{max-width:1080px;padding-top:1rem;}h1{font-weight:700;}</style>",
            unsafe_allow_html=True,
        )
    hp = _assets_dir() / "saat_widget.html"
    if hp.is_file():
        components.html(hp.read_text(encoding="utf-8"), height=52)


def _yukleme_imzalari_temizle() -> None:
    st.session_state.pop(_SEPET_IMZA, None)
    st.session_state.pop(_YUKLENEN_ANAHTAR, None)


def baslik(baslik_metni: str, alt: str | None = None) -> None:
    st.title(baslik_metni)
    if alt:
        st.caption(alt)
    st.divider()


def kenar_cubugu_cikis() -> None:
    u = auth()
    if u:
        st.success("Giriş yapıldı.")
        cikis_butonu()


def page_giris() -> None:
    baslik(
        "Görsel ürün tanıma",
        "Fotoğraf veya QR ile ürün eşlemesi — bitirme projesi (Streamlit + SQLite + YOLO + OCR).",
    )
    sol, sag = st.columns((3, 2), gap="large")
    with sol:
        with st.container(border=True):
            st.subheader("Giriş")
            with st.form("giris_form"):
                email_in = st.text_input("E-posta")
                sifre_in = st.text_input("Şifre", type="password")
                gir_btn = st.form_submit_button("Giriş yap", type="primary", use_container_width=True)
            if gir_btn:
                ok, msg, user = store.login_user(email_in, sifre_in)
                if ok:
                    if user != None:
                        set_auth(user)
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.error(msg)
        with st.container(border=True):
            st.subheader("Kayıt (müşteri)")
            with st.form("kayit_form"):
                ad = st.text_input("Ad soyad", key="kad")
                em = st.text_input("E-posta (kayıt)", key="kem")
                pw = st.text_input("Şifre (en az 8 karakter)", type="password", key="kpw")
                kay_btn = st.form_submit_button("Kayıt ol", use_container_width=True)
            if kay_btn:
                ok2, msg2 = store.register_user(em, pw, ad, as_admin=False)
                (st.success if ok2 else st.error)(msg2)
    with sag:
        with st.container(border=True):
            st.subheader("Demo hesaplar")
            st.markdown(
                "| Rol | E-posta | Şifre |\n|-----|---------|--------|\n"
                "| Yönetici | `admin@demo.local` | `Admin123!` |\n"
                "| Personel | `staff@demo.local` | `Staff123!` |\n"
                "| Müşteri | `user@demo.local` | `User123!` |"
            )
            st.info(
                "İlk çalıştırmada veritabanı: `data/processed/app.db`. "
                "İsteğe bağlı OCR logu: `MONGO_URI` — `README.md`.",
                icon="ℹ️",
            )


def page_magaza() -> None:  # en karisik sayfa burda (sepet+sidebar+foto)
    kullanici = require_login()
    baslik("Mağaza", "Ürün fotoğrafı yükleyin; YOLO + OCR ile eşler. Sepet simülasyondur.")
    st.info("Tanınan ürün (stok varsa) otomatik sepete eklenir.", icon="ℹ️")
    son_fis = st.session_state.get(_SON_FIS)
    if son_fis:
        with st.container(border=True):
            st.subheader("Dijital fiş")
            st.text(store.format_receipt_text(son_fis))
            c_dl, c_kapat = st.columns(2)
            with c_dl:
                st.download_button(
                    label="Fişi indir (.txt)",
                    data=store.format_receipt_text(son_fis).encode("utf-8"),
                    file_name=f"fis_siparis_{son_fis['order_id']}.txt",
                    mime="text/plain",
                )
            with c_kapat:
                if st.button("Fişi kapat", key="btn_fis_kapat"):
                    st.session_state.pop(_SON_FIS, None)
                    st.rerun()
    with st.sidebar:
        st.subheader("Sepet")
        fis = cart_flash_mesaj()
        if fis:
            st.success(fis)
        sepet = cart_get()
        if not sepet:
            st.write("Sepet boş.")
        else:
            ara_toplam = 0.0
            for idx, satir in enumerate(sepet):
                tutar = satir["price"] * satir["qty"]
                ara_toplam += tutar
                st.markdown(f"**{satir['name']}** · *{tutar:.2f} TL*")
                b1, b2, b3 = st.columns(3)
                if b1.button("−", key=f"azalt_{idx}_{satir['product_id']}"):
                    satir["qty"] -= 1
                    if satir["qty"] < 1:
                        sepet.pop(idx)
                    st.rerun()
                b2.caption(f"**{satir['qty']}** adet")
                if b3.button("+", key=f"artir_{idx}_{satir['product_id']}"):
                    urun = store.get_by_id(int(satir["product_id"]))
                    ust = int(urun["stock"]) if urun is not None else 0
                    if satir["qty"] + 1 <= ust:
                        satir["qty"] += 1
                    else:
                        cart_mesaj_ayarla(_STOK_UYARI)
                    st.rerun()
                st.caption(f"Birim fiyat: {satir['price']:.2f} TL")
            st.metric("Ara toplam", f"{ara_toplam:.2f} TL")
            if st.button("Ödemeyi tamamla (simülasyon)", type="primary"):
                basarili, mesaj, fis_dict = store.simulate_checkout(
                    int(kullanici["id"]),
                    str(kullanici.get("full_name") or kullanici["email"]),
                    sepet,
                )
                if basarili and fis_dict:
                    cart_clear()
                    st.session_state[_SON_FIS] = fis_dict
                    _yukleme_imzalari_temizle()
                    st.balloons()
                    cart_mesaj_ayarla(mesaj)
                    st.rerun()
                elif basarili:
                    cart_clear()
                    st.success(mesaj)
                    st.balloons()
                    st.rerun()
                else:
                    st.error(mesaj)
        st.divider()
        if st.button("Sepeti temizle"):
            cart_clear()
            _yukleme_imzalari_temizle()
            st.rerun()
    yuklenen = st.file_uploader("Ürün fotoğrafı", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if yuklenen is not None:
        try:
            boyut = len(yuklenen.getvalue())
        except Exception:
            boyut = id(yuklenen)
        dosya_anahtari = f"{yuklenen.name}:{boyut}"  # dosya degisince reset
        if st.session_state.get(_YUKLENEN_ANAHTAR) != dosya_anahtari:
            st.session_state[_YUKLENEN_ANAHTAR] = dosya_anahtari
            st.session_state.pop(_SEPET_IMZA, None)
        try:
            pil = validate_upload(yuklenen)
        except Exception as exc:
            st.error(public_error(exc))
            return
        k1, k2 = st.columns([1, 1], gap="medium")
        with k1:
            st.image(pil, use_container_width=True)
        with k2:
            with st.spinner("Görüntü işleniyor (YOLO + OCR)…"):
                sonuc = analyze_product_image(pil, user_id=int(kullanici["id"]), log_scan=True)
            m1, m2 = st.columns(2)
            m1.metric("YOLO güveni", f"{sonuc.yolo_conf:.1%}")
            m2.metric("OCR güveni", f"{sonuc.ocr_conf:.1%}")
            st.caption(
                f"YOLO: `{sonuc.yolo_backend}` · **{sonuc.yolo_label}** · OCR motoru: `{sonuc.ocr_backend}`"
            )
            if sonuc.ner_hints:
                st.caption("Metin ipuçları: " + ", ".join(sonuc.ner_hints[:8]))
            if sonuc.yolo_backend == "yolov8_demo_hash":
                st.warning(
                    "Eğitilmiş model bulunamadı (`models/exports/yolov8_product.pt`). "
                    "Önce: `python scripts/tools.py build` ve `python scripts/tools.py train`."
                )
            if sonuc.ocr_text:
                st.text_area("Okunan metin (OCR)", sonuc.ocr_text[:1200], height=100)
            if sonuc.product:
                p = sonuc.product
                pid = int(p["id"])
                guncel = store.get_by_id(pid)
                stok = int(guncel["stock"]) if guncel is not None else 0
                imza = f"{dosya_anahtari}:{p['id']}"
                sepet_once = cart_get()
                sepetteki = sum(x["qty"] for x in sepet_once if x["product_id"] == pid)
                if stok > 0 and sepetteki == 0 and st.session_state.get(_SEPET_IMZA) == imza:
                    st.session_state.pop(_SEPET_IMZA, None)
                if st.session_state.get(_SEPET_IMZA) != imza:
                    if stok > 0:
                        sepet_simdi = cart_get()
                        adet = sum(x["qty"] for x in sepet_simdi if x["product_id"] == pid)
                        if adet >= stok:
                            st.session_state[_SEPET_IMZA] = imza
                            cart_mesaj_ayarla(_STOK_UYARI)
                            st.rerun()
                        else:
                            var_mi = False
                            for satir in sepet_simdi:
                                if satir["product_id"] == pid:
                                    satir["qty"] += 1
                                    var_mi = True
                                    break
                            if not var_mi:
                                sepet_simdi.append(
                                    {"product_id": pid, "name": p["name"], "price": float(p["price"]), "qty": 1}
                                )
                            st.session_state[_SEPET_IMZA] = imza
                            cart_mesaj_ayarla(f"**{p['name']}** sepete eklendi.")
                            st.rerun()
                sepet_son = cart_get()
                sepet_adet = sum(x["qty"] for x in sepet_son if x["product_id"] == pid)
                with st.container(border=True):
                    st.subheader(p["name"])
                    st.write(f"Birim fiyat: {p['price']:.2f} TL · Stok: {stok}")
                    st.write(
                        f"Besin / 100 ml: kalori {p['cal_100ml'] or '—'} · "
                        f"Karbonhidrat {p['carbs_g_100ml'] or '—'} g · şeker {p['sugar_g_100ml'] or '—'} g"
                    )
                    if stok <= 0:
                        st.warning(_STOK_UYARI)
                    elif sepet_adet > 0:
                        st.info(f"Sepette {sepet_adet} adet (stok: {stok}).")
            else:
                st.error("Veritabanında eşleşen ürün bulunamadı.")
    st.divider()
    st.subheader("Alışveriş geçmişi")
    for satir in store.list_user_orders(int(kullanici["id"]), limit=15):
        st.write(
            f"**#{satir['id']}** · {satir['total']:.2f} TL · {satir['status']} · _{format_tr_display(satir['created_at'])}_"
        )


def page_hesabim() -> None:
    kullanici = require_login()
    baslik("Hesabım", "Hesabınızı kalıcı olarak silebilirsiniz.")
    st.warning("Silme işlemi sipariş, tarama ve kullanıcı kaydını tamamen kaldırır.")
    if st.button("Hesabımı kalıcı olarak sil", type="primary"):
        store.delete_user_account(int(kullanici["id"]))
        set_auth(None)
        cart_clear()
        st.success("Hesap silindi.")
        st.rerun()


def page_yonetim() -> None:  # admin ekrani — baya kod var
    kullanici = require_login(roller={"admin", "staff"})
    sadece_admin = kullanici.get("role") == "admin"
    baslik("Yönetim paneli", "Özet, envanter, kayıtlar ve tanıma testi.")
    bag = store.get_connection()  # direkt baglandi manually
    try:
        tarama_say = bag.execute("SELECT COUNT(*) AS c FROM scan_logs").fetchone()["c"]
        siparis_say = bag.execute(
            "SELECT COUNT(*) AS c FROM admin_logs WHERE kind = 'payment_simulation'"
        ).fetchone()["c"]
        ciro = bag.execute("SELECT COALESCE(SUM(total),0) AS s FROM orders").fetchone()["s"]
        dusuk = bag.execute("SELECT COUNT(*) AS c FROM products WHERE stock < 10").fetchone()["c"]
    finally:
        bag.close()
    t1, t2, t3, t4, t5 = st.tabs(
        ["Özet", "Ürün yönetimi", "Tarama günlüğü", "Ödeme kayıtları", "Kontrol"]
    )
    with t1:
        st.metric("Toplam tarama sayısı", tarama_say)
        c1, c2, c3 = st.columns(3)
        c1.metric("Simüle sipariş", siparis_say)
        c2.metric("Toplam ciro (TL)", f"{ciro:.2f}")
        c3.metric("Düşük stok (<10)", dusuk)
    with t2:
        st.subheader("Envanter")
        urunler = store.list_all()
        L = len(urunler)
        if L == 0:
            st.info("Henüz ürün yok.")
        else:
            liste = urunler
            tablo = pd.DataFrame([dict(x) for x in liste])
            st.dataframe(tablo[["id", "sku", "name", "price", "stock", "yolo_class_hint"]], use_container_width=True, hide_index=True)
            for r in liste:
                c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                c1.write(f"**{r['name']}** ({r['sku']})")
                if c2.button("+10 stok", key=f"ekle{r['id']}"):
                    store.adjust_stock(int(r["id"]), 10)
                    store.admin_append("inventory", f"Stok +10: {r['name']}", None)
                    st.rerun()
                if c3.button("-1 stok", key=f"eksilt{r['id']}"):
                    store.adjust_stock(int(r["id"]), -1)
                    st.rerun()
                if sadece_admin and c4.button("Sil", key=f"sil{r['id']}"):
                    store.delete_product(int(r["id"]))
                    store.admin_append("inventory", f"Ürün silindi id={r['id']}", None)
                    st.rerun()
                elif not sadece_admin:
                    c4.caption("Silme yetkisi yok")
        st.divider()
        if not sadece_admin:
            st.info("Yeni ürün ekleme yalnızca yöneticidedir.")
            return
        st.subheader("Yeni ürün")
        with st.form("yeni_urun"):
            sku = st.text_input("SKU kodu")
            ad = st.text_input("Ürün adı")
            sku, ad = sanitize_text(sku, 40), sanitize_text(ad, 80)
            fiyat = st.number_input("Fiyat (TL)", min_value=0.0, value=9.99)
            stok_adet = st.number_input("Stok", min_value=0, value=20)
            kalori = st.number_input("Kalori / 100 ml", value=40.0)
            kh = st.number_input("Karbonhidrat (g) / 100 ml", value=10.0)
            seker = st.number_input("Şeker (g) / 100 ml", value=10.0)
            ipucu = st.text_input(
                "YOLO sınıf ipucu (klasör adıyla aynı olmalı)", placeholder="ornek_urun"
            )
            if st.form_submit_button("Kaydet"):
                if sku and ad:
                    try:
                        yeni_id = store.insert_product(
                            sku, ad, fiyat, int(stok_adet),
                            cal_100ml=kalori, carbs_g_100ml=kh, sugar_g_100ml=seker, yolo_class_hint=ipucu or None,
                        )
                        store.admin_append("inventory", f"Yeni ürün #{yeni_id}: {ad}", None)
                        st.success("Ürün kaydedildi.")
                        st.rerun()
                    except Exception as e:
                        st.error(public_error(e))
                else:
                    st.error("SKU ve ürün adı zorunlu.")
    with t3:
        taramalar = store.scan_list_recent(120)
        if taramalar:
            df_scan = pd.DataFrame([dict(x) for x in taramalar])
            if "created_at" in df_scan.columns:
                df_scan["created_at"] = df_scan["created_at"].map(format_tr_display)
            st.dataframe(df_scan, use_container_width=True, hide_index=True)
        else:
            st.write("Kayıt yok.")
    with t4:
        odemeler = store.admin_list_by_kind("payment_simulation", 80)
        if not odemeler:
            st.info("Henüz simüle ödeme kaydı yok.")
        for r in odemeler:
            with st.expander(f"{format_tr_display(r['created_at'])} — {r['message']}"):
                if r["payload"]:
                    st.json(json.loads(r["payload"]))
    with t5:
        arama = st.text_input("Ürün adı veya SKU ile ara", "")
        if arama.strip():
            bulunan = store.search(arama)
            if bulunan:
                for r in bulunan:
                    st.json(store.row_to_dict(r))
            else:
                st.warning("Sonuç yok.")
        st.divider()
        test_dosya = st.file_uploader(
            "Ürün fotoğrafı (yönetici testi)", type=["jpg", "jpeg", "png"], key="yonetici_test"
        )
        if test_dosya is not None:
            try:
                pil_t = validate_upload(test_dosya)
            except Exception as exc:
                st.error(public_error(exc))
                return
            st.image(pil_t, use_container_width=True)
            with st.spinner("Analiz…"):
                tst = analyze_product_image(pil_t, user_id=None, log_scan=True)
            a, b = st.columns(2)
            a.metric("YOLO güveni", f"{tst.yolo_conf:.1%}")
            b.metric("OCR güveni", f"{tst.ocr_conf:.1%}")
            if tst.product:
                st.json(tst.product)
            else:
                st.warning("Eşleşen ürün yok.")


def page_veri_klasorleri() -> None:
    require_login(roller={"admin", "staff"})
    yollar = ProjectPaths.default()
    baslik("Veri klasörleri", f"Eğitim görselleri ve `{vision.MANIFEST_NAME}`.")
    tb_ozet, tb_sku, tb_qr, tb_duz, tb_man = st.tabs(
        ["Özet", "Çok açılı (SKU klasörü)", "QR okuma", "Düz görüntü / fiş", "Manifest listesi"]
    )
    with tb_ozet:
        st.markdown(
            f"| Klasör | Ne için |\n|---|---|\n"
            f"| `{yollar.data_raw_products}` | Her SKU için alt klasör + fotoğraflar |\n"
            f"| `{yollar.data_raw_images}` | Tek klasörde karışık görseller |\n"
            f"| `{yollar.data_raw_receipts}` | Fiş görüntüleri (deneme) |\n"
            f"| `{yollar.data_manifests}` | CSV manifest |\n"
            f"| `{yollar.models_exports}` | `yolov8_product.pt` (eğitince) |"
        )
    with tb_sku:
        sku_al = st.text_input("SKU", placeholder="SKU-001")
        aci = st.text_input("Açı etiketi (ör. on, yan)", value="on")
        up_sku = st.file_uploader("Görüntü seç", type=["jpg", "jpeg", "png", "webp", "bmp"], key="vh_sku")
        dosya_ad_ops = st.text_input("Dosya adı (isteğe bağlı)", "", key="vh_fn")
        if up_sku and st.button("Kaydet", type="primary", key="vh_kaydet"):
            if not sku_al.strip():
                st.error("SKU gerekli.")
            else:
                s, v = vision.safe_segment(sku_al, "SKU"), vision.safe_segment(aci, "view")
                ad_kaynak = dosya_ad_ops.strip() or up_sku.name
                fname = vision.safe_filename(ad_kaynak)
                if Path(fname).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    fname = f"{Path(fname).stem}.jpg"
                hedef_klasor = yollar.data_raw_products / s
                hedef_klasor.mkdir(parents=True, exist_ok=True)
                son_ad = f"{v}_{Path(fname).stem}.jpg"
                hedef_dosya = hedef_klasor / son_ad
                Image.open(up_sku).convert("RGB").save(hedef_dosya, format="JPEG", quality=92)
                vision.append_manifest(yollar, f"products/{s}/{son_ad}", label=s, view=v, modality="product_multi_view")
                st.success(f"Kaydedildi: {hedef_dosya}")
    with tb_qr:
        up_qr = st.file_uploader("QR içeren görüntü", type=["jpg", "jpeg", "png"], key="vh_qr")
        if up_qr is not None:
            pil_q = Image.open(up_qr).convert("RGB")
            st.image(pil_q, use_container_width=True)
            try:
                qr_son = vision.decode_qr_from_pil(pil_q)
                st.success(qr_son.text) if qr_son.ok else st.warning("QR okunamadı.")
            except ImportError as ex:
                st.error(str(ex))
    with tb_duz:
        hedef_tur = st.radio(
            "Kayıt türü",
            ["product", "receipt"],
            horizontal=True,
            format_func=lambda x: "Ürün görüntüsü" if x == "product" else "Fiş görüntüsü",
        )
        if hedef_tur == "product":
            hedef = yollar.data_raw_images
            up_duz = st.file_uploader("Görüntü", key="vh_duz")
            etiket = st.text_input("Kısa etiket", "", key="vh_etiket")
            if up_duz and st.button("Kaydet", key="vh_duz_kaydet"):
                ad = vision.safe_filename(up_duz.name)
                hedef.mkdir(parents=True, exist_ok=True)
                kayit = hedef / ad
                Image.open(up_duz).convert("RGB").save(kayit, format="JPEG", quality=92)
                vision.append_manifest(yollar, f"images/{ad}", etiket or Path(ad).stem, modality="visual_flat")
                st.success(str(kayit))
        else:
            up_fis = st.file_uploader(
                "Fiş görüntüsü", type=["jpg", "jpeg", "png", "webp", "bmp"], key="vh_fis"
            )
            if up_fis is not None:
                pil_f = Image.open(up_fis).convert("RGB")
                st.image(pil_f, use_container_width=True)
                with st.spinner("Metin okunuyor (OCR)…"):
                    ocr_text, ocr_conf, ocr_back = read_text_ocr(pil_f)
                st.metric("Güven", f"{ocr_conf:.1%}")
                st.caption(f"Kullanılan motor: `{ocr_back}`")
                st.text_area("Çıkan metin", ocr_text or "(boş)", height=220)
    with tb_man:
        mf = vision.load_manifest(yollar)
        if len(mf):
            st.dataframe(
                vision.resolve_manifest_for_display(mf, yollar),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Manifest boş.")


st.set_page_config(page_title="Görsel ürün tanıma", page_icon="🛒", layout="wide")  # basta olmali dediler
configure_logging()
setup_chrome()
store.init_db()  # db yoksa kuruyo

with st.sidebar:  # sayfa secimi
    sayfa = st.radio(
        "Sayfa",
        ["Giriş", "Mağaza", "Yönetim", "Veri klasörleri", "Hesabım"],
        label_visibility="visible",
    )
    kenar_cubugu_cikis()

if sayfa == "Giriş":
    page_giris()
elif sayfa == "Mağaza":
    page_magaza()
elif sayfa == "Yönetim":
    page_yonetim()
elif sayfa == "Hesabım":
    page_hesabim()
else:
    page_veri_klasorleri()
