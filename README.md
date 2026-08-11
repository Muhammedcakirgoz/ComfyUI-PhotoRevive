# ComfyUI-PhotoRevive

Eski/hasarlı fotoğrafları (çizik, solma, siyah-beyaz, düşük çözünürlük) tek
workflow ile onarıp renklendiren ComfyUI custom node paketi.

## Pipeline

1. **Analiz** — B&W/renk, hasar yoğunluğu, yüz var mı otomatik sınıflandırılır;
   sonraki adımların şiddeti buna göre ayarlanır.
2. **Hasar onarımı** — Çizik/leke giderme (BOPBTL) + gürültü azaltma.
3. **Yüz restorasyonu** — GFPGAN / CodeFormer, düşük-orta şiddet.
4. **Renklendirme + upscale** — DDColor renklendirme, ardından ESRGAN tiled
   upscale (hasar temizlenmeden büyütülmez).

## Kurulum

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<kullanici-adi>/ComfyUI-PhotoRevive
cd ComfyUI-PhotoRevive
pip install -r requirements.txt
python scripts/download_models.py
```

## Durum

Bu proje erken MVP aşamasında. Node iskeleti hazır, model entegrasyonu
devam ediyor.

## Referanslar

- https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life
- https://github.com/Skivus/ComfyUI-Bringing-Old-Photos-Back-to-Life
- GFPGAN / CodeFormer
- DDColor
- ESRGAN

## Lisans

MIT
