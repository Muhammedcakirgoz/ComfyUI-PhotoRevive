# ComfyUI-PhotoRevive

Eski/hasarlı fotoğrafları (çizik, solma, siyah-beyaz, düşük çözünürlük) tek
workflow ile onarıp renklendiren ComfyUI custom node paketi.

## Pipeline

1. **Analiz** (`PhotoRevive_Analyze`) — B&W/renk ve hasar yoğunluğu, HSV
   doygunluk ve Canny kenar yoğunluğu analiziyle otomatik sınıflandırılır;
   sonraki adımların şiddeti ve hangi işlemlerin çalışacağı buna göre
   ayarlanır (`PHOTOREVIVE_PRESET` çıktısı).
2. **Hasar onarımı** (`PhotoRevive_ScratchRepair`) — BOPBTL'nin resmi
   tespit modeli (U-Net) ile çizik maskesi çıkarılır, maske güdümlü
   inpainting + gürültü azaltma uygulanır. Hasar düzeyi düşükse (preset)
   atlanır.
3. **Yüz restorasyonu** (`PhotoRevive_FaceRestore`) — GFPGAN, ayarlanabilir
   düşük-orta şiddette (kimlik/yaş/ifade sapmasını sınırlamak için).
4. **Renklendirme + upscale** (`PhotoRevive_ColorizeUpscale`) — Sadece
   siyah-beyaz fotoğraflarda DDColor ile renklendirme, ardından Real-ESRGAN
   tiled upscale (hasar temizlenmeden büyütülmez).
5. **Kalite kontrolü** (`PhotoRevive_QualityCheck`) — Öncesi/sonrası
   görüntüleri karşılaştırır; ArcFace tabanlı yüz kimliği benzerliği ve
   histogram/kenar korelasyonuna dayalı arka plan/kıyafet benzerliği
   hesaplayıp önemli bir sapma varsa uyarır. Yan yana karşılaştırma görseli
   ve metin raporu üretir.

Tüm node'lar ComfyUI'nin doğal `IMAGE` batch tensor'unu (aynı boyuttaki
görüntülerden oluşan bir batch'in tamamını) işler.

## Kurulum

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Muhammedcakirgoz/ComfyUI-PhotoRevive
cd ComfyUI-PhotoRevive
pip install -r requirements.txt
python scripts/download_models.py
```

Model ağırlıkları `ComfyUI/models/photorevive/<aşama>/` altına iner
(~3.3GB toplam: BOPBTL, GFPGAN, CodeFormer, DDColor, Real-ESRGAN). Tek bir
modeli indirmek için: `python scripts/download_models.py --only gfpgan`.
Tüm modelleri listelemek için: `python scripts/download_models.py --list`.

## Kullanım

**ComfyUI içinde:** Node'ları `LoadImage → PhotoRevive_Analyze →
PhotoRevive_ScratchRepair → PhotoRevive_FaceRestore →
PhotoRevive_ColorizeUpscale → SaveImage` sırasıyla bağlayın (`preset`
çıktısını her aşamaya ayrıca iletin). İsteğe bağlı olarak orijinal ve
nihai görüntüyü `PhotoRevive_QualityCheck`'e vererek bir karşılaştırma
raporu alabilirsiniz.

**Toplu (batch) işleme, ComfyUI sunucusu olmadan:**

```bash
python scripts/batch_process.py --input photos/ --output restored/ --quality-check
```

Bir klasördeki tüm fotoğrafları (farklı boyut/en-boy oranlarında olsalar
bile, tek tek işleyerek) pipeline'dan geçirip `restored/` altına yazar;
`--quality-check` ile her fotoğraf için ayrıca karşılaştırma görseli ve
rapor (`.txt`) üretir. `--models-dir` ile modellerin bulunduğu farklı bir
`ComfyUI/models` yolu belirtilebilir.

## Durum

MVP tamamlandı: 5 node de gerçek model entegrasyonlarıyla çalışıyor,
gerçek checkpoint'lerle (state_dict eşleşmesi doğrulanmış) ve gerçek bir
ComfyUI kurulumunda uçtan uca test edildi. Batch modu ve kalite kontrolü
dahil.

## Referanslar

- https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life
- https://github.com/TencentARC/GFPGAN
- https://github.com/sczhou/CodeFormer
- https://github.com/piddnad/DDColor
- https://github.com/xinntao/Real-ESRGAN

## Lisans

MIT
