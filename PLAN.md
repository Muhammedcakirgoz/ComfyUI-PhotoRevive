# Proje: Eski Fotoğraf Restorasyonu — Uçtan Uca ComfyUI Pipeline

## Amaç

Eski/hasarlı fotoğrafları (çizik, solma, siyah-beyaz, düşük çözünürlük) tek
komutla/tek workflow ile onarıp renklendiren, ComfyUI üzerinde çalışan açık
kaynak bir node paketi + hazır workflow'lar üretmek.

## Neden bu niş

ComfyUI ekosisteminde parçalar zaten var (Microsoft'un
Bringing-Old-Photos-Back-to-Life modeli, GFPGAN, CodeFormer, DDColor,
ESRGAN/tiled upscale) ve birkaç dağınık fork mevcut. Ama hepsi manuel,
7-8 node'u elle bağlamayı gerektiriyor, kurulumu zor, toplu/batch
kullanım için uygun değil. Boşluk: bu parçaları **tek tıkla çalışan,
akıllı ve toplu işleyen bir paket** haline getirmek — yeni model eğitmeye
gerek yok, iyi orkestrasyon yeterli.

## Pipeline mimarisi (4 aşama)

1. **Analiz / ön işlem**
   - Fotoğrafı otomatik sınıflandır: siyah-beyaz mı renkli mi, çizik
     yoğunluğu, yüz var mı var mı yok.
   - Bu sınıflandırmaya göre sonraki node'ların sırası ve şiddeti otomatik
     ayarlanır (kullanıcı slider'larla uğraşmaz).

2. **Hasar onarımı**
   - Çizik/leke giderme: BOPBTL (Bringing-Old-Photos-Back-to-Life)
     scratch detection modeli.
   - Gürültü azaltma.
   - Restore edilmiş B&W ara kopyayı checkpoint olarak kaydet (renklendirme
     öncesi geri dönüş noktası).

3. **Yüz restorasyonu**
   - GFPGAN veya CodeFormer, düşük-orta şiddette (kimlik/yaş/ifade
     bozulmasını önlemek için — aşırı restorasyon yüzü "başka biri" gibi
     gösterebilir, bu yüzden şiddet sınırlı tutulmalı).

4. **Renklendirme + üst çözünürlük**
   - DDColor ile renklendirme.
   - ESRGAN / tiled upscale ile büyütme (çizik/leke temizlenmeden büyütme
     yapılmamalı — hasarı da büyütür).

## Farklılaştırıcı özellikler (rakiplerde eksik olanlar)

- **Toplu albüm modu**: Bir klasör dolusu fotoğrafı otomatik, aynı
  kalitede işleme. Mevcut workflow'lar tek fotoğraf için elle ayarlanıyor.
- **Kalite kontrol adımı**: İşlem öncesi/sonrası karşılaştırıp
  kimlik/yüz/kıyafet/arka plan önemli ölçüde değişmiş mi diye otomatik
  uyaran bir node. Şu an bu kontrol manuel yapılıyor.
- **Preset kütüphanesi**: Döneme özel hazır ayarlar (örn. "1950'ler
  stüdyo fotoğrafı", "1980'ler renkli solmuş fotoğraf").
- **Basit kurulum**: Tek `git clone` + ComfyUI Manager üzerinden
  kurulabilen, gerekli modelleri otomatik indiren bir paket.

## MVP kapsamı (ilk sürüm)

- Tek fotoğraf girişi
- Otomatik preset seçimi (adım 1'deki analiz)
- 4 aşamalı pipeline (yukarıdaki sıra)
- Öncesi/sonrası karşılaştırma görseli çıktısı

Batch modu ve otomatik kalite kontrolü ikinci sürüme bırakılabilir.

## Teknoloji

- Python, ComfyUI custom node API
- Mevcut açık kaynak modeller (yeniden eğitim gerekmiyor):
  - Scratch detection: microsoft/Bringing-Old-Photos-Back-to-Life
  - Yüz restorasyonu: GFPGAN / CodeFormer
  - Renklendirme: DDColor
  - Upscale: ESRGAN / tiled upscale

## Referanslar

- https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life
- https://github.com/Skivus/ComfyUI-Bringing-Old-Photos-Back-to-Life
- https://oldphotorestoration.org/blog/comfyui-old-photo-restoration-workflow/
- https://comfy.org/workflows/use-cases/restore-old-photos/

## Sıradaki adımlar (Claude Code'da yapılacak)

1. Proje adı belirle, GitHub'da isim çakışması kontrolü yap.
2. Repo iskeletini kur: `custom_nodes/` altında node paketi, `workflows/`
   altında hazır .json workflow'lar, README.
3. Gerekli modelleri indiren bir kurulum scripti yaz.
4. MVP pipeline'ı tek fotoğrafla test et.
5. Karşılaştırma/kalite kontrol node'unu ekle.
6. Batch modunu ekle.
