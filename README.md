# Gistify: Not Özetleyici

Gistify, terminal üzerinden girilen veya bir dosyadan okunan metinleri özetlemek için tasarlanmış basit bir Python aracıdır. `transformers` kütüphanesini kullanarak `facebook/bart-large-cnn` modelini kullanarak özetleme yapar. Uzun metinler için gelişmiş parçalama ve özyinelemeli özetleme stratejisi kullanır.

## Özellikler

- Metinleri doğrudan komut satırından özetleme.
- **Çeşitli dosya formatlarından (TXT, PDF, DOCX) metin okuma ve özetleme.**
- `facebook/bart-large-cnn` modeli ile yüksek kaliteli özetler.
- Uzun metinler için gelişmiş parçalama ve özyinelemeli özetleme stratejisi.

## Kurulum

Bu projeyi çalıştırmak için aşağıdaki adımları izleyin:

1.  **Python ve pip'in yüklü olduğundan emin olun.**

2.  **Gerekli kütüphaneleri yükleyin:**
    `transformers` kütüphanesi ve modelin çalışması için PyTorch gereklidir. PDF ve DOCX dosyalarını okumak için ek kütüphaneler gereklidir.

    ```bash
    pip install transformers torch PyPDF2 python-docx
    ```

3.  **Gistify dosyasını indirin veya klonlayın:**

    ```bash
    # Eğer git kullanıyorsanız
    git clone <proje_deposu_url>
    cd Gistify/Code/Gistify
    # veya sadece gistify.py dosyasını indirin
    ```

## Kullanım

`gistify.py` betiğini çalıştırmak için aşağıdaki komutları kullanabilirsiniz:

### Doğrudan Metin Özetleme

`-t` veya `--text` argümanını kullanarak doğrudan bir metin özetleyebilirsiniz:

```bash
python gistify.py -t "Buraya özetlemek istediğiniz uzun metni girin. Bu metin, önemli bilgileri içeren ve özetlenmesi gereken bir paragraf veya daha uzun bir yazı olabilir."
```

### Dosyadan Metin Özetleme

`-f` veya `--file` argümanını kullanarak bir metin dosyasının içeriğini özetleyebilirsiniz. Desteklenen formatlar: `.txt`, `.pdf`, `.docx`.

```bash
# TXT dosyası için
python gistify.py -f metin.txt

# PDF dosyası için
python gistify.py -f belge.pdf

# DOCX dosyası (Word belgesi) için
python gistify.py -f rapor.docx
```

`metin.txt`, `belge.pdf` veya `rapor.docx` dosyasının içeriği özetlenecektir.

## Kullanılan Model

Bu proje, özetleme için Hugging Face'in `facebook/bart-large-cnn` modelini kullanır. Bu model, özellikle haber makaleleri gibi uzun metinleri özetlemek için eğitilmiştir ve yüksek kaliteli, akıcı özetler üretir.

Model ilk kez kullanıldığında otomatik olarak indirilecektir. Bu işlem internet bağlantınıza ve modelin boyutuna bağlı olarak biraz zaman alabilir.

**Not:** Modelin Türkçe metinlerde bazen üretebildiği İngilizce kelime halüsinasyonlarını azaltmak için özet üzerinde bir son işlem (post-processing) adımı uygulanmıştır.

## Gelişmiş Özetleme Stratejisi

Betiğin uzun metinleri daha etkili bir şekilde özetleyebilmesi için parçalama (chunking) ve özyinelemeli özetleme (recursive summarization) stratejisi uygulanmıştır. Bu strateji:
1.  Giriş metnini daha küçük parçalara böler.
2.  Her bir parçayı ayrı ayrı özetler.
3.  Bu parça özetlerini birleştirir.
4.  Eğer birleştirilmiş özet hala çok uzunsa, nihai özeti elde etmek için bu özetleri tekrar özetler.

## Hata Yönetimi

Eğer özetleme sırasında bir hata oluşursa (örneğin, modelin indirilememesi veya işleme hatası), betik bir hata mesajı döndürecektir.