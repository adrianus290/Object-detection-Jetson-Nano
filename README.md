# Traffic Sign Recognition dengan NVIDIA Jetson Nano

Sistem ini menggunakan **NVIDIA Jetson Nano Developer Kit**, kamera CSI, model
deteksi objek **YOLO11**, dan tiga LED sebagai aktuator. Kamera membaca video
secara real-time, model mengenali rambu `GO`, `SLOW`, atau `STOP`, kemudian
program mengubah hasil deteksi menjadi keputusan GPIO:

| Sign yang dikenali | Keputusan | LED |
|---|---|---|
| `STOP` | Berhenti | Merah |
| `SLOW` | Kurangi kecepatan | Kuning |
| `GO` | Jalan | Hijau |
| Tidak ada sign atau confidence < 0.50 | Kondisi aman | Semua LED mati |

Alur sistem:

```text
Kamera CSI
    │
    ▼
GStreamer (nvarguscamerasrc)
    │
    ▼
OpenCV + ONNX Runtime (YOLO11)
    │  sign, bounding box, confidence
    ▼
Decision Logic
    │
    ▼
Jetson.GPIO
    │
    ├── LED merah   (STOP)
    ├── LED kuning  (SLOW)
    └── LED hijau   (GO)
```

## Fitur

- Deteksi tiga kelas rambu: `GO`, `SLOW`, dan `STOP`.
- Inferensi model ONNX berukuran input `640 x 640`.
- Confidence threshold: `0.50`.
- Non-maximum suppression dengan IoU threshold `0.45`.
- Tampilan bounding box, nama sign, confidence, FPS, waktu inferensi,
  dan status LED pada jendela OpenCV.
- Kamera diproses menggunakan buffer frame terbaru agar tidak menumpuk
  frame lama.
- Semua LED dimatikan jika tidak ada deteksi yang memenuhi threshold.
- GPIO dibersihkan ketika program dihentikan dengan tombol `q` atau
  `Ctrl+C`.

## Hardware

### Komponen

- NVIDIA Jetson Nano Developer Kit.
- Kamera CSI yang kompatibel dengan Jetson Nano.
- Breadboard dan kabel jumper.
- 3 LED:
  - merah untuk `STOP`,
  - kuning untuk `SLOW`,
  - hijau untuk `GO`.
- 3 resistor pembatas arus, disarankan 220--330 ohm.
- Catu daya Jetson Nano yang sesuai.
- Media sign `GO`, `SLOW`, dan `STOP` untuk pengujian.

Jangan menghubungkan LED langsung tanpa resistor. Pastikan ground rangkaian
LED terhubung ke ground Jetson Nano.

### Wiring GPIO

Program menggunakan `GPIO.BOARD`, sehingga nomor di bawah adalah **nomor pin
fisik**, bukan nomor BCM:

| Fungsi | Variabel program | Pin fisik Jetson | Sambungan |
|---|---|---:|---|
| LED merah | `LED_MERAH = 18` | 11 | GPIO output -> resistor -> anoda LED |
| LED kuning | `LED_KUNING = 12` | 13 | GPIO output -> resistor -> anoda LED |
| LED hijau | `LED_HIJAU = 16` | 16 | GPIO output -> resistor -> anoda LED |
| Ground LED | - | GND, misalnya pin 14 | Katoda LED -> GND |

Di source code, `LED_ACTIVE_LOW = False`, sehingga LED menyala ketika output
GPIO berada pada level HIGH. Jika rangkaian menggunakan konfigurasi active-low,
ubah nilai tersebut menjadi `True`.

Tambahkan foto rangkaian aktual ke folder `Dokumentasi/`, misalnya:

```text
Dokumentasi/wiring-led-jetson.jpg
Dokumentasi/arsitektur-sistem.png
```

## Software dan dependencies

### Software utama

- Ubuntu yang disediakan untuk Jetson Nano melalui JetPack.
- Python 3.
- OpenCV.
- NumPy.
- ONNX Runtime.
- `Jetson.GPIO`.
- GStreamer dan plugin NVIDIA Jetson (`nvarguscamerasrc`,
  `nvvidconv`, dan `fdsink`).

Import yang digunakan oleh [`main.py`](main.py):

- `cv2` dari OpenCV.
- `numpy` dari NumPy.
- `onnxruntime`.
- `Jetson.GPIO` pada perangkat Jetson Nano.
- `subprocess`, `sys`, `threading`, dan `time` dari standard library Python.

> Catatan: `Jetson.GPIO` hanya tersedia dan diperlukan ketika program dijalankan
> pada Jetson Nano. Saat library tersebut tidak tersedia, program tetap dapat
> melakukan inference dan menampilkan video, tetapi LED fisik tidak akan
> dikendalikan.

### Instalasi

Jalankan perintah berikut pada Jetson Nano:

```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv \
    gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good
python3 -m pip install --upgrade pip
python3 -m pip install numpy Jetson.GPIO
```

Pasang ONNX Runtime yang kompatibel dengan versi JetPack, arsitektur
`aarch64`, dan versi Python pada Jetson Nano. Paket ONNX Runtime untuk Jetson
sebaiknya mengikuti wheel/build yang disediakan untuk lingkungan Jetson yang
digunakan, bukan memasang wheel x86 biasa:

```bash
python3 -m pip install onnxruntime
```

Jika perintah terakhir tidak menyediakan wheel yang kompatibel, gunakan build
ONNX Runtime khusus Jetson sesuai versi JetPack yang terpasang. Verifikasi
instalasi dengan:

```bash
python3 -c "import cv2, numpy, onnxruntime; print('dependency OK')"
python3 -c "import Jetson.GPIO as GPIO; print('Jetson.GPIO OK')"
gst-inspect-1.0 nvarguscamerasrc
```

## Struktur repository

Struktur penting saat ini:

```text
.
├── main.py
├── best1.onnx
├── Dataset traffict sign/
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
└── Dokumentasi/
    ├── Jarak dekat/
    ├── Jarak sedang/
    └── Jarak jauh/
```

`main.py` memuat model dari path relatif `best1.onnx`. Karena itu, jalankan
program dari root repository atau ubah `MODEL_ONNX_PATH` jika model disimpan
di lokasi lain.

## Dataset dan Roboflow

Dataset diberi label menggunakan **Roboflow** dan diekspor dalam format YOLO.
Informasi dataset yang tersimpan di repository:

- Project: [Lalu Lintas di Roboflow](https://universe.roboflow.com/adrianus-tukmp/lalu-lintas-k6fep)
- Versi dataset: `2`.
- Kelas: `Go`, `Slow`, `Stop`.
- Jumlah gambar pada export: 231.
- Lisensi dataset: CC BY 4.0.
- Augmentasi yang tercatat: horizontal flip dengan probabilitas 50% dan
  Gaussian blur acak 0--1.2 piksel.

Konfigurasi kelas dapat dilihat pada
[`Dataset traffict sign/data.yaml`](Dataset%20traffict%20sign/data.yaml).
Model hasil training/export yang digunakan aplikasi adalah `best1.onnx`.

## Menjalankan program

1. Hubungkan kamera CSI ke konektor CSI Jetson Nano.
2. Hubungkan ketiga LED sesuai tabel wiring.
3. Pastikan model `best1.onnx` berada di root repository.
4. Buka terminal pada root repository.
5. Jalankan:

   ```bash
   python3 main.py
   ```

6. Arahkan sign ke kamera.
7. Amati jendela **Jetson Nano - Real-Time Traffic Sign** dan LED.
8. Tekan `q` pada jendela video untuk keluar.

Output tampilan menampilkan format confidence pada bounding box, misalnya
`STOP 0.87`. Status bagian atas menunjukkan keputusan yang sedang aktif,
misalnya `PREDIKSI: STOP -> LED MERAH ON`.

### Kondisi aman

Jika tidak ada objek yang terdeteksi atau seluruh confidence berada di bawah
`0.50`, status menjadi:

```text
STATUS: TIDAK ADA RAMBU (All LEDs OFF)
```

Pada kondisi ini semua LED dimatikan. Jika beberapa sign terdeteksi sekaligus,
prioritas keputusan program adalah:

```text
STOP > SLOW > GO
```

## Pengujian minimum

Pengujian perlu dilakukan dengan prosedur dan hasil yang dapat diulang.
Repository ini sudah menyediakan dokumentasi foto pada tiga kategori jarak:

- `Dokumentasi/Jarak dekat/`
- `Dokumentasi/Jarak sedang/`
- `Dokumentasi/Jarak jauh/`


Skenario tambahan yang wajib didokumentasikan:

1. Pergantian `GO -> SLOW -> STOP` dan verifikasi LED ikut berubah.
2. Pergantian `STOP -> tidak ada sign` dan verifikasi semua LED mati.
3. Sign dengan confidence di bawah `0.50` dan verifikasi kondisi aman.
4. Tidak ada sign di area kamera dan verifikasi semua LED mati.
5. Pengujian saat dua sign terlihat sekaligus jika skenario tersebut
   diperlukan; dokumentasikan aturan prioritas `STOP > SLOW > GO`.

### Menampilkan foto di GitHub

File gambar yang berada di repository dapat ditampilkan di README dengan
format Markdown berikut:

```markdown
![Keterangan gambar](Dokumentasi/Jarak%20dekat/STOP%20terang%20dekat.jpeg)
```

Gunakan tanda `%20` untuk setiap spasi pada nama folder atau file. Path harus
relatif terhadap lokasi `README.md`, menggunakan `/` (bukan `\`). Setelah file
README dan gambar sudah di-commit lalu di-push ke GitHub, gambar akan tampil
otomatis di halaman repository.

Contoh galeri dokumentasi pengujian yang sudah tersedia:

#### Jarak dekat

| Sign | Kondisi terang | Kondisi gelap |
|---|---|---|
| GO | ![GO terang dekat](Dokumentasi/Jarak%20dekat/GO%20terang%20dekat.jpeg) | ![GO gelap dekat](Dokumentasi/Jarak%20dekat/GO%20gelap%20dekat.jpeg) |
| SLOW | ![SLOW terang dekat](Dokumentasi/Jarak%20dekat/SLOW%20terang%20dekat.jpeg) | ![SLOW gelap dekat](Dokumentasi/Jarak%20dekat/SLOW%20gelap%20dekat.jpeg) |
| STOP | ![STOP terang dekat](Dokumentasi/Jarak%20dekat/STOP%20terang%20dekat.jpeg) | ![STOP gelap dekat](Dokumentasi/Jarak%20dekat/STOP%20gelap%20dekat.jpeg) |

#### Jarak sedang

| Sign | Kondisi terang | Kondisi gelap |
|---|---|---|
| GO | ![GO terang sedang](Dokumentasi/Jarak%20sedang/GO%20terang%20sedang.jpeg) | ![GO gelap sedang](Dokumentasi/Jarak%20sedang/GO%20gelap%20sedang.jpeg) |
| SLOW | ![SLOW terang sedang](Dokumentasi/Jarak%20sedang/SLOW%20terang%20sedang.jpeg) | ![SLOW gelap sedang](Dokumentasi/Jarak%20sedang/SLOW%20gelap%20sedang.jpeg) |
| STOP | ![STOP terang sedang](Dokumentasi/Jarak%20sedang/STOP%20terang%20sedang.jpeg) | ![STOP gelap sedang](Dokumentasi/Jarak%20sedang/STOP%20gelap%20sedang.jpeg) |

#### Jarak jauh

| Sign | Kondisi terang | Kondisi gelap |
|---|---|---|
| GO | ![GO terang jauh](Dokumentasi/Jarak%20jauh/GO%20terang%20jauh.jpeg) | ![GO gelap jauh](Dokumentasi/Jarak%20jauh/GO%20gelap%20jauh.jpeg) |
| SLOW | ![SLOW terang jauh](Dokumentasi/Jarak%20jauh/SLOW%20terang%20jauh.jpeg) | ![SLOW gelap jauh](Dokumentasi/Jarak%20jauh/SLOW%20gelap%20jauh.jpeg) |
| STOP | ![STOP terang jauh](Dokumentasi/Jarak%20jauh/STOP%20terang%20jauh.jpeg) | ![STOP gelap jauh](Dokumentasi/Jarak%20jauh/STOP%20gelap%20jauh.jpeg) |


