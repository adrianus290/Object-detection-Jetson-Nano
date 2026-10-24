# Traffic Sign Detection dengan Jetson Nano dan Model YOLO11

Sistem ini menggunakan **NVIDIA Jetson Nano Developer Kit**, kamera CSI, model
deteksi objek **YOLO11**, dan tiga LED sebagai aktuator. Kamera membaca video
secara real-time, model mengenali rambu `GO`, `SLOW`, atau `STOP`, kemudian
program mengubah hasil deteksi menjadi keputusan GPIO:

| Sign yang dikenali | LED |
|---|---|
| `STOP` | Merah |
| `SLOW` | Kuning |
| `GO` | Hijau |
| Tidak ada sign atau confidence < 0.50 | Semua LED mati |

Alur sistem:

```text
Kamera CSI
    │
    ▼
GStreamer
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
- 3 resistor pembatas arus 220.
- Gambar sign `GO`, `SLOW`, dan `STOP` untuk pengujian.

### Wiring GPIO

Program menggunakan `GPIO.BOARD`, sehingga nomor di bawah adalah **nomor pin
fisik**, bukan nomor BCM:

| Fungsi | Variabel program | Pin fisik Jetson | Sambungan |
|---|---|---:|---|
| LED merah | `LED_MERAH = 18` | 24 | GPIO output -> resistor -> anoda LED |
| LED kuning | `LED_KUNING = 12` | 13 | GPIO output -> resistor -> anoda LED |
| LED hijau | `LED_HIJAU = 16` | 26 | GPIO output -> resistor -> anoda LED |
| Ground LED | - | GND, misalnya pin 14 | Katoda LED -> GND |

## Software dan dependencies
### Software utama

- Ubuntu  versi 18.8.2 yang disediakan untuk Jetson Nano melalui JetPack.
- Python 3.
- OpenCV.
- NumPy.
- ONNX Runtime.
- `Jetson.GPIO`.
-

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
sudo apt install -y python3-pip python3-opencv
python3 -m pip install --upgrade pip
python3 -m pip install numpy Jetson.GPIO
```

Pasang ONNX Runtime yang kompatibel dengan versi JetPack, arsitektur
`aarch64`, dan versi Python pada Jetson Nano.

```bash
python3 -m pip install onnxruntime
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
Informasi dataset yang tersimpan di  pada folder `Dataset traffict sign`:

- Kelas: `Go`, `Slow`, `Stop`.
- Jumlah dataset sebelum augmentasi: 150 dengan 50 gambar `GO` 50 gambar `SLOW` 50 gambar `STOP`.
- Dataset diaugmentasi: horizontal flip vertikal dan horizontal serta
  Gaussian blur acak 0--1.2 piksel dengan masing masing menghasilkan.
-Total dataset setelah augmentasi adalah: 231 data train, 14 datal validasi dan 7 data testing.

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
prioritas keputusan program adalah pada nilai confident tertinggi.

## Pengujian minimum
Pengujian perlu dilakukan dengan prosedur dan hasil yang dapat diulang.
Repository ini sudah menyediakan dokumentasi foto pada tiga kategori jarak serta pada dua kondisi pencahayaan yaitu gelap dan terang:

- `Dokumentasi/Jarak dekat/`
- `Dokumentasi/Jarak sedang/`
- `Dokumentasi/Jarak jauh/`


Skenario tambahan yang wajib didokumentasikan:

1. Pergantian `GO -> SLOW -> STOP` dan verifikasi LED ikut berubah.
2. Pergantian `STOP -> tidak ada sign` dan verifikasi semua LED mati.
3. Sign dengan confidence di bawah `0.50` dan verifikasi kondisi aman.
4. Tidak ada sign di area kamera dan verifikasi semua LED mati.
5. Pengujian saat dua sign terlihat sekaligus jika skenario tersebut
   diperlukan; dokumentasikan aturan prioritas adalah dengan nilai confident tertinggi.


## Tabel Rekap Pengujian

| Sign | Jarak | Kondisi Cahaya | Confidence | LED | Hasil |
|------|-------|----------------|------------|-----|------|
| GO   | Dekat | Terang | 0.93 | Hijau | Berhasil |
| SLOW | Sedang | Gelap | 0.87 | Kuning | Berhasil |
| STOP | Jauh  | Terang | 0.89 | Merah | Berhasil |
| -    | -     | Terang    | - | Semua LED mati | Berhasil |

|------|-------|----------------|------------|-----|
| GO   | Dekat | Terang | 0.93 | Hijau |
| SLOW | Sedang | Gelap | 0.87 | Kuning |
| STOP | Jauh  | Terang | 0.96 | Merah |
| -    | -     | -      | < 0.50 | Semua LED mati |

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


