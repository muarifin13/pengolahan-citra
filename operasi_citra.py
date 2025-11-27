import cv2                   # library untuk memproses gambar
import numpy as np           # library untuk hitungan matematika (contoh: rata-rata)

# --------------------------------------------------------
# FUNGSI 1 : WHITE BALANCE (menyeimbangkan warna)
# Tujuan: membuat warna foto lebih natural, tidak kekuningan / kebiruan
# --------------------------------------------------------
def white_balance(img):

    b, g, r = cv2.split(img)      # memisahkan gambar menjadi warna Blue, Green, Red

    avg_b = np.mean(b)            # rata-rata warna biru
    avg_g = np.mean(g)            # rata-rata warna hijau
    avg_r = np.mean(r)            # rata-rata warna merah

    k = (avg_b + avg_g + avg_r) / 3     # mencari rata-rata yang seharusnya (nilai ideal)

    # menyeimbangkan warna agar tidak ada yang terlalu dominan
    b = cv2.multiply(b, k / avg_b)  
    g = cv2.multiply(g, k / avg_g)
    r = cv2.multiply(r, k / avg_r)

    balanced = cv2.merge([b, g, r])     # menggabungkan kembali warna menjadi satu gambar

    return np.clip(balanced, 0, 255).astype(np.uint8)   # memastikan nilai piksel tetap 0–255


# --------------------------------------------------------
# FUNGSI 2 : GAMMA CORRECTION (mencerahkan secara halus)
# Tujuan: membuat foto sedikit lebih terang TANPA merusak detail
# --------------------------------------------------------
def gamma_correction(img, gamma=1.1):

    inv_gamma = 1.0 / gamma       # gamma dibalik (rumusnya memang seperti ini)

    # membuat tabel pencerahan dari nilai 0 sampai 255
    table = np.array([(i / 255.0) ** inv_gamma * 255 
                      for i in range(256)]).astype("uint8")

    return cv2.LUT(img, table)    # menerapkan efek pencerahan ke foto


# --------------------------------------------------------
# FUNGSI 3 : CLAHE (meningkatkan kontras agar detail lebih jelas)
# Tujuan: membuat tekstur wajah / objek lebih terlihat
# --------------------------------------------------------
def apply_clahe(img):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)   # ubah ke LAB (L = cahaya)
    l, a, b = cv2.split(lab)                     # pisahkan channel L, A, B

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))   # alat untuk boost kontras
    cl = clahe.apply(l)                          # terapkan ke channel cahaya saja

    merged = cv2.merge((cl, a, b))               # gabung lagi dengan channel warna
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)   # kembalikan ke gambar normal


# --------------------------------------------------------
# FUNGSI UTAMA : menggabungkan semua proses retouch
# --------------------------------------------------------
def color_retouch(img):
    wb = white_balance(img)                      # langkah 1: perbaiki warna
    gamma = gamma_correction(wb, gamma=1.08)     # langkah 2: cerahkan halus
    final = apply_clahe(gamma)                   # langkah 3: naikkan kontras
    return final


# --------------------------------------------------------
# BAGIAN UTAMA PROGRAM (yang dijalankan)
# --------------------------------------------------------
if __name__ == "__main__":

    input_file = "WhatsApp Image.jpeg"                    # gambar asli
    output_retouched = "hasil_retouch.jpg"       # hasil edit
    output_comparison = "perbandingan_before_after.jpg"   # before-after

    original = cv2.imread(input_file)            # buka gambar asli
    if original is None:
        raise Exception("Gambar tidak ditemukan!")   # kalau salah path → error

    retouched = color_retouch(original)          # proses retouching

    cv2.imwrite(output_retouched, retouched)     # simpan hasil edit

    comparison = np.hstack((original, retouched))    # gabungkan kiri-kanan (before-after)

    cv2.imwrite(output_comparison, comparison)       # simpan gabungan

    cv2.imshow("Before (Kiri)   |   After (Kanan)", comparison)  # tampilkan
    cv2.waitKey(0)                                 # tunggu tombol ditekan
    cv2.destroyAllWindows()                        # tutup window

    print("✔ Hasil retouch:", output_retouched)     # info output
    print("✔ Perbandingan before-after:", output_comparison)