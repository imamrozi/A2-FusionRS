# Cell Colab — A2-FusionRS v2 (ekstraksi PyABSA + skoring SA-BERT + token global)

Salin tiap blok ke satu cell Colab, jalankan berurutan.

**Catatan:** cell yang lama memakai `| tee`, BUKAN `| tail`. `tail` menahan
seluruh output sampai perintah selesai, sehingga progres per-run (`MULAI
(4/54)`) tidak terlihat sama sekali selama berjam-jam. `tee` menampilkannya
langsung sekaligus menyimpan log.
Branch: `phase2-a2-fusionrs-v2`.

**Runtime: pilih GPU** (Runtime → Change runtime type → T4/L4). Cell 4
(precompute) butuh GPU; sisanya jalan di CPU tapi jauh lebih lambat.

Urutan ini **tidak boleh diacak**. Cell 8 (commit tabel seleksi) adalah jejak
audit bahwa pemenang ditetapkan sebelum test disentuh — cell 10 akan menolak
jalan kalau dilewati.

---

### Cell 1 — Mount Drive & masuk ke repo

```python
from google.colab import drive
drive.mount('/content/drive')

REPO = '/content/drive/MyDrive/PHD-STUDENT/Code/A2-FusionRS'
%cd $REPO
!pwd && git log --oneline -1
```

---

### Cell 2 — Ambil kode v2 terbaru

```python
!git fetch origin
!git checkout phase2-a2-fusionrs-v2
!git pull origin phase2-a2-fusionrs-v2
!git log --oneline -3
```

Kalau ada konflik karena file hasil run sebelumnya, `git stash` dulu —
jangan `git checkout -- .` (itu membuang perubahan tanpa bisa dikembalikan).

---

### Cell 3 — Verifikasi lingkungan & prasyarat

```python
import torch, subprocess, os
print('GPU        :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'TIDAK ADA (precompute akan sangat lambat)')
print('torch      :', torch.__version__)

# Cache yang WAJIB sudah ada sebelum precompute
need = {
    'yelp_restaurant':    'restaurant',
    'amazon_electronics': 'amazon_electronics',
    'tripadvisor_hotel':  'tripadvisor_hotel',
}
print('\n-- prasyarat --')
for ckpt, label in need.items():
    py  = f'checkpoints/{ckpt}/pyabsa/pyabsa_scores_{label}.csv'
    sa  = f'checkpoints/{ckpt}/sentiment_bert/sentiment_scores.csv'
    mdl = f'checkpoints/{ckpt}/sentiment_bert/model.safetensors'
    for p in (py, sa, mdl):
        print(('  OK   ' if os.path.exists(p) else '  HILANG '), p)
```

Semua harus `OK`. Kalau ada yang hilang, precompute tidak bisa jalan.

---

### Cell 4 — Precompute skor SA-BERT atas aspek PyABSA (GPU, ~20–40 mnt)

Ini prasyarat semua run berikutnya. Resumable: kalau terputus, jalankan lagi —
domain yang sudah selesai dilewati otomatis.

```python
!python scripts/precompute_pyabsa_sabert_scores.py 2>&1 | tee /content/precompute.log
```

Per domain saja (kalau mau bertahap):

```python
!python scripts/precompute_pyabsa_sabert_scores.py --domain tripadvisor_hotel
```

Verifikasi hasilnya:

```python
import pandas as pd, os
for ckpt, label in [('yelp_restaurant','restaurant'),
                    ('amazon_electronics','amazon_electronics'),
                    ('tripadvisor_hotel','tripadvisor_hotel')]:
    f = f'checkpoints/{ckpt}/pyabsa/sabert_aspect_scores_{label}.csv'
    if os.path.exists(f):
        d = pd.read_csv(f)
        print(f'{ckpt:22} {len(d):>8,} baris  {d.aspect_term.nunique():>6,} istilah aspek unik')
    else:
        print(f'{ckpt:22} BELUM ADA')
```

---

### Cell 5 — Seleksi kapasitas AGF di DEV (54 run, ~2–3 jam)

**Tidak menyentuh test set.** Resumable — aman dijalankan ulang kalau Colab
terputus.

```python
!bash scripts/run_agf_v2_selection.sh 2>&1 | tee /content/selection.log
```

Pantau progres dari cell terpisah kalau perlu:

```python
!ls checkpoints/results_phase2_clean_v2/dev/*.yaml 2>/dev/null | wc -l
!echo "target: 54"
!cat checkpoints/results_phase2_clean_v2/logs/selection/FAILED_RUNS.txt 2>/dev/null || echo "(tidak ada kegagalan)"
```

---

### Cell 6 — Terapkan aturan keputusan pra-registrasi

```python
!python scripts/analyze_agf_v2_selection.py
```

Script ini **menolak jalan** kalau grid belum lengkap, ada file dengan
`eval_split != selection_dev`, atau ada sel yang konfigurasi "konstan"-nya
ternyata berbeda. Kalau berhenti dengan pesan error, baca pesannya — itu
disengaja, bukan bug.

Catat baris **PEMENANG** (nilai `d` dan `weight_decay`). Kalau ada peringatan
bahwa pemenang hanya menang lewat tie-break, itu **wajib** masuk manuskrip.

---

### Cell 7 — (opsional) Salin hasil seleksi ke lokal

```python
!ls -la reports/agf_v2_selection/
!cat reports/agf_v2_selection/selection_ranking.csv
```

---

### Cell 8 — COMMIT tabel seleksi ⚠ JANGAN DILEWATI

Ini jejak audit bahwa pemenang ditetapkan **sebelum** test set disentuh.
Tanpa ini, klaim protokol anti-p-hacking tidak bisa dibuktikan.

```python
!git add reports/agf_v2_selection/
!git -c user.email="imam.rozi@gmail.com" -c user.name="imamrozi" commit -m "Seleksi DEV v2: pemenang kapasitas AGF dikunci sebelum test disentuh"
!git push origin phase2-a2-fusionrs-v2
!git log --oneline -1
```

---

### Cell 9 — Kunci pemenang di skrip faktorial

Ganti `NILAI_D` dan `NILAI_WD` dengan hasil Cell 6. **Jangan menebak** —
mengubahnya tanpa seleksi = memilih arsitektur di test set.

```python
NILAI_D  = 64      # <-- isi dari Cell 6
NILAI_WD = 0.0001  # <-- isi dari Cell 6

import re, pathlib
p = pathlib.Path('scripts/run_agf_v2_factorial.sh')
s = p.read_text()
s = re.sub(r'^WINNER_D=.*$',  f'WINNER_D="{NILAI_D}"',  s, count=1, flags=re.M)
s = re.sub(r'^WINNER_WD=.*$', f'WINNER_WD="{NILAI_WD}"', s, count=1, flags=re.M)
p.write_text(s)
!grep -E "^WINNER_(D|WD)=" scripts/run_agf_v2_factorial.sh
```

Commit juga penguncian ini:

```python
!git add scripts/run_agf_v2_factorial.sh
!git -c user.email="imam.rozi@gmail.com" -c user.name="imamrozi" commit -m "Kunci pemenang seleksi v2 di skrip faktorial"
!git push origin phase2-a2-fusionrs-v2
```

---

### Cell 10 — Konfirmasi di TEST (60 run, ~3 jam) ⚠ MENYENTUH TEST SET

Jalankan **sekali**. `save_results_yaml(overwrite=False)` akan gagal keras
kalau hasil sudah ada — itu disengaja, supaya run ulang tidak diam-diam
mengganti angka yang sudah dilaporkan.

```python
!bash scripts/run_agf_v2_factorial.sh 2>&1 | tee /content/factorial.log
```

Progres:

```python
!ls checkpoints/results_phase2_clean_v2/test/*.yaml 2>/dev/null | wc -l
!echo "target: 60"
!cat checkpoints/results_phase2_clean_v2/logs/factorial/FAILED_RUNS.txt 2>/dev/null || echo "(tidak ada kegagalan)"
```

---

### Cell 11 — Analisis efek (dengan pembanding adil)

```python
!python scripts/analyze_agf_v2_factorial.py
```

Yang harus dibaca: **`efek EKSTRAKSI PyABSA (ADIL)` = E − B′**. Itu klaim
kontribusi PyABSA yang sah. `TOTAL vs A2-IRM` (F − A) boleh dilaporkan sebagai
total perbaikan sistem, tapi **bukan** sebagai kontribusi PyABSA — selisih
naif kira-kira 2× selisih adil.

---

### Cell 12 — Analisis robustness

```python
!python scripts/analyze_agf_robustness.py --version v2
```

Perhatikan tabel **worst-case** dan **CV**, bukan hanya SD: sel dengan SD kecil
tapi mean tinggi bisa punya worst-case lebih buruk daripada baseline.

Pembanding dengan faktorial lama:

```python
!python scripts/analyze_agf_robustness.py --version v1
```

---

### Cell 13 — Integritas hasil (jalankan sebelum menulis manuskrip)

```python
import glob, yaml, collections
files = sorted(glob.glob('checkpoints/results_phase2_clean_v2/test/*.yaml'))
print(f'total file: {len(files)} (target 60)\n')

bad, cnt = [], collections.Counter()
for f in files:
    d = yaml.safe_load(open(f))
    cnt[(d.get('run_tag'), d.get('domain'))] += 1
    if d.get('stage') != 'confirm':   bad.append((f, 'stage', d.get('stage')))
    if d.get('eval_split') != 'test': bad.append((f, 'eval_split', d.get('eval_split')))

print('sel x domain (harus 5 seed masing-masing):')
for k, v in sorted(cnt.items()):
    print(f'  {str(k):48} {v}', '' if v == 5 else '  <-- TIDAK 5!')

print('\npelanggaran:', bad if bad else 'tidak ada')
```

---

## Ringkasan waktu

| Cell | Isi | Perkiraan |
|---|---|---|
| 4 | precompute SA-BERT (GPU) | 20–40 mnt |
| 5 | seleksi DEV 54 run | 2–3 jam |
| 10 | faktorial TEST 60 run | ~3 jam |

Cell 5 dan 10 keduanya resumable — kalau Colab memutus sesi, jalankan ulang
cell yang sama dan run yang sudah selesai akan dilewati.
