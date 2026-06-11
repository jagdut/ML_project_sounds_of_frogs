# ML_project-_sound_of_frogs

# 🐸 Rozpoznawanie odgłosów płazów

**Gabriela Marchwica, Jagoda Dutkiewicz**

Projekt z zakresu uczenia maszynowego klasyfikujący odgłosy płazów do jednego z 10 gatunków przy użyciu konwolucyjnych sieci neuronowych (CNN).

---

## 📋 Opis projektu

Celem projektu było stworzenie systemu, który na podstawie nagrań audio potrafi automatycznie rozpoznać gatunek płaza. Dane pochodzą z bazy [Anuran Sound (Frogs or Toads) Dataset](https://www.kaggle.com/) na platformie Kaggle.

---

## 🦎 Dane

Ze zbioru 26 gatunków wybrano 10 pierwszych, obejmując łącznie **565 nagrań** w formacie `.m4a`:

| Gatunek | Liczba nagrań |
|---|---|
| Acris gryllus | 70 |
| Agalychnis callidryas | 85 |
| Colostethus talamancae | 52 |
| Alytes cisternasii | 50 |
| Alytes muletensis | 30 |
| Ameerega cainarachi | 75 |
| Anaxyrus cognatus | 53 |
| Anaxyrus quercicus | 50 |
| Centrolene savagei | 50 |
| Dendrobates leucomelas | 50 |

---

## ⚙️ Preprocessing

1. Konwersja plików `.m4a` → `.wav` (FFMPEG)
2. Odczyt nagrań jako tablice `float32`
3. Zachowanie sygnału stereo (2 kanały)
4. Ujednolicenie długości do **2.5 sekundy** (przycinanie / zero-padding)
5. Standaryzacja sygnałów (`StandardScaler`)
6. Konwersja do tensorów PyTorch

Podział danych: **70% trening / 20% walidacja / 10% test**, batch size = 16.

---

## 🧠 Modele

Zbudowano i porównano 4 modele o rosnącej złożoności:

| Model | Architektura | Accuracy (5 epok) |
|---|---|---|
| FrogNet_1 | Flatten → Linear | 25% |
| FrogNet_2 | Conv1D (1 blok) + BN + Dropout | 30% |
| FrogNet_3 | Conv1D (2 bloki) + BN + Dropout | 51% |
| FrogNet_4 | Conv1D (2 bloki) + AdaptiveAvgPool + Dropout | **72%** |

Najlepszy model (**FrogNet_4**) dotrenowano przez **30 epok** — osiągając dokładność **89%** na zbiorze testowym.

Kluczowym elementem FrogNet_4 jest warstwa `AdaptiveAvgPool1d(16)`, która uniezależnia model od rozmiaru wejścia.

---

## 📊 Wyniki

- Macierz pomyłek na zbiorze testowym pokazuje bardzo wysoką skuteczność klasyfikacji dla większości gatunków.
- Jako dodatek do projektu nagrano własne imitacje odgłosów żab i przetestowano je na modelu:
  - **Gabrysia** → klasa 0 (*Acris gryllus*, pewność 46.3%)
  - **Jagoda** → klasa 1 (*Agalychnis callidryas*, pewność 99.7%)

---

## 🚀 Uruchomienie

### Wymagania

```bash
pip install torch torchaudio scikit-learn numpy matplotlib ffmpeg-python
```

> Wymagany jest też zainstalowany [FFmpeg](https://ffmpeg.org/) w systemie do konwersji plików audio.

### Uruchomienie notebooka

```bash
jupyter notebook Frogs_and_toads_notebook.ipynb
```

---

## 📁 Struktura repozytorium

```
├── Frogs_and_toads_notebook.ipynb   # Główny notebook z całą analizą
├── README.md
└── data/                            # Katalog na dane (patrz niżej)
```

---


## 🌿 Potencjalne zastosowania

- Kontrola gatunków inwazyjnych i ochrona zagrożonych
- Płazy jako bioindykatory stanu środowiska
- Badanie wpływu zmian klimatycznych na populacje płazów

---

## 📚 Literatura

Erhan Akbal et al., *Explainable automated anuran sound classification using improved one-dimensional local binary pattern and Tunable Q Wavelet Transform techniques*, Expert Systems with Applications, Volume 225, 2023. [DOI](https://doi.org/10.1016/j.eswa.2023.120089)
