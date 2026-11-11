# Benchmark summary

2 sentences, 104 transcriptions. WER in %. Mean ± SD across sentences; pooled = total errors / total reference words.
With this few sentences, differences of a few points are indicative only.

## Mean WER by condition, noise reduction OFF

| condition         | Wav2Vec2 Base (960h)   | Whisper base.en   |
|:------------------|:-----------------------|:------------------|
| Clean             | 0.0 ± 0.0              | 0.0 ± 0.0         |
| Rain 10 dB        | 10.9 ± 1.2             | 5.0 ± 7.1         |
| Rain 5 dB         | 62.6 ± 38.7            | 10.0 ± 14.1       |
| Rain 0 dB         | 89.1 ± 1.2             | 80.6 ± 55.7       |
| Traffic 10 dB     | 2.9 ± 4.2              | 0.0 ± 0.0         |
| Traffic 5 dB      | 2.9 ± 4.2              | 0.0 ± 0.0         |
| Traffic 0 dB      | 21.8 ± 2.5             | 27.9 ± 31.2       |
| Cafe babble 10 dB | 7.9 ± 2.9              | 0.0 ± 0.0         |
| Cafe babble 5 dB  | 54.7 ± 35.8            | 50.9 ± 55.3       |
| Cafe babble 0 dB  | 94.1 ± 8.3             | 82.4 ± 25.0       |
| Engine 10 dB      | 0.0 ± 0.0              | 5.0 ± 7.1         |
| Engine 5 dB       | 7.9 ± 2.9              | 10.0 ± 14.1       |
| Engine 0 dB       | 16.8 ± 9.6             | 12.9 ± 10.0       |

## Mean WER by condition, noise reduction ON

| condition         | Wav2Vec2 Base (960h)   | Whisper base.en   |
|:------------------|:-----------------------|:------------------|
| Clean             | 5.0 ± 7.1              | 5.0 ± 7.1         |
| Rain 10 dB        | 10.9 ± 1.2             | 15.9 ± 5.8        |
| Rain 5 dB         | 29.7 ± 0.4             | 27.6 ± 10.8       |
| Rain 0 dB         | 81.5 ± 40.3            | 93.2 ± 23.7       |
| Traffic 10 dB     | 7.9 ± 2.9              | 7.9 ± 2.9         |
| Traffic 5 dB      | 10.9 ± 1.2             | 13.8 ± 5.4        |
| Traffic 0 dB      | 23.8 ± 8.7             | 30.9 ± 27.0       |
| Cafe babble 10 dB | 13.8 ± 5.4             | 16.8 ± 9.6        |
| Cafe babble 5 dB  | 55.6 ± 20.4            | 46.5 ± 9.2        |
| Cafe babble 0 dB  | 110.3 ± 56.2           | 88.2 ± 16.6       |
| Engine 10 dB      | 7.9 ± 2.9              | 5.9 ± 8.3         |
| Engine 5 dB       | 12.9 ± 10.0            | 18.8 ± 1.7        |
| Engine 0 dB       | 26.8 ± 4.6             | 35.6 ± 7.9        |

## Mean WER by SNR (all noises pooled)

|               |   Wav2Vec2 Base (960h) |   Whisper base.en |
|:--------------|-----------------------:|------------------:|
| NR off, 10 dB |                    5.4 |               2.5 |
| NR off, 5 dB  |                   32.1 |              17.7 |
| NR off, 0 dB  |                   55.4 |              51   |
| NR on, 10 dB  |                   10.1 |              11.6 |
| NR on, 5 dB   |                   27.3 |              26.7 |
| NR on, 0 dB   |                   60.6 |              62   |

## Wins per mixture

|                      |   wins |
|:---------------------|-------:|
| Whisper base.en      |     18 |
| Wav2Vec2 Base (960h) |     17 |
| tie                  |     17 |

## Error split (all conditions)

High insertion counts in heavy noise point to hallucinated words.

| model                |   S |   D |   I |   N |   WER pooled % |
|:---------------------|----:|----:|----:|----:|---------------:|
| Wav2Vec2 Base (960h) | 120 |  62 |  15 | 702 |           28.1 |
| Whisper base.en      | 118 |  31 |  25 | 702 |           24.8 |

## Speed (cpu, 4 threads)

Median is the fairer figure: background load on a laptop causes occasional slow runs.

| model                |   time_mean_s |   time_median_s |   rtf_mean |   rtf_median |
|:---------------------|--------------:|----------------:|-----------:|-------------:|
| Wav2Vec2 Base (960h) |         1.878 |           0.896 |      0.382 |        0.178 |
| Whisper base.en      |         6.457 |           2.991 |      1.286 |        0.619 |

## Effect of noise reduction (WER with NR minus without, points; negative = helped)

| noise       |   Wav2Vec2 Base (960h) |   Whisper base.en |
|:------------|-----------------------:|------------------:|
| Cafe babble |                    7.6 |               6.1 |
| Engine      |                    7.6 |              10.8 |
| Rain        |                  -13.5 |              13.7 |
| Traffic     |                    5   |               8.2 |
