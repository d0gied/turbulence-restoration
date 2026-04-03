# Turbulence Restoration Repository

Рабочий PyTorch-репозиторий для курсовой по устранению атмосферной турбулентности.

Что внутри:

- `GPUTurbulenceSimulator`: GPU-симулятор формулы
  `I_dist(x,y,t)=alpha(x,y,t) * Blur_sigma( I_clean(x+u,y+v,t) )`.
- 3D Perlin-like noise на PyTorch для временно-связанных полей `u,v`.
- Time-aware sampler: кадры выбираются по секундам, а не только по `T`.
- `TimeAwareGeoLuckyRestorer`: shared 2D encoder → time FiLM → pyramid alignment → softmax lucky fusion → decoder.
- Tiled inference: `tile + halo + overlap blending`.
- Smoke-test, train loop, PSNR/SSIM.

## Установка

```bash
cd turbulence_restoration_repo
pip install -r requirements.txt
```

## Smoke-test

```bash
python scripts/smoke_test.py --device cuda
```

или:

```bash
python scripts/smoke_test.py --device cpu
```

## Основной temporal policy

Для 60 FPS:

```text
K = 9
offsets = [-6, -4, -2, -1, 0, +1, +2, +4, +6]
span = 200 ms
```

В коде:

```python
TemporalSampler(policy="k9_200ms")
```

## Обучение на smoke dataset

```bash
python -m turbulence_restoration.training.train --config configs/train_smoke.yaml --device cuda
```

## Inference по видео

```bash
python -m turbulence_restoration.inference.infer_video \
  --input input.mp4 \
  --output restored.mp4 \
  --weights checkpoints/best.pt \
  --fps 60 \
  --temporal_policy default_200ms \
  --tile 512 \
  --halo 128 \
  --overlap 128
```

## Структура

```text
turbulence_restoration/
  simulator/gpu_turbulence.py   GPU Perlin, warp, blur, scintillation
  data/temporal_sampler.py      выбор кадров по времени
  data/dataset.py               synthetic/debug dataset
  models/restorer.py            TimeAwareGeoLuckyRestorer
  training/losses.py            Charbonnier + Sobel + auxiliary losses
  training/train.py             базовый train loop
  inference/tiled_inference.py  tile+halo inference
  experiments/metrics.py        PSNR/SSIM
```

## Что доказывать экспериментами

1. `T=5 consecutive` против `K=9 200ms`.
2. `no time embedding` против `time embedding`.
3. `no alignment` против `pyramid alignment`.
4. `no oracle lucky fusion pretrain` против `oracle lucky fusion pretrain`.
5. `full inference` против `tiled inference` на кадре, который помещается в память.
