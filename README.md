# Turbulence Restoration

Код для восстановления видео, испорченного атмосферной турбулентностью.

В репозитории есть:

- симулятор турбулентности;
- модель восстановления по соседним кадрам;
- обучение по YAML-конфигам;
- инференс готового видео.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Данные

Ожидаемый формат:

```text
datasets/clean/
  train/video_0001/000000.png
  train/video_0001/000001.png
  val/video_0001/000000.png
```

## Обучение

Минимальный запуск:

```bash
python -m turbulence_restoration.training.train \
  --config configs/train_stage.yaml \
  --device cuda
```

## Инференс

```bash
python -m turbulence_restoration.inference.infer_video \
  --input input.mp4 \
  --output restored.mp4 \
  --weights checkpoints/train_stage/best.pt \
  --fps 24 \
  --device cuda
```
