# Дарця-бокс

Подарунок: маленький пристрій з екраном і двома кнопками, який веде з Дарцею
короткий діалог і показує теплі фрази та спогади. Фрази генерує крихітна
мовна модель (трансформер ~11M параметрів), що працює прямо на мікроконтролері,
без інтернету.

Перше питання завжди: **«В тебе хороший настрій?»**, далі діалог іде за
сценарієм, а відповіді в потрібних місцях генерує модель.

## Залізо

| Що | Модель |
|---|---|
| Плата | ESP32-S3 **N16R8** (16 МБ flash, 8 МБ PSRAM) |
| Екран | ST7789, 240×240, SPI |
| Кнопки | 2 зовнішні (ліва / права відповідь) |
| Живлення | USB / павербанк |

Спочатку план був на STM32F401CC, але там влазить лише модель ~120K параметрів
(char-level GRU). ESP32-S3 дозволяє модель у ~100 разів більшу.

## Як це працює

```
[сценарій діалогу]  ->  вузол "gen: V"  ->  модель отримує <bos><V>  ->  генерує фразу
 dialog/dialog.json                          (V = спогад про Відень)      -> фільтр -> екран
```

- **Сценарій** ([dialog/dialog.json](dialog/dialog.json)): граф вузлів. Кожен вузол
  має текст (`say`) або категорію для генерації (`gen`), і дві кнопки з переходами.
- **Категорії** (тег на початку фрази, за ним модель знає, про що писати):

  | Тег | Зміст | Тег | Зміст |
  |---|---|---|---|
  | `C` | комплімент | `P` | жарт |
  | `L` | кохання | `V` | спогад: Відень |
  | `S` | підтримка | `D` | спогад: перше побачення |
  | `M` | сумую | `O` | спогад: Олександрія |
  | `W` | побажання / турбота | `T` | стиль з нашого Telegram-чату |

- **Модель**: трансформер у стилі [llama2.c](https://github.com/karpathy/llama2.c)
  (RMSNorm, RoPE, SwiGLU, спільні ваги embedding/виходу), BPE-токенізатор на 4096 токенів.
- **Тренування у 2 етапи**:
  1. *pretrain* на художній прозі (UberText 2.0, fiction): модель вчить українську;
  2. *finetune* на наших фразах з тегами категорій: вчить стиль і теми.

## Бюджет пам'яті (ESP32-S3)

| Пресет | Параметри | Flash (int8 + масштаби) | RAM (KV-кеш float, 256 токенів) | Pretrain на CPU i7-12650H |
|---|---|---|---|---|
| tiny | 3.0M | ~3.4 МБ | ~2 МБ | ~2 год |
| small | 7.2M | ~8.1 МБ | ~3.5 МБ | ~6 год |
| **base** (основний) | **11.2M** | **~12.6 МБ** | **~5.2 МБ** (2.6 МБ при 128 токенах) | **~8–9 год** |

На ESP32 швидкість ~3–6 токенів/с (упирається в читання з flash), фраза за 5–10 с,
поки на екрані б'ється серце.

## Структура

```
dataset/
  generate_dataset.py   генератор доменних фраз -> phrases.tsv
  phrases.tsv           ~5800 фраз "<категорія>\t<текст>"
  memories.txt          спогади (V/D/O), редагуються вручну
dialog/
  dialog.json           сценарій діалогу
tools/
  play_dialog.py        перевірка сценарію + програвання в терміналі
llm/
  charset.py            дозволені символи (= гліфи шрифту на пристрої)
  prepare_pretrain.py   корпус UberText fiction -> data/pretrain.txt
  prepare_telegram.py   експорт Telegram -> data/telegram.tsv (тільки локально!)
  train_tokenizer.py    BPE-токенізатор -> data/tok.model
  tokenize_data.py      тексти -> .bin для тренування
  model.py              модель + пресети + розрахунок бюджету
  train.py              тренування (pretrain / finetune), CPU або GPU
  sample.py             генерація з чекпойнта (PyTorch)
  export.py             чекпойнт -> export/model.bin + export/tokenizer.bin
  c/dlm.h, c/dlm.c      інференс на C (спільний для ПК і ESP32)
  c/main_pc.c           консольна програма для ПК
```

## Встановлення

Python 3.11+.

```
python -m pip install -r requirements.txt
```

Якщо немає NVIDIA GPU, PyTorch краще ставити CPU-версію:

```
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Повний цикл з нуля

Великі файли (корпус, токени, чекпойнти) в git не потрапляють, їх треба згенерувати.

**1. Доменні фрази і діалог**

```
python dataset/generate_dataset.py
python tools/play_dialog.py
```

**2. Корпус для pretrain** (~433 МБ завантаження, ~375 МБ тексту)

```
mkdir llm/data/raw
curl -L -o llm/data/raw/fiction.txt.bz2 "https://lang.org.ua/static/downloads/ubertext2.0/fiction/cleansed/ubertext.fiction.filter_rus_gcld+short.text_only.txt.bz2"
python llm/prepare_pretrain.py --max_chars 300000000
```

**3. (Опційно) Telegram-чат**: Telegram Desktop -> чат -> ⋮ -> Export chat history ->
без медіа -> формат JSON.

```
python llm/prepare_telegram.py --json "C:/шлях/до/result.json"
python llm/prepare_telegram.py --json "C:/шлях/до/result.json" --me "Моє ім'я"
```

Потім **переглянути** `llm/data/telegram.tsv` і видалити все, що не має бути на екрані.

**4. Токенізатор і токени**

```
python llm/train_tokenizer.py --vocab_size 4096
python llm/tokenize_data.py pretrain
python llm/tokenize_data.py finetune
```

> Токенізатор треба тренувати **до** pretrain і більше не міняти: модель прив'язана до нього.
> `llm/data/tok.model` лежить у репозиторії. Після зміни фраз достатньо
> перезапустити тільки `tokenize_data.py finetune`.

**5. Pretrain** (довго, краще на ніч)

```
python llm/train.py --stage pretrain --preset base --max_iters 8500
```

Перервати: Ctrl+C. Продовжити:

```
python llm/train.py --stage pretrain --preset base --max_iters 8500 --resume
```

Перевірити:

```
python llm/sample.py --ckpt llm/out/pretrain_base.pt --prompt "Вона усміхнулась і"
```

**6. Finetune** (~15–30 хв)

```
python llm/train.py --stage finetune --preset base --init llm/out/pretrain_base.pt --max_iters 1500 --eval_every 100
python llm/sample.py --ckpt llm/out/finetune_base.pt --cat V --n 10
```

**7. Експорт для C**

```
python llm/export.py --ckpt llm/out/finetune_base.pt
```

**8. C-інференс на ПК** (Visual Studio 2022, "x64 Native Tools Command Prompt")

```
cl /O2 /openmp /utf-8 /Fe:llm\c\dlm.exe llm\c\main_pc.c llm\c\dlm.c
llm\c\dlm.exe llm\export\model.bin llm\export\tokenizer.bin -c C -n 10
```

або з gcc:

```
gcc -O2 -fopenmp -o llm/c/dlm llm/c/main_pc.c llm/c/dlm.c -lm
```

## Стан

- [x] Датасет фраз (5800) + спогади-чернетки
- [x] Сценарій діалогу (24 вузли) + програвач у терміналі
- [x] Корпус, токенізатор, токени (70M токенів)
- [x] Модель, тренування, генерація (перевірено коротким прогоном)
- [ ] Pretrain base (запущено)
- [ ] Реальні деталі спогадів у `dataset/memories.txt` (і що таке «дд»)
- [ ] Telegram-чат як дані
- [ ] Більший датасет фраз через відкриту модель-«вчителя» (Gemma / MamayLM)
- [ ] Finetune
- [x] Експорт int8 + C-інференс: **написано, ще не скомпільовано і не звірено з PyTorch**
- [ ] `check_c.py`: автоматичне порівняння логітів C vs PyTorch
- [ ] Фільтр якості + fallback на шаблони / записані історії
- [ ] Прошивка ESP32-S3 (ESP-IDF): ST7789, кириличний шрифт 12×24 з ♥, кнопки,
      машина станів діалогу, анімація серця, «друкування» по буквах

## Приватність

Репозиторій варто тримати **приватним**: там ім'я і спогади.
`llm/data/telegram*` в `.gitignore` і в git не потрапляє.

## Джерела

- Корпус: [UberText 2.0](https://lang.org.ua/en/ubertext/), підкорпус fiction (Chaplynskyi, UNLP 2023)
- Архітектура/формат: [karpathy/llama2.c](https://github.com/karpathy/llama2.c)
- LLM на ESP32-S3: [esp32-llm](https://github.com/jstephencorey/esp32-llm)
