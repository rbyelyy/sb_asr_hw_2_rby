This repository provides code for solving the **Automatic Speech Recognition (ASR)** task using **PyTorch**.

---

## ⚙️ Installation

### **Prerequisites:**
- 🐍 Python 3.x (Check with `python3 --version`)
- 📦 pip (Check with `pip --version`)

### **Steps:**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rbyelyy/sb_asr_hw_2_rby.git
   ```

2. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download LM file**
   ```bash
   mkdir -p src/model/lm/ && cd src/model/lm/ && gdown https://drive.google.com/uc?id=1rPFId5MJpA-5eApoRUPrUuihCqo7HTIh
   ```

## 🚀 Training

Model was trained with following duration:
- n_epochs: 70
- epoch_len: 400

```bash
python3 train.py
```

### Data Sets:
 - train-clean-100 (Train)
 - val-clean
 - test-clean

### Training Results:
    epoch          : 70
    loss           : 0.0624021303281188
    grad_norm      : 0.8567263886332512
    val_loss       : 0.7683639847315274
    val_CER_(Argmax): 0.18309797752060733
    val_WER_(Argmax): 0.5383294712728577
    test_loss      : 0.5632555152361209
    test_CER_(Argmax): 0.14324053146386118
    test_WER_(Argmax): 0.4427900451156789
wandb: 🚀 View run testing at: https://wandb.ai/hse_rbyelyy/pytorch_template_asr_example/runs/qq0nvt91

Example of generation on the last epoch:
```text
target_text - as i spoke i made him a gracious bow and i think i showed him by my mode of address that i did not bear any grudge as to my individual self
predicted_text - as i spoke i made him a gracious bow and i thick a showed him bot ma note if adres that i didnot there any grudge ask to my individualself
target_text - dont know well of all things inwardly commented miss taylor literally born in cotton and oh well as much as to ask whats the use she turned again to go
predicted_text - dot now wile of al things in wiuldle comented mhis tailor neter ra bon and cotden and o wel as muches to esk whas the yuse she turned again to go

```

---

### 🔍 Inference
Run inference using the pre-trained model:

```bash
python3 inference.py
```
------
## ОТЧЕТ
Отчет по проекту ASR
1. Как воспроизвести модель:
Обучение в течение 75 эпох:
   - Первые 50 эпох: используя train_config_1.yaml с начальным learning rate
   - Следующие 25 эпох: используя train_config_2.yaml с уменьшенным learning rate и увеличенным количеством итераций на эпоху
2. Использование KenLM языковой модели для декодирования (реализовано в CTCTextEncoder)
2. Что было реализовано и протестировано:
А. Аугментация данных:

Успешно реализовано:

Аугментация громкости (Simple gain/volume)
Добавление гауссовского шума


Рассматривались, но не были реализованы из-за временных ограничений:

Time stretching
Pitch shifting
SpecAugment
Time masking



Б. Языковые модели:

KenLM показала лучшее соотношение производительность/скорость
BPE и GPT2 также работали, но были значительно медленнее
Добавлен параметр в конфигурации для переключения между подходами (KenLM/BPE/GPT2)

В. Архитектурные эксперименты:

Базовая архитектура:

Модифицированный baseline с dropout (0.2)
Инициализация Kaiming для полносвязного слоя
Инициализация bias против blank token
2-слойный двунаправленный LSTM


Неудачные эксперименты:

Попытка использования архитектуры в стиле DeepSpeech v2:

Проблема взрывающихся градиентов около 50-й эпохи
Пробовал исправить уменьшением learning rate и нормализацией
В итоге остался на модифицированном baseline из-за лучшей стабильности


Эксперименты с более сложными архитектурами:

Более медленная сходимость loss
Потенциально требовали >75 эпох для улучшения результатов





3. Процесс обучения:

Оптимальное количество эпох - 75:

Loss выходил на плато после ~75 эпох
Увеличено количество итераций в середине обучения
Стратегия learning rate:

Более высокий начальный rate для первых 50 эпох
Уменьшенный rate для последних 25 эпох





4. Бонусные задачи:

Интеграция языковых моделей:

Реализованы несколько подходов (KenLM, BPE, GPT2)
Добавлен beam search декодинг с настраиваемыми параметрами
Обработка subword токенов для BPE


Улучшения CTCTextEncoder:

Добавлен параметр beam width
Реализован weight scaling для языковой модели
Добавлен length penalty для улучшения декодирования

*** train_log.log (в корне репозитория)
