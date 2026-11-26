// Презентация к защите проекта. Сборка из корня репозитория:
//   typst compile docs/presentation.typ docs/presentation.pdf

#let authors = "Алан Насибуллин, Герман Еделькин"
#let repo = "https://github.com/lm-cyber/efficient_rank"
#let short-title = "Эффективный ранг активаций как критерий выбора ширины слоёв"

#let accent = rgb("#257de0")
#let orange = rgb("#f47230")
#let ink = rgb("#191919")
#let ink2 = rgb("#5c5c5c")
#let soft = rgb("#f3f6fa")
#let rule = rgb("#d9dee5")

#set document(title: short-title, author: authors)
#set text(font: "PT Sans", size: 20pt, fill: ink, lang: "ru")
#set par(leading: 0.55em, justify: false)
#set list(marker: text(fill: accent)[•], spacing: 0.6em, indent: 0.2em)
#set enum(spacing: 0.7em)
#set table(stroke: none)
#show math.equation: set text(size: 22pt)
#show link: set text(fill: accent)

#set page(
  width: 33.867cm,
  height: 19.05cm,
  margin: (x: 1.6cm, top: 1.25cm, bottom: 1.3cm),
  footer: context {
    let n = counter(page).get().first()
    if n > 1 {
      set text(size: 11pt, fill: ink2)
      grid(columns: (1fr, auto), short-title, [#n / #counter(page).final().first()])
    }
  },
)

#let slide(title: none, body) = {
  pagebreak(weak: true)
  if title != none {
    block(below: 0.25em, text(size: 27pt, weight: "bold", title))
    line(length: 100%, stroke: 1.5pt + accent)
    v(0.4em)
  }
  body
}

#let caption(body) = align(center, text(fill: ink2, size: 15pt, body))
#let card(body, stroke: accent) = block(
  width: 100%, inset: (x: 16pt, y: 13pt), radius: 6pt, fill: soft,
  stroke: (left: 4pt + stroke), body,
)
#let stat(value, label, color: accent) = block(
  width: 100%, inset: 16pt, radius: 6pt, fill: soft,
  [#text(size: 44pt, weight: "bold", fill: color, value) \ #v(-0.1em) #text(size: 18pt, label)],
)
#let step(body) = block(
  width: 100%, height: 2.6cm, inset: 10pt, radius: 6pt, fill: soft,
  align(horizon + center, text(size: 17pt, body)),
)
#let arrow = align(horizon + center, text(fill: accent, size: 24pt)[→])

// 1 ─────────────────────────────────────────────────────────────
#page(footer: none)[
  #v(1fr)
  #text(size: 15pt, fill: accent, weight: "bold", upper[Проект курса по эффективным моделям глубокого обучения])
  #v(0.3em)
  #text(size: 40pt, weight: "bold")[Эффективный ранг активаций\ как критерий выбора ширины слоёв]
  #v(0.8em)
  #text(size: 22pt)[#authors]
  #v(1.2fr)
  #text(size: 16pt, fill: ink2)[Защита проекта · декабрь 2026]
]

// 2 ─────────────────────────────────────────────────────────────
#slide(title: [Эффективный ранг активаций и гипотеза об избыточности каналов])[
  *Гипотеза.* Слой с низким эффективным рангом активаций можно сильнее сузить с малой потерей точности.

  *Определение.* Эффективный ранг — экспонента энтропии нормированного спектра матрицы активаций $X$ (Roy & Vetterli, 2007).

  #v(0.6em)
  #grid(columns: (auto, auto, 1fr), row-gutter: 1.1em, column-gutter: 1.2em, align: horizon,
    text(fill: accent, weight: "bold")[ERank-σ],
    $r_sigma = exp(-sum_i p_i log p_i), quad p_i = sigma_i / (sum_j sigma_j)$,
    text(fill: ink2, size: 17pt)[доли сингулярных чисел $sigma_i$],
    text(fill: orange, weight: "bold")[ERank-cov],
    $r_"cov" = exp(-sum_i q_i log q_i), quad q_i = sigma_i^2 / (sum_j sigma_j^2)$,
    text(fill: ink2, size: 17pt)[доли их квадратов\ (спектр ковариации каналов)],
  )
]

// 3 ─────────────────────────────────────────────────────────────
#slide(title: [Процедура выбора ширины слоёв и отбора каналов])[
  #v(0.3em)
  #grid(columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr), gutter: 0.25cm,
    step[Обучить модель], arrow,
    step[Собрать активации], arrow,
    step[Рассчитать ширины слоёв], arrow,
    step[Отобрать каналы], arrow,
    step[Обучить сжатую модель],
  )
  #v(0.9em)
  #table(columns: (auto, 1fr), inset: (x: 8pt, y: 10pt),
    table.hline(stroke: 0.8pt + rule),
    [*Выбор ширины слоёв*], [ERank-σ, ERank-cov, Uniform, Random],
    table.hline(stroke: 0.5pt + rule),
    [*Отбор каналов*], [QR, случайный отбор, L1-норма, Taylor],
    table.hline(stroke: 0.5pt + rule),
    [*Восстановление и обучение*], [дообучение, пересчёт статистик BatchNorm, линейная компенсация, обучение с нуля],
    table.hline(stroke: 0.8pt + rule),
  )
]

// 4 ─────────────────────────────────────────────────────────────
#slide(title: [Архитектуры моделей и наборы данных])[
  #v(0.8em)
  #grid(columns: (1fr, 1fr), gutter: 0.8cm,
    card[
      #text(size: 17pt, fill: ink2)[Классификация]\
      #text(size: 26pt, weight: "bold")[ResNet-20 и Mini-ViT]\
      #text(size: 22pt)[CIFAR-10 и CIFAR-100]
    ],
    card(stroke: orange)[
      #text(size: 17pt, fill: ink2)[Сегментация]\
      #text(size: 26pt, weight: "bold")[SegFormer-B1]\
      #text(size: 22pt)[ADE20K]
    ],
  )
  #v(1em)
  Заранее заданы три уровня сокращения числа операций за проход — *10, 25 и 40%* (ADE20K — 10 и 20%).

  Методы для сравнения: Uniform, Random, L1, Taylor, HRank, Network Slimming, SVD весов.

  Каждая конфигурация — три запуска с разными seed; всего 630 обучений на NVIDIA A100.
]

// 5 ─────────────────────────────────────────────────────────────
#slide(title: [Точность классификации при сокращении числа операций на 25%])[
  #align(center, image("figures/methods_flops25.png", width: 100%, height: 11.6cm, fit: "contain"))
  #caption[100% — точность исходной модели. Точки — средние трёх запусков, отрезки — разброс между запусками.]
]

// 6 ─────────────────────────────────────────────────────────────
#slide(title: [Динамика эффективного ранга активаций в процессе обучения])[
  #align(center, image("figures/rank_dynamics.png", width: 100%, height: 10.2cm, fit: "contain"))
  #v(0.2em)
  #set text(size: 18pt)
  - К концу обучения ERank-σ составляет около 94% ширины у ResNet-20 и 62–74% у Mini-ViT.
  - У ResNet-20 ширины по ERank близки к равномерным; у Mini-ViT ранг ниже, но подтверждённого выигрыша в точности тоже нет.
]

// 7 ─────────────────────────────────────────────────────────────
#slide(title: [Зависимость эффективного ранга от масштаба активаций каналов])[
  *Опыт.* Выход канала умножаем на положительный множитель, а соответствующие веса следующего слоя делим на тот же множитель — например, на 2. Вклад канала в выход сети при этом не меняется.

  #v(0.8em)
  #grid(columns: (1fr, 1fr), gutter: 1cm,
    stat[≤ 0,00001][изменение выхода сети (logits)],
    stat(color: orange)[до 75%][изменение ERank-cov\ (ERank-σ — до 48%)],
  )
  #v(0.6em)
  #caption[ResNet-20 и Mini-ViT на CIFAR-10 и CIFAR-100.]
]

// 8 ─────────────────────────────────────────────────────────────
#slide(title: [Влияние способа отбора каналов на точность до и после дообучения])[
  #align(center, image("figures/immediate_vs_finetuned.png", width: 100%, height: 11.4cm, fit: "contain"))
  #caption[ResNet-20, CIFAR-10, ширины слоёв по ERank-σ, −25% операций; среднее трёх запусков.]
]

// 9 ─────────────────────────────────────────────────────────────
#slide(title: [Обучение сжатых моделей с нуля и итеративное сужение слоёв])[
  #grid(columns: (1fr, 1fr), gutter: 0.9cm,
    card[
      *Обучение с нуля · Mini-ViT, CIFAR-100*\
      #text(size: 17pt, fill: ink2)[75% исходного числа операций]
      #v(0.2em)
      #set text(size: 18pt)
      #table(columns: (1fr, auto), inset: (x: 4pt, y: 6pt),
        [Исходная модель], [58,56%],
        table.hline(stroke: 0.5pt + rule),
        [Равномерное сужение, 150 эпох с нуля], [*58,98%*],
        [ERank-σ, 150 эпох с нуля], [*58,87%*],
        [ERank-σ, 20 эпох дообучения], [57,96%],
      )
    ],
    card(stroke: orange)[
      *Итеративное сжатие · Mini-ViT, CIFAR-10*\
      #text(size: 17pt, fill: ink2)[Раунд: заново оценить ранг → сузить слои → дообучить 20 эпох]
      #v(0.2em)
      #set text(size: 18pt)
      #table(columns: (1fr, auto, auto), inset: (x: 4pt, y: 6pt), align: (left, right, right),
        [Раунд], [Операций], [Точность],
        table.hline(stroke: 0.5pt + rule),
        [1], [78,2%], [99,6%],
        [2], [69,2%], [99,4%],
        [3], [*63,4%*], [*99,1%*],
      )
      #text(size: 15pt, fill: ink2)[В % от исходной модели]
    ],
  )
  #v(0.3em)
  #caption[Точность — среднее трёх запусков.]
]

// 10 ────────────────────────────────────────────────────────────
#slide(title: [Пропускная способность сжатых моделей на NVIDIA A100])[
  #align(center, image("figures/flops_vs_speed.png", width: 100%, height: 11.3cm, fit: "contain"))
  #caption[0 — скорость исходной модели; слева — медленнее, справа — быстрее. CIFAR-100 · A100 · BF16 · 128 изображений в пакете.]
]

// 11 ────────────────────────────────────────────────────────────
#slide(title: [Качество сегментации после сжатия SegFormer-B1 на ADE20K])[
  #align(center, image("figures/ade20k_quality.png", width: 100%, height: 10.4cm, fit: "contain"))
  #v(0.1em)
  #set text(size: 16.5pt)
  #grid(columns: (1fr, 1fr), gutter: 0.8cm,
    [*Low / random / high* — по 512 изображений с низким эффективным рангом активаций модели, случайных и с высоким; по ним рассчитывались ширины слоёв.],
    [Исходная модель — mIoU *40,94*. Дообучение каждого варианта — 1 500 шагов, среднее трёх запусков.],
  )
]

// 12 ────────────────────────────────────────────────────────────
#slide(title: [Основные результаты и ограничения исследования])[
  #set text(size: 19pt)
  + На CIFAR при одинаковом числе операций преимущество выбора ширин по ERank над равномерным сужением *не подтверждено*: 0 из 24 сравнений.
  + Ни в одной из 12 постановок на CIFAR и ни на одном уровне сжатия ADE20K ERank не дал наибольшей средней точности. Лучшими были Network Slimming, SVD весов, Uniform и Taylor.
  + Ранг зависит от масштаба активаций: при неизменном выходе сети ERank-cov меняется до 75%.
  + Сокращение числа операций не равно ускорению: при −25% операций ResNet-20 с ERank-cov обрабатывает на 14,7% меньше изображений в секунду.
  #v(0.2em)
  #text(size: 16pt, fill: ink2)[Ограничения: небольшие модели, три запуска на конфигурацию, на ADE20K — короткое дообучение (0,68 эпохи).]
  #v(1fr)
  #line(length: 100%, stroke: 0.8pt + rule)
  #grid(columns: (1fr, auto),
    text(size: 17pt)[Код, данные и отчёт: #link(repo)[github.com/lm-cyber/efficient_rank]],
    text(size: 17pt, weight: "bold", fill: accent)[Спасибо!],
  )
]

// Запасные слайды ───────────────────────────────────────────────
#slide(title: [Запасной слайд: выбор ширин по ERank и равномерное сужение])[
  #align(center, image("figures/erank_vs_uniform.png", width: 100%, height: 11.2cm, fit: "contain"))
  #caption[Каналы в обоих случаях отбирает QR — проверяется только выбор ширин. Ни в одном из 24 парных сравнений преимущество ERank не подтверждено после поправки Бонферрони.]
]

#slide(title: [Запасной слайд: связь ранга с чувствительностью слоёв])[
  Из каждого блока отдельно удаляли 25% или 50% каналов и измеряли рост loss без дообучения. Затем считали корреляцию Спирмена между нормированным рангом блока и этим ростом.
  #v(0.5em)
  #align(center, table(columns: (auto, auto), inset: (x: 18pt, y: 9pt), align: (left, right),
    [*Модель и данные*], [*Spearman, диапазон*],
    table.hline(stroke: 0.8pt + rule),
    [ResNet-20, CIFAR-10 (отдельный пул v1)], [0,00 … +0,28],
    [ResNet-20, CIFAR-10], [−0,07 … +0,21],
    [ResNet-20, CIFAR-100], [−0,12 … +0,15],
    [Mini-ViT, CIFAR-10], [−0,35 … −0,20],
    [Mini-ViT, CIFAR-100], [−0,39 … +0,50],
    table.hline(stroke: 0.8pt + rule),
  ))
  #caption[Диапазон по долям удаления (25%, 50%) и двум определениям ранга. Строки v2 — средние трёх запусков, строка v1 — значения отдельных запусков.]
]

#slide(title: [Запасной слайд: статистический анализ результатов])[
  #set text(size: 17pt)
  #grid(columns: (1fr, 1fr), gutter: 1cm,
    [
      *Разбиение данных*
      - CIFAR: 45 000 train / 5 000 validation; активации для ERank — подвыборка train без аугментаций
      - Официальный test оценивали один раз, после всех обучений
      - ADE20K: 2 048 изображений для расчёта ширин, 512 для выбора checkpoint, 17 650 для дообучения; итоговая оценка — 2 000 validation

      *Число операций*
      - Учитываются Conv, Linear и умножения матриц во внимании
      - Внутри уровня сжатия у всех методов одинаковое число операций (у SVD отклонение до 0,34 п.п.)
    ],
    [
      *Запуски и поправка*
      - Три независимых обучения исходной модели (seed 42, 123, 2026); разницы считаются парно внутри seed
      - 24 сравнения ERank − Uniform: t-интервалы с поправкой Бонферрони
      - Сравнения со всеми остальными методами — описательные

      *Замер скорости*
      - CUDA events, 30 прогревов + 100 повторов, 5 перемешанных серий; вход уже на GPU
      - Отдельно проверены режимы eager и CUDA Graphs
    ],
  )
]

#slide(title: [Запасной слайд: реализация методов сжатия])[
  #set text(size: 17pt)
  #table(
    columns: (auto, 1fr),
    inset: (x: 8pt, y: 8pt),
    [*Метод*], [*Реализация*],
    table.hline(stroke: 0.8pt + rule),
    [Uniform / Random], [Одинаковая доля сужения всех слоёв / случайное распределение; каналы отбирает тот же QR, что у ERank],
    [L1, Taylor, HRank], [Ширины как у Uniform; каналы по L1-норме фильтров, по оценке Тейлора первого порядка или по рангу карт признаков],
    [Network Slimming], [Отдельное обучение со штрафом L1 на масштабах BatchNorm (в ViT — на множителях после GELU), затем удаление каналов],
    [SVD весов], [Слои заменяются двумя низкоранговыми; ранги выбираются жадно по потере энергии весов на единицу сэкономленных операций],
    [HRank в ViT], [Канал MLP на 64 токенах рассматривается как карта 8×8 — это адаптация метода для ViT],
    table.hline(stroke: 0.8pt + rule),
  )
]
