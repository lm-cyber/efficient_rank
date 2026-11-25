// Презентация к защите проекта. Сборка из корня репозитория:
//   typst compile docs/presentation.typ docs/presentation.pdf

#let authors = "Алан Насибуллин, Герман Еделькин"
#let repo = "https://github.com/lm-cyber/efficient_rank"
#let short-title = "ERank как критерий ширины слоёв"

#let accent = rgb("#257de0")
#let orange = rgb("#f47230")
#let red = rgb("#d93b3c")
#let ink = rgb("#191919")
#let ink2 = rgb("#5c5c5c")
#let soft = rgb("#f3f6fa")
#let rule = rgb("#d9dee5")

#set document(title: "Эффективный ранг активаций как критерий выбора ширины слоёв", author: authors)
#set text(font: "PT Sans", size: 19pt, fill: ink, lang: "ru")
#set par(leading: 0.55em, justify: false)
#set list(marker: text(fill: accent)[•], spacing: 0.55em, indent: 0.2em)
#set table(stroke: none)
#show math.equation: set text(size: 17pt)
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
    block(below: 0.25em, text(size: 28pt, weight: "bold", title))
    line(length: 100%, stroke: 1.5pt + accent)
    v(0.35em)
  }
  body
}

#let muted(body) = text(fill: ink2, size: 14pt, body)
#let callout(body, fill: soft, stroke: accent) = block(
  width: 100%, inset: (x: 14pt, y: 11pt), radius: 6pt, fill: fill,
  stroke: (left: 4pt + stroke), body,
)
#let stat(value, label, color: accent) = block(
  width: 100%, inset: 14pt, radius: 6pt, fill: soft,
  [#text(size: 40pt, weight: "bold", fill: color, value) \ #v(-0.2em) #text(size: 16pt, label)],
)
#let step(n, body) = block(
  width: 100%, height: 3.3cm, inset: 10pt, radius: 6pt, fill: soft,
  [#text(fill: accent, weight: "bold", size: 15pt)[#n] \ #text(size: 15.5pt, body)],
)
#let arrow = align(horizon + center, text(fill: accent, size: 22pt)[→])

// 1 ─────────────────────────────────────────────────────────────
#page(footer: none)[
  #v(1.5cm)
  #text(size: 15pt, fill: accent, weight: "bold", upper[Проект курса по эффективным моделям глубокого обучения])
  #v(0.2em)
  #text(size: 38pt, weight: "bold")[Эффективный ранг активаций\ как критерий выбора ширины слоёв]
  #v(0.4em)
  #text(size: 21pt, fill: ink2)[Помогает ли ERank сжимать сети лучше равномерного сужения и классического прунинга?\ ResNet-20 и Mini-ViT на CIFAR-10/100, SegFormer-B1 на ADE20K]
  #v(1fr)
  #line(length: 100%, stroke: 0.8pt + rule)
  #grid(columns: (1fr, auto), gutter: 1em,
    text(size: 18pt)[*#authors*\ #text(fill: ink2)[Защита проекта · декабрь 2026]],
    align(right, text(size: 16pt)[#link(repo)[github.com/lm-cyber/efficient_rank]]),
  )
]

// 2 ─────────────────────────────────────────────────────────────
#slide(title: [Идея: сузить слой до его «реальной» размерности])[
  #grid(columns: (1.3fr, 1fr), gutter: 1cm,
    [
      *Гипотеза.* Если активации слоя лежат в подпространстве малой размерности, слой можно сузить почти без потери качества.

      *Эффективный ранг* — экспонента энтропии нормированного спектра (Roy & Vetterli, 2007). Для центрированной матрицы активаций $X in RR^(M times C)$ с сингулярными числами $sigma_i$:

      #v(0.2em)
      #grid(columns: (auto, 1fr), row-gutter: 0.7em, column-gutter: 0.6em,
        text(fill: accent, weight: "bold")[ERank-σ], $r_sigma = exp(-sum_i p_i log p_i), quad p_i = sigma_i \/ sum_j sigma_j$,
        text(fill: orange, weight: "bold")[ERank-cov], $r_"cov" = exp(-sum_i q_i log q_i), quad q_i = sigma_i^2 \/ sum_j sigma_j^2$,
      )
      #v(0.2em)
      #muted[ERank-cov — спектр ковариации каналов, как в Smirnov et al. (2026), где ERank использовали как меру сложности изображений. Перенос на выбор ширины сети — гипотеза нашего проекта.]
    ],
    [
      #v(0.6em)
      #callout[
        #text(weight: "bold", size: 20pt)[Главный вопрос]

        Даёт ли распределение ширин по ERank лучшее соотношение *качества и FLOPs*, чем равномерное сужение, случайное распределение и классический структурный прунинг — *при одинаковом бюджете*?
      ]
      #v(0.6em)
      #muted[Цель — проверить гипотезу и найти границы применимости, а не максимизировать accuracy. Отрицательный результат при честном сравнении тоже ответ.]
    ],
  )
]

// 3 ─────────────────────────────────────────────────────────────
#slide(title: [Метод: три отдельных решения вместо одного])[
  #grid(columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr), gutter: 0.25cm,
    step[1][Обучить исходную модель (150 эпох)], arrow,
    step[2][Снять активации на калибровке из train],  arrow,
    step[3][ERank каждого слоя → профиль ширин под бюджет FLOPs], arrow,
    step[4][Выбрать каналы (QR с перестановкой столбцов)], arrow,
    step[5][Физически удалить и дообучить 20 эпох / с нуля],
  )
  #v(0.5em)
  #grid(columns: (1fr, 1fr, 1fr), gutter: 0.6cm,
    callout[*Сколько каналов в слое*\ #text(size: 16pt)[ERank-σ, ERank-cov, Uniform, Random. Ширины масштабируются под общий бюджет MACs.]],
    callout(stroke: orange)[*Какие каналы оставить*\ #text(size: 16pt)[QR, случайные, L1-норма, Taylor. Ранг сам не говорит, какие каналы важны.]],
    callout(stroke: ink2)[*Как восстановить веса*\ #text(size: 16pt)[Fine-tuning, пересчёт BN, линейная компенсация, обучение с нуля.]],
  )
  #v(0.35em)
  Эти три фактора проверяем *по отдельности*: иначе вклад QR или дообучения легко приписать ERank. Сжимаются внутренние каналы блоков (ResNet), скрытый слой MLP (ViT), Mix-FFN (SegFormer).
]

// 4 ─────────────────────────────────────────────────────────────
#slide(title: [Эксперименты: 630 обучений на A100])[
  #grid(columns: (1.25fr, 1fr), gutter: 1cm,
    [
      #set text(size: 16.5pt)
      #table(
        columns: (auto, 1fr, auto),
        inset: (x: 6pt, y: 7pt),
        align: (left, left, left),
        table.header(text(fill: ink2)[Пул], text(fill: ink2)[Модели и данные], text(fill: ink2)[Объём]),
        table.hline(stroke: 0.8pt + rule),
        [*v1*], [ResNet-20 / CIFAR-10], [63 обучения, 2 430 эпох],
        [*v2*], [ResNet-20, Mini-ViT × CIFAR-10/100], [504 обучения, 21 000 эпох],
        [*ADE20K*], [SegFormer-B1 (pretrained), 150 классов, 512×512], [63 fine-tuning],
        [*Аудит*], [72 checkpoint, eager и CUDA Graphs], [432 серии замеров],
        table.hline(stroke: 0.8pt + rule),
      )
      #muted[Mini-ViT: 6 блоков, dim 96, 3 heads, MLP 384, без предобучения. ADE20K заменил запланированный Tiny ImageNet.]
    ],
    [
      - *Бюджеты:* −10 / −25 / −40% FLOPs (ADE20K: −10 / −20%), у всех методов одинаковые MACs
      - *Бейзлайны:* Uniform, Random, L1, Network Slimming, Taylor, HRank, SVD весов
      - *3 seed*, парные сравнения, поправка Бонферрони
      - *Test* открыт только после всех обучений; протокол записан до запуска, поправки задокументированы
      - *Скорость:* A100 80 GB, BF16, batch 1 и 128
    ],
  )
]

// 5 ─────────────────────────────────────────────────────────────
#slide(title: [Главный результат: ERank не лучше равномерного сужения])[
  #grid(columns: (2.05fr, 1fr), gutter: 0.7cm,
    image("figures/erank_vs_uniform.png", width: 100%),
    [
      #v(0.3em)
      #stat[0 из 24][заранее заданных сравнений с Uniform значимы после поправки Бонферрони]
      #v(0.3em)
      - Средние разницы — от *−1,3 до +0,5 п.п.*
      - ERank-cov на Mini-ViT / CIFAR-100 систематически *хуже* Uniform
      - Порог практической пользы (+0,2 п.п.) нигде не достигнут
      #v(0.2em)
      #muted[Отсутствие значимости ≠ эквивалентность: при 3 seed интервалы широкие.]
    ],
  )
]

// 6 ─────────────────────────────────────────────────────────────
#slide(title: [Против классического прунинга: ERank в середине группы])[
  #align(center, image("figures/methods_flops25.png", height: 9.3cm))
  #v(-0.3em)
  #set text(size: 16pt)
  #grid(columns: (1fr, 1fr, 1fr), gutter: 0.6cm,
    [#text(fill: accent, weight: "bold")[ERank] — нигде не лучший; на CIFAR-100 у Mini-ViT ERank-cov последний среди методов прунинга по ширине.],
    [*Network Slimming* — лучший в 3 из 4 постановок, но требует отдельного sparse-обучения.],
    [*SVD весов* ломается на ViT: 82,5% исходного качества на CIFAR-100.],
  )
  #muted[−25% FLOPs; качество = 100 × accuracy / accuracy исходной модели того же seed.]
]

// 7 ─────────────────────────────────────────────────────────────
#slide(title: [Почему ERank почти совпадает с Uniform])[
  #grid(columns: (1.75fr, 1fr), gutter: 0.7cm,
    [
      #image("figures/rank_dynamics.png", width: 100%)
      #muted[Средний нормированный ERank по слоям и 3 seed на фиксированных 256 калибровочных изображениях.]
    ],
    [
      - Обученная ResNet-20 использует *≈94% ширины* по ERank-σ (77% по ERank-cov): «лишних» направлений мало
      - Профиль ERank-σ близок к Uniform: разница *5–11 каналов из 336*
      - ERank-cov даёт другие архитектуры (23–28 каналов разницы), но качество от этого не растёт
      - Mini-ViT использует ширину слабее (σ: 62–74%), но и там выигрыша нет
    ],
  )
]

// 8 ─────────────────────────────────────────────────────────────
#slide(title: [Ранг ≠ «необходимая ширина» слоя])[
  #grid(columns: (1fr, 1fr), gutter: 1cm,
    [
      #stat(color: orange)[до 75%][меняется ERank-cov (ERank-σ — до 48%) при масштабировании каналов, компенсированном в следующем слое]
      #v(0.3em)
      Logits при этом меняются не больше чем на $10^(-5)$: *функция сети та же, а ранг другой*. ERank зависит от масштаба координат, а не только от того, что сеть вычисляет.
    ],
    [
      #stat[ρ ≈ 0][связь нормированного ранга с чувствительностью блока к удалению каналов]
      #v(0.3em)
      Spearman по блокам: *0,00…0,28* в v1 и *−0,39…+0,50* в v2, знак меняется между постановками. Низкий ранг не означает, что блок можно сузить безнаказанно.
    ],
  )
  #v(0.4em)
  #callout[Ранг описывает геометрию представления, а не вклад направлений в loss. Поэтому как критерий ширины он слабее методов, которые смотрят на loss или выход следующего слоя.]
]

// 9 ─────────────────────────────────────────────────────────────
#slide(title: [Выбор каналов важен сразу после удаления — и почти не важен после дообучения])[
  #grid(columns: (1.75fr, 1fr), gutter: 0.7cm,
    image("figures/immediate_vs_finetuned.png", width: 100%),
    [
      #v(0.4em)
      - Линейная компенсация весов: *+24,6 п.п.* сразу после удаления
      - Случайные каналы вместо QR: *−24,6 п.п.*
      - После 20 эпох fine-tuning разница *≤ 0,05 п.п.*
      - В v2 ширины ERank с отбором каналов по L1 или Taylor дают близкий результат (разница в пределах 0,7%)
      #v(0.3em)
      #muted[ResNet-20 / CIFAR-10, ширины ERank-σ, −25% FLOPs, 3 seed (v1).]
    ],
  )
  #callout[Качество сразу после прунинга — плохой прокси итогового качества. Методы нужно сравнивать после одинакового восстановления.]
]

// 10 ────────────────────────────────────────────────────────────
#slide(title: [Обучение с нуля, итерации и архитектура ViT])[
  #grid(columns: (1fr, 1fr, 1fr), gutter: 0.7cm,
    callout[
      *С нуля, 150 эпох*\
      #set text(size: 16pt)
      Архитектура ERank-σ не лучше Uniform ни в одной из 4 постановок v2. В v1 средний плюс +0,11 п.п. дал один seed из трёх.

      Для ERank нужна ещё и обученная исходная сеть, поэтому полные затраты выше.
    ],
    callout(stroke: orange)[
      *Итеративный ceil(ERank), 3 раунда*\
      #set text(size: 16pt)
      ResNet-20 почти не сжимается: ≈ −10% FLOPs, качество 99,4–99,7%.

      Mini-ViT / CIFAR-10: −37% FLOPs при 99,1% качества.

      Парного итеративного контроля нет: это траектория, а не доказательство оптимальности.
    ],
    callout(stroke: ink2)[
      *Форма ViT при −25% FLOPs*\
      #set text(size: 16pt)
      Heads + MLP по ERank: 99,7% / 100,5% качества (CIFAR-10/100).

      4 блока × dim 96: качество 98,9–99,4%, а инференс *на 44% быстрее*.

      Глубина влияет на скорость сильнее любой ширины.
    ],
  )
  #v(0.2em)
  #muted[Все числа — средние 3 seed, качество в % от исходной модели того же seed.]
]

// 11 ────────────────────────────────────────────────────────────
#slide(title: [Меньше FLOPs ≠ быстрее])[
  #align(center, image("figures/flops_vs_speed.png", height: 9.2cm))
  #v(-0.3em)
  #set text(size: 16.5pt)
  #grid(columns: (1fr, 1fr), gutter: 0.9cm,
    [
      - Сжатая ResNet-20 *медленнее* исходной во всех 10 режимах аудита: eager и CUDA Graphs, с автотюнингом cuDNN и без, batch 1 и 128
      - Неровные ширины ERank замедляют сильнее Uniform
    ],
    [
      - У Mini-ViT сужение MLP почти не меняет скорость, а меньше блоков — +44%
      - #text(fill: ink2)[Гипотеза, не проверена: ширины не кратны 8 и плохо ложатся на тензорные ядра. На ADE20K ширины кратны 8, и ускорение есть.]
    ],
  )
]

// 12 ────────────────────────────────────────────────────────────
#slide(title: [Сложная задача: сегментация ADE20K, SegFormer-B1])[
  #grid(columns: (1.75fr, 1fr), gutter: 0.7cm,
    [
      #image("figures/ade20k_quality.png", width: 100%)
      #muted[Исходный mIoU 40,94%. Сжимаются 8 блоков Mix-FFN; ширины кратны 8; 1 500 шагов fine-tuning.]
    ],
    [
      - *Taylor* — лучший: 95,0% и 66,5% исходного mIoU
      - ERank-σ ≈ Uniform на −10%, *хуже* на −20% (−2,6%, интервал не включает 0); ERank-cov хуже на 5–6%
      - Калибровка на изображениях низкого ранга *не помогает*: 0 из 4 сравнений с random
      - Ускорение при −20% FLOPs: *+6–8%* в CUDA Graphs и eager с batch 8; ≈ 0 в eager с batch 1
      #v(0.2em)
      #muted[Ограничение: 0,68 эпохи дообучения, у 51 из 63 запусков лучший checkpoint — последний. Восстановление не завершено.]
    ],
  )
]

// 13 ────────────────────────────────────────────────────────────
#slide(title: [Что не работает — и что делать вместо этого])[
  #set text(size: 17pt)
  #table(
    columns: (1fr, 1fr),
    inset: (x: 10pt, y: 8.5pt),
    table.header(
      text(fill: red, weight: "bold", size: 19pt)[Не стоит],
      text(fill: accent, weight: "bold", size: 19pt)[Лучше],
    ),
    table.hline(stroke: 0.8pt + rule),
    [Считать ERank «необходимой шириной» слоя], [Использовать критерии, учитывающие loss: Taylor, Network Slimming],
    table.hline(stroke: 0.5pt + rule),
    [Сравнивать методы по качеству сразу после удаления каналов], [Сравнивать после одинакового восстановления и одинакового бюджета MACs],
    table.hline(stroke: 0.5pt + rule),
    [Считать, что −25% FLOPs = −25% времени], [Мерить latency на целевом железе; уменьшать глубину и heads; выравнивать ширины кратно 8],
    table.hline(stroke: 0.5pt + rule),
    [Подбирать «простые» калибровочные изображения ради сильного сжатия], [Фиксировать калибровку заранее: low-ранговая выборка качество не меняет],
    table.hline(stroke: 0.5pt + rule),
    [Делать выводы по одному seed и десяткам масок одной сети], [Брать независимые seed, парные сравнения и открывать test только в конце],
  )
]

// 14 ────────────────────────────────────────────────────────────
#slide(title: [Итоги])[
  #grid(columns: (1.35fr, 1fr), gutter: 1cm,
    [
      + Распределение ширин по ERank *не превосходит* равномерное сужение: 0 из 24 значимых сравнений на CIFAR; на ADE20K — хуже Taylor и Uniform.
      + ERank зависит от параметризации и слабо связан с чувствительностью слоёв, поэтому не задаёт «реальную» ширину.
      + Реальное ускорение определяет форма вычислений, а не число FLOPs.
      #v(0.3em)
      #muted[Границы: небольшие модели, 3 seed, короткое восстановление на ADE20K. Отсутствие преимущества не доказывает эквивалентность методов.]
    ],
    [
      #callout[
        *Всё воспроизводимо*\
        #set text(size: 16pt)
        Код, протоколы, split, логи, предсказания, CSV и графики:\
        #link(repo)[github.com/lm-cyber/efficient_rank]

        Отчёт: `docs/report.pdf`
      ]
      #v(1fr)
      #align(center, text(size: 34pt, weight: "bold", fill: accent)[Спасибо! Вопросы?])
      #v(0.6fr)
    ],
  )
]

// Запасные слайды ───────────────────────────────────────────────
#slide(title: [Запасной слайд: честность сравнения])[
  #set text(size: 16.5pt)
  #grid(columns: (1fr, 1fr), gutter: 1cm,
    [
      *Данные*
      - CIFAR: 45 000 train / 5 000 validation; калибровка ERank — подвыборка train без аугментаций
      - Официальный test открыт один раз, после всех обучений; манифест test заморожен
      - ADE20K: 2 048 калибровка, 512 выбор checkpoint, 17 650 fine-tuning; 2 000 validation — итоговая оценка

      *Бюджеты*
      - Все методы внутри бюджета имеют одинаковые MACs: Conv, Linear и matmul внимания
      - SVD весов отклоняется от бюджета не более чем на 0,34 п.п. FLOPs из-за дискретных рангов
    ],
    [
      *Статистика*
      - 3 независимых исходных обучения (seed 42, 123, 2026); разницы считаются внутри seed
      - 24 первичных сравнения ERank − Uniform с поправкой Бонферрони; остальное — описательно
      - Сравнения со всеми бейзлайнами добавлены после результатов и отмечены как описательные

      *Скорость*
      - CUDA events, 30 warmup + 100 повторов, 5 перемешанных серий; вход уже на GPU
      - Отдельный аудит: eager / CUDA Graphs, cuDNN deterministic / autotune
    ],
  )
]

#slide(title: [Запасной слайд: адаптации методов])[
  #set text(size: 16.5pt)
  #table(
    columns: (auto, 1fr),
    inset: (x: 8pt, y: 7pt),
    table.header(text(fill: ink2)[Метод], text(fill: ink2)[Как реализован]),
    table.hline(stroke: 0.8pt + rule),
    [Uniform / Random], [Одна доля сужения на все слои / случайный профиль ширин; каналы выбираются тем же QR, что у ERank],
    [L1, Taylor, HRank], [Ширины Uniform, каналы по L1-норме фильтров, первому порядку Тейлора или рангу карт признаков],
    [Network Slimming], [Отдельное обучение со штрафом L1 на масштабах BN (в ViT — на гейтах MLP), затем прунинг; контроль Sparse → Uniform],
    [SVD весов], [Жадное снижение ранга Conv/Linear по отношению потери энергии весов к сэкономленным MACs],
    [HRank в ViT], [Канал MLP на 64 токенах превращается в карту 8×8 — это адаптация, а не исходный CNN-метод],
    table.hline(stroke: 0.8pt + rule),
  )
  #muted[Подробные протоколы: experiment_pool_v1.md, experiment_pool_v2.md, experiment_pool_segmentation.md.]
]
