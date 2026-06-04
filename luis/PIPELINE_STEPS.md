# DSML Group 11 — Part 2: Predictive Pipeline (passo a passo)

Documento de apoio ao notebook `DSML_Group11_Predictive.ipynb`.
Descreve, por ordem, todos os passos desde a limpeza de valores impossíveis até à
geração da submissão Kaggle. Serve de guião para a defesa do projeto.

> **Métrica primária:** F1-Score (binário).
> **Estratégia de avaliação:** 5-fold Stratified Cross-Validation.
> **Princípio central:** tudo o que usa estatísticas dos dados (imputação,
> winsorização, escala) é ajustado **dentro** de cada fold de CV → sem *data leakage*.

---

## Visão geral do fluxo

```
Dados (3 ficheiros)
   │
   ├─ donors_descriptive.csv  → features de treino   (13 560 linhas)
   ├─ donors_train_target.csv → target (TARGET_B)    (13 560 linhas)
   └─ test.csv                → features de teste     (5 812 linhas)
   │
   ▼
1. Loading + join (por CONTROL_NUMBER, que é índice)
2. Limpeza determinística (clean_before_split, do Keni)  ← antes do split, SEM leakage
3. EDA (missing, classe, skewness, outliers, correlações)
4. Feature engineering (2 novas features mantidas)
5. Feature selection (11 numéricas + 3 categóricas, |corr| ≥ 0.08)
6. Pipeline de preprocessing coluna-a-coluna (dentro do CV)
7. Baseline de 3 modelos (KNN, DT, MLP)
8. Tuning (RandomizedSearchCV, 5-fold)
9. CV final + seleção de modelo
10. Otimização do threshold (out-of-fold)
11. Retreino + previsão + submissão
12. Open-ended: importância de features + análise de erros
```

---

## 1. Loading dos dados

- Caminhos definidos para **Colab** (Google Drive) e **local** (pastas irmãs do repo).
- Os 3 CSV são lidos com `index_col='CONTROL_NUMBER'` → o ID passa a ser **índice**, não coluna.
- `df_train = df_features.join(target)` faz o merge pelo índice (à la David, sem `merge` explícito).

| Ficheiro | Papel | Linhas |
|----------|-------|--------|
| `Descriptive/donors_descriptive.csv` | features de treino | 13 560 |
| `Predictive/donors_train_target.csv` | target | 13 560 |
| `Predictive/test.csv` | features de teste | 5 812 |

---

## 2. Limpeza de valores impossíveis — `clean_before_split` (Keni)

Função em `keni_utils.py`. É a **divisão da função original do Keni**: ficou só a
parte **determinística** (não calcula estatísticas dos dados), portanto é seguro
aplicá-la ao dataset inteiro **antes** do split — não há leakage.

> A imputação por mediana/moda/grupo (que *usa* estatísticas) foi **deliberadamente
> removida** desta função e empurrada para o pipeline de CV (passo 6).

O que `clean_before_split` faz, coluna a coluna:

### Categóricas (correções determinísticas)
- `DONOR_GENDER` → `fillna('U')` + uppercase.
- `HOME_OWNER` → `fillna('U')` + mapeia `{'H':1, 'U':0}` (binário).
- `RECENCY_STATUS_96NK` → `fillna('U')` + uppercase.
- `SES` → `replace('?', NaN)` + `float` (NaN fica para imputar no pipeline).
- `URBANICITY` → `replace('?', None)` + `fillna('Unknown')` + uppercase.

### Numéricas com clipping de valores impossíveis + flag de missingness
- `PEP_STAR` → clip a [0,1], flag `PEP_STAR_IS_MISSING`, binariza (>0.5 → 1).
- `FREQUENCY_STATUS_97NK` → clip a [1,4], flag, NaN→-1.
- `CHILDREN` → clip a [0,4], flag, NaN→0.
- `INCOME_GROUP` → clip a [1,7], fora-do-range/NaN → -1.
- `WEALTH_RATING` → clip a [0,9], negativos/NaN → **-1** (importante: missing fica como -1, não NaN).
- `RECENT_CARD_RESPONSE_COUNT` → arredonda, negativos→0, flag.
- `RECENT_RESPONSE_COUNT` → arredonda, negativos→0, flag.
- `CARD_PROM_12` → arredonda, NaN→0, flag.
- `RECENT_STAR_STATUS` → binariza.
- `DONOR_AGE` → só cria a flag `DONOR_AGE_IS_MISSING` (imputação fica para o pipeline).

### Grupos de colunas (apenas clipping, **sem** imputação)
- **Discretas** (`MONTHS_SINCE_LAST_GIFT`, `FILE_CARD_GIFT`, `NUMBER_PROM_12`, …)
  → `clip(lower=0)` (não impute aqui).
- **Proporções/percentagens** (`RECENT_RESPONSE_PROP`, `PCT_ATTRIBUTE1-4`, …)
  → clip a [0,1] se for proporção, ou [0,100] se for percentagem.
- **Monetárias** (`LAST_GIFT_AMT`, `LIFETIME_GIFT_AMOUNT`, `MEDIAN_HOME_VALUE`, …)
  → `clip(lower=0.01)` (uma doação não pode ser ≤ 0).

**Resultado:** valores impossíveis tratados; restam ~10 700 NaN que serão imputados
**dentro do CV** (sem leakage). Flags `*_IS_MISSING` ficam disponíveis como features.

---

## 3. EDA — Exploratory Data Analysis

Cada análise leva a uma decisão de preprocessing:

| Análise | Resultado | Decisão |
|---------|-----------|---------|
| **Missing values** | `DONOR_AGE` 26%, `SES` 4%, restantes ~2% | imputação no pipeline |
| **Distribuição da target** | 75% não-doadores / 25% doadores (3:1) | F1 (não accuracy), `class_weight='balanced'` |
| **Skewness** | muitas features com \|skew\|>1 (gift amounts, counts) | `log1p` nessas features |
| **Outliers (IQR)** | financeiras com muitos outliers | Winsorização (1%/99%) |
| **Correlação com TARGET_B** | a mais forte é `RECENCY_X_FREQ` (~0.14) | guia a feature selection |
| **Categóricas vs target** | diferenças por género/urbanicity | manter para one-hot |

---

## 4. Feature Engineering — `engineer_features`

Foram experimentadas 6 features engineered, mas **só se mantêm as que têm
\|corr\| ≥ 0.08** com TARGET_B (o mesmo critério aplicado às originais). Sobrevivem 2:

| Feature | Fórmula | Racional | \|corr\| | Estado |
|---------|---------|----------|--------|--------|
| `RECENCY_X_FREQ` | `FREQUENCY_STATUS_97NK / (MONTHS_SINCE_LAST_GIFT + 1)` | doador ativo = dá com frequência **e** recentemente | **0.14** (a melhor) | ✅ mantida |
| `CARD_RESP_RATE` | `RECENT_CARD_RESPONSE_COUNT / (CARD_PROM_12 + 1)` | taxa de resposta a cartões | 0.08 | ✅ mantida |
| `LIFETIME_VALUE_TIER` | `LIFETIME_GIFT_AMOUNT * FREQUENCY_STATUS_97NK` | valor total ponderado pela frequência | 0.078 | ❌ descartada |
| `RECENT_AVG_GIFT_AMT` (orig) | — | valor médio recente | 0.077 | ❌ descartada |
| `IS_RECENT_DONOR` | `MONTHS_SINCE_LAST_GIFT <= 12` | deu no último ano | 0.067 | ❌ descartada |
| `RESPONSE_RATE` | `RECENT_RESPONSE_COUNT / (NUMBER_PROM_12 + 1)` | taxa de resposta a promoções | 0.063 | ❌ descartada |
| `WEALTH_KNOWN` | `WEALTH_RATING != -1` | missingness de riqueza | 0.027 | ❌ descartada |

> **Nota:** removeram-se as features abaixo de 0.08 — testou-se que o F1 em CV **não muda**
> (DT ~0.40, MLP ~0.41), confirmando que eram ruído.

Aplicado a `df_train` **e** `df_test` (mesma transformação).

---

## 5. Feature Selection

Mantêm-se **apenas as features com \|corr\| ≥ 0.08** com TARGET_B.

- **11 numéricas** (ordenadas por \|corr\|):

  | # | Feature | \|corr\| |
  |---|---------|--------|
  | 1 | `RECENCY_X_FREQ` (engineered) | 0.144 |
  | 2 | `FREQUENCY_STATUS_97NK` | 0.123 |
  | 3 | `RECENT_RESPONSE_COUNT` | 0.121 |
  | 4 | `RECENT_CARD_RESPONSE_COUNT` | 0.121 |
  | 5 | `RECENT_RESPONSE_PROP` | 0.119 |
  | 6 | `LIFETIME_GIFT_COUNT` | 0.100 |
  | 7 | `PEP_STAR` | 0.100 |
  | 8 | `FILE_CARD_GIFT` | 0.099 |
  | 9 | `RECENT_CARD_RESPONSE_PROP` | 0.098 |
  | 10 | `MONTHS_SINCE_LAST_GIFT` | 0.089 |
  | 11 | `CARD_RESP_RATE` (engineered) | 0.083 |

- **3 categóricas**: `DONOR_GENDER`, `RECENCY_STATUS_96NK`, `URBANICITY`
  (mantidas para one-hot; a correlação de Pearson não se aplica a nominais).
- **Descartadas** (< 0.08, ruído): todas as gift amounts/lifetime/demográficas de baixo sinal
  (`LAST_GIFT_AMT`, `LIFETIME_GIFT_AMOUNT`, `DONOR_AGE`, `SES`, `WEALTH_RATING`, `MEDIAN_HOME_VALUE`,
  `NUMBER_PROM_12`, `MONTHS_SINCE_FIRST_GIFT`, etc.), `PCT_ATTRIBUTE1-4`, `RECENT_STAR_STATUS`,
  e as flags `*_IS_MISSING`.
- **Total: 14 features (11 num + 3 cat).**

---

## 6. Pipeline de Preprocessing — coluna a coluna

Paradigma escolhido: **um transformer por coluna** (em vez de grupos), para ser
trivial ajustar qualquer coluna individualmente.

```
numérica  : median impute → [log1p se skewed] → winsorize(1%,99%) → MinMax scale
categórica: most-frequent impute → OneHotEncoder(drop='first')
```

- `Log1pTransformer` — custom, aplica `log1p(clip(x,0,None))`.
- `WinsorizationTransformer` — custom, percentis ajustados **só no treino** (`fit`).
- `num_pipe(log1p=True/False)` — fábrica de pipeline numérico; o flag liga/desliga o log1p.
- Tudo montado num `ColumnTransformer` (1 transformer por coluna).
- **14 transformers no total** (11 numéricos + 3 categóricos).

> **Por que dentro do CV?** O `ColumnTransformer` é colocado dentro de um `Pipeline`
> com o modelo. No loop de CV, `pipe.fit(treino)` ajusta imputers/winsor/scaler
> **apenas no fold de treino** → o fold de validação nunca "vê" estatísticas dele.

---

## 7. Baseline — 3 modelos

Três algoritmos vistos nas aulas:

| Modelo | Configuração baseline |
|--------|----------------------|
| **KNN** | `n_neighbors=7, weights='distance'` |
| **Decision Tree** | `max_depth=5, class_weight='balanced'` |
| **MLP** | `(100,), activation='tanh', early_stopping` |

- **LogReg removido**: na versão com 4 modelos, o LogReg não superava o Decision Tree
  nem o MLP (ficava empatado a ~0.40) e adicionava complexidade — foi retirado.
- Avaliação: 5-fold CV manual, F1 calculado ao **THRESHOLD=0.25** (não 0.5, por causa do desbalanceamento).
- Reporta Val F1, Train F1, e *overfit gap*.

**Resultados baseline (Val F1):** DT ≈ 0.40, MLP ≈ 0.39, KNN ≈ 0.35.

> **Nota guideline:** o guideline sugere ≥ 4 algoritmos. Aqui ficámos com 3 por decisão
> de simplicidade. Se for necessário cumprir o mínimo, reintroduzir o LogReg (o código
> está documentado no histórico) ou adicionar outro classificador *vanilla* sklearn.

---

## 8. Hyperparameter Tuning — `RandomizedSearchCV`

- `n_iter=20`, **5-fold CV completa** (fiável), scoring = F1 @ THRESHOLD.
- Grids por modelo (KNN: vizinhos/métrica; DT: profundidade/splits/critério;
  MLP: arquitetura/alpha/lr; LogReg: C/penalty L1-L2).
- Guarda `tuned_best_params` por modelo + top-10 configurações globais.

---

## 9. CV Final + Seleção de Modelo

- Re-avalia cada modelo com os melhores params em **5-fold CV completa**.
- **Critério de seleção:** maior Val F1; desempate pelo menor *overfit gap*.
- Resultado típico: **Decision Tree / LogReg** empatados a ~0.40.

---

## 10. Otimização do Threshold — out-of-fold (sem leakage)

- `cross_val_predict` gera probabilidades **out-of-fold** (cada amostra pontuada por
  um modelo que nunca a viu) → escolha de threshold honesta.
- `precision_recall_curve` → escolhe o threshold que **maximiza F1** (`FINAL_THRESHOLD`).
- Mais fiável que um único split 80/20.

> ⚠️ **Aviso:** maximizar F1 num dataset 25% positivo empurra o threshold para
> **alto recall** → a taxa de doadores prevista no test fica acima dos 25% reais.
> Isto é esperado para otimização de F1, mas convém confirmar no Kaggle. Se o score
> público for baixo, experimentar `FINAL_THRESHOLD = 0.5`.

---

## 11. Deployment — Retreino, Previsão e Submissão

1. `final_pipeline.fit(X, y)` — retreina no **dataset de treino completo**.
2. Previsão no test: `predict_proba ≥ FINAL_THRESHOLD`.
3. Submissão: `CONTROL_NUMBER` (vem do índice `df_test.index`) + `TARGET_B`.
4. Validação de formato contra `sample_submission.csv` (colunas e nº de linhas).
5. Guarda em `Predictive/DSML_Group11_submission.csv`.

---

## 12. Open-Ended — Interpretação

- **Permutation importance** (funciona para qualquer modelo): mede a queda de F1 ao
  baralhar cada feature. Top features esperadas: `RECENCY_X_FREQ`, `FREQUENCY_STATUS_97NK`, recência.
- **Matriz de confusão + análise de erros**: separa False Negatives (doadores perdidos)
  de False Positives (contactados em vão).
- **Distribuição de probabilidades** por classe verdadeira: mostra a (in)confiança do modelo.

---

## Resumo dos resultados

| Etapa | Melhor modelo | Val F1 (CV) |
|-------|--------------|-------------|
| Baseline (3 modelos) | DT / MLP | ~0.40 |
| Após tuning | Decision Tree / MLP | ~0.40–0.41 |
| Threshold OOF otimizado | — | ~0.40–0.41 |

**Teto natural** deste dataset com modelos *vanilla* sklearn: **F1 ≈ 0.40–0.41**.
Reduzir de 32 → 14 features **não baixou o F1**, confirmando que o que se removeu era ruído.

---

## Decisões-chave para a defesa

1. **`clean_before_split` vs imputação no pipeline** — separámos limpeza determinística
   (antes do split) de imputação estatística (dentro do CV) para evitar leakage. Esta foi
   a divisão da função original do Keni.
2. **Feature selection por correlação (≥ 0.08)** — só 11 numéricas + 3 categóricas;
   o F1 manteve-se, logo as features fracas eram ruído.
3. **Pipeline coluna-a-coluna** — escolhido para flexibilidade de ajuste por feature.
4. **Threshold tunado em OOF** — mais honesto que um único holdout.
5. **3 modelos (KNN, DT, MLP)** — o LogReg foi removido por não superar DT/MLP.
6. **F1 @ 0.25 em vez de 0.5** — alinha a avaliação com o desbalanceamento da target.
