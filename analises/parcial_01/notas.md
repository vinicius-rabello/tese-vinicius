# Análise parcial 01: regime A, semente 0 (SRCNN, DSC/MS) e baselines

Data: 2026-10-01. Runs na tag campanha-v1 (b25f1f3). Teste: 974 dias (2022-01-01 a 2024-08-31).

A avaliação foi feita em CPU, com o mesmo código da avaliação em GPU. Os resultados ficaram gravados em `results/<run>/`, como prevê a política do `collect_results.py`, e a fila não vai reavaliar essas runs.

## Arquivos

- `report/`: saída do `collect_results.py --runs "A_srcnn*" "A_dscms*"` (figuras, `all_runs.csv`, `baselines.csv`, `report.md`); zip em `report_report.zip`, sem `.pt`.
- `tabela_parcial_01.csv`: tabela completa, gerada por `analise.py`.
- `consistencia.txt`: verificações de PSNR e das constantes.

## Resultados principais (teste)

| | MSE (m²/s²) | PSNR (dB) | SSIM | KE | eps_hf | k_c | k_c_01 | inclinação (alvo −2,10) |
|---|---|---|---|---|---|---|---|---|
| identidade | 4,55e-3 | 38,07 | 0,961 | 42,7 | 0,874 | 15 | 8 | −16,0 |
| SRCNN | 8,30e-4 | 45,57 | 0,985 | 0,26 | 0,248 | 63+ | 20 | −2,59 |
| DSC/MS | 7,81e-4 | 45,82 | 0,986 | 0,35 | 0,260 | 63+ | 22 | −2,70 |
| Wiener | 3,24e-4 | 49,55 | 0,997 | 5,32 | 0,218 | 51 | 49 | −2,10 |

- **Distância percorrida:** a SRCNN fecha 88,0% do intervalo de MSE entre a identidade e o Wiener; a DSC/MS, 89,2%.
- **Distância ao Wiener:** as redes ficam com 2,6× e 2,4× o MSE do Wiener, ou −4,0 e −3,7 dB de PSNR.
- **DSC/MS contra SRCNN:** 6,0% de MSE a menos, tanto no teste quanto no melhor MSE de validação (7,20e-4 contra 7,65e-4).

## Onde está o erro

**SRCNN e DSC/MS:**
- **Até k≈20,** as redes recuperam quase toda a amplitude: T ≥ 0,96 e eps ≤ 0,014.
- **Na banda 20–40,** que concentra 63–64% do MSE, a amplitude fica em T = 0,56 (SRCNN) e 0,68 (DSC/MS), com eps de 0,28 e 0,26.
- **Na banda k≥40,** que concentra 27–32% do MSE, T fica em 0,58 e 0,46, com eps de 0,28 e 0,32.

**Wiener:**
- **Até k≈40** ele é essencialmente exato: eps < 3e-7.
- **Todo o seu MSE** está na banda k≥40, onde T = 0,64 e eps = 0,41, e acima de k≈52 ele não recupera nada.

**Comparação na banda k≥40:** as redes têm menos erro que o Wiener nessa banda (MSE de 2,2–2,5e-4 contra 3,2e-4). O Wiener vence no total porque é praticamente exato entre 10 e 40.

**Fase e amplitude:**
- **Fase:** k_c_fase ≈ 63 nas redes e 52,7 no Wiener. A coerência de fase não enxerga o déficit de amplitude.
- **Amplitude:** o déficit aparece em T(k) e na inclinação espectral, que fica mais íngreme que o alvo.
- **O erro alto em k≥20 tem as duas partes:** se a fase fosse perfeita, T = 0,56 daria eps ≈ 0,06; o medido é 0,28.

## Cautelas

- **k_c (eps ≤ 1/2) está censurado em 63 nas duas redes**, porque o eps conjunto nunca passa de 1/2. k_c_01 separa melhor: 20 (SRCNN) e 22 (DSC/MS).
- **KE e WKE favorecem fortemente as redes em relação ao Wiener** (0,3 contra 5,3). São métricas log-espectrais e punem o espectro quase nulo do Wiener acima de k≈52. Interpretar com o MSE ao lado.
- **eps_hf (média simples sobre as shells k ≥ 10) e eps ponderado pela potência ordenam as duas redes ao contrário:**
  - eps_hf: SRCNN 0,248 < DSC/MS 0,260;
  - ponderado: DSC/MS 0,110 < SRCNN 0,117.

  O eps_hf dá peso grande às 24 shells com k ≥ 40, onde a SRCNN é melhor. É a métrica da regra de seleção do λ, que está fixada a priori.
- **Regra do teto:** a DSC/MS bateu 300 épocas e fica 6% à frente da arquitetura mais próxima até agora (SRCNN), acima do limiar de ~2–3%. Pela regra, por ora ela não seria estendida. A decisão é provisória até PRUSR, SRNO e SoFoNO terminarem.
- **Ainda não há variância entre sementes:** é uma semente só e são dois modelos. Diferenças de alguns por cento ainda não têm intervalo de confiança.

## Complemento (itens 4b–6)

**T(k) e ε(k) por shell (teste):**

| | T < 0,9 a partir de | T < 0,5 a partir de | ε > 0,1 a partir de | ε máximo |
|---|---|---|---|---|
| SRCNN | k=18 | k=25 | k=21 | 0,43 em k=31 |
| DSC/MS | k=20 | k=44 | k=23 | 0,45 em k=31 |
| Wiener | k=52 | k=53 | – | ~1 acima de k≈53 |

- **ε(k) das redes não é monotônico:** sobe até k≈31 e cai para 0,22–0,29 em k≈60.
- **A coerência normalizada também volta a subir:** c(k) = (1 + T − ε)/(2√T) desce a 0,75 em k≈30 e sobe para 0,88 em k=60.
- **Validação × teste:** razão teste/validação do MSE de 1,085 nas redes, contra 1,13 (identidade) e 1,14 (Wiener). O período de teste é mais difícil; não há sinal de overfitting.
- **Treino × validação:** razão train/val ≈ 0,98 no fim das duas runs.
- **Piso do LR:** o espectro do LR satura em ~1e-16 m²/s² acima de k≈53, que é o piso de precisão float32 dos dados. Isso explica por que o Wiener (γ = 1e-12) não inverte acima de k≈52.
