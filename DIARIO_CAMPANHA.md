# Diário da campanha

Registro cronológico do que foi feito, verificado e decidido, com os números que importam para a dissertação. Cada passo novo entra no fim, com data. As regras formais estão em `DECISIONS.md`, o estado atual em `STATUS.md`, os comandos em `RUNNING.md` e as dúvidas abertas em `OPEN_QUESTIONS.md`.

**Convenções:**
- Teste: 2022-01-01 a 2024-08-31 (974 dias).
- Validação: 2020–2021 (731 dias).
- MSE em m²/s².
- "Semente 0", a menos que se diga outra coisa.

---

## 1. Infraestrutura e regras (29/09 a 01/10/2026)

- **Pipeline:** reprodutibilidade, checkpoints completos com retomada automática, early stopping relativo e opção de resíduo global igual para as cinco arquiteturas. Perda MSE + λ·LPIPS, métricas espectrais, estatística por bootstrap em blocos e multiestágio. Commits de 3580c63 a 6e87591.
- **Dados e degradação:**
  - um recorte 128×128 por dia, às 12:00 UTC, numa janela fixa do HYCOM GOMb0.01 (22,9–27,5°N, 90,1–84,9°W);
  - média 4×4, malha isotrópica de ~4,02 km;
  - degradação: só o filtro gaussiano H(k) = exp(−k²/(2·10²)).
- **Piloto:** 2 épocas por arquitetura, para medir tempos e memória.
- **Tag `campanha-v1`** (b25f1f3) e worktree `_run/`. O treino roda sempre a partir dele.
- **Regras fixadas antes de qualquer resultado** (`DECISIONS.md`):
  - parada: early stopping relativo de 0,1% em 15 épocas, com 50 a 300 épocas;
  - seleção sempre na validação;
  - arquitetura do bloco E: menor MSE de validação no bloco A;
  - regra do λ;
  - regra do teto de épocas.

## 2. Fase1: bloco A (MSE, semente 0) e bloco B (varredura de λ), 01/10 a 15/10

**Bloco A, as cinco arquiteturas com λ = 0, residual:**

| Run | Épocas | Parada | Horas | MSE val | MSE teste | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|---|---|
| SRCNN | 213 | early stop | 0,37 | 7,65e-4 | 8,30e-4 | 45,57 | 0,9846 | 0,156 |
| DSC/MS | 300 | teto | 1,72 | 7,20e-4 | 7,81e-4 | 45,82 | 0,9857 | 0,138 |
| PRUSR | 192 | early stop | 5,47 | 5,98e-4 | 6,49e-4 | 46,64 | 0,9870 | 0,124 |
| SRNO | 89 | early stop | 4,43 | 5,35e-4 | 5,82e-4 | 47,11 | 0,9889 | 0,115 |
| **SoFoNO** | 174 | early stop | 11,74 | **4,65e-4** | **5,08e-4** | **47,72** | **0,9902** | **0,102** |
| *Wiener (γ = 1e-12)* | – | – | – | *2,83e-4* | *3,24e-4* | *49,55* | *0,9968* | – |
| *identidade* | – | – | – | *4,03e-3* | *4,55e-3* | *38,07* | *0,9614* | – |

- **DSC/MS bateu o teto de 300 épocas.** Pela regra a priori (estender só se ficar a menos de 2–3% da arquitetura mais próxima), não é estendida: está 6% à frente da SRCNN. Análise em `DECISIONS.md`.
- **Validação × teste:** o teste é ~8,5% mais difícil que a validação em todas as redes, e o mesmo vale para os baselines. Não há overfitting.

**Bloco B, varredura de λ só na SRCNN e na SRNO**, uma CNN e um operador, como previsto no plano:

| Run | Épocas | Horas | MSE val | MSE teste | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|---|
| SRCNN λ=1e-3 | 211 | 1,87 | 7,42e-4 | 8,05e-4 | 45,70 | 0,9852 | 0,146 |
| SRCNN λ=1e-2 | 198 | 1,76 | 6,93e-4 | 7,53e-4 | 45,97 | 0,9862 | 0,090 |
| SRCNN λ=1e-1 | 223 | 1,97 | 5,92e-4 | 6,43e-4 | 46,65 | 0,9879 | 0,060 |
| SRCNN λ=1 | 134 | 1,19 | 6,21e-4 | 6,79e-4 | 46,39 | 0,9878 | 0,056 |
| SRNO λ=1e-3 | 128 | 7,34 | 5,04e-4 | 5,49e-4 | 47,35 | 0,9897 | 0,095 |
| SRNO λ=1e-2 | 96 | 5,51 | 3,99e-4 | 4,35e-4 | 48,36 | 0,9916 | 0,047 |
| SRNO λ=1e-1 | 125 | 7,17 | 2,12e-4 | 2,32e-4 | 51,08 | 0,9958 | 0,017 |
| **SRNO λ=1** | 170 | 9,75 | **1,07e-4** | **1,18e-4** | **54,03** | **0,9980** | **0,006** |

**Resultado central:** o termo LPIPS **reduz o MSE**. Na SRNO o efeito é grande e monotônico: o MSE com λ = 1 é 5× menor que com λ = 0 e fica **abaixo do Wiener**. Na SRCNN o ganho é menor, com o ótimo de MSE em λ = 0,1.

## 3. Achado metodológico: vazamento de borda nas métricas espectrais (05 a 14/10)

- **Causa:** os recortes não são periódicos. O salto entre bordas opostas é ~12,5× a diferença entre vizinhos, e a FFT o espalha em ~k⁻² por todo o espectro (uma cruz nos eixos). 59% da energia do alvo em k = 40–63 vem disso. Detalhes em `analises/parcial_02`.
- **Efeito sem janela:** as métricas de k alto ficam infladas: k_c aparece como "63+", a razão E_pred/E_HR fica alta em k ≳ 25 e a inclinação do HR sai −2,09.
- **Correção:** janela Hann opcional nas métricas espectrais (`--window hann`, commit 8f91f87). O padrão continua sem janela, bit a bit idêntico ao anterior.
- **Com janela:**
  - a inclinação do HR passa a **−2,66** (k = 11–46);
  - o k_c real das redes do bloco A fica entre **25 e 29**;
  - sem LPIPS, as redes quase não têm energia acima de k ≈ 30;
  - comparação completa em `tables/metricas_espectrais_hann.csv` e `analises/hann/comparacao.md`.
- **A regra do λ foi emendada antes de olhar o bloco B** (59edf2f): passou a usar eps_w_10_40, robusta ao vazamento. O eps_hf ficou só como sensibilidade.

## 4. Verificações de dados para a dissertação (14/10)

- **Grade:** isotrópica, dx = dy ≈ 4,02 km, de 4,10 km no sul a 3,95 km no norte.
- **Tempo:** cada amostra é um instante às 12:00 UTC, não uma média diária. Conferido no eixo do servidor HYCOM: em 2003–2008, 2014 e 2022 o índice cai às 12h.
- **2001:** datas deslocadas 15 dias, porque o servidor começa em 16/01/2001. Só afeta o treino.
- **Amostra ruim:** 2014-12-01 (treino) está inteira zerada, porque u e v estão ausentes no servidor nesse instante. Não há outras amostras zeradas, constantes ou com NaN.
- **Figuras e scripts:** `analises/dissertacao/`.

## 5. Seleção do λ e da arquitetura do bloco E (15/10)

Comando: `collect_results.py --split val --runs "A_*_seed0" "B_*"`. Relatório em `report/20261015-1025_val/`.

| λ | SRCNN: MSE val / razão ao λ=0 / eps_w_10_40 | SRNO: MSE val / razão / eps_w_10_40 |
|---|---|---|
| 0 (referência) | 7,65e-4 / 1,00 / 0,1007 | 5,35e-4 / 1,00 / 0,0624 |
| 1e-3 | 7,42e-4 / 0,97 / 0,0961 | 5,04e-4 / 0,94 / 0,0569 |
| 1e-2 | 6,93e-4 / 0,91 / 0,0876 | 3,99e-4 / 0,75 / 0,0367 |
| 1e-1 | 5,92e-4 / 0,77 / 0,0670 | 2,12e-4 / 0,40 / 0,0090 |
| **1** | 6,21e-4 / 0,81 / **0,0650** | 1,07e-4 / 0,20 / **0,0019** |

- **Regra aplicada:** todos os λ > 0 passam no limite de MSE.
  - **SRCNN:** λ = 1. O λ = 0,1 ficou 3,1% acima do mínimo, fora da tolerância de 2%.
  - **SRNO:** λ = 1, com folga.
  - **Mesmo λ nas duas varreduras:** **λ = 1 para as cinco arquiteturas** no bloco C (`make_block_c.py --lambda 1 --sweep-arch srno srcnn`).
  - A seleção foi conferida manualmente contra o CSV.
- **Sensibilidade (eps_hf):** na SRCNN escolheria λ = 0,1, que também tem MSE 4,6% menor. Na SRNO concorda.
- **Arquitetura do bloco E:** **SoFoNO** (MSE de validação 4,65e-4), depois SRNO (5,35e-4) e PRUSR (5,98e-4).
- **Ressalvas para o texto:**
  1. A decisão na SRCNN foi apertada, e as duas métricas discordam.
  2. λ = 1 está na borda da grade, e na SRNO a tendência continua caindo, então o ótimo pode estar acima de 1. A regra não prevê estender a grade; λ = 10 na SRNO pode entrar só como análise de sensibilidade.
- **Verificação do ganho do SRNO com λ = 1:** é real, não artefato de borda. Feita no teste, com SRNO λ = 0, 0,1 e 1, SoFoNO λ = 0 e Wiener:

  | Teste | MSE | MSE no interior (≥8 px da borda) | ε 10–40 com Hann | ε k≥40 com Hann | T k≥40 com Hann |
  |---|---|---|---|---|---|
  | Wiener | 3,24e-4 | 1,15e-4 | ~0 | 0,56 | 0,97 |
  | SRNO λ=0 | 5,82e-4 | 4,61e-4 | 0,083 | 0,98 | 0,006 |
  | SRNO λ=0,1 | 2,32e-4 | 1,65e-4 | 0,006 | 0,73 | 0,41 |
  | SRNO λ=1 | 1,18e-4 | 7,9e-5 | 0,0004 | 0,35 | 0,90 |
  | SoFoNO λ=0 | 5,08e-4 | 4,16e-4 | 0,069 | 0,97 | 0,007 |

  A SRNO com λ = 1 supera o Wiener também no interior e recupera energia real em k ≥ 40. A verificar ainda: mapas de erro e espectros de alguns dias, antes de entrar no texto.

**Próximo:** gerar os configs do bloco C com λ = 1 e lançar a fase2 (C semente 0: DSC/MS, PRUSR, SoFoNO).
