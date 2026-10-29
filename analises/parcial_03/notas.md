# Análise parcial 03: regime A, semente 0 (SRCNN, DSC/MS, PRUSR, SRNO), teste

Data: 2026-10-05. CPU; runs na tag campanha-v1. Falta a SoFoNO (em treino).

## Arquivos

- `tabela_parcial_03.csv`: métricas padrão.
- `janela_teste.csv`: métricas com janela Hann e sem os eixos.
- `report/`: relatório do `collect_results.py`.
- `analise.py`, `janela.py`: scripts.

## Métricas padrão (teste)

| | MSE | PSNR | ε pond. k≥10 | k_c real (Hann) | épocas | horas |
|---|---|---|---|---|---|---|
| SRCNN | 8,30e-4 | 45,57 | 0,117 | 25 | 213 | 0,37 |
| DSC/MS | 7,81e-4 | 45,82 | 0,110 | 26 | 300 (teto) | 1,72 |
| PRUSR | 6,49e-4 | 46,64 | 0,092 | 27 | 192 | 5,47 |
| SRNO | 5,82e-4 | 47,11 | 0,082 | 29 | 89 | 4,43 |
| Wiener | 3,24e-4 | 49,55 | 0,044 | 49 | – | – |

## Pontos principais

- **Ordem:** SRNO < PRUSR < DSC/MS < SRCNN, em MSE, ε ponderado (com e sem janela) e LPIPS.
- **Validação × teste:** teste/val ≈ 1,085 nas quatro redes; sem overfitting.
- **Erro:** dominado pela banda 20–40. Com janela, nenhuma rede recupera o interior acima de k≈35, e o k_c real fica entre 25 e 29.
- **eps_hf atual:** é a única métrica que inverte SRCNN/DSC/MS e que põe SRNO/PRUSR à frente do Wiener. Isso é artefato de borda em k≥40.
- **Teto:** a DSC/MS não é estendida, porque a arquitetura mais próxima está a 6%.
- **Bloco E:** a escolha provisória é a SRNO (menor MSE de validação), até a SoFoNO terminar.
