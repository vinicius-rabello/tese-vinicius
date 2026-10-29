# Pacote de resultados: campanha de super-resolução HYCOM (estado em 15/10/2026)

**Ordem de leitura sugerida:**

1. `DIARIO_CAMPANHA.md`: o que foi feito, verificado e decidido, em ordem, com as tabelas principais.
2. `DECISIONS.md`: regras fixadas a priori (parada, seleção do λ e emenda, teto de épocas, janela Hann).
3. `report/*_fase1/report.md`: métricas de teste das 13 runs da fase1, sem janela. `report/*_fase1_hann/`: as mesmas com janela Hann nas métricas espectrais.
4. `report/*_val/selecao_lambda_*.csv`: aplicação da regra do λ na validação.
5. `tables/metricas_espectrais_hann.csv` e `analises/hann/comparacao.md`: comparação sem janela × Hann.
6. `analises/parcial_02/notas.md`: diagnóstico do vazamento de borda (figuras `item2_variantes.png`, `item3_mapas.png`).
7. `analises/dissertacao/`: espectro médio do HR, figura HR/LR e razão espectral com e sem janela.

**Estado:**
- **Fase1 completa:** bloco A, as 5 arquiteturas na semente 0 com MSE; bloco B, varredura de λ na SRCNN e na SRNO.
- **Seleção:** λ = 1 para todas as arquiteturas; o bloco E fica com a SoFoNO.
- **Fase2 em andamento:** bloco C semente 0 com λ = 1 (DSC/MS, PRUSR, SoFoNO). Ainda não há resultados de teste dela.

**Ainda não feito:**
- sementes 1 e 2 (fases 4 e 5);
- ablação do skip (bloco D);
- multiestágio (bloco E);
- regime com ruído;
- estatística com bootstrap (`stats.py`), que exige as 3 sementes.

**Observação:** `STATUS.md` pode estar desatualizado; o diário é a referência.

**Unidades e definições:**
- MSE em m²/s²; PSNR com R fixo = 5,3268 m/s.
- k em ciclos por domínio de 128 px (λ ≈ 515/k km).
- eps_w_10_40: erro espectral relativo ponderado pela potência, em 10 ≤ k < 40.
- k_c: maior k com ε(k') ≤ 1/2 para todo k' ≤ k.
