# Análise parcial 02: por que ε(k) não é monotônico em k alto

Data: 2026-10-01. Só leitura, CPU. As runs são A_srcnn e A_dscms (semente 0, tag campanha-v1). Script: `diagnostico.py`; números em `resultados.json`, `item2_variantes.csv` e `item5_metricas.csv`; figuras em `item2_variantes.*` e `item3_mapas.*`.

## Conclusão

O ε(k) não monotônico e a coerência alta em k≈60 são **vazamento de borda**, com confiança alta.

- **Por que há vazamento:** os recortes 128×128 não são periódicos. O salto entre bordas opostas é ~0,44 m/s, 12,5× a diferença típica entre pixels vizinhos. A FFT trata esse salto como descontinuidade.
- **Onde ele aparece no espectro:** forma uma cruz nos eixos kx=0 e ky=0. Ela contém 59% da energia do alvo em k=40–63, ocupando só 3,7% dos bins. No espaço, 77% da energia k≥40 está na faixa de 8 px junto às bordas.
- **Por que as redes "acertam" essa energia:** o LR é gerado por filtro gaussiano circular (FFT), então a versão borrada do salto está na entrada. As redes, e também o Wiener, reconstroem essa assinatura determinística.
- **Com janela Hann ou Tukey:**
  - ε das redes fica monotônico e vai a ~1 em k≥35, com T→0 e coerência→0. As redes não recuperam praticamente nada acima de k≈35 no interior.
  - A banda de ε > 0,5 começa em k≈25–27.
- **Sem terra:** a janela não tem nenhum pixel de terra (máscara bruta de 1/100° conferida), então a hipótese de máscara ou costa está descartada.
- **Previsibilidade linear:** é zero. O R² linear da componente k≥40 contra k<20 dá 0,000 em todas as máscaras, como esperado pela ortogonalidade das bandas. No interior (≥8 px das bordas), as redes explicam 19% da componente k≥40 e o Wiener, 48%.

## Outros achados

- **Recorte zerado no treino:** um recorte do treino é inteiro zero (índice global 5067, ano 2014, índice 334 no ano, ≈2014-12-01 assumindo um recorte por dia). Provavelmente é um dia sem dados preenchido com 0. Validação e teste não têm nenhum.
- **Métrica da regra do λ:** a ordem dada pelo eps_hf (média simples, k≥10) se inverte com janela.
  - Sem janela: SRCNN 0,248 < DSC/MS 0,260.
  - Com Hann: DSC/MS 0,692 < SRCNN 0,705.

  As métricas ponderadas pela potência mantêm a ordem (DSC/MS melhor) com e sem janela. O eps_hf atual é dominado pelas shells k≥40, onde o sinal é sobretudo artefato de borda.
