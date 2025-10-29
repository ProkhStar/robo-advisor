# Design Decisions / Mapeamento de Perfil de Risco

## Objetivo
Documentar decisões funcionais e mapeamentos usados pelo protótipo Robo-Advisor. Este ficheiro serve para justificar escolhas técnicas no portefólio.

## Questionário curto (input)
1. Idade: \<30 / 30–45 / 45–60 / \>60  
2. Horizonte: \<1 ano / 1–3 anos / 3–7 anos / \>7 anos  
3. Tolerância à perda: Alta / Média / Baixa  
4. Experiência com investimentos: Nenhuma / Alguma / Experiente  
5. Objetivo: Crescimento / Rendimento / Proteção

## Mapeamento para `risk_aversion` (0 = agressivo, 1 = conservador)
- Idade:
  - \<30 → 0.1
  - 30–45 → 0.25
  - 45–60 → 0.45
  - \>60 → 0.7
- Horizonte:
  - \<1 ano → 0.8
  - 1–3 anos → 0.6
  - 3–7 anos → 0.35
  - \>7 anos → 0.15
- Tolerância à perda:
  - Alta → 0.1
  - Média → 0.45
  - Baixa → 0.85
- Experiência:
  - Nenhuma → +0.05
  - Alguma → 0.0
  - Experiente → -0.05
- Objetivo:
  - Crescimento → -0.05
  - Rendimento → +0.05
  - Proteção → +0.1

## Fórmula de Agregação (exemplo)
Consolidamos os valores acima com pesos:  
`risk_aversion = 0.35*age + 0.30*horizon + 0.25*tolerance + 0.05*experience + 0.05*objective`  
Depois normalizamos para [0,1]. Documenta no notebook como ajustaste estes pesos.

## Notas sobre trade-offs e limites
- Este mapeamento é heurístico; para produção deverias calibrar com dados/estudos de comportamento.
- Explicar a alocação ao utilizador (usando linguagem simples) é obrigatório — inclui um parágrafo no UI.
- Testes: valida com cenários contrafactuais (stress test) e mostra sensibilidade da alocação à `risk_aversion`.

