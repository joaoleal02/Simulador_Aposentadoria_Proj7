Simulador de Aposentadoria — README
Visão geral

Aplicativo Streamlit para planejar aposentadoria com:

Simulações Monte Carlo do portfólio (RF/RV),

Glide path (alocação em renda variável diminuindo ao longo do tempo),

Curva salarial com decaimento (cresce mais no início da carreira e desacelera),

Compra de imóvel (à vista ou financiado), com ajuste “balloon” no fim do horizonte,

Métricas claras: distribuição final, probabilidade de atingir meta e gráficos explicativos.

Meta de aposentadoria no app (padrão atual): viver somente dos juros (perpetuidade nominal) — o principal é preservado. A renda desejada mensal é paga com o rendimento do portfólio investido a uma taxa nominal (por padrão, a Selic escolhida nas entradas).

Principais recursos

Monte Carlo com Moving Block Bootstrap (MBB) sobre retornos mensais do Ibovespa (preserva dependência temporal de curto prazo).

Renda fixa (Selic) como proxy da perna conservadora.

Glide path: informe RV inicial e final (p. ex. 60% → 40%).

Curva salarial com decaimento: crescimento anual inicial, meia-vida e piso de longo prazo.

Habitação:

À vista: compra quando o portfólio tem caixa para o preço.

Financiada: compra quando há entrada; prestação (PRICE) primeiro abate a contribuição mensal; se faltar, retira do portfólio proporcionalmente.

Balloon no fim do horizonte: qualquer saldo devedor remanescente é descontado do patrimônio final (simulando quitação).

Gráficos: % de simulações com parcela ativa por mês (financiada) e CDF de probabilidade acumulada de compra (à vista).

Análises visuais:

Fan chart (p5–p95),

Histograma/ECDF do valor final com linha da meta e tooltip de probabilidade,

Tabela de estatísticas chave com coloração condicional.

Importação flexível de dados do Ibovespa (CSV diário ou mensal).

Como funciona (por dentro)
1) Retornos

Ibovespa: você fornece retornos mensais (ou um CSV diário/mensal; o app converte para mensal).

MBB (Moving Block Bootstrap): amostra blocos de tamanho b meses para cada simulação, preservando autocorrelação intra-bloco.

2) Glide path

Fração de RV gp[t] varia linearmente do RV inicial → RV final ao longo dos meses.

Rebalanceamento mensal opcional para acompanhar o glide path.

3) Salário e contribuições

Curva salarial com decaimento: crescimento anual começa em g0, decai com meia-vida definida e converge para um piso.

Contribuição mensal = % contrib × salário do mês.

4) Habitação

À vista: quando total_portfólio ≥ preço, ocorre saque imediato (composição proporcional RF/RV).

Financiada:

Entrada = (1 − loan_pct)*preço. Compra assim que houver caixa para a entrada.

Prestação (PRICE) começa um mês após a compra, por n_pay = anos*12 meses; parcela fixa pmt.

Ordem dos fluxos mensais:

contribuições entram,

parcela abate contribuição;

faltou? retira do portfólio proporcionalmente (RF/RV).

Na aposentadoria (fim do horizonte): se ainda houver parcelas pendentes, calcula-se o saldo devedor remanescente e desconta do portfólio (ajuste balloon). As estatísticas finais usam esse patrimônio líquido.

5) Meta de aposentadoria (perpetuidade nominal)

Viver dos juros, preservando o principal.

Capital necessário: PV = renda_mensal / r_m, onde r_m é a taxa nominal mensal (por padrão, convertida a partir da Selic definida nas entradas).

Sem correção por inflação no cálculo atual — meta e retornos estão em termos nominais.

Instalação

Pré-requisitos

Python 3.9+ recomendado

pip / venv

Clonar e instalar dependências

git clone <seu-repo>.git
cd <seu-repo>
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt


Se não tiver um requirements.txt, comece com:

streamlit>=1.32
pandas>=2.0
numpy>=1.24
altair>=5.0

Execução
streamlit run app.py


O navegador abrirá em http://localhost:8501.

Entrada de dados (CSV do Ibovespa)

Arquivo com colunas de data e preço:

Data: Date ou Data

Preço: Adj Close, AdjClose, Close, Fechamento ou Price

O app detecta frequência; se diário, reamostra para último preço do mês e calcula pct_change mensal.

Também é possível editar manualmente uma série mensal semente (placeholders).

Parâmetros importantes

Selic (anual, nominal): taxa usada como retorno da perna conservadora e como taxa de aposentadoria na perpetuidade.

Glide path (RV inicial/final): controla o risco ao longo do tempo.

Contribuições: % do salário.

Curva salarial: g0 (início), meia-vida (anos) e piso.

Habitação:

À vista: preço.

Financiada: loan_pct, spread sobre Selic, prazo em anos.

Simulações: número de cenários (n_sims) e tamanho do bloco no MBB.

Saídas e gráficos

Visão Geral

Métricas: Mediana/Média/P5–P95 e Prob(≥ meta).

Fan chart do portfólio.

Texto com glide path e parâmetros de meta (perpetuidade).

Distribuições

Histograma do valor final.

ECDF com linha da meta e probabilidade (P(≥ meta)).

Estatísticas chave com formatação BR e faixa de cor por probabilidade.

Premissas

Médias e desvios do Ibovespa (mensal e anualizado).

Selic, glide path e parâmetros da curva salarial.

Habitação

Financiada: parcela estimada, nº de parcelas, juros totais, tabela de amortização (download CSV) e % de simulações com parcela ativa por mês.

À vista: probabilidade acumulada de compra (CDF).

Dados

Série de retornos usada, parâmetros e meta.

Interpretação da meta

Perpetuidade nominal (padrão): o app calcula o capital para pagar a renda mensal nominal indefinidamente, sem consumir principal, com base na taxa Selic definida.
Ex.: com Selic de 12% a.a. → r_m = (1+0,12)^(1/12)-1 ≈ 0,9489%/mês.
Para renda de R$ 30.000/mês: PV ≈ 30.000 / 0,009489 ≈ R$ 3,16 milhões.

Se desejar trabalhar em termos reais (descontando inflação), seria preciso reintroduzir sliders de taxa real e crescimento real dos pagamentos — não habilitado nesta versão.

Desempenho

Comece com n_sims entre 2.000 e 5.000.

Aumente o tamanho do bloco do MBB (3–6) para preservar mais dependência, mas note que isso pode aumentar a variância dos cenários.

Para horizontes longos (300–480 meses), considere reduzir n_sims.

Limitações e suposições

Taxas nominais fixas ao longo do horizonte.

Sem impostos, taxas, fricções ou aportes extraordinários além da regra definida.

Sem aluguel / valorização do imóvel — o foco é no fluxo de caixa (entrada, prestação, balloon).

Compra financiada considera parcela constante PRICE com taxa fixa (Selic + spread como proxy).

Perpetuidade nominal: não há ajuste explícito por inflação (meta e retornos são nominais).

FAQ

Q: Por que a probabilidade de compra à vista pode ser baixa mesmo com crescimento do portfólio?
A: Porque depende do primeiro mês em que o portfólio cruza o preço total do imóvel. Em horizontes curtos ou preços altos, parte dos cenários pode não atingir o valor.

Q: Por que algumas simulações ainda têm parcelas no último mês?
A: Porque a compra financiada pode ocorrer tardiamente; nesses casos, o app aplica o balloon (desconta o saldo devedor do patrimônio final) para comparar cenários de forma justa.

Q: O fan chart usa média/percentis entre simulações?
A: Sim — por mês, calculamos p5, p25, p50, p75, p95 e a média do valor de portfólio.

Solução de problemas

CSV inválido: verifique nomes de colunas (Date/Data e Close/Adj Close/etc.).

Rendimentos “vazios”: confira se há pelo menos 12 meses de dados após a limpeza.

Execução lenta: reduza n_sims ou o horizonte, desative gráficos auxiliares, feche outras abas.

Extensões futuras (ideias)

Meta real (inflação explícita) e pagamentos indexados.

Custos de transação, taxas e impostos.

Renda variável além do IBOV (ex.: MSCI ACWI, small caps, etc.).

Stress testing e análise de sensibilidade interativa.

Considerar valor do imóvel como ativo (liquidez baixa) e cenários de venda.

Licença

Defina a licença do projeto (ex.: MIT).

Agradecimentos

Comunidade Streamlit e Altair.

Usuários que contribuíram com feedback para ajustes de lógica de financiamento e visualizações.
