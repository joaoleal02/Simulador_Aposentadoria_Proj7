
# 🧮 Simulador de Aposentadoria — README

## 📘 Visão Geral

Aplicativo **Streamlit** para planejamento de aposentadoria com:
- **Simulações Monte Carlo** (RF/RV);
- **Glide path** (reduz risco com o tempo);
- **Curva salarial com decaimento** (cresce mais no início, desacelera depois);
- **Compra de imóvel** (à vista ou financiado com ajuste *balloon*);
- Visualizações interativas: distribuição final, probabilidade de meta e mais.

> **Meta atual:** viver **somente dos juros (perpetuidade nominal)** — o principal é preservado e o rendimento mensal cobre a renda-alvo.

---

## ⚙️ Funcionalidades Principais

- **Monte Carlo + Moving Block Bootstrap (MBB)** para capturar dependência temporal nos retornos;
- **Renda Fixa (Selic)** como proxy da perna conservadora;
- **Glide path** personalizável;
- **Habitação**:
  - À vista → compra quando o portfólio atinge o valor;
  - Financiada → começa ao ter o valor da entrada;
  - “**Balloon**” no final → saldo devedor descontado no patrimônio;
  - Gráficos: % de simulações pagando parcelas / probabilidade acumulada de compra;
- **Meta de aposentadoria** baseada em perpetuidade nominal (renda vitalícia pelos juros).

---

## 🧠 Como Funciona

### 1️⃣ Retornos
- Usa retornos **mensais do Ibovespa** (ou CSV diário → convertido).
- **MBB** amostra blocos consecutivos para manter autocorrelação.

### 2️⃣ Glide Path
- Alocação em renda variável decresce linearmente do valor inicial ao final.

### 3️⃣ Salário e Contribuições
- Crescimento salarial decrescente com meia-vida ajustável.
- Contribuição mensal = % salário.

### 4️⃣ Habitação
- **À vista:** compra ao atingir o preço total.
- **Financiada:**
  - Entrada = (1 − financiamento) × preço;
  - Financia o restante (PRICE fixo);
  - Prestação abate contribuição e/ou portfólio;
  - Se restar dívida no final → *balloon* (deduzido do patrimônio).

### 5️⃣ Meta de Aposentadoria
- Capital alvo:  
  \\( PV = \frac{renda\_mensal}{r\_m} \\)  
  onde \( r\_m = (1 + selic)^{1/12} - 1 \)

---

## 🧩 Instalação

```bash
git clone <seu-repo>.git
cd <seu-repo>
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
pip install -r requirements.txt
```

Ou manualmente:
```bash
pip install streamlit pandas numpy altair
```

---

## ▶️ Execução

```bash
streamlit run app.py
```

Abrirá em: [http://localhost:8501](http://localhost:8501)

---

## 📊 Gráficos e Saídas

| Aba | Conteúdo |
|-----|-----------|
| **Visão Geral** | Fan chart (p5–p95), métricas e probabilidade de meta |
| **Distribuições** | Histograma, ECDF com linha de meta e tooltip |
| **Premissas** | Estatísticas de retornos, Selic, glide path, salário |
| **Habitação** | Parcelas médias / % sims com parcelas ativas / CDF de compra |
| **Dados** | Parâmetros e retornos usados |

---

## 🧾 Exemplo de Meta

Com Selic = 12% a.a. → \( r_m ≈ 0,009489 \)  
Renda mensal desejada = **R$ 30.000**  
\( PV = 30.000 / 0.009489 ≈ 3,16 milhões \)

---

## ⚡ Dicas de Desempenho

- `n_sims`: 2000–5000 é suficiente.
- `block_size`: 3–6 meses.
- Reduza `n_sims` para horizontes longos (30–40 anos).

---

## ⚠️ Limitações

- Tudo em **termos nominais** (sem inflação real).  
- Sem impostos, taxas ou custos de transação.  
- Imóvel sem valorização.  
- Sem rebalanceamentos fora da regra mensal.  

---

## 🧮 FAQ

**Q:** Por que algumas simulações ainda têm parcelas no último mês?  
**A:** Porque o financiamento começou tarde; o *balloon* resolve isso no cálculo do patrimônio final.

**Q:** O que o fan chart mostra?  
**A:** A dispersão do valor do portfólio ao longo do tempo — p5 a p95 e mediana.

**Q:** Por que a meta usa perpetuidade?  
**A:** Porque o objetivo é **viver dos juros**, não consumir o principal.

---

## 📜 Licença

MIT (recomendado).

---

## 🏁 Citação

> “Simulador de Aposentadoria (Streamlit). Monte Carlo com MBB, glide path, curva salarial decrescente e módulo habitacional com balloon. Meta nominal de perpetuidade (juros).”
