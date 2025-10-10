import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# =============================
# Utilitários e premissas
# =============================

@dataclass
class SalaryCurve:
    name: str
    annual_growth_pct: float

DEFAULT_CURVES = {
    "Ruim":  SalaryCurve("Ruim", 0.02),
    "Médio": SalaryCurve("Médio", 0.05),
    "Bom":   SalaryCurve("Bom", 0.08),
}

DEFAULT_IBOV_LAST10 = [-0.1331, 0.3894, 0.2686, 0.1503, 0.3158, -0.1009, -0.1193, 0.0469, 0.2228, 0.1200]

def annual_to_monthly_rate(annual_rate: float) -> float:
    return (1 + annual_rate) ** (1 / 12) - 1

def monthly_to_annual_rate(monthly_rate: float) -> float:
    return (1 + monthly_rate) ** 12 - 1

def expand_annual_to_monthly_series(annual_returns: List[float], years_each: int = 1, noise: float = 0.0, seed: Optional[int] = None) -> List[float]:
    rng = np.random.default_rng(seed)
    monthly = []
    for r in annual_returns:
        m = annual_to_monthly_rate(r)
        base = np.full(12 * years_each, m, dtype=float)
        if noise > 0:
            base = base + rng.normal(0.0, noise, size=base.size)
        monthly.extend(base.tolist())
    return monthly

def compute_salary_path_decay(base_salary: float, months: int, annual_growth_initial: float, half_life_years: float, annual_growth_floor: float) -> np.ndarray:
    t = np.arange(months, dtype=float)
    half_life_m = max(1.0, half_life_years * 12.0)
    g_a_t = annual_growth_floor + (annual_growth_initial - annual_growth_floor) * np.exp(-np.log(2) * t / half_life_m)
    g_m_t = (1.0 + g_a_t) ** (1.0 / 12.0) - 1.0
    salary = np.empty(months, dtype=float)
    salary[0] = base_salary
    for i in range(1, months):
        salary[i] = salary[i - 1] * (1.0 + g_m_t[i - 1])
    return salary

def robust_monthly_returns_from_csv(df: pd.DataFrame) -> List[float]:
    close_col = next((c for c in ["Adj Close", "AdjClose", "Close", "Fechamento", "Price"] if c in df.columns), None)
    date_col  = next((c for c in ["Date", "Data"] if c in df.columns), None)
    if close_col is None or date_col is None:
        raise ValueError("CSV precisa ter colunas de data e preço (ex.: 'Date' e 'Adj Close'/'Close').")
    dfx = df[[date_col, close_col]].dropna().copy()
    dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce")
    dfx = dfx.dropna().sort_values(date_col)
    per_m = dfx[date_col].dt.to_period("M")
    counts = per_m.value_counts()
    if (counts.median() if len(counts) else 0) > 3:
        dfx = dfx.set_index(date_col).resample("M").last().reset_index()
    dfx["ret_mensal"] = dfx[close_col].pct_change()
    ret = dfx["ret_mensal"].dropna().astype(float).tolist()
    if len(ret) < 12:
        st.warning("Série mensal tem poucas observações (<12). Considere carregar histórico mais longo.")
    return ret

# =============================
# Modelo RV: Moving Block Bootstrap (MBB)
# =============================

def mbb_generate(months: int, n_sims: int, monthly_returns: np.ndarray, block_size: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = monthly_returns.astype(float)
    T = x.shape[0]
    b = max(1, int(block_size))
    if T < b:
        return rng.choice(x, size=(months, n_sims), replace=True)
    n_blocks = int(np.ceil(months / b))
    starts = np.arange(0, T - b + 1)
    out = np.zeros((months, n_sims), dtype=float)
    for s in range(n_sims):
        seq = []
        for _ in range(n_blocks):
            i0 = int(rng.choice(starts))
            seq.extend(x[i0:i0 + b])
        out[:, s] = np.array(seq[:months], dtype=float)
    return out

# =============================
# Simulador principal
# =============================

def simulate_portfolio(
    months: int,
    n_sims: int,
    selic_annual: float,
    ibov_monthly_returns: List[float],
    salary_path: np.ndarray,
    contrib_pct: float,
    initial_investment: float,
    eq_share_start: float,
    eq_share_end: float,
    rebalance_monthly: bool = True,
    seed: Optional[int] = None,
    block_size: int = 3,
    enable_house: bool = False,
    house_price: float = 0.0,
    house_mode: str = "avista",
    loan_pct: float = 0.8,
    loan_spread: float = 0.03,
    loan_years: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:

    rng = np.random.default_rng(seed)

    rf_m = annual_to_monthly_rate(selic_annual)
    eq_monthly = np.array(ibov_monthly_returns, dtype=float)
    if eq_monthly.size == 0:
        raise ValueError("Lista de retornos mensais do Ibovespa está vazia.")

    eq_rets = mbb_generate(months=months, n_sims=n_sims, monthly_returns=eq_monthly, block_size=block_size, seed=seed)

    eq_share_start = float(np.clip(eq_share_start, 0.0, 1.0))
    eq_share_end   = float(np.clip(eq_share_end,   0.0, 1.0))
    gp = np.linspace(eq_share_start, eq_share_end, months)
    rf_gp = 1.0 - gp

    total0  = float(initial_investment)
    rf_hold = np.full(n_sims, rf_gp[0] * total0, dtype=float)
    eq_hold = np.full(n_sims, gp[0]   * total0, dtype=float)

    paths         = np.zeros((months, n_sims), dtype=float)
    contribs      = np.zeros((months, n_sims), dtype=float)
    mortgage_pays = np.zeros((months, n_sims), dtype=float)
    house_withdraw= np.zeros((months, n_sims), dtype=float)

    purchased  = np.zeros(n_sims, dtype=bool)
    purchase_m = np.full(n_sims, -1, dtype=int)

    if enable_house and house_price > 0:
        if house_mode == "avista":
            dp        = house_price
            principal = 0.0
            loan_m    = 0.0
            n_pay     = 0
            pmt       = 0.0
        else:
            dp        = house_price * (1.0 - loan_pct)
            principal = house_price * loan_pct
            annual_rate = max(0.0, selic_annual + loan_spread)
            loan_m    = annual_to_monthly_rate(annual_rate)
            n_pay     = max(1, int(loan_years * 12))
            pmt       = principal * loan_m / (1 - (1 + loan_m) ** (-n_pay)) if loan_m > 0 else principal / n_pay
    else:
        dp = principal = loan_m = 0.0
        n_pay = 0
        pmt   = 0.0

    house_stats = {
        "prob_compra": 0.0,
        "mediana_mes_compra": None,
        "pmt": pmt,
        "n_parcelas": n_pay,
        "principal": principal,
        "taxa_mensal_emprestimo": loan_m,
        "total_pago_por_contrato": 0.0,
        "juros_totais_por_contrato": 0.0,
        "amort_schedule": None,
    }

    if enable_house and house_price > 0 and house_mode == "divida" and pmt > 0 and principal > 0 and n_pay > 0:
        amort = amortization_table(principal=principal, monthly_rate=loan_m, n_pay=n_pay)
        house_stats["amort_schedule"] = amort
        house_stats["total_pago_por_contrato"] = float(amort["parcela"].sum())
        house_stats["juros_totais_por_contrato"] = float(amort["juros"].sum())

    for t in range(months):
        if enable_house and house_price > 0:
            current_total = rf_hold + eq_hold
            # CORREÇÃO: permitir compra imediata quando entrada dp == 0
            mask = (~purchased) & ((current_total >= dp) if dp > 0 else True)
            if np.any(mask):
                total_before_w = rf_hold[mask] + eq_hold[mask]
                with np.errstate(divide="ignore", invalid="ignore"):
                    rf_ratio = np.where(total_before_w > 0, rf_hold[mask] / total_before_w, rf_gp[t])
                    eq_ratio = np.where(total_before_w > 0, eq_hold[mask] / total_before_w, gp[t])
                withdraw = np.minimum(dp, total_before_w)
                rf_hold[mask] -= withdraw * rf_ratio
                eq_hold[mask] -= withdraw * eq_ratio
                house_withdraw[t, mask] = withdraw
                purchased[mask] = True
                purchase_m[mask] = t

        base_contr = contrib_pct * salary_path[t]
        contr_vec  = np.full(n_sims, base_contr, dtype=float)

        if enable_house and house_price > 0 and house_mode == "divida" and pmt > 0:
            active = purchased
            if np.any(active):
                months_since = np.where(active, t - purchase_m, -1)
                pay_active   = (months_since >= 1) & (months_since <= n_pay)
                current_pays = np.where(pay_active, pmt, 0.0)

                shortfall = current_pays - contr_vec
                shortfall = np.where(shortfall > 0, shortfall, 0.0)
                if np.any(shortfall > 0):
                    tot_before = rf_hold + eq_hold
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rf_ratio_all = np.where(tot_before > 0, rf_hold / tot_before, rf_gp[t])
                        eq_ratio_all = np.where(tot_before > 0, eq_hold / tot_before, gp[t])
                    rf_hold -= shortfall * rf_ratio_all
                    eq_hold -= shortfall * eq_ratio_all
                    rf_hold = np.maximum(rf_hold, 0.0)
                    eq_hold = np.maximum(eq_hold, 0.0)

                contr_vec = np.maximum(0.0, contr_vec - current_pays)
                mortgage_pays[t, :] = current_pays
            else:
                mortgage_pays[t, :] = 0.0
        else:
            mortgage_pays[t, :] = 0.0

        contribs[t, :] = contr_vec

        if rebalance_monthly:
            total_before = rf_hold + eq_hold + contr_vec
            rf_hold = rf_gp[t] * total_before
            eq_hold = gp[t]   * total_before
        else:
            rf_hold += rf_gp[t] * contr_vec
            eq_hold += gp[t]   * contr_vec

        rf_hold *= (1 + rf_m)
        eq_hold *= (1 + eq_rets[t, :])

        paths[t, :] = rf_hold + eq_hold

    month_index = pd.RangeIndex(1, months + 1, name="mês")
    paths_df = pd.DataFrame(paths, index=month_index)
    components_df = pd.DataFrame({
        "mês": np.repeat(np.arange(1, months + 1), n_sims),
        "sim": np.tile(np.arange(1, n_sims + 1), months),
        "total": paths.flatten(order="C"),
        "contribution": contribs.flatten(order="C"),
        "prestacao": mortgage_pays.flatten(order="C"),
        "saque_casa": house_withdraw.flatten(order="C"),
    })

    if enable_house and house_price > 0:
        comprou = purchased
        house_stats["prob_compra"] = float(np.mean(comprou))
        house_stats["mediana_mes_compra"] = int(np.median(purchase_m[comprou])) + 1 if np.any(comprou) else None

    return paths_df, components_df, house_stats

def amortization_table(principal: float, monthly_rate: float, n_pay: int) -> pd.DataFrame:
    if n_pay <= 0 or principal <= 0:
        return pd.DataFrame(columns=["mes", "parcela", "juros", "amortizacao", "saldo"])
    if monthly_rate <= 0:
        pmt = principal / n_pay
        rows, bal = [], principal
        for m in range(1, n_pay + 1):
            j = 0.0
            a = pmt
            bal = max(0.0, bal - a)
            rows.append((m, pmt, j, a, bal))
        return pd.DataFrame(rows, columns=["mes", "parcela", "juros", "amortizacao", "saldo"])
    pmt = principal * monthly_rate / (1 - (1 + monthly_rate) ** (-n_pay))
    rows, bal = [], principal
    for m in range(1, n_pay + 1):
        j = bal * monthly_rate
        a = pmt - j
        bal = max(0.0, bal - a)
        rows.append((m, pmt, j, a, bal))
    return pd.DataFrame(rows, columns=["mes", "parcela", "juros", "amortizacao", "saldo"])

def summarize_simulations(paths_df: pd.DataFrame) -> pd.DataFrame:
    percentiles = [5, 25, 50, 75, 95]
    summary = paths_df.quantile(q=[p / 100 for p in percentiles], axis=1).T
    summary.columns = [f"p{p}" for p in percentiles]
    summary["mean"] = paths_df.mean(axis=1)
    summary["mês"] = summary.index
    summary = summary.set_index("mês")
    return summary

def final_distribution_stats(final_values: np.ndarray, target: Optional[float] = None) -> dict:
    stats = {
        "mean": float(np.mean(final_values)),
        "median": float(np.median(final_values)),
        "p5": float(np.percentile(final_values, 5)),
        "p25": float(np.percentile(final_values, 25)),
        "p75": float(np.percentile(final_values, 75)),
        "p95": float(np.percentile(final_values, 95)),
        "min": float(np.min(final_values)),
        "max": float(np.max(final_values)),
    }
    if target is not None and np.isfinite(target):
        stats["prob_ge_target"] = float(np.mean(final_values >= target))
    else:
        stats["prob_ge_target"] = np.nan
    return stats

def pv_required_for_income(monthly_income: float, years: Optional[int], real_annual_return: float, growth_annual: float = 0.0) -> float:
    r_m = annual_to_monthly_rate(real_annual_return)
    g_m = annual_to_monthly_rate(growth_annual)
    pmt = monthly_income
    if years is None:
        if r_m <= g_m:
            return float("inf")
        return pmt / (r_m - g_m)
    n = int(years * 12)
    if abs(r_m - g_m) < 1e-12:
        return pmt * n / (1 + r_m)
    pv = pmt * (1 - ((1 + g_m) / (1 + r_m)) ** n) / (r_m - g_m)
    return pv

# =============================
# App Streamlit
# =============================

def main():
    st.set_page_config(page_title="Simulador de Aposentadoria", layout="wide")
    st.title("Simulador de Aposentadoria")

    with st.sidebar:
        st.header("Entradas")

        base_salary = st.number_input("Salário mensal atual (R$)", min_value=0.0, value=8000.0, step=500.0)
        curve_name  = st.selectbox("Trajetória de carreira (base)", list(DEFAULT_CURVES.keys()), index=1)
        curve       = DEFAULT_CURVES[curve_name]
        annual_growth0 = st.slider(f"Crescimento anual inicial para {curve.name}", -0.05, 0.25, curve.annual_growth_pct, 0.005)
        half_life      = st.slider("Meia-vida do crescimento (anos)", 1.0, 15.0, 5.0, 0.5)
        growth_floor   = st.slider("Piso do crescimento anual no longo prazo", -0.02, 0.10, 0.02, 0.0025)

        contrib_pct = st.slider("% do salário investido por mês", 0, 100, 25, 1, format="%d%%") / 100

        st.subheader("Alocação (glide path)")
        eq_start = st.slider("RV (Ibovespa) inicial", 0, 100, 60, 1, format="%d%%") / 100
        eq_end   = st.slider("RV (Ibovespa) final",   0, 100, 40, 1, format="%d%%") / 100

        initial_investment = st.number_input("Investimento inicial (R$)", min_value=0.0, value=30000.0, step=1000.0)
        years  = st.slider("Horizonte de investimento (anos)", 1, 40, 20)
        months = years * 12

        st.subheader("Premissas de retorno")
        selic_annual = st.slider("Selic (anual, nominal)", 0.00, 0.20, 0.10, 0.0025, format="%0.2f")

        st.caption("Retornos do Ibovespa (mensais) — carregue CSV (Yahoo/B3) ou edite manualmente.")
        uploaded = st.file_uploader("CSV (Date, Close/Adj Close)", type=["csv"])
        ibov_monthly_returns = None
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded)
                ibov_monthly_returns = robust_monthly_returns_from_csv(df_up)
                st.success(f"Carregado {len(ibov_monthly_returns)} retornos mensais do CSV.")
            except Exception as e:
                st.error(f"Falha ao ler CSV: {e}")

        use_monthly_editor = st.toggle("Editar manualmente (se não carregar CSV)", value=False)
        if ibov_monthly_returns is None:
            if use_monthly_editor:
                seed_monthly = expand_annual_to_monthly_series(DEFAULT_IBOV_LAST10, years_each=1, noise=0.01)
                mdf = pd.DataFrame({"mês": list(range(1, len(seed_monthly)+1)), "retorno_mensal": seed_monthly})
                edited = st.data_editor(
                    mdf, num_rows="dynamic", use_container_width=True, hide_index=True,
                    column_config={"retorno_mensal": st.column_config.NumberColumn(format="%0.4f", step=0.001)}
                )
                ibov_monthly_returns = [float(x) for x in edited["retorno_mensal"].tolist()]
            else:
                ibov_monthly_returns = expand_annual_to_monthly_series(DEFAULT_IBOV_LAST10, years_each=1, noise=0.01)
                st.warning("Usando placeholders mensais (com leve ruído). Prefira CSV real.")

        n_sims  = st.slider("Número de simulações", 100, 20000, 5000, 100)
        seed_opt= st.text_input("Seed aleatória (opcional)", value="")
        seed    = int(seed_opt) if seed_opt.strip().isdigit() else None

        advanced = st.expander("Opções avançadas")
        with advanced:
            rebalance  = st.toggle("Rebalancear mensalmente conforme glide path", value=True)
            block_size = st.slider("Tamanho do bloco (MBB)", 1, 12, 3, 1, help="Blocos maiores preservam mais dependência temporal.")
            show_salary= st.toggle("Mostrar gráficos de salário e contribuições", value=True)

        st.subheader("Habitação (opcional)")
        enable_house = st.toggle("Ativar compra de casa", value=False)
        house_params: Dict[str, float | int | str] = {}
        if enable_house:
            house_price = st.number_input("Valor da casa (R$)", min_value=0.0, value=500_000.0, step=10_000.0)
            house_mode  = st.radio("Modo de compra", ["à vista quando houver caixa", "financiada (dívida)"])
            if "financiada" in house_mode:
                loan_pct    = st.slider("Percentual financiado", 0, 100, 80, 5, format="%d%%") / 100
                loan_spread = st.slider("Spread anual sobre a Selic", -0.05, 0.30, 0.03, 0.0025, format="%0.3f")
                loan_years  = st.slider("Prazo (anos)", 5, 35, 30, 1)
                annual_rate = max(0.0, selic_annual + loan_spread)
                loan_m      = annual_to_monthly_rate(annual_rate)
                principal   = house_price * loan_pct
                n_pay       = int(loan_years * 12)
                pmt         = principal * loan_m / (1 - (1 + loan_m) ** (-n_pay)) if (loan_m > 0 and n_pay > 0) else (principal / n_pay if n_pay > 0 else 0.0)
                st.info(f"Parcela estimada: R$ {pmt:,.2f} | Total pago: R$ {pmt*n_pay:,.0f} | Juros: R$ {(pmt*n_pay - principal):,.0f}")
                house_params.update(dict(house_price=house_price, house_mode="divida", loan_pct=loan_pct, loan_spread=loan_spread, loan_years=loan_years))
            else:
                st.caption("Compra à vista assim que houver caixa suficiente.")
                house_params.update(dict(house_price=house_price, house_mode="avista", loan_pct=0.0, loan_spread=0.0, loan_years=0))
        else:
            house_params.update(dict(house_price=0.0, house_mode="avista", loan_pct=0.0, loan_spread=0.0, loan_years=0))

        st.subheader("Meta de aposentadoria (nominal, viver só dos juros)")
        desired_income = st.number_input("Renda mensal desejada (R$)", min_value=0.0, value=30_000.0, step=1_000.0)
        treat_as_perpetuity   = True
        retire_years_income   = None
        income_growth_real    = 0.0  # ignorado
        r_m_nom = annual_to_monthly_rate(selic_annual)
        if r_m_nom <= 0:
            required_pv = float("inf")
            st.caption("Capital necessário (perpetuidade): **∞** (taxa de juros ≤ 0).")
        else:
            required_pv = desired_income / r_m_nom
            st.caption(f"Capital necessário (perpetuidade; principal preservado): **R$ {required_pv:,.0f}**")

        if n_sims >= 10000 or months >= 360:
            st.info("Dica de performance: se o app ficar lento, reduza o número de simulações ou o horizonte.")

        run_btn = st.button("Rodar simulação", type="primary")

    salary_path = compute_salary_path_decay(
        base_salary=base_salary,
        months=months,
        annual_growth_initial=annual_growth0,
        half_life_years=half_life,
        annual_growth_floor=growth_floor
    )

    tab_overview, tab_dist, tab_assump, tab_house, tab_sens, tab_data = st.tabs(
        ["Visão Geral", "Distribuições", "Premissas", "Habitação", "Sensibilidade", "Dados"]
    )

    if run_btn:
        with st.spinner("Simulando..."):
            paths_df, components_df, house_info = simulate_portfolio(
                months=months,
                n_sims=n_sims,
                selic_annual=selic_annual,
                ibov_monthly_returns=ibov_monthly_returns,
                salary_path=salary_path,
                contrib_pct=contrib_pct,
                initial_investment=initial_investment,
                eq_share_start=eq_start,
                eq_share_end=eq_end,
                rebalance_monthly=rebalance,
                seed=seed,
                block_size=block_size,
                enable_house=enable_house,
                **house_params,
            )

        summary = summarize_simulations(paths_df)

        # ===== Balloon: descontar saldo devedor remanescente na aposentadoria =====
        final_vals = paths_df.iloc[-1, :].to_numpy(dtype=float)
        saldo_remanescente = np.zeros_like(final_vals)
        if enable_house and house_params.get("house_mode") == "divida":
            n_pay = int(house_info.get("n_parcelas", 0) or 0)
            pmt   = float(house_info.get("pmt", 0.0) or 0.0)
            i     = float(house_info.get("taxa_mensal_emprestimo", 0.0) or 0.0)
            P     = float(house_info.get("principal", 0.0) or 0.0)
            if n_pay > 0 and pmt > 0 and P > 0:
                paying_flag = (components_df["prestacao"] > 0).astype(int)
                pagos_por_sim = (
                    components_df.assign(pagando=paying_flag)
                    .groupby("sim", as_index=False)["pagando"].sum()
                    .set_index("sim")
                    .reindex(range(1, final_vals.size + 1))
                    .fillna(0)
                    .astype(int)
                )
                k_vec = pagos_por_sim["pagando"].to_numpy()
                if i > 0:
                    fator = (1.0 + i) ** k_vec
                    saldo = P * fator - pmt * (fator - 1.0) / i
                else:
                    saldo = np.maximum(P - pmt * k_vec, 0.0)
                saldo[k_vec >= n_pay] = 0.0
                saldo_remanescente = saldo
        final_vals = final_vals - saldo_remanescente
        pend = np.mean(saldo_remanescente > 0) * 100
        if pend > 0:
            st.markdown(
                f"**Nota:** {pend:.1f}% das simulações chegam à aposentadoria ainda com saldo devedor; "
                f"aplicado *balloon* (desconto do saldo) no patrimônio final."
            )

        stats = final_distribution_stats(final_vals, target=required_pv)

        with tab_overview:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mediana final (R$)", f"{stats['median']:,.0f}")
            c2.metric("Média final (R$)", f"{stats['mean']:,.0f}")
            c3.metric("P5–P95 (R$)", f"{stats['p5']:,.0f} – {stats['p95']:,.0f}")
            prob_txt = "—" if (stats.get("prob_ge_target") is None or not np.isfinite(stats["prob_ge_target"])) else f"{stats['prob_ge_target']*100:0.1f}%"
            c4.metric("P(≥ meta)", prob_txt)

            st.subheader("Portfólio ao longo do tempo")
            fan = summary.reset_index().melt(id_vars=["mês"], value_vars=["p5", "p25", "p50", "p75", "p95"],
                                             var_name="percentil", value_name="valor")
            fan_chart = (
                alt.Chart(fan)
                .mark_line()
                .encode(
                    x=alt.X("mês:Q", title="Mês"),
                    y=alt.Y("valor:Q", title="Valor do Portfólio (R$)", scale=alt.Scale(zero=False)),
                    color="percentil:N",
                )
                .properties(height=380)
            )
            st.altair_chart(fan_chart, use_container_width=True)

            st.markdown(f"**Glide path:** RV {int(eq_start*100)}% → {int(eq_end*100)}% | **Contribuição:** {int(contrib_pct*100)}% do salário")
            st.markdown("**Todas as métricas estão em valores nominais.** A meta de renda é nominal e assume investimento 100% na taxa de juros indicada (Selic) durante a aposentadoria, preservando o principal.")

            if show_salary:
                st.subheader("Salário e contribuições")
                salary_df = pd.DataFrame({"mês": np.arange(1, months + 1), "salário": salary_path, "contribuição": salary_path * contrib_pct})
                salary_chart = (
                    alt.Chart(salary_df.melt(id_vars=["mês"], value_vars=["salário", "contribuição"], var_name="tipo", value_name="valor"))
                    .mark_line()
                    .encode(x="mês:Q", y="valor:Q", color="tipo:N")
                    .properties(height=260)
                )
                st.altair_chart(salary_chart, use_container_width=True)

        with tab_dist:
            st.subheader("Distribuição final do portfólio")
            hist_df = pd.DataFrame({"valor_final": final_vals})
            _range = float(hist_df["valor_final"].max() - hist_df["valor_final"].min())
            step   = max(_range / 60.0, 10_000.0)  # CORREÇÃO: bins com passo mínimo grande
            hist = (
                alt.Chart(hist_df)
                .transform_bin("bin_final", field="valor_final", bin=alt.Bin(step=step))
                .transform_aggregate(count="count()", groupby=["bin_final"])
                .mark_bar()
                .encode(x=alt.X("bin_final:Q", title="Portfólio final (R$)"),
                        y=alt.Y("count:Q", title="Frequência"))
                .properties(height=320)
            )
            st.altair_chart(hist, use_container_width=True)

            st.subheader("ECDF do valor final")
            vals = np.sort(final_vals)
            ecdf_vals = np.arange(1, len(vals) + 1) / len(vals)
            ecdf_df = pd.DataFrame({"valor_final": vals, "ecdf": ecdf_vals})
            base = (
                alt.Chart(ecdf_df)
                .mark_line()
                .encode(
                    x=alt.X("valor_final:Q", title="Portfólio final (R$)"),
                    y=alt.Y("ecdf:Q", title="F(Valor Final)"),
                    tooltip=[
                        alt.Tooltip("valor_final:Q", title="Portfólio (R$)", format=",.0f"),
                        alt.Tooltip("ecdf:Q",        title="P(≤ x)", format=".1%"),
                        alt.Tooltip("ecdf:Q",        title="P(≥ x) = 1 - P(≤ x)", format=".1%")
                    ]
                )
                .properties(height=240)
            )
            layers = [base]
            if np.isfinite(required_pv):
                p_le = float(np.mean(final_vals <= required_pv))
                p_ge = 1.0 - p_le
                rule = alt.Chart(pd.DataFrame({"x": [required_pv]})).mark_rule(strokeDash=[6,3]).encode(x="x:Q")
                point = alt.Chart(pd.DataFrame({"x":[required_pv], "y":[p_le]})).mark_point(size=60).encode(x="x:Q", y="y:Q")
                label = alt.Chart(pd.DataFrame({"x":[required_pv], "y":[p_le], "txt":[f"P(≥ meta) = {p_ge*100:.1f}%"]})) \
                        .mark_text(align="left", dx=8, dy=-8) \
                        .encode(x="x:Q", y="y:Q", text="txt:N")
                layers += [rule, point, label]
            st.altair_chart(alt.layer(*layers), use_container_width=True)

            gap = final_vals - (required_pv if np.isfinite(required_pv) else np.nan)
            gap_stats = {
                "prob_atinge_meta": float(np.mean(gap >= 0)) if np.isfinite(required_pv) else np.nan,
                "mediana_gap": float(np.nanmedian(gap)) if np.isfinite(required_pv) else np.nan,
                "p5_gap": float(np.nanpercentile(gap, 5)) if np.isfinite(required_pv) else np.nan,
                "p95_gap": float(np.nanpercentile(gap, 95)) if np.isfinite(required_pv) else np.nan,
            }

            st.subheader("Estatísticas chave")
            df_stats = pd.DataFrame([{**stats, **gap_stats}])

            def fmt_money(s: pd.Series) -> pd.Series:
                return s.apply(lambda x: "" if pd.isna(x) else f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

            def fmt_percent(s: pd.Series) -> pd.Series:
                return s.apply(lambda x: "" if pd.isna(x) else f"{x*100:,.1f}%".replace(",", "X").replace(".", ",").replace("X", "."))

            styled = df_stats.copy()
            money_cols = ["mean", "median", "p5", "p25", "p75", "p95", "min", "max", "mediana_gap", "p5_gap", "p95_gap"]
            prob_cols  = ["prob_ge_target", "prob_atinge_meta"]
            for c in money_cols:
                if c in styled.columns:
                    styled[c] = fmt_money(styled[c])
            for c in prob_cols:
                if c in styled.columns:
                    styled[c] = fmt_percent(styled[c])

            rename_map = {
                "mean": "Média final (R$)",
                "median": "Mediana final (R$)",
                "p5": "P5 (R$)", "p25": "P25 (R$)", "p75": "P75 (R$)", "p95": "P95 (R$)",
                "min": "Mínimo (R$)", "max": "Máximo (R$)",
                "prob_ge_target": "Prob(≥ meta)", "prob_atinge_meta": "Prob(atinge meta)",
                "mediana_gap": "Mediana do gap (R$)", "p5_gap": "P5 do gap (R$)", "p95_gap": "P95 do gap (R$)"
            }
            styled = styled.rename(columns=rename_map)

            base_prob = stats.get("prob_ge_target", np.nan)
            color = "gray"
            if np.isfinite(base_prob):
                if base_prob >= 0.70:
                    color = "#2e7d32"
                elif base_prob >= 0.40:
                    color = "#f9a825"
                else:
                    color = "#c62828"
            st.markdown(
                f"<div style='border-left:8px solid {color}; padding:8px 12px; background:#1111; border-radius:6px;'>"
                f"<b>Status meta:</b> {'' if not np.isfinite(base_prob) else f'{base_prob*100:.1f}%'}"
                f"</div>", unsafe_allow_html=True
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.download_button("Baixar distribuição final CSV",
                               data=hist_df.to_csv(index=False).encode("utf-8"),
                               file_name="distribuicao_final.csv", mime="text/csv")

            st.subheader("Exemplo de caminhos de simulação")
            sample_cols = min(20, paths_df.shape[1])
            sample = paths_df.iloc[:, :sample_cols].copy()
            sample["mês"] = sample.index
            sample_long = sample.melt(id_vars=["mês"], var_name="simulação", value_name="valor")
            line = (
                alt.Chart(sample_long)
                .mark_line(opacity=0.5)
                .encode(x="mês:Q", y=alt.Y("valor:Q", scale=alt.Scale(zero=False)), color="simulação:N")
                .properties(height=300)
            )
            st.altair_chart(line, use_container_width=True)

        with tab_assump:
            st.subheader("Resumo das premissas de retorno")
            eq_monthly = np.array(ibov_monthly_returns, dtype=float)
            mu_m   = float(np.mean(eq_monthly))
            sigma_m= float(np.std(eq_monthly, ddof=1)) if len(eq_monthly) > 1 else 0.0
            mu_a   = monthly_to_annual_rate(mu_m)
            sigma_a= sigma_m * math.sqrt(12)

            a1, a2, a3 = st.columns(3)
            a1.metric("Média Ibovespa (mensal)", f"{mu_m*100:0.2f}%")
            a2.metric("Desv. padrão Ibovespa (mensal)", f"{sigma_m*100:0.2f}%")
            a3.metric("Selic (anual)", f"{selic_annual*100:0.2f}%")

            b1, b2 = st.columns(2)
            b1.metric("Média Ibovespa (anual equiv.)", f"{mu_a*100:0.2f}%")
            b2.metric("Desv. padrão Ibovespa (anual ~)", f"{sigma_a*100:0.2f}%")

            st.markdown("#### Crescimento salarial (com decaimento):")
            st.write(f"Inicial: {annual_growth0*100:0.2f}% | Meia-vida: {half_life:.1f} anos | Piso: {growth_floor*100:0.2f}%")
            st.write(f"Glide path: RV {int(eq_start*100)}% → {int(eq_end*100)}%")

        with tab_house:
            st.subheader("Habitação – métricas")
            if not (enable_house and house_params["house_price"] > 0):
                st.info("Habitação desativada ou sem valor de casa.")
            else:
                st.metric("Prob. de compra no horizonte", f"{house_info['prob_compra']*100:0.1f}%")
                st.metric("Mediana do mês de compra", f"{house_info['mediana_mes_compra'] if house_info['mediana_mes_compra'] is not None else '—'}")
                if house_params["house_mode"] == "divida":
                    st.metric("Parcela estimada (fixa)", f"R$ {house_info['pmt']:,.2f}")
                    st.metric("Nº de parcelas", f"{house_info['n_parcelas']}")
                    st.metric("Total pago (aprox.)", f"R$ {house_info['total_pago_por_contrato']:,.0f}")
                    st.metric("Juros totais (aprox.)", f"R$ {house_info['juros_totais_por_contrato']:,.0f}")

                    st.subheader("Tabela de amortização (contrato)")
                    if house_info["amort_schedule"] is not None:
                        amort = house_info["amort_schedule"].copy()
                        show_amort = amort.copy()
                        for col in ["parcela", "juros", "amortizacao", "saldo"]:
                            show_amort[col] = show_amort[col].map(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                        st.dataframe(show_amort, use_container_width=True, hide_index=True, height=300)
                        st.download_button("Baixar tabela de amortização (CSV)",
                                           data=amort.to_csv(index=False).encode("utf-8"),
                                           file_name="amortizacao.csv", mime="text/csv")
                else:
                    st.metric("Valor da compra à vista", f"R$ {house_params['house_price']:,.0f}")

                st.subheader("Percentual de simulações com prestação ativa por mês")
                if house_params.get("house_mode") == "divida":
                    paying_flag = components_df.copy()
                    paying_flag["pagando"] = (paying_flag["prestacao"] > 0).astype(int)
                    paying_share = paying_flag.groupby("mês", as_index=False)["pagando"].mean()
                    paying_share["pct"] = paying_share["pagando"] * 100.0
                    if paying_share["pct"].max() == 0:
                        st.info("Nenhuma simulação entrou em fase de pagamento dentro do horizonte.")
                    else:
                        paying_chart = (
                            alt.Chart(paying_share)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X("mês:Q", title="Mês"),
                                y=alt.Y("pct:Q", title="% de simulações com parcela ativa", scale=alt.Scale(domain=[0, 100])),
                                tooltip=[alt.Tooltip("mês:Q"), alt.Tooltip("pct:Q", format=".1f")]
                            )
                            .properties(height=240)
                        )
                        st.altair_chart(paying_chart, use_container_width=True)
                else:
                    st.info("Financiamento não selecionado — gráfico de prestação ativa não se aplica.")

                if house_params.get("house_mode") == "avista":
                    st.subheader("Probabilidade acumulada de compra (à vista)")
                    # CORREÇÃO: cálculo direto do primeiro mês de compra por simulação
                    first_purchase = (
                        components_df.loc[components_df["saque_casa"] > 0, ["sim", "mês"]]
                        .groupby("sim", as_index=False)["mês"]
                        .min()
                    )
                    if first_purchase.empty:
                        st.info("Nenhuma simulação atingiu o valor da casa no horizonte — probabilidade de compra nula.")
                    else:
                        total_sims = components_df["sim"].nunique()
                        cdf = (
                            first_purchase["mês"]
                            .value_counts()
                            .sort_index()
                            .rename_axis("mês")
                            .reset_index(name="compras_no_mes")
                        )
                        cdf["compras_acumuladas"] = cdf["compras_no_mes"].cumsum()
                        cdf["prob_acumulada"] = cdf["compras_acumuladas"] / total_sims
                        chart = (
                            alt.Chart(cdf)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X("mês:Q", title="Mês"),
                                y=alt.Y("prob_acumulada:Q", title="Probabilidade acumulada de compra", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                                tooltip=[alt.Tooltip("mês:Q", title="Mês"), alt.Tooltip("prob_acumulada:Q", title="Prob. acumulada", format=".1%")],
                            )
                            .properties(height=240)
                        )
                        st.altair_chart(chart, use_container_width=True)
                else:
                    st.caption("Compra financiada: gráfico de probabilidade omitido. Use as métricas acima.")

        with tab_sens:
            st.subheader("Análise de sensibilidade (em breve)")
            st.caption("Varia cada parâmetro ± e mede impacto em Prob(≥ meta) e Mediana final. Roda com menos simulações para agilidade.")

        with tab_data:
            st.subheader("Retornos mensais do Ibovespa usados")
            st.dataframe(pd.DataFrame({"retorno_mensal": ibov_monthly_returns}), use_container_width=True)
            st.subheader("Parâmetros da simulação")
            st.json({
                "salário_base": base_salary,
                "crescimento_inicial_anual": annual_growth0,
                "meia_vida_anos": half_life,
                "piso_crescimento_anual": growth_floor,
                "pct_contribuição": contrib_pct,
                "glide_path": {"eq_start": eq_start, "eq_end": eq_end},
                "investimento_inicial": initial_investment,
                "anos": years,
                "meses": months,
                "selic_anual": selic_annual,
                "n_sims": n_sims,
                "rebalancear_mensalmente": rebalance,
                "block_size": block_size,
                "seed": seed,
                "habitação": house_params,
                "meta": {
                    "modo": "perpetuidade_nominal",
                    "renda_mensal": desired_income,
                    "anos": retire_years_income,
                    "taxa_nominal_aposentadoria": selic_annual,
                    "PV_necessário": None if not np.isfinite(required_pv) else round(required_pv, 2),
                },
            })

    else:
        with tab_overview:
            st.info("Defina entradas, carregue retornos mensais (CSV) e clique em **Rodar simulação**.")

if __name__ == "__main__":
    main()
