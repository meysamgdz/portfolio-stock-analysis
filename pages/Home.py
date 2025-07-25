import streamlit as st

st.set_page_config(page_title="Stock & Portfolio App", layout="centered")

st.title("📊 Stock & Portfolio Analyzer")

st.markdown("""
This app helps you:
- 🔍 Analyze individual stock performance using statistical modeling of normal and fat-tail distributions 
(powerlaw and lognormal) 
""")

# One-line LaTeX formulas
st.latex(r"""
\begin{aligned}
&\text{Normal:} && f_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}} \\
&\text{Lognormal:} && f_X(x) = \frac{1}{x\sigma\sqrt{2\pi}} e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}} \\
&\text{Power-law:} && f_X(x) = \mu x_{min}^{\mu}x^{-(\mu+1)}, \quad \text{or} \\
&&&  f_X(x) = \frac{\alpha-1}{x_{min}}\left(\frac{x}{x_{min}}\right)^{-\alpha}
\end{aligned}
""")
st.markdown(r"**NOTE 1**: the first powerlaw formulation is what is used in the portfolio analysis. However, "
            "the powerlaw package in python has been developed based on the second notation. Therefore, wherever "
            "needed in the code, there will be transformations in form of $\mu = \\alpha - 1$. Furthermore, $x_{min}$ is "
            "sometimes replaced by $A$ in the literature.")

st.markdown("**NOTE 2**: The concepts and module developed in this section rely on "
            "[Statistical Consequences of Fat Tails (Nassim Taleb)]"
            "(https://codowd.com/bigdata/misc/Taleb_Statistical_Consequences_of_Fat_Tails.pdf)")

st.markdown("""
- 🧠 Run Markowitz-based portfolio optimization for distribution-aware investment, In particular, we will solve variations of the following fundamental optimization problem 
    (see [Boyd](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), page 155):
""")

st.latex(r"""
\begin{aligned}
    &\text{minimize}\ && w^T\Sigma w \\
    &\text{subject to}\ && \bar{r}w\geq r_{\min}, \\
    &&& \mathbf{1}^T w = 1,\quad w \succeq 0
\end{aligned}
""")
st.markdown("""where
- $w_i$ denotes **the amount of asset** $i$ held throughout the period,
- $r_i$ denote the **relative price change of asset** $i$ over the period,
- and variance $\mathrm{Var}(r) = w^T \Sigma w$
""")

st.markdown("**NOTE 3**: The conceptual explanations and implementations throughout the code rely on the book "
            "[Theory of Financial Risks and Derivative Pricing (Jean-Philippe Bouchaud)]"
            "(https://opac.feb.uinjkt.ac.id/repository/bc6c1f019d05503c7e1ec4dfde4f6dc6.pdf)")

st.markdown("Use the **sidebar** to navigate between pages.")
