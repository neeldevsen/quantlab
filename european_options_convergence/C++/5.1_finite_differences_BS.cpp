
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>
#include <pybind11/pybind11.h>
#include "functions.hpp"
#include "black_scholes_struct.hpp"

namespace py = pybind11;

template <typename S_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
struct Coefficients
{
    using commonType = std::common_type_t<S_t, K_t, r_t, sigma_t, T_t>;
    std::vector<commonType> alphas{};
    std::vector<commonType> betas{};
    std::vector<commonType> gammas{};

    explicit Coefficients(int M)
    : alphas(M-1), betas(M-1), gammas(M-1)
    {}
};

template <typename S_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
struct Diagonals
{
    using commonType = std::common_type_t<S_t, K_t, r_t, sigma_t, T_t>;
    std::vector<commonType> A_lower {};
    std::vector<commonType> A_main {};
    std::vector<commonType> A_upper {};

    explicit Diagonals(int M)
    : A_lower(M-2), A_main(M-1), A_upper(M-2)
    {}
};

template <typename S_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
Coefficients<S_t, K_t, r_t, sigma_t, T_t> generateCoefficientsBS(const OptionDataBS<S_t, K_t, r_t, sigma_t, T_t>& data, const auto deltaS, const int M)
{
    Coefficients<S_t, K_t, r_t, sigma_t, T_t> coeff {M};
    using commonType = decltype(data.spot);

    for (int i{0}; i < M-1; ++i)
    {
        commonType S_i {(i+1)*deltaS};
        coeff.alphas[i] = 0.5 * (std::pow(data.volatility*S_i, 2.0)) / (std::pow(deltaS, 2.0)) - (data.rate*S_i) / (2*deltaS);
        coeff.betas[i] = -(std::pow(data.volatility*S_i, 2.0)) / (std::pow(deltaS, 2.0)) - data.rate;
        coeff.gammas[i] = 0.5 * (std::pow(data.volatility*S_i, 2.0)) / (std::pow(deltaS, 2.0)) + (data.rate*S_i) / (2*deltaS);
    }

    return coeff;
}

template <typename S_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
Diagonals<S_t, K_t, r_t, sigma_t, T_t> generateDiagonalsBS(const Coefficients<S_t, K_t, r_t, sigma_t, T_t>& coeff, const auto deltaT, const int M)
{
    Diagonals<S_t, K_t, r_t, sigma_t, T_t> diag {M};

    for (int i {0}; i < M-2; ++i)
    {
        diag.A_main[i] = 1 - 0.5 * deltaT * coeff.betas[i];
        diag.A_lower[i] = -0.5 * deltaT * coeff.alphas[i+1];
        diag.A_upper[i] = -0.5 * deltaT * coeff.gammas[i];
    }

    diag.A_main[M-2] = 1 - 0.5 * deltaT * coeff.betas[M-2];

    return diag;
}

template <typename S_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
auto generateRHSCall(const OptionDataBS<S_t, K_t, r_t, sigma_t, T_t>& data, Coefficients<S_t, K_t, r_t, sigma_t, T_t>& coeff, auto V, const int n, const int M, const auto deltaT, const auto Smax)
{
    using commonType = decltype(data.spot);

    V[M] = Smax - data.strike * std::exp(-data.rate*static_cast<commonType>(n)*deltaT);
    V[0] = 0;

    std::vector<commonType> RHS (M-1);

    for (int i{}; i < M-1; ++i)
    {
        RHS[i] = static_cast<commonType>(0.5)*deltaT*coeff.alphas[i]*V[i] + (static_cast<commonType>(1) + static_cast<commonType>(0.5)*deltaT*coeff.betas[i])*V[i+1] + static_cast<commonType>(0.5)*deltaT*coeff.gammas[i]*V[i+2];
    }

    RHS[M-2] += static_cast<commonType>(0.5) * deltaT * coeff.gammas[M-2] * (Smax - data.strike * std::exp(-data.rate * (static_cast<commonType>(n) + static_cast<commonType>(1)) * deltaT));

    return RHS;
}

template <typename S_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
auto generateRHSPut(const OptionDataBS<S_t, K_t, r_t, sigma_t, T_t>& data, Coefficients<S_t, K_t, r_t, sigma_t, T_t>& coeff, auto V, const int n, const int M, const auto deltaT, const auto Smax)
{
    using commonType = decltype(data.spot);

    V[0] = data.strike * std::exp(-data.rate*static_cast<commonType>(n)*deltaT);
    V[M] = 0;

    std::vector<commonType> RHS (M-1);

    for (int i{0}; i < M-1; ++i)
    {
        RHS[i] = static_cast<commonType>(0.5)*deltaT*coeff.alphas[i]*V[i] + (static_cast<commonType>(1) + static_cast<commonType>(0.5)*deltaT*coeff.betas[i])*V[i+1] + static_cast<commonType>(0.5)*deltaT*coeff.gammas[i]*V[i+2];
    }

    RHS[0] += static_cast<commonType>(0.5) * deltaT * coeff.alphas[0] * (data.strike * std::exp(-data.rate * (static_cast<commonType>(n) + static_cast<commonType>(1)) * deltaT));

    return RHS;
}

template <typename S_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
auto thomasAlgorithm(Diagonals<S_t, K_t, r_t, sigma_t, T_t>& diag, const auto& RHS, const int M)
{
    using commonType = std::remove_reference_t<decltype(diag.A_upper[0])>;

    std::vector<commonType> rho_p (M-2);
    std::vector<commonType> mu_p (M-1);

    rho_p[0] = diag.A_upper[0] / diag.A_main[0];
    mu_p[0] = RHS[0] / diag.A_main[0];

    for (int i {1}; i < M-2; ++i)
    {
        mu_p[i] = (RHS[i] / diag.A_main[i] - diag.A_lower[i-1] / diag.A_main[i] * mu_p[i-1]) / (static_cast<commonType>(1) - diag.A_lower[i-1] / diag.A_main[i] * rho_p[i-1]);
        rho_p[i] = diag.A_upper[i] / diag.A_main[i] / (static_cast<commonType>(1) - diag.A_lower[i-1] / diag.A_main[i] * rho_p[i-1]);
    }

    mu_p[M-2] = (RHS[M-2] / diag.A_main[M-2] - diag.A_lower[M-3] / diag.A_main[M-2] * mu_p[M-3]) / (static_cast<commonType>(1) - diag.A_lower[M-3] / diag.A_main[M-2] * rho_p[M-3]);

    std::vector<commonType> Vn (M);
    Vn[M-1] = mu_p[M-2];

    for (int i{M-2}; i >= 1; --i)
    {
        Vn[i] = mu_p[i-1] - rho_p[i-1] * Vn[i+1];
    }

    return Vn;
}

template <typename S_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
auto stepCrankNicolsonCall(const OptionDataBS<S_t, K_t, r_t, sigma_t, T_t>& data, Coefficients<S_t, K_t, r_t, sigma_t, T_t>& coeff, Diagonals<S_t, K_t, r_t, sigma_t, T_t>& diag, const auto deltaT, const int M, const int n, const auto Smax, auto V)
{
    using commonType = decltype(data.spot);

    std::vector RHS {generateRHSCall(data, coeff, V, n, M, deltaT, Smax)};
    std::vector V_next {thomasAlgorithm(diag, RHS, M)};

    V_next[0] = 0;
    V_next.push_back(Smax - data.strike * std::exp(-data.rate*deltaT*static_cast<commonType>(n+1)));

    return V_next;
}

template <typename S_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
auto stepCrankNicolsonPut(const OptionDataBS<S_t, K_t, r_t, sigma_t, T_t>& data, Coefficients<S_t, K_t, r_t, sigma_t, T_t>& coeff, Diagonals<S_t, K_t, r_t, sigma_t, T_t>& diag, const auto deltaT, const int M, const int n, const auto Smax, auto V)
{
    using commonType = decltype(data.spot);

    std::vector RHS {generateRHSPut(data, coeff, V, n, M, deltaT, Smax)};
    std::vector V_next {thomasAlgorithm(diag, RHS, M)};

    V_next[0] = data.strike * std::exp(-data.rate*deltaT*static_cast<commonType>(n+1));
    V_next.push_back(0);

    return V_next;
}

template <typename S_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
auto finiteDifferencesCallBS(const OptionDataBS<S_t, K_t, r_t, sigma_t, T_t>& data, const int M, const int N, const double multiplier=4.0)
{
    using commonType = decltype(data.spot);

    commonType Smax {multiplier * std::max(data.spot,data.strike)};
    commonType deltaS {Smax / static_cast<commonType>(M)};
    commonType deltaT {data.maturity / static_cast<commonType>(N)};

    std::vector<commonType> S (M + 1);
    for (int i{}; i < M + 1; ++i)
    {
        S[i] = static_cast<commonType>(i) * deltaS;
    }

    std::vector<commonType> V (M + 1);
    for (int i{}; i < M + 1; ++i)
    {
        V[i] = std::max(S[i] - data.strike, static_cast<commonType>(0));
    }

    Coefficients coeff {generateCoefficientsBS(data, deltaS, M)};
    Diagonals diag {generateDiagonalsBS(coeff, deltaT, M)};

    for (int n{}; n < N; ++n)
    {
        auto V_new {stepCrankNicolsonCall(data, coeff, diag, deltaT, M, n, Smax, V)};
        std::swap(V_new, V);
    }

    int j {static_cast<int>(data.spot / deltaS)};
    j = std::max(static_cast<int>(0), static_cast<int>(std::min(M-1, static_cast<int>(j))));

    double price {V[j] + (data.spot - S[j]) * (V[j+1] - V[j]) / (S[j+1] - S[j])};

    return price;
}

template <typename S_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
auto finiteDifferencesPutBS(const OptionDataBS<S_t, K_t, r_t, sigma_t, T_t>& data, const int M, const int N, const double multiplier=4.0)
{
    using commonType = decltype(data.spot);

    commonType Smax {multiplier * std::max(data.spot,data.strike)};
    commonType deltaS {Smax / static_cast<commonType>(M)};
    commonType deltaT {data.maturity / static_cast<commonType>(N)};

    std::vector<commonType> S (M + 1);
    for (int i{}; i < M + 1; ++i)
    {
        S[i] = static_cast<commonType>(i) * deltaS;
    }

    std::vector<commonType> V (M + 1);
    for (int i{}; i < M + 1; ++i)
    {
        V[i] = std::max(data.strike - S[i], static_cast<commonType>(0));
    }

    Coefficients coeff {generateCoefficientsBS(data, deltaS, M)};
    Diagonals diag {generateDiagonalsBS(coeff, deltaT, M)};

    for (int n{}; n < N; ++n)
    {
        auto V_new {stepCrankNicolsonPut(data, coeff, diag, deltaT, M, n, Smax, V)};
        std::swap(V_new, V);
    }

    int j {static_cast<int>(data.spot / deltaS)};
    j = std::max(static_cast<int>(0), static_cast<int>(std::min(M-1, static_cast<int>(j))));

    double price {V[j] + (data.spot - S[j]) * (V[j+1] - V[j]) / (S[j+1] - S[j])};

    return price;
}

PYBIND11_MODULE(bs_fdm, m)
{
    m.doc() = "Black-Scholes Crank-Nicolson finite difference pricing engine";

    m.def("call",
        [](double spot, double strike, double rate, double volatility, double maturity, int M, int N, double multiplier)
        {
            OptionDataBS option {spot, strike, rate, volatility, maturity};
            return finiteDifferencesCallBS(option, M, N, multiplier);
        },
        py::arg("spot"),
        py::arg("strike"),
        py::arg("rate"),
        py::arg("volatility"),
        py::arg("maturity"),
        py::arg("M"),
        py::arg("N"),
        py::arg("multiplier") = 4.0
    );

    m.def("put",
        [](double spot, double strike, double rate, double volatility, double maturity, int M, int N, double multiplier)
        {
            OptionDataBS option {spot, strike, rate, volatility, maturity};
            return finiteDifferencesPutBS(option, M, N, multiplier);
        },
        py::arg("spot"),
        py::arg("strike"),
        py::arg("rate"),
        py::arg("volatility"),
        py::arg("maturity"),
        py::arg("M"),
        py::arg("N"),
        py::arg("multiplier") = 4.0
    );
}

