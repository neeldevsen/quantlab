
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>
#include <pybind11/pybind11.h>
#include "black_scholes_struct.hpp"

namespace py = pybind11;

template <typename S_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
auto binomialTreeBSCall(const OptionDataBS<S_t, K_t, r_t, sigma_t, T_t>& data, const int N) -> decltype(data.spot)
{
    using commonType = decltype(data.spot);

    auto sqrt_deltat {std::sqrt(data.maturity / static_cast<commonType>(N))};

    auto u {std::exp(data.volatility * sqrt_deltat)};
    auto d {std::exp(-data.volatility * sqrt_deltat)};
    auto p {(std::exp(data.rate * data.maturity / N) - d) / (u - d)};

    std::vector<commonType> V_current (N + 1);
    for (int j {}; j < N + 1; ++j)
    {
        V_current[j] = std::max(data.spot * std::pow(static_cast<commonType>(u), static_cast<commonType>(j)) * std::pow(static_cast<commonType>(d), static_cast<commonType>(N-j)) - data.strike, static_cast<commonType>(0));
    }

    for (int i {}; i < N; ++i)
    {
        std::vector<commonType> V_new (N-i);
        for (int k {}; k < N-i; ++k)
        {
            V_new[k] = p * V_current[k+1] + (1-p) * V_current[k];
        }
        std::swap(V_current, V_new);
    }

    return V_current[0] * std::exp(-data.rate * data.maturity);
}

template <typename S_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
auto binomialTreeBSPut(const OptionDataBS<S_t, K_t, r_t, sigma_t, T_t>& data, const int N) -> decltype(data.spot)
{
    using commonType = decltype(data.spot);

    auto sqrt_deltat {std::sqrt(data.maturity / static_cast<commonType>(N))};

    auto u {std::exp(data.volatility * sqrt_deltat)};
    auto d {std::exp(-data.volatility * sqrt_deltat)};
    auto p {(std::exp(data.rate * data.maturity / N) - d) / (u - d)};

    std::vector<commonType> V_current (N + 1);
    for (int j {}; j < N + 1; ++j)
    {
        V_current[j] = std::max(data.strike - data.spot * std::pow(static_cast<commonType>(u), static_cast<commonType>(j)) * std::pow(static_cast<commonType>(d), static_cast<commonType>(N-j)), static_cast<commonType>(0));
    }

    for (int i {}; i < N; ++i)
    {
        std::vector<commonType> V_new (N-i);
        for (int k {}; k < N-i; ++k)
        {
            V_new[k] = p * V_current[k+1] + (1-p) * V_current[k];
        }
        std::swap(V_current, V_new);
    }

    return V_current[0] * std::exp(-data.rate * data.maturity);
}

PYBIND11_MODULE(bs_binomial, m)
{
    m.doc() = "Black-Scholes binomial tree pricing engine";

    m.def("call",
        [](double spot, double strike, double rate, double volatility, double maturity, int N)
        {
            OptionDataBS option {spot, strike, rate, volatility, maturity};
            return binomialTreeBSCall(option, N);
        },
        py::arg("spot"),
        py::arg("strike"),
        py::arg("rate"),
        py::arg("volatility"),
        py::arg("maturity"),
        py::arg("N")
    );

    m.def("put",
        [](double spot, double strike, double rate, double volatility, double maturity, int N)
        {
            OptionDataBS option {spot, strike, rate, volatility, maturity};
            return binomialTreeBSPut(option, N);
        },
        py::arg("spot"),
        py::arg("strike"),
        py::arg("rate"),
        py::arg("volatility"),
        py::arg("maturity"),
        py::arg("N")
    );
}

