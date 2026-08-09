
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>
#include <pybind11/pybind11.h>
#include "black76_structs.hpp"

namespace py = pybind11;

template <typename F_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
auto binomialTreeB76Call(const OptionDataB76<F_t, K_t, r_t, sigma_t, T_t>& data, const int N) -> decltype(data.future)
{
    using commonType = decltype(data.future);

    auto sqrt_deltat {std::sqrt(data.maturity / static_cast<commonType>(N))};

    auto u {std::exp(data.volatility * sqrt_deltat)};
    auto d {std::exp(-data.volatility * sqrt_deltat)};
    auto p {(1 - d) / (u - d)};

    std::vector<commonType> V_current (N + 1);
    for (int j {}; j < N + 1; ++j)
    {
        V_current[j] = std::max(data.future * std::pow(static_cast<commonType>(u), static_cast<commonType>(j)) * std::pow(static_cast<commonType>(d), static_cast<commonType>(N-j)) - data.strike, static_cast<commonType>(0));
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

template <typename F_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
auto binomialTreeB76Put(const OptionDataB76<F_t, K_t, r_t, sigma_t, T_t>& data, const int N) -> decltype(data.future)
{
    using commonType = decltype(data.future);

    auto sqrt_deltat {std::sqrt(data.maturity / static_cast<commonType>(N))};

    auto u {std::exp(data.volatility * sqrt_deltat)};
    auto d {std::exp(-data.volatility * sqrt_deltat)};
    auto p {(1 - d) / (u - d)};

    std::vector<commonType> V_current (N + 1);
    for (int j {}; j < N + 1; ++j)
    {
        V_current[j] = std::max(data.strike - data.future * std::pow(static_cast<commonType>(u), static_cast<commonType>(j)) * std::pow(static_cast<commonType>(d), static_cast<commonType>(N-j)), static_cast<commonType>(0));
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

PYBIND11_MODULE(b76_binomial, m)
{
    m.doc() = "Black-76 binomial tree pricing engine";

    m.def("call",
        [](double future, double strike, double rate, double volatility, double maturity, int N)
        {
            OptionDataB76 option {future, strike, rate, volatility, maturity};
            return binomialTreeB76Call(option, N);
        },
        py::arg("future"),
        py::arg("strike"),
        py::arg("rate"),
        py::arg("volatility"),
        py::arg("maturity"),
        py::arg("N")
    );

    m.def("put",
        [](double future, double strike, double rate, double volatility, double maturity, int N)
        {
            OptionDataB76 option {future, strike, rate, volatility, maturity};
            return binomialTreeB76Put(option, N);
        },
        py::arg("future"),
        py::arg("strike"),
        py::arg("rate"),
        py::arg("volatility"),
        py::arg("maturity"),
        py::arg("N")
    );
}

