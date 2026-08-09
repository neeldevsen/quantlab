
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>
#include <pybind11/pybind11.h>
#include "black_scholes_struct.hpp"

namespace py = pybind11;

template <typename S_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
auto trinomialTreeBSCall(const OptionDataBS<S_t, K_t, r_t, sigma_t, T_t>& data, const int N) -> decltype(data.spot)
{
    using commonType = decltype(data.spot);

    auto sqrt_2deltat {std::sqrt(2 * data.maturity / static_cast<commonType>(N))};

    auto u {std::exp(data.volatility * sqrt_2deltat)};
    auto d {std::exp(-data.volatility * sqrt_2deltat)};
    auto p_u {std::pow((std::exp(data.rate * data.maturity / (2 * N)) - std::exp(-data.volatility * sqrt_2deltat / static_cast<commonType>(2))) / (std::exp(data.volatility * sqrt_2deltat / static_cast<commonType>(2)) - std::exp(-data.volatility * sqrt_2deltat / static_cast<commonType>(2))), static_cast<commonType>(2))};
    auto p_d {std::pow((std::exp(data.volatility * sqrt_2deltat / static_cast<commonType>(2)) - std::exp(data.rate * data.maturity / (2 * N))) / (std::exp(data.volatility * sqrt_2deltat / static_cast<commonType>(2)) - std::exp(-data.volatility * sqrt_2deltat / static_cast<commonType>(2))), static_cast<commonType>(2))};
    auto p_m {static_cast<commonType>(1) - p_u - p_d};

    std::vector<commonType> V_current (2 * N + 1);
    for (int j {}; j < 2 * N + 1; ++j)
    {
        V_current[j] = std::max(data.spot * std::pow(static_cast<commonType>(u), static_cast<commonType>(j-N)) - data.strike, static_cast<commonType>(0));
    }

    for (int i {}; i < N; ++i)
    {
        std::vector<commonType> V_new (2 * N - 2 * i - 1);
        for (int k {}; k < 2 * N - 2 * i - 1; ++k)
        {
            V_new[k] = p_u * V_current[k+2] + p_m * V_current[k+1] + p_d * V_current[k];
        }
        std::swap(V_current, V_new);
    }

    return V_current[0] * std::exp(-data.rate * data.maturity);
}

template <typename S_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
auto trinomialTreeBSPut(const OptionDataBS<S_t, K_t, r_t, sigma_t, T_t>& data, const int N) -> decltype(data.spot)
{
    using commonType = decltype(data.spot);

    auto sqrt_2deltat {std::sqrt(2 * data.maturity / static_cast<commonType>(N))};

    auto u {std::exp(data.volatility * sqrt_2deltat)};
    auto d {std::exp(-data.volatility * sqrt_2deltat)};
    auto p_u {std::pow((std::exp(data.rate * data.maturity / (2 * N)) - std::exp(-data.volatility * sqrt_2deltat / static_cast<commonType>(2))) / (std::exp(data.volatility * sqrt_2deltat / static_cast<commonType>(2)) - std::exp(-data.volatility * sqrt_2deltat / static_cast<commonType>(2))), static_cast<commonType>(2))};
    auto p_d {std::pow((std::exp(data.volatility * sqrt_2deltat / static_cast<commonType>(2)) - std::exp(data.rate * data.maturity / (2 * N))) / (std::exp(data.volatility * sqrt_2deltat / static_cast<commonType>(2)) - std::exp(-data.volatility * sqrt_2deltat / static_cast<commonType>(2))), static_cast<commonType>(2))};
    auto p_m {static_cast<commonType>(1) - p_u - p_d};

    std::vector<commonType> V_current (2 * N + 1);
    for (int j {}; j < 2 * N + 1; ++j)
    {
        V_current[j] = std::max(data.strike - data.spot * std::pow(static_cast<commonType>(u), static_cast<commonType>(j-N)), static_cast<commonType>(0));
    }

    for (int i {}; i < N; ++i)
    {
        std::vector<commonType> V_new (2 * N - 2 * i - 1);
        for (int k {}; k < 2 * N - 2 * i - 1; ++k)
        {
            V_new[k] = p_u * V_current[k+2] + p_m * V_current[k+1] + p_d * V_current[k];
        }
        std::swap(V_current, V_new);
    }

    return V_current[0] * std::exp(-data.rate * data.maturity);
}

PYBIND11_MODULE(bs_trinomial, m)
{
    m.doc() = "Black-Scholes trinomial tree pricing engine";

    m.def("call",
        [](double spot, double strike, double rate, double volatility, double maturity, int N)
        {
            OptionDataBS option {spot, strike, rate, volatility, maturity};
            return trinomialTreeBSCall(option, N);
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
            return trinomialTreeBSPut(option, N);
        },
        py::arg("spot"),
        py::arg("strike"),
        py::arg("rate"),
        py::arg("volatility"),
        py::arg("maturity"),
        py::arg("N")
    );
}

