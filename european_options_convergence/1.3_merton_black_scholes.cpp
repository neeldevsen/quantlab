
#include <cmath>
#include <type_traits>
#include <pybind11/pybind11.h>
#include "functions.hpp"
#include "merton_struct.hpp"

namespace py = pybind11;

template <typename S_t, typename K_t, typename r_t, typename q_t, typename sigma_t, typename T_t>
auto bsDividendCall(const OptionDataMerton<S_t, K_t, r_t, q_t, sigma_t, T_t>& data) -> decltype(data.spot)
{
    using CommonType = decltype(data.spot);
    auto sqrt_maturity {std::sqrt(data.maturity)};

    auto d1 {(std::log(data.spot / data.strike) + (data.rate - data.dividend + (static_cast<CommonType>(0.5) * data.volatility * data.volatility)) * data.maturity) / (data.volatility * sqrt_maturity)};
    auto d2 {d1 - data.volatility * sqrt_maturity};

    auto C {data.spot * std::exp(-data.dividend * data.maturity) * normalCDF(d1) - data.strike * std::exp(-data.rate * data.maturity) * normalCDF(d2)};
    return C;
}

template <typename S_t, typename K_t, typename r_t, typename q_t, typename sigma_t, typename T_t>
auto bsDividendPut(const OptionDataMerton<S_t, K_t, r_t, q_t, sigma_t, T_t>& data) -> decltype(data.spot)
{
    using CommonType = decltype(data.spot);
    auto sqrt_maturity {std::sqrt(data.maturity)};

    auto d1 {(std::log(data.spot / data.strike) + (data.rate - data.dividend + (static_cast<CommonType>(0.5) * data.volatility * data.volatility)) * data.maturity) / (data.volatility * sqrt_maturity)};
    auto d2 {d1 - data.volatility * sqrt_maturity};

    auto P {data.strike * std::exp(-data.rate * data.maturity) * normalCDF(-d2) - data.spot * std::exp(-data.dividend * data.maturity) * normalCDF(-d1)};
    return P;
}

PYBIND11_MODULE(merton, m)
{
    m.doc() = "Merton dividend-adjusted Black-Scholes pricing engine";

    m.def("call",
        [](double spot, double strike, double rate, double dividend, double volatility, double maturity)
        {
            OptionDataMerton option {spot, strike, rate, dividend, volatility, maturity};
            return bsDividendCall(option);
        },
        py::arg("spot"),
        py::arg("strike"),
        py::arg("rate"),
        py::arg("dividend"),
        py::arg("volatility"),
        py::arg("maturity")
    );

    m.def("put",
        [](double spot, double strike, double rate, double dividend, double volatility, double maturity)
        {
            OptionDataMerton option {spot, strike, rate, dividend, volatility, maturity};
            return bsDividendPut(option);
        },
        py::arg("spot"),
        py::arg("strike"),
        py::arg("rate"),
        py::arg("dividend"),
        py::arg("volatility"),
        py::arg("maturity")
    );
}

