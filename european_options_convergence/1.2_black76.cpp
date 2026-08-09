
#include <cmath>
#include <type_traits>
#include <pybind11/pybind11.h>
#include "functions.hpp"
#include "black76_structs.hpp"

namespace py = pybind11;

template <typename F_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
auto black76Call(const OptionDataB76<F_t, K_t, r_t, sigma_t, T_t>& data) -> decltype(data.future)
{
    using commonType = decltype(data.future);
    auto sqrt_maturity {std::sqrt(data.maturity)};

    auto d1 {(std::log(data.future / data.strike) + (static_cast<commonType>(0.5) * data.volatility * data.volatility) * data.maturity) / (data.volatility * sqrt_maturity)};
    auto d2 {d1 - data.volatility * sqrt_maturity};

    auto C {std::exp(-data.rate * data.maturity) * (data.future * normalCDF(d1) - data.strike * normalCDF(d2))};
    return C;
}

template <typename F_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
auto black76Put(const OptionDataB76<F_t, K_t, r_t, sigma_t, T_t>& data) -> decltype(data.future)
{
    using commonType = decltype(data.future);
    auto sqrt_maturity {std::sqrt(data.maturity)};

    auto d1 {(std::log(data.future / data.strike) + (static_cast<commonType>(0.5) * data.volatility * data.volatility) * data.maturity) / (data.volatility * sqrt_maturity)};
    auto d2 {d1 - data.volatility * sqrt_maturity};

    auto P {std::exp(-data.rate * data.maturity) * (data.strike * normalCDF(-d2) - data.future * normalCDF(-d1))};
    return P;
}

PYBIND11_MODULE(black76, m)
{
    m.doc() = "Black-76 closed-form pricing engine";

    m.def("call",
        [](double future, double strike, double rate, double volatility, double maturity)
        {
            OptionDataB76 option {future, strike, rate, volatility, maturity};
            return black76Call(option);
        },
        py::arg("future"),
        py::arg("strike"),
        py::arg("rate"),
        py::arg("volatility"),
        py::arg("maturity")
    );

    m.def("put",
        [](double future, double strike, double rate, double volatility, double maturity)
        {
            OptionDataB76 option {future, strike, rate, volatility, maturity};
            return black76Put(option);
        },
        py::arg("future"),
        py::arg("strike"),
        py::arg("rate"),
        py::arg("volatility"),
        py::arg("maturity")
    );
}

