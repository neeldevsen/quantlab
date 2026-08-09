#ifndef MERTONSTRUCTS
#define MERTONSTRUCTS
#include <type_traits>

template <typename S_t, typename K_t, typename r_t, typename q_t, typename sigma_t, typename T_t>

struct OptionDataMerton
{
    using CommonType = std::common_type_t <double, S_t, K_t, r_t, q_t, sigma_t, T_t>;
    CommonType spot {};
    CommonType strike {};
    CommonType rate {};
    CommonType dividend {};
    CommonType volatility {};
    CommonType maturity {};
};

template <typename S_t, typename K_t, typename r_t, typename q_t, typename sigma_t, typename T_t>
OptionDataMerton(S_t, K_t, r_t, q_t, sigma_t, T_t) -> OptionDataMerton<S_t, K_t, r_t, q_t, sigma_t, T_t>;

#endif