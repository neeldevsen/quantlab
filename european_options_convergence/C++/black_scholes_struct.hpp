#ifndef GBMOPTIONS
#define GBMOPTIONS
#include <type_traits>

template <typename S_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
struct OptionDataBS
{
    using CommonType = std::common_type_t <double, S_t, K_t, r_t, sigma_t, T_t>;
    CommonType spot {};
    CommonType strike {};
    CommonType rate {};
    CommonType volatility {};
    CommonType maturity {};
};

template <typename S_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
OptionDataBS(S_t, K_t, r_t, sigma_t, T_t) -> OptionDataBS<S_t, K_t, r_t, sigma_t, T_t>;
#endif