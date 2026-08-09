#ifndef BLACK76
#define BLACK76
#include <type_traits>

template <typename F_t, typename K_t, typename r_t, typename sigma_t, typename T_t>

struct OptionDataB76
{
    using commonType = std::common_type_t <double, F_t, K_t, r_t, sigma_t, T_t>;
    commonType future {};
    commonType strike {};
    commonType rate {};
    commonType volatility {};
    commonType maturity {};
};

template <typename F_t, typename K_t, typename r_t, typename sigma_t, typename T_t>
OptionDataB76(F_t, K_t, r_t, sigma_t, T_t) -> OptionDataB76<F_t, K_t, r_t, sigma_t, T_t>;

#endif