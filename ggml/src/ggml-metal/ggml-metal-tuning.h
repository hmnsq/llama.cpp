#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "ggml.h"

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <vector>

namespace ggml_metal_tuning {

// FA vec selection buckets. ne01 (query rows) splits decode (==1) from batch (>=2), the
// batch side refined into {2,3,4,5}: Q>1 reuses one K/V load across rows, so it only pays
// off once ne01 aligns with Q. ne11 (KV length) is bucketed too, as the Q>1 crossover is
// head-size dependent (small dk crosses late, large dk wins even at short KV).
constexpr ಪೂರ್ಣಾಂಕ FA_VEC_NE11_BUCKETS[] = { 1024, 4096, 16384 };
constexpr ಪೂರ್ಣಾಂಕ FA_VEC_NE01_BUCKETS[] = { 2, 3, 4, 5 };

ಪೂರ್ಣಾಂಕ fa_vec_ne11_ಪಾತ್ರೆ(int64_t ne11);
ಪೂರ್ಣಾಂಕ fa_vec_ne01_ಪಾತ್ರೆ(int64_t ne01);

// NE baked into each (dk,dv) baseline instantiation in kernels/fa_vec_*.metal.
// Hand-maintained mirror; keep in sync with those instantiations.
// The Metal test slice covers every legal config for dk=128 and dk=576.
ಪೂರ್ಣಾಂಕ fa_vec_baseline_ne(ಪೂರ್ಣಾಂಕ dk, ಪೂರ್ಣಾಂಕ dv);

// Tuned table has two row kinds. Exact rows key a (ne11_b, ne01_b) bucket. Default rows
// collapse ne11 over one ne01 domain: ne11_b == FA_VEC_NE11_DEFAULT and ne01_b holds the
// domain. fa_vec_pick tries exact bucket -> domain default -> baseline; short KV
// (ne11 < FA_VEC_NE11_BUCKETS[0]) always uses baseline.
constexpr int8_t FA_VEC_NE11_DEFAULT  = -1;
constexpr int8_t FA_VEC_DOMAIN_DECODE = 0;  // ne01 == 1
constexpr int8_t FA_VEC_DOMAIN_BATCH  = 1;  // ne01 >= 2

ರಚನೆ fa_vec_ಕೀಲಿ_t {
    int8_t  family;
    int8_t  dtype;
    int16_t dk;
    int16_t dv;
    int8_t  ne11_b;
    int8_t  ne01_b;
};

static_assert(ಗಾತ್ರ(fa_vec_ಕೀಲಿ_t) == 8, "fa_vec_key_t must be tightly packed for memcmp");

ರಚನೆ fa_vec_cfg_t {
    int8_t Q;
    int8_t NE;
};

ರಚನೆ fa_vec_ನಮೂದು_t {
    fa_vec_ಕೀಲಿ_t ಕೀಲಿ;
    fa_vec_cfg_t cfg;
};

// legal NE values for a (dk,dv): NL = 32/NE, require (dk/4)%NL==0 && (dv/4)%NL==0.
// single source shared by the offline tuner and test-backend-ops.
ಸಾಲೊಳಗೆ std::ಸದಿಶ<ಪೂರ್ಣಾಂಕ> fa_vec_legal_ne(ಪೂರ್ಣಾಂಕ dk, ಪೂರ್ಣಾಂಕ dv) {
    std::ಸದಿಶ<ಪೂರ್ಣಾಂಕ> r;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ne : { 1, 2, 4 }) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nl = 32 / ne;
        ಆದರೆ ((dk / 4) % nl == 0 && (dv / 4) % nl == 0) {
            r.ತಳ್ಳು_ಹಿಂಭಾಗ(ne);
        }
    }
    ಹಿಂತಿರುಗಿ r;
}

// test/tune-only override; when set, fa_vec_pick returns it directly.
ಶೂನ್ಯ         fa_vec_ಗಣ_override(fa_vec_cfg_t cfg);
ಶೂನ್ಯ         fa_vec_clear_override();
fa_vec_cfg_t fa_vec_baseline_cfg(ಪೂರ್ಣಾಂಕ dk, ಪೂರ್ಣಾಂಕ dv);

// Keyed by Apple GPU family; an untuned family matches no row and gets the baseline.
fa_vec_cfg_t fa_vec_pick(ಪೂರ್ಣಾಂಕ gpu_family, ಪೂರ್ಣಾಂಕ dtype, ಪೂರ್ಣಾಂಕ dk, ಪೂರ್ಣಾಂಕ dv, int64_t ne11, int64_t ne01);

}  // namespace ggml_metal_tuning
