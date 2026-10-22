#pragma once

#ضمن "ggml.h"

#ضمن <cstdint>
#ضمن <vector>

namespace ggml_metal_tuning {

// FA vec selection buckets. ne01 (query rows) splits decode (==1) from batch (>=2), the
// batch side refined into {2,3,4,5}: Q>1 reuses one K/V load across rows, so it only pays
// off once ne01 aligns with Q. ne11 (KV length) is bucketed too, as the Q>1 crossover is
// head-size dependent (small dk crosses late, large dk wins even at short KV).
constexpr صحيح FA_VEC_NE11_BUCKETS[] = { 1024, 4096, 16384 };
constexpr صحيح FA_VEC_NE01_BUCKETS[] = { 2, 3, 4, 5 };

صحيح fa_vec_ne11_bucket(int64_t ne11);
صحيح fa_vec_ne01_bucket(int64_t ne01);

// NE baked into each (dk,dv) baseline instantiation in kernels/fa_vec_*.metal.
// Hand-maintained mirror; keep in sync with those instantiations.
// The Metal test slice covers every legal config for dk=128 and dk=576.
صحيح fa_vec_baseline_ne(صحيح dk, صحيح dv);

// Tuned table has two row kinds. Exact rows key a (ne11_b, ne01_b) bucket. Default rows
// collapse ne11 over one ne01 domain: ne11_b == FA_VEC_NE11_DEFAULT and ne01_b holds the
// domain. fa_vec_pick tries exact bucket -> domain default -> baseline; short KV
// (ne11 < FA_VEC_NE11_BUCKETS[0]) always uses baseline.
constexpr int8_t FA_VEC_NE11_DEFAULT  = -1;
constexpr int8_t FA_VEC_DOMAIN_DECODE = 0;  // ne01 == 1
constexpr int8_t FA_VEC_DOMAIN_BATCH  = 1;  // ne01 >= 2

بنية fa_vec_key_t {
    int8_t  family;
    int8_t  dtype;
    int16_t dk;
    int16_t dv;
    int8_t  ne11_b;
    int8_t  ne01_b;
};

static_assert(حجم(fa_vec_key_t) == 8, "fa_vec_key_t must be tightly packed for memcmp");

بنية fa_vec_cfg_t {
    int8_t Q;
    int8_t NE;
};

بنية fa_vec_entry_t {
    fa_vec_key_t key;
    fa_vec_cfg_t cfg;
};

// legal NE values for a (dk,dv): NL = 32/NE, require (dk/4)%NL==0 && (dv/4)%NL==0.
// single source shared by the offline tuner and test-backend-ops.
مضمن std::vector<صحيح> fa_vec_legal_ne(صحيح dk, صحيح dv) {
    std::vector<صحيح> r;
    لكل (صحيح ne : { 1, 2, 4 }) {
        ثابت صحيح nl = 32 / ne;
        𐡄𐡍 ((dk / 4) % nl == 0 && (dv / 4) % nl == 0) {
            r.push_back(ne);
        }
    }
    أرجع r;
}

// test/tune-only override; when set, fa_vec_pick returns it directly.
فراغ         fa_vec_set_override(fa_vec_cfg_t cfg);
فراغ         fa_vec_clear_override();
fa_vec_cfg_t fa_vec_baseline_cfg(صحيح dk, صحيح dv);

// Keyed by Apple GPU family; an untuned family matches no row and gets the baseline.
fa_vec_cfg_t fa_vec_pick(صحيح gpu_family, صحيح dtype, صحيح dk, صحيح dv, int64_t ne11, int64_t ne01);

}  // namespace ggml_metal_tuning
