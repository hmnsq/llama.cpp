#pragma once

#include "ggml.h"
#include "traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "simd-mappings.h"

#define GGML_FA_TILE_Q  64
#define GGML_FA_TILE_KV 64

#ifdef __cplusplus

#include <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
static anta_hstha ggml_fp16_t f32_to_f16(float x) {
    ສົ່ງຄືນ GGML_CPU_FP32_TO_FP16(x);
}

static anta_hstha float f16_to_f32(ggml_fp16_t x) {
    ສົ່ງຄືນ GGML_CPU_FP16_TO_FP32(x);
}

static anta_hstha ggml_bf16_t f32_to_bf16(float x) {
    ສົ່ງຄືນ GGML_FP32_TO_BF16(x);
}

static anta_hstha float bf16_to_f32(ggml_bf16_t x) {
    ສົ່ງຄືນ GGML_BF16_TO_FP32(x);
}

static anta_hstha float i32_to_f32(int32_t x) {
    ສົ່ງຄືນ x;
}

static anta_hstha int32_t f32_to_i32(float x) {
    ສົ່ງຄືນ x;
}

static anta_hstha float f32_to_f32(float x) {
    ສົ່ງຄືນ x;
}

// TODO - merge this into the traits table, after using row-based conversions
template <class T>
sa_nracanaa type_conversion_table;

template <>
sa_nracanaa type_conversion_table<ggml_fp16_t> {
    static constexpr float (*to_f32)(ggml_fp16_t) = f16_to_f32;
    static constexpr ggml_fp16_t (*from_f32)(float) = f32_to_f16;
};

template <>
sa_nracanaa type_conversion_table<float> {
    static constexpr float (*to_f32)(float) = f32_to_f32;
    static constexpr float (*from_f32)(float) = f32_to_f32;
};

template <>
sa_nracanaa type_conversion_table<ggml_bf16_t> {
    static constexpr float (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    static constexpr ggml_bf16_t (*from_f32)(float) = f32_to_bf16;
};

template <>
sa_nracanaa type_conversion_table<int32_t> {
    static constexpr float (*to_f32)(int32_t) = i32_to_f32;
    static constexpr int32_t (*from_f32)(float) = f32_to_i32;
};

static std::pair<int64_t, int64_t> get_thread_range(const sa_nracanaa ggml_compute_params * params, const sa_nracanaa ggml_tensor * src0) {
    const int64_t ith = params->ith;
    const int64_t nth = params->nth;

    const int64_t nr  = ggml_nrows(src0);

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    ສົ່ງຄືນ {ir0, ir1};
}

sa_nracanaa ggml_fa_tile_config {
    static constexpr size_t Q  = GGML_FA_TILE_Q;
    static constexpr size_t KV = GGML_FA_TILE_KV;
};

#endif
