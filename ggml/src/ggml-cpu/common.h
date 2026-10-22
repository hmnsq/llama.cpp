#సూచన once

#అన్తర్భావ "ggml.h"
#అన్తర్భావ "traits.h"
#అన్తర్భావ "ggml-cpu-impl.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "simd-mappings.h"

#నిర్వచించు GGML_FA_TILE_Q  64
#నిర్వచించు GGML_FA_TILE_KV 64

#నిర్వచితమైతే __cplusplus

#అన్తర్భావ <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
స్థిర అంతఃవరుస ggml_fp16_t f32_గమ్యం_f16(దశాంశం x) {
    తిరిగి GGML_CPU_FP32_గమ్యం_FP16(x);
}

స్థిర అంతఃవరుస దశాంశం f16_గమ్యం_f32(ggml_fp16_t x) {
    తిరిగి GGML_CPU_FP16_గమ్యం_FP32(x);
}

స్థిర అంతఃవరుస ggml_bf16_t f32_గమ్యం_bf16(దశాంశం x) {
    తిరిగి GGML_FP32_గమ్యం_BF16(x);
}

స్థిర అంతఃవరుస దశాంశం bf16_గమ్యం_f32(ggml_bf16_t x) {
    తిరిగి GGML_BF16_గమ్యం_FP32(x);
}

స్థిర అంతఃవరుస దశాంశం i32_గమ్యం_f32(int32_t x) {
    తిరిగి x;
}

స్థిర అంతఃవరుస int32_t f32_గమ్యం_i32(దశాంశం x) {
    తిరిగి x;
}

స్థిర అంతఃవరుస దశాంశం f32_గమ్యం_f32(దశాంశం x) {
    తిరిగి x;
}

// TODO - merge this into the traits table, after using row-based conversions
మూస <class T>
నిర్మాణం type_conversion_పట్టిక;

మూస <>
నిర్మాణం type_conversion_పట్టిక<ggml_fp16_t> {
    స్థిర constexpr దశాంశం (*గమ్యం_f32)(ggml_fp16_t) = f16_గమ్యం_f32;
    స్థిర constexpr ggml_fp16_t (*from_f32)(దశాంశం) = f32_గమ్యం_f16;
};

మూస <>
నిర్మాణం type_conversion_పట్టిక<దశాంశం> {
    స్థిర constexpr దశాంశం (*గమ్యం_f32)(దశాంశం) = f32_గమ్యం_f32;
    స్థిర constexpr దశాంశం (*from_f32)(దశాంశం) = f32_గమ్యం_f32;
};

మూస <>
నిర్మాణం type_conversion_పట్టిక<ggml_bf16_t> {
    స్థిర constexpr దశాంశం (*గమ్యం_f32)(ggml_bf16_t) = bf16_గమ్యం_f32;
    స్థిర constexpr ggml_bf16_t (*from_f32)(దశాంశం) = f32_గమ్యం_bf16;
};

మూస <>
నిర్మాణం type_conversion_పట్టిక<int32_t> {
    స్థిర constexpr దశాంశం (*గమ్యం_f32)(int32_t) = i32_గమ్యం_f32;
    స్థిర constexpr int32_t (*from_f32)(దశాంశం) = f32_గమ్యం_i32;
};

స్థిర std::pair<int64_t, int64_t> get_thread_range(స్థిరరాశి నిర్మాణం ggml_compute_params * params, స్థిరరాశి నిర్మాణం ggml_tensor * src0) {
    స్థిరరాశి int64_t ith = params->ith;
    స్థిరరాశి int64_t nth = params->nth;

    స్థిరరాశి int64_t nr  = ggml_nrows(src0);

    // rows per thread
    స్థిరరాశి int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి int64_t ir0 = dr*ith;
    స్థిరరాశి int64_t ir1 = MIN(ir0 + dr, nr);

    తిరిగి {ir0, ir1};
}

నిర్మాణం ggml_fa_tile_config {
    స్థిర constexpr size_t Q  = GGML_FA_TILE_Q;
    స్థిర constexpr size_t KV = GGML_FA_TILE_KV;
};

#అయితే_ముగింపు
