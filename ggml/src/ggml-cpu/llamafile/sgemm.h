#pragma once
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <stdbool.h>

#எனில் defined(__VXE__) || defined(__VXE2__)
#அந்தர்பாவ <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#இல்லையெனில்
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
வெளி "C" {
#endif

bool llamafile_sgemm(மாறிலி கட்டமைப்பு ggml_compute_params * params, int64_t, int64_t, int64_t,
                     மாறிலி வெறுமை *, int64_t, மாறிலி வெறுமை *, int64_t, வெறுமை *, int64_t,
                     முழு, முழு, முழு);

#ifdef __cplusplus
}
#endif
