#pragma once
#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <stdbool.h>

#જો defined(__VXE__) || defined(__VXE2__)
#અન્તર્ભાવ <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#નહીંતર
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
બાહ્ય "C" {
#endif

bool llamafile_sgemm(અચલ સંરચના ggml_compute_params * params, int64_t, int64_t, int64_t,
                     અચલ શૂન્ય *, int64_t, અચલ શૂન્ય *, int64_t, શૂન્ય *, int64_t,
                     પૂર્ણાંક, પૂર્ણાંક, પૂર્ણાંક);

#ifdef __cplusplus
}
#endif
