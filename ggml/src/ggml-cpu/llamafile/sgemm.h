#pragma once
#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <stdbool.h>

#എങ്കിൽ defined(__VXE__) || defined(__VXE2__)
#അന്തര്ഭാവ <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#അല്ലെങ്കിൽ
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
ബാഹ്യ "C" {
#endif

bool llamafile_sgemm(സ്ഥിരം ഘടന ggml_compute_params * params, int64_t, int64_t, int64_t,
                     സ്ഥിരം ശൂന്യം *, int64_t, സ്ഥിരം ശൂന്യം *, int64_t, ശൂന്യം *, int64_t,
                     പൂർണ്ണസംഖ്യ, പൂർണ്ണസംഖ്യ, പൂർണ്ണസംഖ്യ);

#ifdef __cplusplus
}
#endif
