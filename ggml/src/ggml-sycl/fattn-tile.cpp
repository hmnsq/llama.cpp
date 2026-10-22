#അന്തര്ഭാവ <sycl/sycl.hpp>
#അന്തര്ഭാവ <sycl/ext/oneapi/work_group_static.hpp>
#അന്തര്ഭാവ "dpct/helper.hpp"
#അന്തര്ഭാവ "common.hpp"
#അന്തര്ഭാവ "fattn-common.hpp"
#അന്തര്ഭാവ "fattn-tile.hpp"
#അന്തര്ഭാവ <cmath>
#അന്തര്ഭാവ <float.h>
നാമാകാശ syclex = sycl::ext::oneapi::experimental;

ശൂന്യം ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * K = dst->src[1];
    സ്ഥിരം ggml_tensor * V = dst->src[2];
    തിരഞ്ഞെടുപ്പ് (K->ne[0]) {
        അവസ്ഥ  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } നിർത്തുക;
        അവസ്ഥ  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } നിർത്തുക;
        അവസ്ഥ  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } നിർത്തുക;
        അവസ്ഥ  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } നിർത്തുക;
        അവസ്ഥ  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } നിർത്തുക;
        അവസ്ഥ 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } നിർത്തുക;
        അവസ്ഥ 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } നിർത്തുക;
        അവസ്ഥ 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } നിർത്തുക;
        അവസ്ഥ 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } നിർത്തുക;
        അവസ്ഥ 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_sycl_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } നിർത്തുക;
        സ്ഥിരസ്ഥിതി: {
            GGML_ABORT("Unsupported head size");
        }
    }
}
