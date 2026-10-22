#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/sycl.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/ext/oneapi/work_group_static.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ "dpct/helper.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "fattn-common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "fattn-tile.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <float.h>
ਨਾਮਾਕਾਸ਼ syclex = sycl::ext::oneapi::experimental;

ਖਾਲੀ ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * K = dst->src[1];
    ਅਚਲ ggml_tensor * V = dst->src[2];
    ਚੋਣ (K->ne[0]) {
        ਹਾਲਤ  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } ਰੋਕੋ;
        ਹਾਲਤ  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } ਰੋਕੋ;
        ਹਾਲਤ  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } ਰੋਕੋ;
        ਹਾਲਤ  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } ਰੋਕੋ;
        ਹਾਲਤ  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } ਰੋਕੋ;
        ਹਾਲਤ 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } ਰੋਕੋ;
        ਹਾਲਤ 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } ਰੋਕੋ;
        ਹਾਲਤ 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } ਰੋਕੋ;
        ਹਾਲਤ 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } ਰੋਕੋ;
        ਹਾਲਤ 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_sycl_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } ਰੋਕੋ;
        ਮੂਲ: {
            GGML_ABORT("Unsupported head size");
        }
    }
}
