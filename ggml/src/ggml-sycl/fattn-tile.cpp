#شامل <sycl/sycl.hpp>
#شامل <sycl/ext/oneapi/work_group_static.hpp>
#شامل "dpct/helper.hpp"
#شامل "common.hpp"
#شامل "fattn-common.hpp"
#شامل "fattn-tile.hpp"
#شامل <cmath>
#شامل <float.h>
فضاینام syclex = sycl::ext::oneapi::experimental;

تهی ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * K = dst->src[1];
    ثابت ggml_tensor * V = dst->src[2];
    انتخاب (K->ne[0]) {
        حالت  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } بشکن;
        حالت  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } بشکن;
        حالت  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } بشکن;
        حالت  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } بشکن;
        حالت  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } بشکن;
        حالت 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } بشکن;
        حالت 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } بشکن;
        حالت 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } بشکن;
        حالت 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } بشکن;
        حالت 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_sycl_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } بشکن;
        پیشفرض: {
            GGML_ABORT("Unsupported head size");
        }
    }
}
