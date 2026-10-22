#অন্তর্ভাৱ <sycl/sycl.hpp>
#অন্তর্ভাৱ <sycl/ext/oneapi/work_group_static.hpp>
#অন্তর্ভাৱ "dpct/helper.hpp"
#অন্তর্ভাৱ "common.hpp"
#অন্তর্ভাৱ "fattn-common.hpp"
#অন্তর্ভাৱ "fattn-tile.hpp"
#অন্তর্ভাৱ <cmath>
#অন্তর্ভাৱ <float.h>
নামাকাশ syclex = sycl::ext::oneapi::experimental;

শূন্য ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুৱ ggml_tensor * K = dst->src[1];
    ধ্রুৱ ggml_tensor * V = dst->src[2];
    ৱিকল্প (K->ne[0]) {
        পক্ষ  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } ভঙ্গ;
        পক্ষ  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } ভঙ্গ;
        পক্ষ  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } ভঙ্গ;
        পক্ষ  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } ভঙ্গ;
        পক্ষ  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } ভঙ্গ;
        পক্ষ 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } ভঙ্গ;
        পক্ষ 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } ভঙ্গ;
        পক্ষ 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } ভঙ্গ;
        পক্ষ 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } ভঙ্গ;
        পক্ষ 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_sycl_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } ভঙ্গ;
        স্ৱভাৱ: {
            GGML_ABORT("Unsupported head size");
        }
    }
}
