#ଅନ୍ତର୍ଭାଵ <sycl/sycl.hpp>
#ଅନ୍ତର୍ଭାଵ <sycl/ext/oneapi/work_group_static.hpp>
#ଅନ୍ତର୍ଭାଵ "dpct/helper.hpp"
#ଅନ୍ତର୍ଭାଵ "common.hpp"
#ଅନ୍ତର୍ଭାଵ "fattn-common.hpp"
#ଅନ୍ତର୍ଭାଵ "fattn-tile.hpp"
#ଅନ୍ତର୍ଭାଵ <cmath>
#ଅନ୍ତର୍ଭାଵ <float.h>
ନାମାକାଶ syclex = sycl::ext::oneapi::experimental;

ଶୂନ୍ୟ ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * K = dst->src[1];
    ଅଚଳ ggml_tensor * V = dst->src[2];
    ବାଛ (K->ne[0]) {
        ସ୍ଥିତି  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } ଭାଙ୍ଗ;
        ସ୍ଥିତି 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_sycl_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: {
            GGML_ABORT("Unsupported head size");
        }
    }
}
