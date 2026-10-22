#అన్తర్భావ <sycl/sycl.hpp>
#అన్తర్భావ <sycl/ext/oneapi/work_group_static.hpp>
#అన్తర్భావ "dpct/helper.hpp"
#అన్తర్భావ "common.hpp"
#అన్తర్భావ "fattn-common.hpp"
#అన్తర్భావ "fattn-tile.hpp"
#అన్తర్భావ <cmath>
#అన్తర్భావ <float.h>
నామాకాశ syclex = sycl::ext::oneapi::experimental;

శూన్య ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * K = dst->src[1];
    స్థిరరాశి ggml_tensor * V = dst->src[2];
    ఎంపిక (K->ne[0]) {
        స్థితి  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } ఆపు;
        స్థితి  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } ఆపు;
        స్థితి  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } ఆపు;
        స్థితి  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } ఆపు;
        స్థితి  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } ఆపు;
        స్థితి 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } ఆపు;
        స్థితి 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } ఆపు;
        స్థితి 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } ఆపు;
        స్థితి 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } ఆపు;
        స్థితి 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_sycl_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } ఆపు;
        అప్రమేయ: {
            GGML_ABORT("Unsupported head size");
        }
    }
}
