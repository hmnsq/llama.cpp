#અન્તર્ભાવ <sycl/sycl.hpp>
#અન્તર્ભાવ <sycl/ext/oneapi/work_group_static.hpp>
#અન્તર્ભાવ "dpct/helper.hpp"
#અન્તર્ભાવ "common.hpp"
#અન્તર્ભાવ "fattn-common.hpp"
#અન્તર્ભાવ "fattn-tile.hpp"
#અન્તર્ભાવ <cmath>
#અન્તર્ભાવ <float.h>
નામાકાશ syclex = sycl::ext::oneapi::experimental;

શૂન્ય ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    અચલ ggml_tensor * K = dst->src[1];
    અચલ ggml_tensor * V = dst->src[2];
    પસંદગી (K->ne[0]) {
        સ્થિતિ  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } તોડો;
        સ્થિતિ  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } તોડો;
        સ્થિતિ  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } તોડો;
        સ્થિતિ  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } તોડો;
        સ્થિતિ  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } તોડો;
        સ્થિતિ 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } તોડો;
        સ્થિતિ 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } તોડો;
        સ્થિતિ 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } તોડો;
        સ્થિતિ 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } તોડો;
        સ્થિતિ 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_sycl_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } તોડો;
        મૂળભૂત: {
            GGML_ABORT("Unsupported head size");
        }
    }
}
