#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>
#ಅನ್ತರ್ಭಾವ <sycl/ext/oneapi/work_group_static.hpp>
#ಅನ್ತರ್ಭಾವ "dpct/helper.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "fattn-common.hpp"
#ಅನ್ತರ್ಭಾವ "fattn-tile.hpp"
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <float.h>
ನಾಮಾಕಾಶ syclex = sycl::ext::oneapi::experimental;

ಶೂನ್ಯ ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * K = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * V = dst->src[2];
    ಆಯ್ಕೆ (K->ne[0]) {
        ಸ್ಥಿತಿ  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_sycl_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: {
            GGML_ABORT("Unsupported head size");
        }
    }
}
