#அந்தர்பாவ <sycl/sycl.hpp>
#அந்தர்பாவ <sycl/ext/oneapi/work_group_static.hpp>
#அந்தர்பாவ "dpct/helper.hpp"
#அந்தர்பாவ "common.hpp"
#அந்தர்பாவ "fattn-common.hpp"
#அந்தர்பாவ "fattn-tile.hpp"
#அந்தர்பாவ <cmath>
#அந்தர்பாவ <float.h>
நாமாகாஶ syclex = sycl::ext::oneapi::experimental;

வெறுமை ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    மாறிலி ggml_tensor * K = dst->src[1];
    மாறிலி ggml_tensor * V = dst->src[2];
    தேர்வு (K->ne[0]) {
        நிலை  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } நிறுத்து;
        நிலை  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } நிறுத்து;
        நிலை  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } நிறுத்து;
        நிலை  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } நிறுத்து;
        நிலை  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } நிறுத்து;
        நிலை 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } நிறுத்து;
        நிலை 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } நிறுத்து;
        நிலை 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } நிறுத்து;
        நிலை 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } நிறுத்து;
        நிலை 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_sycl_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } நிறுத்து;
        இயல்பு: {
            GGML_ABORT("Unsupported head size");
        }
    }
}
