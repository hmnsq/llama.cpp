#অন্তর্ভাব <sycl/sycl.hpp>
#অন্তর্ভাব <sycl/ext/oneapi/work_group_static.hpp>
#অন্তর্ভাব "dpct/helper.hpp"
#অন্তর্ভাব "common.hpp"
#অন্তর্ভাব "fattn-common.hpp"
#অন্তর্ভাব "fattn-tile.hpp"
#অন্তর্ভাব <cmath>
#অন্তর্ভাব <float.h>
নামাকাশ syclex = sycl::ext::oneapi::experimental;

শূন্য ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুবক ggml_tensor * K = dst->src[1];
    ধ্রুবক ggml_tensor * V = dst->src[2];
    নির্বাচন (K->ne[0]) {
        ক্ষেত্র  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } বিরতি;
        ক্ষেত্র  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } বিরতি;
        ক্ষেত্র  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } বিরতি;
        ক্ষেত্র  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } বিরতি;
        ক্ষেত্র  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } বিরতি;
        ক্ষেত্র 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } বিরতি;
        ক্ষেত্র 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } বিরতি;
        ক্ষেত্র 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } বিরতি;
        ক্ষেত্র 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } বিরতি;
        ক্ষেত্র 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_sycl_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } বিরতি;
        পূর্বনির্ধারিত: {
            GGML_ABORT("Unsupported head size");
        }
    }
}
