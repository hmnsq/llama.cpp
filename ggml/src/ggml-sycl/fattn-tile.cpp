#समावेश <sycl/sycl.hpp>
#समावेश <sycl/ext/oneapi/work_group_static.hpp>
#समावेश "dpct/helper.hpp"
#समावेश "common.hpp"
#समावेश "fattn-common.hpp"
#समावेश "fattn-tile.hpp"
#समावेश <cmath>
#समावेश <float.h>
namespace syclex = sycl::ext::oneapi::experimental;

व्योम ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * K = dst->src[1];
    स्थिर ggml_tensor * V = dst->src[2];
    चयन (K->ne[0]) {
        हाल  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } अवरोध;
        हाल  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } अवरोध;
        हाल  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } अवरोध;
        हाल  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } अवरोध;
        हाल  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } अवरोध;
        हाल 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } अवरोध;
        हाल 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } अवरोध;
        हाल 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } अवरोध;
        हाल 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } अवरोध;
        हाल 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_sycl_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } अवरोध;
        शेष: {
            GGML_ABORT("Unsupported head size");
        }
    }
}
