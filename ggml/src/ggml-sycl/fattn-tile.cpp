#अन्तर्भाव <sycl/sycl.hpp>
#अन्तर्भाव <sycl/ext/oneapi/work_group_static.hpp>
#अन्तर्भाव "dpct/helper.hpp"
#अन्तर्भाव "common.hpp"
#अन्तर्भाव "fattn-common.hpp"
#अन्तर्भाव "fattn-tile.hpp"
#अन्तर्भाव <cmath>
#अन्तर्भाव <float.h>
नामाकाश syclex = sycl::ext::oneapi::experimental;

शून्य ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * K = dst->src[1];
    स्थिरांक ggml_tensor * V = dst->src[2];
    निवड (K->ne[0]) {
        स्थिती  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } थांबा;
        स्थिती  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } थांबा;
        स्थिती  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } थांबा;
        स्थिती  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } थांबा;
        स्थिती  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } थांबा;
        स्थिती 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } थांबा;
        स्थिती 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } थांबा;
        स्थिती 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } थांबा;
        स्थिती 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } थांबा;
        स्थिती 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_sycl_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } थांबा;
        पूर्वनिर्धारित: {
            GGML_ABORT("Unsupported head size");
        }
    }
}
