#ہدایت once

#شامل "node_context.h"

namespace ov {
namespace frontend {
namespace ggml {

namespace op {

#تعریف_کرو GGML_OP_CONVERTER(op) Output_سمتیہ op(ثابت گرہ_Context & context)

GGML_OP_CONVERTER(translate_جوڑو);
GGML_OP_CONVERTER(translate_cont);
GGML_OP_CONVERTER(translate_concat);
GGML_OP_CONVERTER(translate_جوڑو_id);
GGML_OP_CONVERTER(translate_div);
GGML_OP_CONVERTER(translate_fill);
GGML_OP_CONVERTER(translate_get_صفیں);
GGML_OP_CONVERTER(translate_im2col);
GGML_OP_CONVERTER(translate_im2col_3d);
GGML_OP_CONVERTER(translate_mulmat);
GGML_OP_CONVERTER(translate_mul_mat_id);
GGML_OP_CONVERTER(translate_ترتیب_بدلو);
GGML_OP_CONVERTER(translate_reshape);
GGML_OP_CONVERTER(translate_rms_norm);
GGML_OP_CONVERTER(translate_norm);
GGML_OP_CONVERTER(translate_l2_norm);
GGML_OP_CONVERTER(translate_مجموع_صفیں);
GGML_OP_CONVERTER(translate_sqr);
GGML_OP_CONVERTER(translate_rope);
GGML_OP_CONVERTER(translate_scale);
GGML_OP_CONVERTER(translate_sqrt);
GGML_OP_CONVERTER(translate_unary_softplus);
GGML_OP_CONVERTER(translate_soft_max);
GGML_OP_CONVERTER(translate_transpose);
GGML_OP_CONVERTER(translate_view);
GGML_OP_CONVERTER(translate_glu_swiglu);
GGML_OP_CONVERTER(translate_glu_swiglu_oai);
GGML_OP_CONVERTER(translate_glu_swiglu_clamp);
GGML_OP_CONVERTER(translate_glu_geglu);
GGML_OP_CONVERTER(translate_glu_geglu_تیز);
GGML_OP_CONVERTER(translate_سیٹ_صفیں);
GGML_OP_CONVERTER(translate_cpy);
GGML_OP_CONVERTER(translate_argsort);
GGML_OP_CONVERTER(translate_flash_attn_ext);
GGML_OP_CONVERTER(translate_clamp);
GGML_OP_CONVERTER(translate_pad);
GGML_OP_CONVERTER(translate_ssm_conv);
GGML_OP_CONVERTER(translate_gated_delta_net);
GGML_OP_CONVERTER(translate_repeat);
GGML_OP_CONVERTER(translate_cumsum);
GGML_OP_CONVERTER(translate_fill);
GGML_OP_CONVERTER(translate_سیٹ);
GGML_OP_CONVERTER(translate_diag);
GGML_OP_CONVERTER(translate_tri);
GGML_OP_CONVERTER(translate_solve_tri);
GGML_OP_CONVERTER(translate_pool_2d);
GGML_OP_CONVERTER(translate_roll);
GGML_OP_CONVERTER(translate_upscale);
GGML_OP_CONVERTER(translate_اوسط);
GGML_OP_CONVERTER(translate_مجموع);
GGML_OP_CONVERTER(translate_unary_gelu);
GGML_OP_CONVERTER(translate_unary_gelu_erf);
GGML_OP_CONVERTER(translate_unary_gelu_تیز);
GGML_OP_CONVERTER(translate_unary_elu);
GGML_OP_CONVERTER(translate_unary_hardsigmoid);
GGML_OP_CONVERTER(translate_unary_قدم);
GGML_OP_CONVERTER(translate_unary_باری);
GGML_OP_CONVERTER(translate_unary_expm1);
GGML_OP_CONVERTER(translate_conv_2d);
GGML_OP_CONVERTER(translate_conv_2d_dw);
GGML_OP_CONVERTER(translate_conv_transpose_1d);
GGML_OP_CONVERTER(translate_conv_transpose_2d);
GGML_OP_CONVERTER(translate_conv_3d);

}  // namespace op

std::unordered_map<std::string, CreatorFunction> get_supported_ops();

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
