#ضمن "utils.h"

#ضمن <cmath>
#ضمن <cstddef>
#ضمن <ctime>
#ضمن <memory>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/clamp.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/cos.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/maximum.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/sin.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/split.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/subtract.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <string>

نطاق ov {
نطاق frontend {
نطاق ggml {

فراغ انگ_inputs_check(ثابت ڳنڍ_Context & context, size_t min_inputs, size_t max_inputs) {
    خودڪار داخلا_سائيز = context.get_داخلا_سائيز();
    FRONT_پڇاڙي_OP_CONVERSION_CHECK(داخلا_سائيز >= min_inputs, "Got less inputs than expected");
    FRONT_پڇاڙي_OP_CONVERSION_CHECK(داخلا_سائيز <= max_inputs, "Got more inputs than expected");
}

صحيح non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    صحيح dim = nb.size() - 1;
    size_t بائيٽون = nb[dim];
    لكل (صحيح i = dim; i > 0; i--) {
        بائيٽون *= ne[i];
        إذا (بائيٽون != nb[i - 1]) {
            أرجع i;
        }
    }
    أرجع 0;
}

std::shared_ptr<ov::Node> get_dimensions(ثابت std::shared_ptr<ov::op::v3::Shape_جو> & shape,
                                         ثابت std::vector<صحيح> & dims) {
    استعمال نطاق ov::op;
    ثابت خودڪار zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ثابت خودڪار dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    أرجع std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(ثابت std::shared_ptr<ov::Node> & ڳنڍ, ثابت std::vector<صحيح> & dims) {
    أرجع get_dimensions(std::make_shared<ov::op::v3::Shape_جو>(ڳنڍ), dims);
}

Output_ويڪٽر rename_outputs_with_suffix(ثابت Output_ويڪٽر & outputs, ثابت std::string & suffix) {
    لكل (ثابت خودڪار & output : outputs) {
        خودڪار ڳنڍ = output.get_ڳنڍ_shared_ptr();
        std::string name = ڳنڍ->get_friendly_name();
        name += "_";
        name += suffix;
        ڳنڍ->سيٽ_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    أرجع outputs;
}

نطاق {
ov::Output<ov::Node> rope_yarn_ramp_mix(صحيح انگ_elements, ثابت عائم corr_dims[2], عائم ext_ضرب_ڪندڙ) {
    std::vector<عائم> dim_ids_vec(انگ_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    خودڪار dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) انگ_elements}, dim_ids_vec);
    خودڪار corr_هيٺيون = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    خودڪار corr_مٿاهون = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    خودڪار denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::ڪٽيو>(corr_مٿاهون, corr_هيٺيون),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    خودڪار ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::ڪٽيو>(dim_ids, corr_هيٺيون), denom);
    خودڪار ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    خودڪار one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    خودڪار ramp_inverted = std::make_shared<ov::op::v1::ڪٽيو>(one, ramp_clamped);
    خودڪار ext_ضرب_ڪندڙ_ڳنڍ = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_ضرب_ڪندڙ});
    خودڪار ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_ضرب_ڪندڙ_ڳنڍ);
    أرجع ramp_mix;
}

عائم ggml_rope_yarn_corr_dim(صحيح n_dims, صحيح n_ctx_orig, عائم n_rot, عائم بنياد) {
#جيڪڏهن_وصف_نه_ٿيل M_PI
#    define M_PI 3.14159265358979323846
#جيڪڏهن_پورو
    أرجع n_dims * logf(n_ctx_orig / (n_rot * 2 * (عائم) M_PI)) / (2 * logf(بنياد));
}

فراغ ggml_rope_yarn_corr_dims(صحيح n_dims,
                              صحيح n_ctx_orig,
                              عائم freq_بنياد,
                              عائم beta_fast,
                              عائم beta_slow,
                              عائم dims[2]) {
    عائم start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_بنياد));
    عائم end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_بنياد));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<عائم>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_وزن,
                                                           صحيح mode,
                                                           منطقي stateful,
                                                           int64_t سر_dim) {
    constexpr صحيح TYPE_IMROPE = 2;
    constexpr صحيح TYPE_VISION = 3;
    constexpr صحيح TYPE_MROPE = 4;

    ثابت منطقي is_imrope = (mode == TYPE_IMROPE);
    ثابت منطقي is_vision = (mode == TYPE_VISION);
    ثابت منطقي is_mrope = (mode == TYPE_MROPE);
    ثابت منطقي is_multi = is_imrope || is_vision || is_mrope;

    إذا (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        خودڪار pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } وإلا إذا (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        خودڪار pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, سچ);
        خودڪار pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } وإلا {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        خودڪار pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    عائم freq_بنياد;
    عائم freq_scale;
    عائم ext_ضرب_ڪندڙ;
    عائم attn_ضرب_ڪندڙ;
    عائم beta_fast;
    عائم beta_slow;
    ثابت صحيح n_dims = rope_params[1];
    ثابت size_t n_dims_half = n_dims >> 1;
    ثابت صحيح n_ctx_orig = rope_params[4];
    يادگيري_نقل(&freq_بنياد, rope_params + 5, حجم(عائم));
    يادگيري_نقل(&freq_scale, rope_params + 6, حجم(عائم));
    يادگيري_نقل(&ext_ضرب_ڪندڙ, rope_params + 7, حجم(عائم));
    يادگيري_نقل(&attn_ضرب_ڪندڙ, rope_params + 8, حجم(عائم));
    يادگيري_نقل(&beta_fast, rope_params + 9, حجم(عائم));
    يادگيري_نقل(&beta_slow, rope_params + 10, حجم(عائم));

    ثابت عائم theta_scale = powf(freq_بنياد, -2.0f / n_dims);

    Output<Node> theta;
    عائم mscale = attn_ضرب_ڪندڙ;
    إذا (is_multi) {
        ثابت size_t انگ_جوڙا = is_vision ? (n_dims > 0 ? (size_t) n_dims : (size_t) (سر_dim / 2)) : n_dims_half;
        صحيح sections[4];
        يادگيري_نقل(sections, rope_params + 11, حجم(صحيح) * 4);
        صحيح sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        إذا (sect_dims <= 0) {
            sect_dims = انگ_جوڙا;
        }
        صحيح sec_w = sections[0] + sections[1];
        صحيح sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(انگ_جوڙا);
        std::vector<عائم> ضرب_ڪندڙ(انگ_جوڙا);
        لكل (size_t j = 0; j < انگ_جوڙا; j++) {
            صحيح sector = j % sect_dims;
            صحيح plane = 0;
            إذا (is_imrope) {
                إذا (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } وإلا إذا (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } وإلا إذا (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } وإلا {
                    plane = 3;
                }
                ضرب_ڪندڙ[j] = std::قوت(theta_scale, j);
            } وإلا إذا (is_vision) {
                صحيح p_idx = 0;
                إذا (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } وإلا إذا (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } وإلا إذا (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } وإلا {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                ضرب_ڪندڙ[j] = std::قوت(theta_scale, p_idx);
            } وإلا {
                إذا (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } وإلا إذا (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } وإلا إذا (sector >= sec_e) {
                    plane = 3;
                } وإلا {
                    plane = 0;
                }
                ضرب_ڪندڙ[j] = std::قوت(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        خودڪار gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{انگ_جوڙا}, gather_indices);
        خودڪار gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> ضرب_ڪندڙ_ڳنڍ =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{انگ_جوڙا}, ضرب_ڪندڙ);
        إذا (rope_freqs_وزن) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::ٽڪرو>(
                rope_freqs_وزن,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) انگ_جوڙا}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_وزن->get_output_partial_shape(0).rank().get_ڊيگهائي() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) انگ_جوڙا}),
                خطأ);
            ضرب_ڪندڙ_ڳنڍ = std::make_shared<ov::op::v1::Divide>(ضرب_ڪندڙ_ڳنڍ, rope_factors);
        }
        خودڪار theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, ضرب_ڪندڙ_ڳنڍ);
        خودڪار theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        إذا (ext_ضرب_ڪندڙ == 0.0f) {
            theta = theta_interp;
        } وإلا {
            عائم corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_بنياد, beta_fast, beta_slow, corr_dims);
            خودڪار ramp_mix = rope_yarn_ramp_mix(انگ_جوڙا, corr_dims, ext_ضرب_ڪندڙ);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            خودڪار one_minus_ramp = std::make_shared<ov::op::v1::ڪٽيو>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::جوڙيو>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::لاگرٿم(1.0f / freq_scale));
        }
    } وإلا {
        std::vector<عائم> ضرب_ڪندڙ(n_dims_half);
        Output<Node> freq_factors;
        عائم corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_بنياد, beta_fast, beta_slow, corr_dims);
        ضرب_ڪندڙ[0] = 1.0f;
        لكل (size_t i = 1; i < ضرب_ڪندڙ.size(); i++) {
            ضرب_ڪندڙ[i] = theta_scale * ضرب_ڪندڙ[i - 1];
        }
        إذا (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, ضرب_ڪندڙ.size()}, ضرب_ڪندڙ);
        } وإلا {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, ضرب_ڪندڙ.size()}, ضرب_ڪندڙ);
        }
        إذا (rope_freqs_وزن) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::ٽڪرو>(
                rope_freqs_وزن,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_وزن->get_output_partial_shape(0).rank().get_ڊيگهائي() - 1}));
            إذا (stateful) {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {3}, {(int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    خطأ);
            } وإلا {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    خطأ);
            }
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_factors);
        }

        خودڪار theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        خودڪار theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        إذا (ext_ضرب_ڪندڙ == 0.0f) {
            theta = theta_interp;
        } وإلا {
            خودڪار ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_ضرب_ڪندڙ);
            Output<Node> one;
            إذا (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } وإلا {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            }
            خودڪار one_minus_ramp = std::make_shared<ov::op::v1::ڪٽيو>(one, ramp_mix);

            theta =
                std::make_shared<ov::op::v1::جوڙيو>(std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                                                  std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::لاگرٿم(1.0f / freq_scale));
        }
    }

    Output<Node> cos_theta = std::make_shared<ov::op::v0::Cos>(theta);
    Output<Node> sin_theta = std::make_shared<ov::op::v0::Sin>(theta);

    إذا (mscale != 1.0f) {
        خودڪار mscale_ڳنڍ = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_ڳنڍ);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_ڳنڍ);
    }

    أرجع std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_داخلا(ثابت ڳنڍ_Context & context, صحيح داخلا_اشاريو, صحيح ٽڪرو_len, صحيح axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    خودڪار داخلا_2 = context.get_داخلا(داخلا_اشاريو);
    خودڪار * op_params = (size_t *) context.get_داخلا_op_params(داخلا_اشاريو);
    خودڪار src_stride = context.get_داخلا_stride(داخلا_اشاريو);

    int64_t ٽڪرو_شروعات = op_params[0] / src_stride[3];
    إذا (ٽڪرو_len == 0) {
        ٽڪرو_len = context.get_داخلا_shape(داخلا_اشاريو)[3].get_ڊيگهائي();
    }
    int64_t ٽڪرو_پڇاڙي = ٽڪرو_شروعات + ٽڪرو_len;

    خودڪار begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {ٽڪرو_شروعات});
    خودڪار end = ov::op::v0::Constant::create(ov::element::i64, {1}, {ٽڪرو_پڇاڙي});
    خودڪار stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    إذا (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } وإلا {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    خودڪار sliced = std::make_shared<ov::op::v8::ٽڪرو>(داخلا_2, begin, end, stride, axes);
    أرجع sliced;
}

ov::Output<ov::Node> process_view_داخلا_new(ثابت ڳنڍ_Context & context, صحيح داخلا_اشاريو) {
    خودڪار داخلا_2 = context.get_داخلا(داخلا_اشاريو);

    // Check if this input has view inputs
    size_t view_داخلا_سائيز = context.get_view_داخلا_سائيز(داخلا_اشاريو);
    إذا (view_داخلا_سائيز == 0) {
        // No view inputs, return the input as is
        أرجع داخلا_2;
    }

    // If translate_view already resolved this VIEW (produced a Slice), the input
    // will already have the expected shape — skip re-slicing.
    //
    // Two notions of "matches" are accepted per axis:
    //   - both dims static and equal, OR
    //   - both dims dynamic.
    // The dynamic case matters for the MoE expert-plane views: translate_view now emits a
    // DYNAMIC-token slice (so the token dim is not frozen). An all-static-only check would
    // see the dynamic token dim, decide the shapes "don't match", and fall through to
    // re-slice/flatten the already-resolved view (a Reshape to the full flattened
    // n_expert_used*n_embd tail, which then conflicts with the single-plane input). Treat a
    // dynamic-vs-dynamic axis as matching so the already-resolved view is reused as-is.
    //
    // A third case matters for split-model MoE fragments: translate_view resolves the
    // expert-plane view against the fragment's INPUT parameter. When the graph is split
    // the token axis of that parameter may already be concrete (static n_tokens) even
    // though get_view_input_ov_shape() still reports it as dynamic (-1). The resolved
    // view is then static [1,1,n_tokens,n_embd] while `expected` is [1,1,?,n_embd].
    // An "expected dynamic, actual static" axis is a valid concretization of the SAME
    // resolved view, so treat it as matching too. Falling through to process_single_view
    // here would re-slice/re-flatten the already-resolved single-plane view against the
    // recorded (multi-plane) source strides and emit a constant-target Reshape whose baked
    // dims no longer divide the concretized input -> "dimensions do not evenly divide".
    // A fourth case matters for stateful execution: `expected` comes from ggml metadata and is
    // always GGML_MAX_DIMS ranks, but stateful drops the leading size-1 batch dim, so the
    // resolved view is one rank lower. Compare the common trailing dims and require the
    // leading expected dims we skip to be 1. Without this the rank test below never matches on
    // the stateful path and every already-resolved MoE expert-plane view is re-sliced.
    خودڪار expected_ov_shape = context.get_view_داخلا_ov_shape(داخلا_اشاريو, 0);
    خودڪار actual_shape = داخلا_2.get_partial_shape();
    إذا (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_ڊيگهائي() >= actual_shape.rank().get_ڊيگهائي()) {
        ثابت int64_t n_actual = actual_shape.rank().get_ڊيگهائي();
        ثابت int64_t سرڪايو = expected_ov_shape.rank().get_ڊيگهائي() - n_actual;
        منطقي shapes_ميلاپ = سچ;
        لكل (int64_t i = 0; i < سرڪايو; ++i) {
            إذا (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_ڊيگهائي() != 1) {
                shapes_ميلاپ = خطأ;
                اكسر;
            }
        }
        لكل (int64_t i = 0; i < n_actual && shapes_ميلاپ; ++i) {
            ثابت خودڪار & قوت_نما = expected_ov_shape[i + سرڪايو];
            ثابت منطقي both_dynamic = قوت_نما.is_dynamic() && actual_shape[i].is_dynamic();
            ثابت منطقي both_static_equal =
                قوت_نما.is_static() && actual_shape[i].is_static() && قوت_نما == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            ثابت منطقي expected_dyn_actual_static = قوت_نما.is_dynamic() && actual_shape[i].is_static();
            إذا (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_ميلاپ = خطأ;
                اكسر;
            }
        }
        إذا (shapes_ميلاپ) {
            أرجع داخلا_2;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    إذا (context.is_static() && view_داخلا_سائيز == 1) {
        خودڪار view_stride_v = context.get_view_داخلا_stride(داخلا_اشاريو, 0);
        خودڪار view_src_stride_v = context.get_view_داخلا_src_stride(داخلا_اشاريو, 0);
        خودڪار view_ggml_shape = context.get_view_داخلا_ggml_shape(داخلا_اشاريو, 0);
        خودڪار view_src_ggml_shape = context.get_view_داخلا_src_ggml_shape(داخلا_اشاريو, 0);
        خودڪار view_وٿي = context.get_view_داخلا_وٿي(داخلا_اشاريو, 0);
        خودڪار view_src_وٿي = context.get_view_داخلا_src_وٿي(داخلا_اشاريو, 0);

        size_t ndims = view_ggml_shape.size();
        std::vector<صحيح> diff_dims;
        إذا (view_src_ggml_shape.size() == ndims) {
            لكل (size_t i = 0; i < ndims; ++i) {
                إذا (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<صحيح>(i));
                }
            }
        }

        إذا (diff_dims.size() == 1) {
            صحيح ورهايو_dim = diff_dims[0];
            int64_t انگ_splits = static_cast<int64_t>(view_src_ggml_shape[ورهايو_dim]);
            int64_t chunk_سائيز = static_cast<int64_t>(view_ggml_shape[ورهايو_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            إذا (chunk_سائيز == 1 && انگ_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                منطقي suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                إذا (suffix_ok) {
                    لكل (size_t i = static_cast<size_t>(ورهايو_dim) + 1; i < ndims; ++i) {
                        إذا (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = خطأ;
                            اكسر;
                        }
                    }
                }

                إذا (suffix_ok && view_src_stride_v[ورهايو_dim] > 0) {
                    size_t relative_وٿي = view_وٿي >= view_src_وٿي ? view_وٿي - view_src_وٿي : 0;
                    int64_t ورهايو_اشاريو = static_cast<int64_t>(relative_وٿي / view_src_stride_v[ورهايو_dim]);

                    إذا (ورهايو_اشاريو >= 0 && ورهايو_اشاريو < انگ_splits) {
                        خودڪار src_ڳنڍ = داخلا_2.get_ڳنڍ_shared_ptr();
                        std::string rt_ڪنجي = "split_dim_" + std::to_string(ورهايو_dim);
                        خودڪار & rt_info = src_ڳنڍ->get_rt_info();

                        إذا (rt_info.find(rt_ڪنجي) == rt_info.end()) {
                            خودڪار axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(ورهايو_dim)});
                            خودڪار ورهايو_ڳنڍ =
                                std::make_shared<ov::op::v1::ورهايو_2>(داخلا_2, axis_const, static_cast<size_t>(انگ_splits));
                            ورهايو_ڳنڍ->سيٽ_friendly_name(src_ڳنڍ->get_friendly_name() + "_split");
                            rt_info[rt_ڪنجي] = ورهايو_ڳنڍ;
                        }

                        خودڪار ورهايو_ڳنڍ = rt_info[rt_ڪنجي].as<std::shared_ptr<ov::op::v1::ورهايو_2>>();
                        أرجع ورهايو_ڳنڍ->output(static_cast<size_t>(ورهايو_اشاريو));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    خودڪار process_single_view =
        [](ov::Output<ov::Node> current, size_t view_وٿي, ثابت std::vector<size_t> & view_stride,
           ثابت ov::Shape & view_ggml_shape, ثابت ov::PartialShape & view_ov_shape, ثابت std::string & view_name,
           size_t view_src_وٿي, ثابت std::vector<size_t> & view_src_stride, ثابت ov::Shape & view_src_ggml_shape,
           ثابت ov::PartialShape & view_src_ov_shape, ثابت std::string & view_src_name) -> ov::Output<ov::Node> {
        // Stateful execution drops a leading size-1 axis, so `current` can be one rank lower
        // than the ggml shape metadata (view_stride/view_ggml_shape, always GGML_MAX_DIMS
        // entries) assumes. Shift any axis index built from that metadata down by the
        // difference before using it as an OV Slice axis.
        ثابت خودڪار current_rank = current.get_partial_shape().rank();
        ثابت صحيح axis_سرڪايو = current_rank.is_static() ?
                                    static_cast<صحيح>(view_stride.size()) - static_cast<صحيح>(current_rank.get_ڊيگهائي()) :
                                    0;

        خودڪار build_reshape_نمونو = [](ثابت ov::PartialShape & هدف_ov_shape,
                                        ثابت ov::Shape & هدف_ggml_shape) -> std::vector<int64_t> {
            ثابت size_t ndims = هدف_ggml_shape.size();
            std::vector<int64_t> reshape_نمونو(ndims);
            size_t dynamic_dims = 0;

            إذا (هدف_ov_shape.rank().is_static() &&
                هدف_ov_shape.rank().get_ڊيگهائي() == static_cast<int64_t>(ndims)) {
                لكل (size_t i = 0; i < ndims; ++i) {
                    إذا (هدف_ov_shape[i].is_static()) {
                        reshape_نمونو[i] = هدف_ov_shape[i].get_ڊيگهائي();
                    } وإلا {
                        reshape_نمونو[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } وإلا {
                dynamic_dims = 2;
            }

            إذا (dynamic_dims > 1) {
                لكل (size_t i = 0; i < ndims; ++i) {
                    reshape_نمونو[i] = static_cast<int64_t>(هدف_ggml_shape[i]);
                }
            }

            أرجع reshape_نمونو;
        };

        خودڪار build_اڳياڙي_پڇ_reshape_نمونو = [](ثابت ov::PartialShape & هدف_ov_shape,
                                                    ثابت ov::Shape & هدف_ggml_shape, size_t اڳياڙي_dims,
                                                    int64_t پڇ_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_نمونو(اڳياڙي_dims + 1);
            size_t dynamic_dims = 0;

            إذا (هدف_ov_shape.rank().is_static() &&
                هدف_ov_shape.rank().get_ڊيگهائي() == static_cast<int64_t>(هدف_ggml_shape.size())) {
                لكل (size_t i = 0; i < اڳياڙي_dims; ++i) {
                    إذا (هدف_ov_shape[i].is_static()) {
                        reshape_نمونو[i] = هدف_ov_shape[i].get_ڊيگهائي();
                    } وإلا {
                        reshape_نمونو[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } وإلا {
                dynamic_dims = 2;
            }

            إذا (dynamic_dims > 1) {
                لكل (size_t i = 0; i < اڳياڙي_dims; ++i) {
                    reshape_نمونو[i] = static_cast<int64_t>(هدف_ggml_shape[i]);
                }
            }

            reshape_نمونو[اڳياڙي_dims] = پڇ_dim;
            أرجع reshape_نمونو;
        };

        منطقي same_stride = view_stride.size() == view_src_stride.size();
        إذا (same_stride) {
            لكل (size_t i = 0; i < view_stride.size(); ++i) {
                إذا (view_stride[i] != view_src_stride[i]) {
                    same_stride = خطأ;
                    اكسر;
                }
            }
        }

        منطقي same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        إذا (same_ggml_shape) {
            لكل (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                إذا (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = خطأ;
                    اكسر;
                }
            }
        }

        إذا (same_stride && same_ggml_shape) {
            أرجع current;
        }

        إذا (same_stride) {
            ثابت size_t relative_وٿي = view_وٿي >= view_src_وٿي ? view_وٿي - view_src_وٿي : 0;
            ثابت size_t ndims = view_stride.size();

            std::vector<صحيح> diff_dims;
            إذا (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                لكل (size_t i = 0; i < ndims; ++i) {
                    إذا (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<صحيح>(i));
                    }
                }
            }

            إذا (diff_dims.size() == 1) {
                ثابت صحيح ٽڪرو_dim = diff_dims[0];
                ثابت int64_t dim_سائيز = static_cast<int64_t>(view_src_ggml_shape[ٽڪرو_dim]);

                إذا (view_stride[ٽڪرو_dim] > 0 && relative_وٿي % view_stride[ٽڪرو_dim] == 0) {
                    ثابت int64_t begin_val = static_cast<int64_t>((relative_وٿي / view_stride[ٽڪرو_dim]) %
                                                                   static_cast<size_t>(dim_سائيز));
                    ثابت int64_t پڇاڙي_val = begin_val + static_cast<int64_t>(view_ggml_shape[ٽڪرو_dim]);

                    إذا (begin_val >= 0 && پڇاڙي_val <= dim_سائيز) {
                        خودڪار sliced = std::make_shared<ov::op::v8::ٽڪرو>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {پڇاڙي_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {ٽڪرو_dim - axis_سرڪايو}));

                        إذا (view_ov_shape.is_static()) {
                            خودڪار reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.منزل_shape()),
                                خطأ);
                            reshaped->سيٽ_friendly_name(view_name);
                            أرجع reshaped;
                        }

                        sliced->سيٽ_friendly_name(view_name);
                        أرجع sliced;
                    }
                }

                int64_t پڇ_src_elems = 1;
                int64_t پڇ_dst_elems = 1;
                لكل (size_t i = ٽڪرو_dim; i < ndims; ++i) {
                    پڇ_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    پڇ_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                ثابت size_t elem_stride = view_stride[ndims - 1];
                int64_t پڇ_begin = 0;
                إذا (elem_stride > 0) {
                    پڇ_begin =
                        static_cast<int64_t>((relative_وٿي / elem_stride) % static_cast<size_t>(پڇ_src_elems));
                }
                ثابت int64_t پڇ_پڇاڙي = پڇ_begin + پڇ_dst_elems;

                إذا (پڇ_begin >= 0 && پڇ_پڇاڙي <= پڇ_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(ٽڪرو_dim);
                    لكل (صحيح i = 0; i < ٽڪرو_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(پڇ_src_elems);
                    ثابت size_t flat_ndims = flat_shape.size();

                    خودڪار flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), خطأ);

                    خودڪار sliced = std::make_shared<ov::op::v8::ٽڪرو>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {پڇ_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {پڇ_پڇاڙي}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {ٽڪرو_dim}));

                    إذا (view_ov_shape.is_static()) {
                        خودڪار reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.منزل_shape()),
                            خطأ);
                        reshaped->سيٽ_friendly_name(view_name);
                        أرجع reshaped;
                    }

                    sliced->سيٽ_friendly_name(view_name);
                    أرجع sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> قدم(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            size_t بچيل_وٿي = relative_وٿي;
            لكل (size_t i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                إذا (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(بچيل_وٿي / view_stride[i]);
                    بچيل_وٿي %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            منطقي اندريون_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            إذا (اندريون_bounds) {
                لكل (size_t i = 0; i < ndims; ++i) {
                    إذا (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        اندريون_bounds = خطأ;
                        اكسر;
                    }
                }
            }

            إذا (اندريون_bounds && بچيل_وٿي == 0) {
                خودڪار sliced = std::make_shared<ov::op::v8::ٽڪرو>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, قدم),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->سيٽ_friendly_name(view_name);
                أرجع sliced;
            }
        } وإلا {
            منطقي same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            ثابت size_t relative_وٿي = view_وٿي >= view_src_وٿي ? view_وٿي - view_src_وٿي : 0;

            إذا (same_rank) {
                ثابت size_t ndims = view_ggml_shape.size();
                std::vector<صحيح> diff_dims;
                لكل (size_t i = 0; i < ndims; ++i) {
                    إذا (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<صحيح>(i));
                    }
                }

                إذا (diff_dims.size() == 1) {
                    ثابت size_t ٽڪرو_dim = static_cast<size_t>(diff_dims[0]);
                    منطقي suffix_stride_ميلاپ = سچ;
                    لكل (size_t i = ٽڪرو_dim + 1; i < ndims; ++i) {
                        إذا (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_ميلاپ = خطأ;
                            اكسر;
                        }
                    }

                    إذا (suffix_stride_ميلاپ && view_src_stride[ٽڪرو_dim] > 0 &&
                        relative_وٿي % view_src_stride[ٽڪرو_dim] == 0) {
                        ثابت int64_t begin_val = static_cast<int64_t>(relative_وٿي / view_src_stride[ٽڪرو_dim]);
                        ثابت int64_t پڇاڙي_val = begin_val + static_cast<int64_t>(view_ggml_shape[ٽڪرو_dim]);
                        ثابت int64_t dim_سائيز = static_cast<int64_t>(view_src_ggml_shape[ٽڪرو_dim]);

                        إذا (begin_val >= 0 && پڇاڙي_val <= dim_سائيز) {
                            خودڪار sliced = std::make_shared<ov::op::v8::ٽڪرو>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {پڇاڙي_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(ٽڪرو_dim)}));
                            sliced->سيٽ_friendly_name(view_name);
                            أرجع sliced;
                        }
                    }
                }
            }

            size_t view_elems = 1;
            size_t src_elems = 1;
            إذا (same_rank) {
                لكل (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            منطقي same_انگ_elements = same_rank && view_elems == src_elems;

            إذا (same_rank && relative_وٿي == 0 && same_انگ_elements) {
                خودڪار reshape_نمونو = build_reshape_نمونو(view_ov_shape, view_ggml_shape);

                خودڪار reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_نمونو.size()}, reshape_نمونو),
                    خطأ);
                reshaped->سيٽ_friendly_name(view_name);
                أرجع reshaped;
            }

            إذا (same_rank) {
                ثابت size_t ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                منطقي is_regular_ٽڪرو = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> قدم(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                size_t بچيل_وٿي = relative_وٿي;

                إذا (is_regular_ٽڪرو) {
                    لكل (size_t i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        إذا (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_ٽڪرو = خطأ;
                            اكسر;
                        }

                        قدم[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        إذا (قدم[i] <= 0) {
                            is_regular_ٽڪرو = خطأ;
                            اكسر;
                        }

                        begin[i] = static_cast<int64_t>(بچيل_وٿي / view_src_stride[i]);
                        بچيل_وٿي %= view_src_stride[i];

                        إذا (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            استمر;
                        }

                        end[i] = begin[i] + قدم[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        إذا (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_ٽڪرو = خطأ;
                            اكسر;
                        }
                    }
                }

                إذا (is_regular_ٽڪرو && بچيل_وٿي == 0) {
                    خودڪار sliced = std::make_shared<ov::op::v8::ٽڪرو>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, قدم),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->سيٽ_friendly_name(view_name);
                    أرجع sliced;
                }

                ثابت size_t elem_stride = view_src_stride.back();
                ثابت منطقي aligned_وٿي = elem_stride > 0 && relative_وٿي % elem_stride == 0;

                إذا (aligned_وٿي) {
                    size_t suffix_شروعات = 0;
                    size_t expected_stride = elem_stride;
                    لكل (صحيح i = static_cast<صحيح>(ndims) - 1; i >= 0; --i) {
                        إذا (view_stride[i] != expected_stride) {
                            suffix_شروعات = static_cast<size_t>(i + 1);
                            اكسر;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    size_t اڳياڙي_elems = 1;
                    size_t suffix_elems = 1;
                    لكل (size_t i = 0; i < suffix_شروعات; ++i) {
                        اڳياڙي_elems *= view_ggml_shape[i];
                    }
                    لكل (size_t i = suffix_شروعات; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    إذا (اڳياڙي_elems > 0 && src_elems % اڳياڙي_elems == 0) {
                        ثابت size_t src_پڇ_elems = src_elems / اڳياڙي_elems;
                        ثابت int64_t پڇ_begin = static_cast<int64_t>(relative_وٿي / elem_stride);
                        ثابت int64_t پڇ_پڇاڙي = پڇ_begin + static_cast<int64_t>(suffix_elems);

                        إذا (پڇ_begin >= 0 && پڇ_پڇاڙي <= static_cast<int64_t>(src_پڇ_elems)) {
                            خودڪار اڳياڙي_پڇ_نمونو = build_اڳياڙي_پڇ_reshape_نمونو(
                                view_ov_shape, view_ggml_shape, suffix_شروعات, static_cast<int64_t>(src_پڇ_elems));

                            خودڪار اڳياڙي_پڇ = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {اڳياڙي_پڇ_نمونو.size()},
                                                             اڳياڙي_پڇ_نمونو),
                                خطأ);

                            ov::Output<ov::Node> selected = اڳياڙي_پڇ;
                            إذا (پڇ_begin != 0 || پڇ_پڇاڙي != static_cast<int64_t>(src_پڇ_elems)) {
                                selected = std::make_shared<ov::op::v8::ٽڪرو>(
                                    اڳياڙي_پڇ, ov::op::v0::Constant::create(ov::element::i64, {1}, {پڇ_begin}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {پڇ_پڇاڙي}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {static_cast<int64_t>(suffix_شروعات)}));
                            }

                            خودڪار reshape_نمونو = build_reshape_نمونو(view_ov_shape, view_ggml_shape);
                            خودڪار reshaped = std::make_shared<ov::op::v1::Reshape>(
                                selected,
                                ov::op::v0::Constant::create(ov::element::i64, {reshape_نمونو.size()},
                                                             reshape_نمونو),
                                خطأ);
                            reshaped->سيٽ_friendly_name(view_name);
                            أرجع reshaped;
                        }
                    }
                }
            }

            أرجع current;
        }

        (فراغ) view_name;
        (فراغ) view_src_ov_shape;
        (فراغ) view_src_name;

        أرجع current;
    };

    // Special case: ggml collapses VIEW-of-VIEW chains so that `view_offs` is always an
    // ABSOLUTE offset from the true root allocation, regardless of how many VIEW levels
    // are in between (see ggml_new_tensor_impl). `src[0]` is still the immediate op-graph
    // parent though, which can be a DIFFERENT (already narrowed) VIEW with the SAME ggml
    // shape as this one but a different absolute offset -- e.g. a per-layer deepstack
    // slice `view_2d(embd, n_embd, n_tokens, embd->nb[1], layer*n_embd*sizeof(float))`
    // whose src[0] ("embd") is itself already a zero-offset VIEW of the true root (the
    // padded embedding). Chaining through "embd" here would try to re-slice an already
    // 2-narrowed tensor using a root-relative offset, going out of bounds and silently
    // falling back to a no-op (returning the wrong, already-resolved sibling slice).
    // Detect this (same shape as the immediate src, but different absolute offset) and
    // re-slice directly from the untouched root using the innermost view's absolute
    // offset against the ROOT's own shape/stride instead of chaining through src[0].
    {
        خودڪار innermost_وٿي = context.get_view_داخلا_وٿي(داخلا_اشاريو, 0);
        خودڪار innermost_src_وٿي = context.get_view_داخلا_src_وٿي(داخلا_اشاريو, 0);
        خودڪار innermost_shape = context.get_view_داخلا_ggml_shape(داخلا_اشاريو, 0);
        خودڪار innermost_src_shape = context.get_view_داخلا_src_ggml_shape(داخلا_اشاريو, 0);
        إذا (innermost_وٿي != innermost_src_وٿي && innermost_shape == innermost_src_shape) {
            size_t پاڙ_view_idx = view_داخلا_سائيز - 1;
            خودڪار پاڙ_ggml_shape = context.get_view_داخلا_src_ggml_shape(داخلا_اشاريو, پاڙ_view_idx);
            خودڪار پاڙ_stride = context.get_view_داخلا_src_stride(داخلا_اشاريو, پاڙ_view_idx);
            خودڪار پاڙ_وٿي = context.get_view_داخلا_src_وٿي(داخلا_اشاريو, پاڙ_view_idx);
            خودڪار پاڙ_ov_shape = context.get_view_داخلا_src_ov_shape(داخلا_اشاريو, پاڙ_view_idx);
            خودڪار پاڙ_name = context.get_view_داخلا_src_name(داخلا_اشاريو, پاڙ_view_idx);
            خودڪار innermost_stride = context.get_view_داخلا_stride(داخلا_اشاريو, 0);
            خودڪار innermost_ov_shape = context.get_view_داخلا_ov_shape(داخلا_اشاريو, 0);
            خودڪار innermost_name = context.get_view_داخلا_name(داخلا_اشاريو, 0);

            أرجع process_single_view(داخلا_2, innermost_وٿي, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, پاڙ_وٿي, پاڙ_stride, پاڙ_ggml_shape, پاڙ_ov_shape,
                                       پاڙ_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = داخلا_2;

    // Process each view in reverse order (from base to current)
    لكل (صحيح view_idx = view_داخلا_سائيز - 1; view_idx >= 0; view_idx--) {
        خودڪار view_وٿي = context.get_view_داخلا_وٿي(داخلا_اشاريو, view_idx);
        خودڪار view_stride = context.get_view_داخلا_stride(داخلا_اشاريو, view_idx);
        خودڪار view_ggml_shape = context.get_view_داخلا_ggml_shape(داخلا_اشاريو, view_idx);
        خودڪار view_ov_shape = context.get_view_داخلا_ov_shape(داخلا_اشاريو, view_idx);
        خودڪار view_name = context.get_view_داخلا_name(داخلا_اشاريو, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        خودڪار view_src_وٿي = context.get_view_داخلا_src_وٿي(داخلا_اشاريو, view_idx);
        خودڪار view_src_stride = context.get_view_داخلا_src_stride(داخلا_اشاريو, view_idx);
        خودڪار view_src_ggml_shape = context.get_view_داخلا_src_ggml_shape(داخلا_اشاريو, view_idx);
        خودڪار view_src_ov_shape = context.get_view_داخلا_src_ov_shape(داخلا_اشاريو, view_idx);
        خودڪار view_src_name = context.get_view_داخلا_src_name(داخلا_اشاريو, view_idx);
        // print source view info
        // std::cout << "View " << view_idx << ": source name = " << view_src_name
        //           << ", source offset = " << view_src_offset << ", source stride = [" << view_src_stride[0] << ","
        //           << view_src_stride[1] << "," << view_src_stride[2] << "," << view_src_stride[3]
        //           << "], source ggml shape = [" << view_src_ggml_shape[0] << "," << view_src_ggml_shape[1] << ","
        //           << view_src_ggml_shape[2] << "," << view_src_ggml_shape[3]
        //           << "], source ov shape = " << view_src_ov_shape << std::endl;

        current = process_single_view(current, view_وٿي, view_stride, view_ggml_shape, view_ov_shape, view_name,
                                      view_src_وٿي, view_src_stride, view_src_ggml_shape, view_src_ov_shape,
                                      view_src_name);
    }

    أرجع current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
