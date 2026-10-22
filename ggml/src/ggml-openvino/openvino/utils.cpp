#شامل "utils.h"

#شامل <cmath>
#شامل <cstddef>
#شامل <ctime>
#شامل <memory>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/clamp.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/cos.hpp>
#شامل <openvino/op/divide.hpp>
#شامل <openvino/op/gather.hpp>
#شامل <openvino/op/maximum.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/shape_of.hpp>
#شامل <openvino/op/sin.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/split.hpp>
#شامل <openvino/op/squeeze.hpp>
#شامل <openvino/op/subtract.hpp>
#شامل <openvino/op/transpose.hpp>
#شامل <string>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {

خالی عدد_inputs_check(ثابت گرہ_Context & context, size_t min_inputs, size_t max_inputs) {
    خودکار داخلہ_سائز = context.get_داخلہ_سائز();
    FRONT_انتہا_OP_CONVERSION_CHECK(داخلہ_سائز >= min_inputs, "Got less inputs than expected");
    FRONT_انتہا_OP_CONVERSION_CHECK(داخلہ_سائز <= max_inputs, "Got more inputs than expected");
}

صحیح non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    صحیح dim = nb.size() - 1;
    size_t بائٹیں = nb[dim];
    برائے (صحیح i = dim; i > 0; i--) {
        بائٹیں *= ne[i];
        اگر (بائٹیں != nb[i - 1]) {
            واپس i;
        }
    }
    واپس 0;
}

std::shared_ptr<ov::Node> get_dimensions(ثابت std::shared_ptr<ov::op::v3::Shape_کا> & shape,
                                         ثابت std::vector<صحیح> & dims) {
    استعمال نامخلا ov::op;
    ثابت خودکار zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ثابت خودکار dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    واپس std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(ثابت std::shared_ptr<ov::Node> & گرہ, ثابت std::vector<صحیح> & dims) {
    واپس get_dimensions(std::make_shared<ov::op::v3::Shape_کا>(گرہ), dims);
}

Output_سمتیہ rename_outputs_with_suffix(ثابت Output_سمتیہ & outputs, ثابت std::string & suffix) {
    برائے (ثابت خودکار & output : outputs) {
        خودکار گرہ = output.get_گرہ_shared_ptr();
        std::string name = گرہ->get_friendly_name();
        name += "_";
        name += suffix;
        گرہ->سیٹ_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    واپس outputs;
}

نامخلا {
ov::Output<ov::Node> rope_yarn_ramp_mix(صحیح عدد_elements, ثابت اعشاری corr_dims[2], اعشاری ext_ضربی_عدد) {
    std::vector<اعشاری> dim_ids_vec(عدد_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    خودکار dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) عدد_elements}, dim_ids_vec);
    خودکار corr_کم = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    خودکار corr_بلند = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    خودکار denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::تفریق>(corr_بلند, corr_کم),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    خودکار ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::تفریق>(dim_ids, corr_کم), denom);
    خودکار ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    خودکار one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    خودکار ramp_inverted = std::make_shared<ov::op::v1::تفریق>(one, ramp_clamped);
    خودکار ext_ضربی_عدد_گرہ = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_ضربی_عدد});
    خودکار ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_ضربی_عدد_گرہ);
    واپس ramp_mix;
}

اعشاری ggml_rope_yarn_corr_dim(صحیح n_dims, صحیح n_ctx_orig, اعشاری n_rot, اعشاری بنیاد) {
#اگر_غیر_تعریف_شدہ M_PI
#    define M_PI 3.14159265358979323846
#اگر_ختم
    واپس n_dims * logf(n_ctx_orig / (n_rot * 2 * (اعشاری) M_PI)) / (2 * logf(بنیاد));
}

خالی ggml_rope_yarn_corr_dims(صحیح n_dims,
                              صحیح n_ctx_orig,
                              اعشاری freq_بنیاد,
                              اعشاری beta_fast,
                              اعشاری beta_slow,
                              اعشاری dims[2]) {
    اعشاری start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_بنیاد));
    اعشاری end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_بنیاد));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<اعشاری>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_وزن,
                                                           صحیح mode,
                                                           منطقی stateful,
                                                           int64_t سر_dim) {
    constexpr صحیح TYPE_IMROPE = 2;
    constexpr صحیح TYPE_VISION = 3;
    constexpr صحیح TYPE_MROPE = 4;

    ثابت منطقی is_imrope = (mode == TYPE_IMROPE);
    ثابت منطقی is_vision = (mode == TYPE_VISION);
    ثابت منطقی is_mrope = (mode == TYPE_MROPE);
    ثابت منطقی is_multi = is_imrope || is_vision || is_mrope;

    اگر (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        خودکار pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } ورنہ اگر (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        خودکار pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, سچ);
        خودکار pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } ورنہ {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        خودکار pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    اعشاری freq_بنیاد;
    اعشاری freq_scale;
    اعشاری ext_ضربی_عدد;
    اعشاری attn_ضربی_عدد;
    اعشاری beta_fast;
    اعشاری beta_slow;
    ثابت صحیح n_dims = rope_params[1];
    ثابت size_t n_dims_half = n_dims >> 1;
    ثابت صحیح n_ctx_orig = rope_params[4];
    حافظہ_نقل(&freq_بنیاد, rope_params + 5, حجم(اعشاری));
    حافظہ_نقل(&freq_scale, rope_params + 6, حجم(اعشاری));
    حافظہ_نقل(&ext_ضربی_عدد, rope_params + 7, حجم(اعشاری));
    حافظہ_نقل(&attn_ضربی_عدد, rope_params + 8, حجم(اعشاری));
    حافظہ_نقل(&beta_fast, rope_params + 9, حجم(اعشاری));
    حافظہ_نقل(&beta_slow, rope_params + 10, حجم(اعشاری));

    ثابت اعشاری theta_scale = powf(freq_بنیاد, -2.0f / n_dims);

    Output<Node> theta;
    اعشاری mscale = attn_ضربی_عدد;
    اگر (is_multi) {
        ثابت size_t عدد_جوڑے = is_vision ? (n_dims > 0 ? (size_t) n_dims : (size_t) (سر_dim / 2)) : n_dims_half;
        صحیح sections[4];
        حافظہ_نقل(sections, rope_params + 11, حجم(صحیح) * 4);
        صحیح sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        اگر (sect_dims <= 0) {
            sect_dims = عدد_جوڑے;
        }
        صحیح sec_w = sections[0] + sections[1];
        صحیح sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(عدد_جوڑے);
        std::vector<اعشاری> ضربی_عدد(عدد_جوڑے);
        برائے (size_t j = 0; j < عدد_جوڑے; j++) {
            صحیح sector = j % sect_dims;
            صحیح plane = 0;
            اگر (is_imrope) {
                اگر (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } ورنہ اگر (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } ورنہ اگر (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } ورنہ {
                    plane = 3;
                }
                ضربی_عدد[j] = std::قوت(theta_scale, j);
            } ورنہ اگر (is_vision) {
                صحیح p_idx = 0;
                اگر (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } ورنہ اگر (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } ورنہ اگر (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } ورنہ {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                ضربی_عدد[j] = std::قوت(theta_scale, p_idx);
            } ورنہ {
                اگر (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } ورنہ اگر (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } ورنہ اگر (sector >= sec_e) {
                    plane = 3;
                } ورنہ {
                    plane = 0;
                }
                ضربی_عدد[j] = std::قوت(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        خودکار gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{عدد_جوڑے}, gather_indices);
        خودکار gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> ضربی_عدد_گرہ =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{عدد_جوڑے}, ضربی_عدد);
        اگر (rope_freqs_وزن) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::قاش>(
                rope_freqs_وزن,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) عدد_جوڑے}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_وزن->get_output_partial_shape(0).rank().get_لمبائی() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) عدد_جوڑے}),
                جھوٹ);
            ضربی_عدد_گرہ = std::make_shared<ov::op::v1::Divide>(ضربی_عدد_گرہ, rope_factors);
        }
        خودکار theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, ضربی_عدد_گرہ);
        خودکار theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        اگر (ext_ضربی_عدد == 0.0f) {
            theta = theta_interp;
        } ورنہ {
            اعشاری corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_بنیاد, beta_fast, beta_slow, corr_dims);
            خودکار ramp_mix = rope_yarn_ramp_mix(عدد_جوڑے, corr_dims, ext_ضربی_عدد);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            خودکار one_minus_ramp = std::make_shared<ov::op::v1::تفریق>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::جوڑو>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::لوگارتھم(1.0f / freq_scale));
        }
    } ورنہ {
        std::vector<اعشاری> ضربی_عدد(n_dims_half);
        Output<Node> freq_factors;
        اعشاری corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_بنیاد, beta_fast, beta_slow, corr_dims);
        ضربی_عدد[0] = 1.0f;
        برائے (size_t i = 1; i < ضربی_عدد.size(); i++) {
            ضربی_عدد[i] = theta_scale * ضربی_عدد[i - 1];
        }
        اگر (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, ضربی_عدد.size()}, ضربی_عدد);
        } ورنہ {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, ضربی_عدد.size()}, ضربی_عدد);
        }
        اگر (rope_freqs_وزن) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::قاش>(
                rope_freqs_وزن,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_وزن->get_output_partial_shape(0).rank().get_لمبائی() - 1}));
            اگر (stateful) {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {3}, {(int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    جھوٹ);
            } ورنہ {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    جھوٹ);
            }
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_factors);
        }

        خودکار theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        خودکار theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        اگر (ext_ضربی_عدد == 0.0f) {
            theta = theta_interp;
        } ورنہ {
            خودکار ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_ضربی_عدد);
            Output<Node> one;
            اگر (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } ورنہ {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            }
            خودکار one_minus_ramp = std::make_shared<ov::op::v1::تفریق>(one, ramp_mix);

            theta =
                std::make_shared<ov::op::v1::جوڑو>(std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                                                  std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::لوگارتھم(1.0f / freq_scale));
        }
    }

    Output<Node> cos_theta = std::make_shared<ov::op::v0::Cos>(theta);
    Output<Node> sin_theta = std::make_shared<ov::op::v0::Sin>(theta);

    اگر (mscale != 1.0f) {
        خودکار mscale_گرہ = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_گرہ);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_گرہ);
    }

    واپس std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_داخلہ(ثابت گرہ_Context & context, صحیح داخلہ_اشاریہ, صحیح قاش_len, صحیح axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    خودکار داخلہ = context.get_داخلہ(داخلہ_اشاریہ);
    خودکار * op_params = (size_t *) context.get_داخلہ_op_params(داخلہ_اشاریہ);
    خودکار src_stride = context.get_داخلہ_stride(داخلہ_اشاریہ);

    int64_t قاش_آغاز = op_params[0] / src_stride[3];
    اگر (قاش_len == 0) {
        قاش_len = context.get_داخلہ_shape(داخلہ_اشاریہ)[3].get_لمبائی();
    }
    int64_t قاش_انتہا = قاش_آغاز + قاش_len;

    خودکار begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {قاش_آغاز});
    خودکار end = ov::op::v0::Constant::create(ov::element::i64, {1}, {قاش_انتہا});
    خودکار stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    اگر (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } ورنہ {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    خودکار sliced = std::make_shared<ov::op::v8::قاش>(داخلہ, begin, end, stride, axes);
    واپس sliced;
}

ov::Output<ov::Node> process_view_داخلہ_new(ثابت گرہ_Context & context, صحیح داخلہ_اشاریہ) {
    خودکار داخلہ = context.get_داخلہ(داخلہ_اشاریہ);

    // Check if this input has view inputs
    size_t view_داخلہ_سائز = context.get_view_داخلہ_سائز(داخلہ_اشاریہ);
    اگر (view_داخلہ_سائز == 0) {
        // No view inputs, return the input as is
        واپس داخلہ;
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
    خودکار expected_ov_shape = context.get_view_داخلہ_ov_shape(داخلہ_اشاریہ, 0);
    خودکار actual_shape = داخلہ.get_partial_shape();
    اگر (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_لمبائی() >= actual_shape.rank().get_لمبائی()) {
        ثابت int64_t n_actual = actual_shape.rank().get_لمبائی();
        ثابت int64_t سرکاؤ = expected_ov_shape.rank().get_لمبائی() - n_actual;
        منطقی shapes_مطابقت = سچ;
        برائے (int64_t i = 0; i < سرکاؤ; ++i) {
            اگر (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_لمبائی() != 1) {
                shapes_مطابقت = جھوٹ;
                توڑو;
            }
        }
        برائے (int64_t i = 0; i < n_actual && shapes_مطابقت; ++i) {
            ثابت خودکار & قوت_نما = expected_ov_shape[i + سرکاؤ];
            ثابت منطقی both_dynamic = قوت_نما.is_dynamic() && actual_shape[i].is_dynamic();
            ثابت منطقی both_static_equal =
                قوت_نما.is_static() && actual_shape[i].is_static() && قوت_نما == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            ثابت منطقی expected_dyn_actual_static = قوت_نما.is_dynamic() && actual_shape[i].is_static();
            اگر (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_مطابقت = جھوٹ;
                توڑو;
            }
        }
        اگر (shapes_مطابقت) {
            واپس داخلہ;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    اگر (context.is_static() && view_داخلہ_سائز == 1) {
        خودکار view_stride_v = context.get_view_داخلہ_stride(داخلہ_اشاریہ, 0);
        خودکار view_src_stride_v = context.get_view_داخلہ_src_stride(داخلہ_اشاریہ, 0);
        خودکار view_ggml_shape = context.get_view_داخلہ_ggml_shape(داخلہ_اشاریہ, 0);
        خودکار view_src_ggml_shape = context.get_view_داخلہ_src_ggml_shape(داخلہ_اشاریہ, 0);
        خودکار view_انحراف = context.get_view_داخلہ_انحراف(داخلہ_اشاریہ, 0);
        خودکار view_src_انحراف = context.get_view_داخلہ_src_انحراف(داخلہ_اشاریہ, 0);

        size_t ndims = view_ggml_shape.size();
        std::vector<صحیح> diff_dims;
        اگر (view_src_ggml_shape.size() == ndims) {
            برائے (size_t i = 0; i < ndims; ++i) {
                اگر (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<صحیح>(i));
                }
            }
        }

        اگر (diff_dims.size() == 1) {
            صحیح بانٹو_dim = diff_dims[0];
            int64_t عدد_splits = static_cast<int64_t>(view_src_ggml_shape[بانٹو_dim]);
            int64_t chunk_سائز = static_cast<int64_t>(view_ggml_shape[بانٹو_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            اگر (chunk_سائز == 1 && عدد_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                منطقی suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                اگر (suffix_ok) {
                    برائے (size_t i = static_cast<size_t>(بانٹو_dim) + 1; i < ndims; ++i) {
                        اگر (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = جھوٹ;
                            توڑو;
                        }
                    }
                }

                اگر (suffix_ok && view_src_stride_v[بانٹو_dim] > 0) {
                    size_t relative_انحراف = view_انحراف >= view_src_انحراف ? view_انحراف - view_src_انحراف : 0;
                    int64_t بانٹو_اشاریہ = static_cast<int64_t>(relative_انحراف / view_src_stride_v[بانٹو_dim]);

                    اگر (بانٹو_اشاریہ >= 0 && بانٹو_اشاریہ < عدد_splits) {
                        خودکار src_گرہ = داخلہ.get_گرہ_shared_ptr();
                        std::string rt_کلید = "split_dim_" + std::to_string(بانٹو_dim);
                        خودکار & rt_info = src_گرہ->get_rt_info();

                        اگر (rt_info.find(rt_کلید) == rt_info.end()) {
                            خودکار axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(بانٹو_dim)});
                            خودکار بانٹو_گرہ =
                                std::make_shared<ov::op::v1::بانٹو_2>(داخلہ, axis_const, static_cast<size_t>(عدد_splits));
                            بانٹو_گرہ->سیٹ_friendly_name(src_گرہ->get_friendly_name() + "_split");
                            rt_info[rt_کلید] = بانٹو_گرہ;
                        }

                        خودکار بانٹو_گرہ = rt_info[rt_کلید].as<std::shared_ptr<ov::op::v1::بانٹو_2>>();
                        واپس بانٹو_گرہ->output(static_cast<size_t>(بانٹو_اشاریہ));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    خودکار process_single_view =
        [](ov::Output<ov::Node> current, size_t view_انحراف, ثابت std::vector<size_t> & view_stride,
           ثابت ov::Shape & view_ggml_shape, ثابت ov::PartialShape & view_ov_shape, ثابت std::string & view_name,
           size_t view_src_انحراف, ثابت std::vector<size_t> & view_src_stride, ثابت ov::Shape & view_src_ggml_shape,
           ثابت ov::PartialShape & view_src_ov_shape, ثابت std::string & view_src_name) -> ov::Output<ov::Node> {
        // Stateful execution drops a leading size-1 axis, so `current` can be one rank lower
        // than the ggml shape metadata (view_stride/view_ggml_shape, always GGML_MAX_DIMS
        // entries) assumes. Shift any axis index built from that metadata down by the
        // difference before using it as an OV Slice axis.
        ثابت خودکار current_rank = current.get_partial_shape().rank();
        ثابت صحیح axis_سرکاؤ = current_rank.is_static() ?
                                    static_cast<صحیح>(view_stride.size()) - static_cast<صحیح>(current_rank.get_لمبائی()) :
                                    0;

        خودکار build_reshape_نمونہ = [](ثابت ov::PartialShape & ہدف_ov_shape,
                                        ثابت ov::Shape & ہدف_ggml_shape) -> std::vector<int64_t> {
            ثابت size_t ndims = ہدف_ggml_shape.size();
            std::vector<int64_t> reshape_نمونہ(ndims);
            size_t dynamic_dims = 0;

            اگر (ہدف_ov_shape.rank().is_static() &&
                ہدف_ov_shape.rank().get_لمبائی() == static_cast<int64_t>(ndims)) {
                برائے (size_t i = 0; i < ndims; ++i) {
                    اگر (ہدف_ov_shape[i].is_static()) {
                        reshape_نمونہ[i] = ہدف_ov_shape[i].get_لمبائی();
                    } ورنہ {
                        reshape_نمونہ[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } ورنہ {
                dynamic_dims = 2;
            }

            اگر (dynamic_dims > 1) {
                برائے (size_t i = 0; i < ndims; ++i) {
                    reshape_نمونہ[i] = static_cast<int64_t>(ہدف_ggml_shape[i]);
                }
            }

            واپس reshape_نمونہ;
        };

        خودکار build_سابقہ_دم_reshape_نمونہ = [](ثابت ov::PartialShape & ہدف_ov_shape,
                                                    ثابت ov::Shape & ہدف_ggml_shape, size_t سابقہ_dims,
                                                    int64_t دم_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_نمونہ(سابقہ_dims + 1);
            size_t dynamic_dims = 0;

            اگر (ہدف_ov_shape.rank().is_static() &&
                ہدف_ov_shape.rank().get_لمبائی() == static_cast<int64_t>(ہدف_ggml_shape.size())) {
                برائے (size_t i = 0; i < سابقہ_dims; ++i) {
                    اگر (ہدف_ov_shape[i].is_static()) {
                        reshape_نمونہ[i] = ہدف_ov_shape[i].get_لمبائی();
                    } ورنہ {
                        reshape_نمونہ[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } ورنہ {
                dynamic_dims = 2;
            }

            اگر (dynamic_dims > 1) {
                برائے (size_t i = 0; i < سابقہ_dims; ++i) {
                    reshape_نمونہ[i] = static_cast<int64_t>(ہدف_ggml_shape[i]);
                }
            }

            reshape_نمونہ[سابقہ_dims] = دم_dim;
            واپس reshape_نمونہ;
        };

        منطقی same_stride = view_stride.size() == view_src_stride.size();
        اگر (same_stride) {
            برائے (size_t i = 0; i < view_stride.size(); ++i) {
                اگر (view_stride[i] != view_src_stride[i]) {
                    same_stride = جھوٹ;
                    توڑو;
                }
            }
        }

        منطقی same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        اگر (same_ggml_shape) {
            برائے (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                اگر (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = جھوٹ;
                    توڑو;
                }
            }
        }

        اگر (same_stride && same_ggml_shape) {
            واپس current;
        }

        اگر (same_stride) {
            ثابت size_t relative_انحراف = view_انحراف >= view_src_انحراف ? view_انحراف - view_src_انحراف : 0;
            ثابت size_t ndims = view_stride.size();

            std::vector<صحیح> diff_dims;
            اگر (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                برائے (size_t i = 0; i < ndims; ++i) {
                    اگر (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<صحیح>(i));
                    }
                }
            }

            اگر (diff_dims.size() == 1) {
                ثابت صحیح قاش_dim = diff_dims[0];
                ثابت int64_t dim_سائز = static_cast<int64_t>(view_src_ggml_shape[قاش_dim]);

                اگر (view_stride[قاش_dim] > 0 && relative_انحراف % view_stride[قاش_dim] == 0) {
                    ثابت int64_t begin_val = static_cast<int64_t>((relative_انحراف / view_stride[قاش_dim]) %
                                                                   static_cast<size_t>(dim_سائز));
                    ثابت int64_t انتہا_val = begin_val + static_cast<int64_t>(view_ggml_shape[قاش_dim]);

                    اگر (begin_val >= 0 && انتہا_val <= dim_سائز) {
                        خودکار sliced = std::make_shared<ov::op::v8::قاش>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {انتہا_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {قاش_dim - axis_سرکاؤ}));

                        اگر (view_ov_shape.is_static()) {
                            خودکار reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.مقصد_shape()),
                                جھوٹ);
                            reshaped->سیٹ_friendly_name(view_name);
                            واپس reshaped;
                        }

                        sliced->سیٹ_friendly_name(view_name);
                        واپس sliced;
                    }
                }

                int64_t دم_src_elems = 1;
                int64_t دم_dst_elems = 1;
                برائے (size_t i = قاش_dim; i < ndims; ++i) {
                    دم_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    دم_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                ثابت size_t elem_stride = view_stride[ndims - 1];
                int64_t دم_begin = 0;
                اگر (elem_stride > 0) {
                    دم_begin =
                        static_cast<int64_t>((relative_انحراف / elem_stride) % static_cast<size_t>(دم_src_elems));
                }
                ثابت int64_t دم_انتہا = دم_begin + دم_dst_elems;

                اگر (دم_begin >= 0 && دم_انتہا <= دم_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(قاش_dim);
                    برائے (صحیح i = 0; i < قاش_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(دم_src_elems);
                    ثابت size_t flat_ndims = flat_shape.size();

                    خودکار flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), جھوٹ);

                    خودکار sliced = std::make_shared<ov::op::v8::قاش>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {دم_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {دم_انتہا}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {قاش_dim}));

                    اگر (view_ov_shape.is_static()) {
                        خودکار reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.مقصد_shape()),
                            جھوٹ);
                        reshaped->سیٹ_friendly_name(view_name);
                        واپس reshaped;
                    }

                    sliced->سیٹ_friendly_name(view_name);
                    واپس sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> قدم(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            size_t بقایا_انحراف = relative_انحراف;
            برائے (size_t i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                اگر (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(بقایا_انحراف / view_stride[i]);
                    بقایا_انحراف %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            منطقی اندر_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            اگر (اندر_bounds) {
                برائے (size_t i = 0; i < ndims; ++i) {
                    اگر (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        اندر_bounds = جھوٹ;
                        توڑو;
                    }
                }
            }

            اگر (اندر_bounds && بقایا_انحراف == 0) {
                خودکار sliced = std::make_shared<ov::op::v8::قاش>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, قدم),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->سیٹ_friendly_name(view_name);
                واپس sliced;
            }
        } ورنہ {
            منطقی same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            ثابت size_t relative_انحراف = view_انحراف >= view_src_انحراف ? view_انحراف - view_src_انحراف : 0;

            اگر (same_rank) {
                ثابت size_t ndims = view_ggml_shape.size();
                std::vector<صحیح> diff_dims;
                برائے (size_t i = 0; i < ndims; ++i) {
                    اگر (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<صحیح>(i));
                    }
                }

                اگر (diff_dims.size() == 1) {
                    ثابت size_t قاش_dim = static_cast<size_t>(diff_dims[0]);
                    منطقی suffix_stride_مطابقت = سچ;
                    برائے (size_t i = قاش_dim + 1; i < ndims; ++i) {
                        اگر (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_مطابقت = جھوٹ;
                            توڑو;
                        }
                    }

                    اگر (suffix_stride_مطابقت && view_src_stride[قاش_dim] > 0 &&
                        relative_انحراف % view_src_stride[قاش_dim] == 0) {
                        ثابت int64_t begin_val = static_cast<int64_t>(relative_انحراف / view_src_stride[قاش_dim]);
                        ثابت int64_t انتہا_val = begin_val + static_cast<int64_t>(view_ggml_shape[قاش_dim]);
                        ثابت int64_t dim_سائز = static_cast<int64_t>(view_src_ggml_shape[قاش_dim]);

                        اگر (begin_val >= 0 && انتہا_val <= dim_سائز) {
                            خودکار sliced = std::make_shared<ov::op::v8::قاش>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {انتہا_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(قاش_dim)}));
                            sliced->سیٹ_friendly_name(view_name);
                            واپس sliced;
                        }
                    }
                }
            }

            size_t view_elems = 1;
            size_t src_elems = 1;
            اگر (same_rank) {
                برائے (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            منطقی same_عدد_elements = same_rank && view_elems == src_elems;

            اگر (same_rank && relative_انحراف == 0 && same_عدد_elements) {
                خودکار reshape_نمونہ = build_reshape_نمونہ(view_ov_shape, view_ggml_shape);

                خودکار reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_نمونہ.size()}, reshape_نمونہ),
                    جھوٹ);
                reshaped->سیٹ_friendly_name(view_name);
                واپس reshaped;
            }

            اگر (same_rank) {
                ثابت size_t ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                منطقی is_regular_قاش = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> قدم(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                size_t بقایا_انحراف = relative_انحراف;

                اگر (is_regular_قاش) {
                    برائے (size_t i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        اگر (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_قاش = جھوٹ;
                            توڑو;
                        }

                        قدم[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        اگر (قدم[i] <= 0) {
                            is_regular_قاش = جھوٹ;
                            توڑو;
                        }

                        begin[i] = static_cast<int64_t>(بقایا_انحراف / view_src_stride[i]);
                        بقایا_انحراف %= view_src_stride[i];

                        اگر (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            جاری;
                        }

                        end[i] = begin[i] + قدم[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        اگر (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_قاش = جھوٹ;
                            توڑو;
                        }
                    }
                }

                اگر (is_regular_قاش && بقایا_انحراف == 0) {
                    خودکار sliced = std::make_shared<ov::op::v8::قاش>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, قدم),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->سیٹ_friendly_name(view_name);
                    واپس sliced;
                }

                ثابت size_t elem_stride = view_src_stride.back();
                ثابت منطقی aligned_انحراف = elem_stride > 0 && relative_انحراف % elem_stride == 0;

                اگر (aligned_انحراف) {
                    size_t suffix_آغاز = 0;
                    size_t expected_stride = elem_stride;
                    برائے (صحیح i = static_cast<صحیح>(ndims) - 1; i >= 0; --i) {
                        اگر (view_stride[i] != expected_stride) {
                            suffix_آغاز = static_cast<size_t>(i + 1);
                            توڑو;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    size_t سابقہ_elems = 1;
                    size_t suffix_elems = 1;
                    برائے (size_t i = 0; i < suffix_آغاز; ++i) {
                        سابقہ_elems *= view_ggml_shape[i];
                    }
                    برائے (size_t i = suffix_آغاز; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    اگر (سابقہ_elems > 0 && src_elems % سابقہ_elems == 0) {
                        ثابت size_t src_دم_elems = src_elems / سابقہ_elems;
                        ثابت int64_t دم_begin = static_cast<int64_t>(relative_انحراف / elem_stride);
                        ثابت int64_t دم_انتہا = دم_begin + static_cast<int64_t>(suffix_elems);

                        اگر (دم_begin >= 0 && دم_انتہا <= static_cast<int64_t>(src_دم_elems)) {
                            خودکار سابقہ_دم_نمونہ = build_سابقہ_دم_reshape_نمونہ(
                                view_ov_shape, view_ggml_shape, suffix_آغاز, static_cast<int64_t>(src_دم_elems));

                            خودکار سابقہ_دم = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {سابقہ_دم_نمونہ.size()},
                                                             سابقہ_دم_نمونہ),
                                جھوٹ);

                            ov::Output<ov::Node> selected = سابقہ_دم;
                            اگر (دم_begin != 0 || دم_انتہا != static_cast<int64_t>(src_دم_elems)) {
                                selected = std::make_shared<ov::op::v8::قاش>(
                                    سابقہ_دم, ov::op::v0::Constant::create(ov::element::i64, {1}, {دم_begin}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {دم_انتہا}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {static_cast<int64_t>(suffix_آغاز)}));
                            }

                            خودکار reshape_نمونہ = build_reshape_نمونہ(view_ov_shape, view_ggml_shape);
                            خودکار reshaped = std::make_shared<ov::op::v1::Reshape>(
                                selected,
                                ov::op::v0::Constant::create(ov::element::i64, {reshape_نمونہ.size()},
                                                             reshape_نمونہ),
                                جھوٹ);
                            reshaped->سیٹ_friendly_name(view_name);
                            واپس reshaped;
                        }
                    }
                }
            }

            واپس current;
        }

        (خالی) view_name;
        (خالی) view_src_ov_shape;
        (خالی) view_src_name;

        واپس current;
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
        خودکار innermost_انحراف = context.get_view_داخلہ_انحراف(داخلہ_اشاریہ, 0);
        خودکار innermost_src_انحراف = context.get_view_داخلہ_src_انحراف(داخلہ_اشاریہ, 0);
        خودکار innermost_shape = context.get_view_داخلہ_ggml_shape(داخلہ_اشاریہ, 0);
        خودکار innermost_src_shape = context.get_view_داخلہ_src_ggml_shape(داخلہ_اشاریہ, 0);
        اگر (innermost_انحراف != innermost_src_انحراف && innermost_shape == innermost_src_shape) {
            size_t جڑ_view_idx = view_داخلہ_سائز - 1;
            خودکار جڑ_ggml_shape = context.get_view_داخلہ_src_ggml_shape(داخلہ_اشاریہ, جڑ_view_idx);
            خودکار جڑ_stride = context.get_view_داخلہ_src_stride(داخلہ_اشاریہ, جڑ_view_idx);
            خودکار جڑ_انحراف = context.get_view_داخلہ_src_انحراف(داخلہ_اشاریہ, جڑ_view_idx);
            خودکار جڑ_ov_shape = context.get_view_داخلہ_src_ov_shape(داخلہ_اشاریہ, جڑ_view_idx);
            خودکار جڑ_name = context.get_view_داخلہ_src_name(داخلہ_اشاریہ, جڑ_view_idx);
            خودکار innermost_stride = context.get_view_داخلہ_stride(داخلہ_اشاریہ, 0);
            خودکار innermost_ov_shape = context.get_view_داخلہ_ov_shape(داخلہ_اشاریہ, 0);
            خودکار innermost_name = context.get_view_داخلہ_name(داخلہ_اشاریہ, 0);

            واپس process_single_view(داخلہ, innermost_انحراف, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, جڑ_انحراف, جڑ_stride, جڑ_ggml_shape, جڑ_ov_shape,
                                       جڑ_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = داخلہ;

    // Process each view in reverse order (from base to current)
    برائے (صحیح view_idx = view_داخلہ_سائز - 1; view_idx >= 0; view_idx--) {
        خودکار view_انحراف = context.get_view_داخلہ_انحراف(داخلہ_اشاریہ, view_idx);
        خودکار view_stride = context.get_view_داخلہ_stride(داخلہ_اشاریہ, view_idx);
        خودکار view_ggml_shape = context.get_view_داخلہ_ggml_shape(داخلہ_اشاریہ, view_idx);
        خودکار view_ov_shape = context.get_view_داخلہ_ov_shape(داخلہ_اشاریہ, view_idx);
        خودکار view_name = context.get_view_داخلہ_name(داخلہ_اشاریہ, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        خودکار view_src_انحراف = context.get_view_داخلہ_src_انحراف(داخلہ_اشاریہ, view_idx);
        خودکار view_src_stride = context.get_view_داخلہ_src_stride(داخلہ_اشاریہ, view_idx);
        خودکار view_src_ggml_shape = context.get_view_داخلہ_src_ggml_shape(داخلہ_اشاریہ, view_idx);
        خودکار view_src_ov_shape = context.get_view_داخلہ_src_ov_shape(داخلہ_اشاریہ, view_idx);
        خودکار view_src_name = context.get_view_داخلہ_src_name(داخلہ_اشاریہ, view_idx);
        // print source view info
        // std::cout << "View " << view_idx << ": source name = " << view_src_name
        //           << ", source offset = " << view_src_offset << ", source stride = [" << view_src_stride[0] << ","
        //           << view_src_stride[1] << "," << view_src_stride[2] << "," << view_src_stride[3]
        //           << "], source ggml shape = [" << view_src_ggml_shape[0] << "," << view_src_ggml_shape[1] << ","
        //           << view_src_ggml_shape[2] << "," << view_src_ggml_shape[3]
        //           << "], source ov shape = " << view_src_ov_shape << std::endl;

        current = process_single_view(current, view_انحراف, view_stride, view_ggml_shape, view_ov_shape, view_name,
                                      view_src_انحراف, view_src_stride, view_src_ggml_shape, view_src_ov_shape,
                                      view_src_name);
    }

    واپس current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
