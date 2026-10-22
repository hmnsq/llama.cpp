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

فضاینام ov {
فضاینام frontend {
فضاینام ggml {

تهی رقم_inputs_check(ثابت گره_Context & context, size_t min_inputs, size_t max_inputs) {
    auto درون_داد_size = context.get_درون_داد_size();
    FRONT_END_OP_CONVERSION_CHECK(درون_داد_size >= min_inputs, "Got less inputs than expected");
    FRONT_END_OP_CONVERSION_CHECK(درون_داد_size <= max_inputs, "Got more inputs than expected");
}

صحیح non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    صحیح dim = nb.size() - 1;
    size_t bytes = nb[dim];
    برای (صحیح i = dim; i > 0; i--) {
        bytes *= ne[i];
        اگر (bytes != nb[i - 1]) {
            بازگشت i;
        }
    }
    بازگشت 0;
}

std::shared_ptr<ov::Node> get_dimensions(ثابت std::shared_ptr<ov::op::v3::ShapeOf> & shape,
                                         ثابت std::vector<صحیح> & dims) {
    using فضاینام ov::op;
    ثابت auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ثابت auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    بازگشت std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(ثابت std::shared_ptr<ov::Node> & گره, ثابت std::vector<صحیح> & dims) {
    بازگشت get_dimensions(std::make_shared<ov::op::v3::ShapeOf>(گره), dims);
}

OutputVector rename_outputs_with_suffix(ثابت OutputVector & outputs, ثابت std::string & suffix) {
    برای (ثابت auto & output : outputs) {
        auto گره = output.get_گره_shared_ptr();
        std::string name = گره->get_friendly_name();
        name += "_";
        name += suffix;
        گره->set_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    بازگشت outputs;
}

فضاینام {
ov::Output<ov::Node> rope_yarn_ramp_mix(صحیح رقم_elements, ثابت اعشاری corr_dims[2], اعشاری ext_factor) {
    std::vector<اعشاری> dim_ids_vec(رقم_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    auto dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) رقم_elements}, dim_ids_vec);
    auto corr_کم = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    auto corr_بالا = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    auto denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::Subtract>(corr_بالا, corr_کم),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    auto ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Subtract>(dim_ids, corr_کم), denom);
    auto ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    auto one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    auto ramp_inverted = std::make_shared<ov::op::v1::Subtract>(one, ramp_clamped);
    auto ext_factor_گره = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_factor});
    auto ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_factor_گره);
    بازگشت ramp_mix;
}

اعشاری ggml_rope_yarn_corr_dim(صحیح n_dims, صحیح n_ctx_orig, اعشاری n_rot, اعشاری base) {
#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
    بازگشت n_dims * logf(n_ctx_orig / (n_rot * 2 * (اعشاری) M_PI)) / (2 * logf(base));
}

تهی ggml_rope_yarn_corr_dims(صحیح n_dims,
                              صحیح n_ctx_orig,
                              اعشاری freq_base,
                              اعشاری beta_fast,
                              اعشاری beta_slow,
                              اعشاری dims[2]) {
    اعشاری start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    اعشاری end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<اعشاری>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight,
                                                           صحیح mode,
                                                           منطقی stateful,
                                                           int64_t head_dim) {
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
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } وگرنه اگر (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, درست);
        auto pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } وگرنه {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    اعشاری freq_base;
    اعشاری freq_scale;
    اعشاری ext_factor;
    اعشاری attn_factor;
    اعشاری beta_fast;
    اعشاری beta_slow;
    ثابت صحیح n_dims = rope_params[1];
    ثابت size_t n_dims_half = n_dims >> 1;
    ثابت صحیح n_ctx_orig = rope_params[4];
    memcpy(&freq_base, rope_params + 5, اندازه(اعشاری));
    memcpy(&freq_scale, rope_params + 6, اندازه(اعشاری));
    memcpy(&ext_factor, rope_params + 7, اندازه(اعشاری));
    memcpy(&attn_factor, rope_params + 8, اندازه(اعشاری));
    memcpy(&beta_fast, rope_params + 9, اندازه(اعشاری));
    memcpy(&beta_slow, rope_params + 10, اندازه(اعشاری));

    ثابت اعشاری theta_scale = powf(freq_base, -2.0f / n_dims);

    Output<Node> theta;
    اعشاری mscale = attn_factor;
    اگر (is_multi) {
        ثابت size_t رقم_pairs = is_vision ? (n_dims > 0 ? (size_t) n_dims : (size_t) (head_dim / 2)) : n_dims_half;
        صحیح sections[4];
        memcpy(sections, rope_params + 11, اندازه(صحیح) * 4);
        صحیح sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        اگر (sect_dims <= 0) {
            sect_dims = رقم_pairs;
        }
        صحیح sec_w = sections[0] + sections[1];
        صحیح sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(رقم_pairs);
        std::vector<اعشاری> factor(رقم_pairs);
        برای (size_t j = 0; j < رقم_pairs; j++) {
            صحیح sector = j % sect_dims;
            صحیح plane = 0;
            اگر (is_imrope) {
                اگر (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } وگرنه اگر (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } وگرنه اگر (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } وگرنه {
                    plane = 3;
                }
                factor[j] = std::pow(theta_scale, j);
            } وگرنه اگر (is_vision) {
                صحیح p_idx = 0;
                اگر (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } وگرنه اگر (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } وگرنه اگر (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } وگرنه {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                factor[j] = std::pow(theta_scale, p_idx);
            } وگرنه {
                اگر (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } وگرنه اگر (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } وگرنه اگر (sector >= sec_e) {
                    plane = 3;
                } وگرنه {
                    plane = 0;
                }
                factor[j] = std::pow(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        auto gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{رقم_pairs}, gather_indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> factor_گره =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{رقم_pairs}, factor);
        اگر (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) رقم_pairs}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) رقم_pairs}),
                نادرست);
            factor_گره = std::make_shared<ov::op::v1::Divide>(factor_گره, rope_factors);
        }
        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, factor_گره);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        اگر (ext_factor == 0.0f) {
            theta = theta_interp;
        } وگرنه {
            اعشاری corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
            auto ramp_mix = rope_yarn_ramp_mix(رقم_pairs, corr_dims, ext_factor);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            auto one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::Add>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
        }
    } وگرنه {
        std::vector<اعشاری> factor(n_dims_half);
        Output<Node> freq_factors;
        اعشاری corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        factor[0] = 1.0f;
        برای (size_t i = 1; i < factor.size(); i++) {
            factor[i] = theta_scale * factor[i - 1];
        }
        اگر (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, factor.size()}, factor);
        } وگرنه {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, factor.size()}, factor);
        }
        اگر (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            اگر (stateful) {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {3}, {(int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    نادرست);
            } وگرنه {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    نادرست);
            }
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_factors);
        }

        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        اگر (ext_factor == 0.0f) {
            theta = theta_interp;
        } وگرنه {
            auto ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_factor);
            Output<Node> one;
            اگر (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } وگرنه {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            }
            auto one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);

            theta =
                std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                                                  std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
        }
    }

    Output<Node> cos_theta = std::make_shared<ov::op::v0::Cos>(theta);
    Output<Node> sin_theta = std::make_shared<ov::op::v0::Sin>(theta);

    اگر (mscale != 1.0f) {
        auto mscale_گره = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_گره);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_گره);
    }

    بازگشت std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_درون_داد(ثابت گره_Context & context, صحیح درون_داد_index, صحیح slice_len, صحیح axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    auto درون_داد = context.get_درون_داد(درون_داد_index);
    auto * op_params = (size_t *) context.get_درون_داد_op_params(درون_داد_index);
    auto src_stride = context.get_درون_داد_stride(درون_داد_index);

    int64_t slice_start = op_params[0] / src_stride[3];
    اگر (slice_len == 0) {
        slice_len = context.get_درون_داد_shape(درون_داد_index)[3].get_length();
    }
    int64_t slice_end = slice_start + slice_len;

    auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_start});
    auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_end});
    auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    اگر (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } وگرنه {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    auto sliced = std::make_shared<ov::op::v8::Slice>(درون_داد, begin, end, stride, axes);
    بازگشت sliced;
}

ov::Output<ov::Node> process_view_درون_داد_new(ثابت گره_Context & context, صحیح درون_داد_index) {
    auto درون_داد = context.get_درون_داد(درون_داد_index);

    // Check if this input has view inputs
    size_t view_درون_داد_size = context.get_view_درون_داد_size(درون_داد_index);
    اگر (view_درون_داد_size == 0) {
        // No view inputs, return the input as is
        بازگشت درون_داد;
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
    auto expected_ov_shape = context.get_view_درون_داد_ov_shape(درون_داد_index, 0);
    auto actual_shape = درون_داد.get_partial_shape();
    اگر (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_length() >= actual_shape.rank().get_length()) {
        ثابت int64_t n_actual = actual_shape.rank().get_length();
        ثابت int64_t shift = expected_ov_shape.rank().get_length() - n_actual;
        منطقی shapes_match = درست;
        برای (int64_t i = 0; i < shift; ++i) {
            اگر (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_length() != 1) {
                shapes_match = نادرست;
                بشکن;
            }
        }
        برای (int64_t i = 0; i < n_actual && shapes_match; ++i) {
            ثابت auto & exp = expected_ov_shape[i + shift];
            ثابت منطقی both_dynamic = exp.is_dynamic() && actual_shape[i].is_dynamic();
            ثابت منطقی both_static_equal =
                exp.is_static() && actual_shape[i].is_static() && exp == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            ثابت منطقی expected_dyn_actual_static = exp.is_dynamic() && actual_shape[i].is_static();
            اگر (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_match = نادرست;
                بشکن;
            }
        }
        اگر (shapes_match) {
            بازگشت درون_داد;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    اگر (context.is_static() && view_درون_داد_size == 1) {
        auto view_stride_v = context.get_view_درون_داد_stride(درون_داد_index, 0);
        auto view_src_stride_v = context.get_view_درون_داد_src_stride(درون_داد_index, 0);
        auto view_ggml_shape = context.get_view_درون_داد_ggml_shape(درون_داد_index, 0);
        auto view_src_ggml_shape = context.get_view_درون_داد_src_ggml_shape(درون_داد_index, 0);
        auto view_offset = context.get_view_درون_داد_offset(درون_داد_index, 0);
        auto view_src_offset = context.get_view_درون_داد_src_offset(درون_داد_index, 0);

        size_t ndims = view_ggml_shape.size();
        std::vector<صحیح> diff_dims;
        اگر (view_src_ggml_shape.size() == ndims) {
            برای (size_t i = 0; i < ndims; ++i) {
                اگر (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<صحیح>(i));
                }
            }
        }

        اگر (diff_dims.size() == 1) {
            صحیح split_dim = diff_dims[0];
            int64_t رقم_splits = static_cast<int64_t>(view_src_ggml_shape[split_dim]);
            int64_t chunk_size = static_cast<int64_t>(view_ggml_shape[split_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            اگر (chunk_size == 1 && رقم_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                منطقی suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                اگر (suffix_ok) {
                    برای (size_t i = static_cast<size_t>(split_dim) + 1; i < ndims; ++i) {
                        اگر (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = نادرست;
                            بشکن;
                        }
                    }
                }

                اگر (suffix_ok && view_src_stride_v[split_dim] > 0) {
                    size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
                    int64_t split_index = static_cast<int64_t>(relative_offset / view_src_stride_v[split_dim]);

                    اگر (split_index >= 0 && split_index < رقم_splits) {
                        auto src_گره = درون_داد.get_گره_shared_ptr();
                        std::string rt_کلید = "split_dim_" + std::to_string(split_dim);
                        auto & rt_info = src_گره->get_rt_info();

                        اگر (rt_info.find(rt_کلید) == rt_info.end()) {
                            auto axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(split_dim)});
                            auto split_گره =
                                std::make_shared<ov::op::v1::Split>(درون_داد, axis_const, static_cast<size_t>(رقم_splits));
                            split_گره->set_friendly_name(src_گره->get_friendly_name() + "_split");
                            rt_info[rt_کلید] = split_گره;
                        }

                        auto split_گره = rt_info[rt_کلید].as<std::shared_ptr<ov::op::v1::Split>>();
                        بازگشت split_گره->output(static_cast<size_t>(split_index));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    auto process_single_view =
        [](ov::Output<ov::Node> current, size_t view_offset, ثابت std::vector<size_t> & view_stride,
           ثابت ov::Shape & view_ggml_shape, ثابت ov::PartialShape & view_ov_shape, ثابت std::string & view_name,
           size_t view_src_offset, ثابت std::vector<size_t> & view_src_stride, ثابت ov::Shape & view_src_ggml_shape,
           ثابت ov::PartialShape & view_src_ov_shape, ثابت std::string & view_src_name) -> ov::Output<ov::Node> {
        // Stateful execution drops a leading size-1 axis, so `current` can be one rank lower
        // than the ggml shape metadata (view_stride/view_ggml_shape, always GGML_MAX_DIMS
        // entries) assumes. Shift any axis index built from that metadata down by the
        // difference before using it as an OV Slice axis.
        ثابت auto current_rank = current.get_partial_shape().rank();
        ثابت صحیح axis_shift = current_rank.is_static() ?
                                    static_cast<صحیح>(view_stride.size()) - static_cast<صحیح>(current_rank.get_length()) :
                                    0;

        auto build_reshape_pattern = [](ثابت ov::PartialShape & target_ov_shape,
                                        ثابت ov::Shape & target_ggml_shape) -> std::vector<int64_t> {
            ثابت size_t ndims = target_ggml_shape.size();
            std::vector<int64_t> reshape_pattern(ndims);
            size_t dynamic_dims = 0;

            اگر (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(ndims)) {
                برای (size_t i = 0; i < ndims; ++i) {
                    اگر (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } وگرنه {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } وگرنه {
                dynamic_dims = 2;
            }

            اگر (dynamic_dims > 1) {
                برای (size_t i = 0; i < ndims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            بازگشت reshape_pattern;
        };

        auto build_prefix_tail_reshape_pattern = [](ثابت ov::PartialShape & target_ov_shape,
                                                    ثابت ov::Shape & target_ggml_shape, size_t prefix_dims,
                                                    int64_t tail_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_pattern(prefix_dims + 1);
            size_t dynamic_dims = 0;

            اگر (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(target_ggml_shape.size())) {
                برای (size_t i = 0; i < prefix_dims; ++i) {
                    اگر (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } وگرنه {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } وگرنه {
                dynamic_dims = 2;
            }

            اگر (dynamic_dims > 1) {
                برای (size_t i = 0; i < prefix_dims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            reshape_pattern[prefix_dims] = tail_dim;
            بازگشت reshape_pattern;
        };

        منطقی same_stride = view_stride.size() == view_src_stride.size();
        اگر (same_stride) {
            برای (size_t i = 0; i < view_stride.size(); ++i) {
                اگر (view_stride[i] != view_src_stride[i]) {
                    same_stride = نادرست;
                    بشکن;
                }
            }
        }

        منطقی same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        اگر (same_ggml_shape) {
            برای (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                اگر (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = نادرست;
                    بشکن;
                }
            }
        }

        اگر (same_stride && same_ggml_shape) {
            بازگشت current;
        }

        اگر (same_stride) {
            ثابت size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
            ثابت size_t ndims = view_stride.size();

            std::vector<صحیح> diff_dims;
            اگر (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                برای (size_t i = 0; i < ndims; ++i) {
                    اگر (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<صحیح>(i));
                    }
                }
            }

            اگر (diff_dims.size() == 1) {
                ثابت صحیح slice_dim = diff_dims[0];
                ثابت int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                اگر (view_stride[slice_dim] > 0 && relative_offset % view_stride[slice_dim] == 0) {
                    ثابت int64_t begin_val = static_cast<int64_t>((relative_offset / view_stride[slice_dim]) %
                                                                   static_cast<size_t>(dim_size));
                    ثابت int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);

                    اگر (begin_val >= 0 && end_val <= dim_size) {
                        auto sliced = std::make_shared<ov::op::v8::Slice>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim - axis_shift}));

                        اگر (view_ov_shape.is_static()) {
                            auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                                نادرست);
                            reshaped->set_friendly_name(view_name);
                            بازگشت reshaped;
                        }

                        sliced->set_friendly_name(view_name);
                        بازگشت sliced;
                    }
                }

                int64_t tail_src_elems = 1;
                int64_t tail_dst_elems = 1;
                برای (size_t i = slice_dim; i < ndims; ++i) {
                    tail_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    tail_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                ثابت size_t elem_stride = view_stride[ndims - 1];
                int64_t tail_begin = 0;
                اگر (elem_stride > 0) {
                    tail_begin =
                        static_cast<int64_t>((relative_offset / elem_stride) % static_cast<size_t>(tail_src_elems));
                }
                ثابت int64_t tail_end = tail_begin + tail_dst_elems;

                اگر (tail_begin >= 0 && tail_end <= tail_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(slice_dim);
                    برای (صحیح i = 0; i < slice_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(tail_src_elems);
                    ثابت size_t flat_ndims = flat_shape.size();

                    auto flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), نادرست);

                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_end}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

                    اگر (view_ov_shape.is_static()) {
                        auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                            نادرست);
                        reshaped->set_friendly_name(view_name);
                        بازگشت reshaped;
                    }

                    sliced->set_friendly_name(view_name);
                    بازگشت sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> step(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            size_t remaining_offset = relative_offset;
            برای (size_t i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                اگر (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(remaining_offset / view_stride[i]);
                    remaining_offset %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            منطقی in_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            اگر (in_bounds) {
                برای (size_t i = 0; i < ndims; ++i) {
                    اگر (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        in_bounds = نادرست;
                        بشکن;
                    }
                }
            }

            اگر (in_bounds && remaining_offset == 0) {
                auto sliced = std::make_shared<ov::op::v8::Slice>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->set_friendly_name(view_name);
                بازگشت sliced;
            }
        } وگرنه {
            منطقی same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            ثابت size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;

            اگر (same_rank) {
                ثابت size_t ndims = view_ggml_shape.size();
                std::vector<صحیح> diff_dims;
                برای (size_t i = 0; i < ndims; ++i) {
                    اگر (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<صحیح>(i));
                    }
                }

                اگر (diff_dims.size() == 1) {
                    ثابت size_t slice_dim = static_cast<size_t>(diff_dims[0]);
                    منطقی suffix_stride_match = درست;
                    برای (size_t i = slice_dim + 1; i < ndims; ++i) {
                        اگر (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_match = نادرست;
                            بشکن;
                        }
                    }

                    اگر (suffix_stride_match && view_src_stride[slice_dim] > 0 &&
                        relative_offset % view_src_stride[slice_dim] == 0) {
                        ثابت int64_t begin_val = static_cast<int64_t>(relative_offset / view_src_stride[slice_dim]);
                        ثابت int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);
                        ثابت int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                        اگر (begin_val >= 0 && end_val <= dim_size) {
                            auto sliced = std::make_shared<ov::op::v8::Slice>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(slice_dim)}));
                            sliced->set_friendly_name(view_name);
                            بازگشت sliced;
                        }
                    }
                }
            }

            size_t view_elems = 1;
            size_t src_elems = 1;
            اگر (same_rank) {
                برای (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            منطقی same_رقم_elements = same_rank && view_elems == src_elems;

            اگر (same_rank && relative_offset == 0 && same_رقم_elements) {
                auto reshape_pattern = build_reshape_pattern(view_ov_shape, view_ggml_shape);

                auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_pattern.size()}, reshape_pattern),
                    نادرست);
                reshaped->set_friendly_name(view_name);
                بازگشت reshaped;
            }

            اگر (same_rank) {
                ثابت size_t ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                منطقی is_regular_slice = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> step(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                size_t remaining_offset = relative_offset;

                اگر (is_regular_slice) {
                    برای (size_t i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        اگر (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_slice = نادرست;
                            بشکن;
                        }

                        step[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        اگر (step[i] <= 0) {
                            is_regular_slice = نادرست;
                            بشکن;
                        }

                        begin[i] = static_cast<int64_t>(remaining_offset / view_src_stride[i]);
                        remaining_offset %= view_src_stride[i];

                        اگر (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            ادامه;
                        }

                        end[i] = begin[i] + step[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        اگر (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_slice = نادرست;
                            بشکن;
                        }
                    }
                }

                اگر (is_regular_slice && remaining_offset == 0) {
                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->set_friendly_name(view_name);
                    بازگشت sliced;
                }

                ثابت size_t elem_stride = view_src_stride.back();
                ثابت منطقی aligned_offset = elem_stride > 0 && relative_offset % elem_stride == 0;

                اگر (aligned_offset) {
                    size_t suffix_start = 0;
                    size_t expected_stride = elem_stride;
                    برای (صحیح i = static_cast<صحیح>(ndims) - 1; i >= 0; --i) {
                        اگر (view_stride[i] != expected_stride) {
                            suffix_start = static_cast<size_t>(i + 1);
                            بشکن;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    size_t prefix_elems = 1;
                    size_t suffix_elems = 1;
                    برای (size_t i = 0; i < suffix_start; ++i) {
                        prefix_elems *= view_ggml_shape[i];
                    }
                    برای (size_t i = suffix_start; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    اگر (prefix_elems > 0 && src_elems % prefix_elems == 0) {
                        ثابت size_t src_tail_elems = src_elems / prefix_elems;
                        ثابت int64_t tail_begin = static_cast<int64_t>(relative_offset / elem_stride);
                        ثابت int64_t tail_end = tail_begin + static_cast<int64_t>(suffix_elems);

                        اگر (tail_begin >= 0 && tail_end <= static_cast<int64_t>(src_tail_elems)) {
                            auto prefix_tail_pattern = build_prefix_tail_reshape_pattern(
                                view_ov_shape, view_ggml_shape, suffix_start, static_cast<int64_t>(src_tail_elems));

                            auto prefix_tail = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {prefix_tail_pattern.size()},
                                                             prefix_tail_pattern),
                                نادرست);

                            ov::Output<ov::Node> selected = prefix_tail;
                            اگر (tail_begin != 0 || tail_end != static_cast<int64_t>(src_tail_elems)) {
                                selected = std::make_shared<ov::op::v8::Slice>(
                                    prefix_tail, ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_begin}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_end}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {static_cast<int64_t>(suffix_start)}));
                            }

                            auto reshape_pattern = build_reshape_pattern(view_ov_shape, view_ggml_shape);
                            auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                                selected,
                                ov::op::v0::Constant::create(ov::element::i64, {reshape_pattern.size()},
                                                             reshape_pattern),
                                نادرست);
                            reshaped->set_friendly_name(view_name);
                            بازگشت reshaped;
                        }
                    }
                }
            }

            بازگشت current;
        }

        (تهی) view_name;
        (تهی) view_src_ov_shape;
        (تهی) view_src_name;

        بازگشت current;
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
        auto innermost_offset = context.get_view_درون_داد_offset(درون_داد_index, 0);
        auto innermost_src_offset = context.get_view_درون_داد_src_offset(درون_داد_index, 0);
        auto innermost_shape = context.get_view_درون_داد_ggml_shape(درون_داد_index, 0);
        auto innermost_src_shape = context.get_view_درون_داد_src_ggml_shape(درون_داد_index, 0);
        اگر (innermost_offset != innermost_src_offset && innermost_shape == innermost_src_shape) {
            size_t ریشه_view_idx = view_درون_داد_size - 1;
            auto ریشه_ggml_shape = context.get_view_درون_داد_src_ggml_shape(درون_داد_index, ریشه_view_idx);
            auto ریشه_stride = context.get_view_درون_داد_src_stride(درون_داد_index, ریشه_view_idx);
            auto ریشه_offset = context.get_view_درون_داد_src_offset(درون_داد_index, ریشه_view_idx);
            auto ریشه_ov_shape = context.get_view_درون_داد_src_ov_shape(درون_داد_index, ریشه_view_idx);
            auto ریشه_name = context.get_view_درون_داد_src_name(درون_داد_index, ریشه_view_idx);
            auto innermost_stride = context.get_view_درون_داد_stride(درون_داد_index, 0);
            auto innermost_ov_shape = context.get_view_درون_داد_ov_shape(درون_داد_index, 0);
            auto innermost_name = context.get_view_درون_داد_name(درون_داد_index, 0);

            بازگشت process_single_view(درون_داد, innermost_offset, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, ریشه_offset, ریشه_stride, ریشه_ggml_shape, ریشه_ov_shape,
                                       ریشه_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = درون_داد;

    // Process each view in reverse order (from base to current)
    برای (صحیح view_idx = view_درون_داد_size - 1; view_idx >= 0; view_idx--) {
        auto view_offset = context.get_view_درون_داد_offset(درون_داد_index, view_idx);
        auto view_stride = context.get_view_درون_داد_stride(درون_داد_index, view_idx);
        auto view_ggml_shape = context.get_view_درون_داد_ggml_shape(درون_داد_index, view_idx);
        auto view_ov_shape = context.get_view_درون_داد_ov_shape(درون_داد_index, view_idx);
        auto view_name = context.get_view_درون_داد_name(درون_داد_index, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        auto view_src_offset = context.get_view_درون_داد_src_offset(درون_داد_index, view_idx);
        auto view_src_stride = context.get_view_درون_داد_src_stride(درون_داد_index, view_idx);
        auto view_src_ggml_shape = context.get_view_درون_داد_src_ggml_shape(درون_داد_index, view_idx);
        auto view_src_ov_shape = context.get_view_درون_داد_src_ov_shape(درون_داد_index, view_idx);
        auto view_src_name = context.get_view_درون_داد_src_name(درون_داد_index, view_idx);
        // print source view info
        // std::cout << "View " << view_idx << ": source name = " << view_src_name
        //           << ", source offset = " << view_src_offset << ", source stride = [" << view_src_stride[0] << ","
        //           << view_src_stride[1] << "," << view_src_stride[2] << "," << view_src_stride[3]
        //           << "], source ggml shape = [" << view_src_ggml_shape[0] << "," << view_src_ggml_shape[1] << ","
        //           << view_src_ggml_shape[2] << "," << view_src_ggml_shape[3]
        //           << "], source ov shape = " << view_src_ov_shape << std::endl;

        current = process_single_view(current, view_offset, view_stride, view_ggml_shape, view_ov_shape, view_name,
                                      view_src_offset, view_src_stride, view_src_ggml_shape, view_src_ov_shape,
                                      view_src_name);
    }

    بازگشت current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
