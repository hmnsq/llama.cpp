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

فراغ num_inputs_check(ثابت NodeContext & context, size_t min_inputs, size_t max_inputs) {
    auto input_size = context.get_input_size();
    FRONT_END_OP_CONVERSION_CHECK(input_size >= min_inputs, "Got less inputs than expected");
    FRONT_END_OP_CONVERSION_CHECK(input_size <= max_inputs, "Got more inputs than expected");
}

صحيح non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    صحيح dim = nb.size() - 1;
    size_t bytes = nb[dim];
    لكل (صحيح i = dim; i > 0; i--) {
        bytes *= ne[i];
        𐤀𐤌 (bytes != nb[i - 1]) {
            أرجع i;
        }
    }
    أرجع 0;
}

std::shared_ptr<ov::Node> get_dimensions(ثابت std::shared_ptr<ov::op::v3::ShapeOf> & shape,
                                         ثابت std::vector<صحيح> & dims) {
    using نطاق ov::op;
    ثابت auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ثابت auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    أرجع std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(ثابت std::shared_ptr<ov::Node> & node, ثابت std::vector<صحيح> & dims) {
    أرجع get_dimensions(std::make_shared<ov::op::v3::ShapeOf>(node), dims);
}

OutputVector rename_outputs_with_suffix(ثابت OutputVector & outputs, ثابت std::string & suffix) {
    لكل (ثابت auto & output : outputs) {
        auto node = output.get_node_shared_ptr();
        std::string name = node->get_friendly_name();
        name += "_";
        name += suffix;
        node->set_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    أرجع outputs;
}

نطاق {
ov::Output<ov::Node> rope_yarn_ramp_mix(صحيح num_elements, ثابت عائم corr_dims[2], عائم ext_factor) {
    std::vector<عائم> dim_ids_vec(num_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    auto dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) num_elements}, dim_ids_vec);
    auto corr_low = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    auto corr_high = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    auto denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::Subtract>(corr_high, corr_low),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    auto ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Subtract>(dim_ids, corr_low), denom);
    auto ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    auto one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    auto ramp_inverted = std::make_shared<ov::op::v1::Subtract>(one, ramp_clamped);
    auto ext_factor_node = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_factor});
    auto ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_factor_node);
    أرجع ramp_mix;
}

عائم ggml_rope_yarn_corr_dim(صحيح n_dims, صحيح n_ctx_orig, عائم n_rot, عائم base) {
#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
    أرجع n_dims * logf(n_ctx_orig / (n_rot * 2 * (عائم) M_PI)) / (2 * logf(base));
}

فراغ ggml_rope_yarn_corr_dims(صحيح n_dims,
                              صحيح n_ctx_orig,
                              عائم freq_base,
                              عائم beta_fast,
                              عائم beta_slow,
                              عائم dims[2]) {
    عائم start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    عائم end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<عائم>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight,
                                                           صحيح mode,
                                                           منطقي stateful,
                                                           int64_t head_dim) {
    constexpr صحيح TYPE_IMROPE = 2;
    constexpr صحيح TYPE_VISION = 3;
    constexpr صحيح TYPE_MROPE = 4;

    ثابت منطقي is_imrope = (mode == TYPE_IMROPE);
    ثابت منطقي is_vision = (mode == TYPE_VISION);
    ثابت منطقي is_mrope = (mode == TYPE_MROPE);
    ثابت منطقي is_multi = is_imrope || is_vision || is_mrope;

    𐤀𐤌 (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } وإلا 𐤀𐤌 (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, true);
        auto pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } وإلا {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    عائم freq_base;
    عائم freq_scale;
    عائم ext_factor;
    عائم attn_factor;
    عائم beta_fast;
    عائم beta_slow;
    ثابت صحيح n_dims = rope_params[1];
    ثابت size_t n_dims_half = n_dims >> 1;
    ثابت صحيح n_ctx_orig = rope_params[4];
    memcpy(&freq_base, rope_params + 5, حجم(عائم));
    memcpy(&freq_scale, rope_params + 6, حجم(عائم));
    memcpy(&ext_factor, rope_params + 7, حجم(عائم));
    memcpy(&attn_factor, rope_params + 8, حجم(عائم));
    memcpy(&beta_fast, rope_params + 9, حجم(عائم));
    memcpy(&beta_slow, rope_params + 10, حجم(عائم));

    ثابت عائم theta_scale = powf(freq_base, -2.0f / n_dims);

    Output<Node> theta;
    عائم mscale = attn_factor;
    𐤀𐤌 (is_multi) {
        ثابت size_t num_pairs = is_vision ? (n_dims > 0 ? (size_t) n_dims : (size_t) (head_dim / 2)) : n_dims_half;
        صحيح sections[4];
        memcpy(sections, rope_params + 11, حجم(صحيح) * 4);
        صحيح sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        𐤀𐤌 (sect_dims <= 0) {
            sect_dims = num_pairs;
        }
        صحيح sec_w = sections[0] + sections[1];
        صحيح sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(num_pairs);
        std::vector<عائم> factor(num_pairs);
        لكل (size_t j = 0; j < num_pairs; j++) {
            صحيح sector = j % sect_dims;
            صحيح plane = 0;
            𐤀𐤌 (is_imrope) {
                𐤀𐤌 (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } وإلا 𐤀𐤌 (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } وإلا 𐤀𐤌 (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } وإلا {
                    plane = 3;
                }
                factor[j] = std::pow(theta_scale, j);
            } وإلا 𐤀𐤌 (is_vision) {
                صحيح p_idx = 0;
                𐤀𐤌 (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } وإلا 𐤀𐤌 (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } وإلا 𐤀𐤌 (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } وإلا {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                factor[j] = std::pow(theta_scale, p_idx);
            } وإلا {
                𐤀𐤌 (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } وإلا 𐤀𐤌 (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } وإلا 𐤀𐤌 (sector >= sec_e) {
                    plane = 3;
                } وإلا {
                    plane = 0;
                }
                factor[j] = std::pow(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        auto gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{num_pairs}, gather_indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> factor_node =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{num_pairs}, factor);
        𐤀𐤌 (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) num_pairs}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) num_pairs}),
                خطأ);
            factor_node = std::make_shared<ov::op::v1::Divide>(factor_node, rope_factors);
        }
        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, factor_node);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        𐤀𐤌 (ext_factor == 0.0f) {
            theta = theta_interp;
        } وإلا {
            عائم corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
            auto ramp_mix = rope_yarn_ramp_mix(num_pairs, corr_dims, ext_factor);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            auto one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::Add>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
        }
    } وإلا {
        std::vector<عائم> factor(n_dims_half);
        Output<Node> freq_factors;
        عائم corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        factor[0] = 1.0f;
        لكل (size_t i = 1; i < factor.size(); i++) {
            factor[i] = theta_scale * factor[i - 1];
        }
        𐤀𐤌 (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, factor.size()}, factor);
        } وإلا {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, factor.size()}, factor);
        }
        𐤀𐤌 (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            𐤀𐤌 (stateful) {
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

        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        𐤀𐤌 (ext_factor == 0.0f) {
            theta = theta_interp;
        } وإلا {
            auto ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_factor);
            Output<Node> one;
            𐤀𐤌 (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } وإلا {
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

    𐤀𐤌 (mscale != 1.0f) {
        auto mscale_node = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_node);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_node);
    }

    أرجع std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_input(ثابت NodeContext & context, صحيح input_index, صحيح slice_len, صحيح axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    auto input = context.get_input(input_index);
    auto * op_params = (size_t *) context.get_input_op_params(input_index);
    auto src_stride = context.get_input_stride(input_index);

    int64_t slice_start = op_params[0] / src_stride[3];
    𐤀𐤌 (slice_len == 0) {
        slice_len = context.get_input_shape(input_index)[3].get_length();
    }
    int64_t slice_end = slice_start + slice_len;

    auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_start});
    auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_end});
    auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    𐤀𐤌 (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } وإلا {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    auto sliced = std::make_shared<ov::op::v8::Slice>(input, begin, end, stride, axes);
    أرجع sliced;
}

ov::Output<ov::Node> process_view_input_new(ثابت NodeContext & context, صحيح input_index) {
    auto input = context.get_input(input_index);

    // Check if this input has view inputs
    size_t view_input_size = context.get_view_input_size(input_index);
    𐤀𐤌 (view_input_size == 0) {
        // No view inputs, return the input as is
        أرجع input;
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
    auto expected_ov_shape = context.get_view_input_ov_shape(input_index, 0);
    auto actual_shape = input.get_partial_shape();
    𐤀𐤌 (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_length() >= actual_shape.rank().get_length()) {
        ثابت int64_t n_actual = actual_shape.rank().get_length();
        ثابت int64_t shift = expected_ov_shape.rank().get_length() - n_actual;
        منطقي shapes_match = true;
        لكل (int64_t i = 0; i < shift; ++i) {
            𐤀𐤌 (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_length() != 1) {
                shapes_match = خطأ;
                اكسر;
            }
        }
        لكل (int64_t i = 0; i < n_actual && shapes_match; ++i) {
            ثابت auto & exp = expected_ov_shape[i + shift];
            ثابت منطقي both_dynamic = exp.is_dynamic() && actual_shape[i].is_dynamic();
            ثابت منطقي both_static_equal =
                exp.is_static() && actual_shape[i].is_static() && exp == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            ثابت منطقي expected_dyn_actual_static = exp.is_dynamic() && actual_shape[i].is_static();
            𐤀𐤌 (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_match = خطأ;
                اكسر;
            }
        }
        𐤀𐤌 (shapes_match) {
            أرجع input;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    𐤀𐤌 (context.is_static() && view_input_size == 1) {
        auto view_stride_v = context.get_view_input_stride(input_index, 0);
        auto view_src_stride_v = context.get_view_input_src_stride(input_index, 0);
        auto view_ggml_shape = context.get_view_input_ggml_shape(input_index, 0);
        auto view_src_ggml_shape = context.get_view_input_src_ggml_shape(input_index, 0);
        auto view_offset = context.get_view_input_offset(input_index, 0);
        auto view_src_offset = context.get_view_input_src_offset(input_index, 0);

        size_t ndims = view_ggml_shape.size();
        std::vector<صحيح> diff_dims;
        𐤀𐤌 (view_src_ggml_shape.size() == ndims) {
            لكل (size_t i = 0; i < ndims; ++i) {
                𐤀𐤌 (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<صحيح>(i));
                }
            }
        }

        𐤀𐤌 (diff_dims.size() == 1) {
            صحيح split_dim = diff_dims[0];
            int64_t num_splits = static_cast<int64_t>(view_src_ggml_shape[split_dim]);
            int64_t chunk_size = static_cast<int64_t>(view_ggml_shape[split_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            𐤀𐤌 (chunk_size == 1 && num_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                منطقي suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                𐤀𐤌 (suffix_ok) {
                    لكل (size_t i = static_cast<size_t>(split_dim) + 1; i < ndims; ++i) {
                        𐤀𐤌 (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = خطأ;
                            اكسر;
                        }
                    }
                }

                𐤀𐤌 (suffix_ok && view_src_stride_v[split_dim] > 0) {
                    size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
                    int64_t split_index = static_cast<int64_t>(relative_offset / view_src_stride_v[split_dim]);

                    𐤀𐤌 (split_index >= 0 && split_index < num_splits) {
                        auto src_node = input.get_node_shared_ptr();
                        std::string rt_key = "split_dim_" + std::to_string(split_dim);
                        auto & rt_info = src_node->get_rt_info();

                        𐤀𐤌 (rt_info.find(rt_key) == rt_info.end()) {
                            auto axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(split_dim)});
                            auto split_node =
                                std::make_shared<ov::op::v1::Split>(input, axis_const, static_cast<size_t>(num_splits));
                            split_node->set_friendly_name(src_node->get_friendly_name() + "_split");
                            rt_info[rt_key] = split_node;
                        }

                        auto split_node = rt_info[rt_key].as<std::shared_ptr<ov::op::v1::Split>>();
                        أرجع split_node->output(static_cast<size_t>(split_index));
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
        ثابت صحيح axis_shift = current_rank.is_static() ?
                                    static_cast<صحيح>(view_stride.size()) - static_cast<صحيح>(current_rank.get_length()) :
                                    0;

        auto build_reshape_pattern = [](ثابت ov::PartialShape & target_ov_shape,
                                        ثابت ov::Shape & target_ggml_shape) -> std::vector<int64_t> {
            ثابت size_t ndims = target_ggml_shape.size();
            std::vector<int64_t> reshape_pattern(ndims);
            size_t dynamic_dims = 0;

            𐤀𐤌 (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(ndims)) {
                لكل (size_t i = 0; i < ndims; ++i) {
                    𐤀𐤌 (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } وإلا {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } وإلا {
                dynamic_dims = 2;
            }

            𐤀𐤌 (dynamic_dims > 1) {
                لكل (size_t i = 0; i < ndims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            أرجع reshape_pattern;
        };

        auto build_prefix_tail_reshape_pattern = [](ثابت ov::PartialShape & target_ov_shape,
                                                    ثابت ov::Shape & target_ggml_shape, size_t prefix_dims,
                                                    int64_t tail_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_pattern(prefix_dims + 1);
            size_t dynamic_dims = 0;

            𐤀𐤌 (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(target_ggml_shape.size())) {
                لكل (size_t i = 0; i < prefix_dims; ++i) {
                    𐤀𐤌 (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } وإلا {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } وإلا {
                dynamic_dims = 2;
            }

            𐤀𐤌 (dynamic_dims > 1) {
                لكل (size_t i = 0; i < prefix_dims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            reshape_pattern[prefix_dims] = tail_dim;
            أرجع reshape_pattern;
        };

        منطقي same_stride = view_stride.size() == view_src_stride.size();
        𐤀𐤌 (same_stride) {
            لكل (size_t i = 0; i < view_stride.size(); ++i) {
                𐤀𐤌 (view_stride[i] != view_src_stride[i]) {
                    same_stride = خطأ;
                    اكسر;
                }
            }
        }

        منطقي same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        𐤀𐤌 (same_ggml_shape) {
            لكل (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                𐤀𐤌 (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = خطأ;
                    اكسر;
                }
            }
        }

        𐤀𐤌 (same_stride && same_ggml_shape) {
            أرجع current;
        }

        𐤀𐤌 (same_stride) {
            ثابت size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
            ثابت size_t ndims = view_stride.size();

            std::vector<صحيح> diff_dims;
            𐤀𐤌 (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                لكل (size_t i = 0; i < ndims; ++i) {
                    𐤀𐤌 (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<صحيح>(i));
                    }
                }
            }

            𐤀𐤌 (diff_dims.size() == 1) {
                ثابت صحيح slice_dim = diff_dims[0];
                ثابت int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                𐤀𐤌 (view_stride[slice_dim] > 0 && relative_offset % view_stride[slice_dim] == 0) {
                    ثابت int64_t begin_val = static_cast<int64_t>((relative_offset / view_stride[slice_dim]) %
                                                                   static_cast<size_t>(dim_size));
                    ثابت int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);

                    𐤀𐤌 (begin_val >= 0 && end_val <= dim_size) {
                        auto sliced = std::make_shared<ov::op::v8::Slice>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim - axis_shift}));

                        𐤀𐤌 (view_ov_shape.is_static()) {
                            auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                                خطأ);
                            reshaped->set_friendly_name(view_name);
                            أرجع reshaped;
                        }

                        sliced->set_friendly_name(view_name);
                        أرجع sliced;
                    }
                }

                int64_t tail_src_elems = 1;
                int64_t tail_dst_elems = 1;
                لكل (size_t i = slice_dim; i < ndims; ++i) {
                    tail_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    tail_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                ثابت size_t elem_stride = view_stride[ndims - 1];
                int64_t tail_begin = 0;
                𐤀𐤌 (elem_stride > 0) {
                    tail_begin =
                        static_cast<int64_t>((relative_offset / elem_stride) % static_cast<size_t>(tail_src_elems));
                }
                ثابت int64_t tail_end = tail_begin + tail_dst_elems;

                𐤀𐤌 (tail_begin >= 0 && tail_end <= tail_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(slice_dim);
                    لكل (صحيح i = 0; i < slice_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(tail_src_elems);
                    ثابت size_t flat_ndims = flat_shape.size();

                    auto flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), خطأ);

                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_end}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

                    𐤀𐤌 (view_ov_shape.is_static()) {
                        auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                            خطأ);
                        reshaped->set_friendly_name(view_name);
                        أرجع reshaped;
                    }

                    sliced->set_friendly_name(view_name);
                    أرجع sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> step(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            size_t remaining_offset = relative_offset;
            لكل (size_t i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                𐤀𐤌 (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(remaining_offset / view_stride[i]);
                    remaining_offset %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            منطقي in_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            𐤀𐤌 (in_bounds) {
                لكل (size_t i = 0; i < ndims; ++i) {
                    𐤀𐤌 (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        in_bounds = خطأ;
                        اكسر;
                    }
                }
            }

            𐤀𐤌 (in_bounds && remaining_offset == 0) {
                auto sliced = std::make_shared<ov::op::v8::Slice>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->set_friendly_name(view_name);
                أرجع sliced;
            }
        } وإلا {
            منطقي same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            ثابت size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;

            𐤀𐤌 (same_rank) {
                ثابت size_t ndims = view_ggml_shape.size();
                std::vector<صحيح> diff_dims;
                لكل (size_t i = 0; i < ndims; ++i) {
                    𐤀𐤌 (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<صحيح>(i));
                    }
                }

                𐤀𐤌 (diff_dims.size() == 1) {
                    ثابت size_t slice_dim = static_cast<size_t>(diff_dims[0]);
                    منطقي suffix_stride_match = true;
                    لكل (size_t i = slice_dim + 1; i < ndims; ++i) {
                        𐤀𐤌 (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_match = خطأ;
                            اكسر;
                        }
                    }

                    𐤀𐤌 (suffix_stride_match && view_src_stride[slice_dim] > 0 &&
                        relative_offset % view_src_stride[slice_dim] == 0) {
                        ثابت int64_t begin_val = static_cast<int64_t>(relative_offset / view_src_stride[slice_dim]);
                        ثابت int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);
                        ثابت int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                        𐤀𐤌 (begin_val >= 0 && end_val <= dim_size) {
                            auto sliced = std::make_shared<ov::op::v8::Slice>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(slice_dim)}));
                            sliced->set_friendly_name(view_name);
                            أرجع sliced;
                        }
                    }
                }
            }

            size_t view_elems = 1;
            size_t src_elems = 1;
            𐤀𐤌 (same_rank) {
                لكل (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            منطقي same_num_elements = same_rank && view_elems == src_elems;

            𐤀𐤌 (same_rank && relative_offset == 0 && same_num_elements) {
                auto reshape_pattern = build_reshape_pattern(view_ov_shape, view_ggml_shape);

                auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_pattern.size()}, reshape_pattern),
                    خطأ);
                reshaped->set_friendly_name(view_name);
                أرجع reshaped;
            }

            𐤀𐤌 (same_rank) {
                ثابت size_t ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                منطقي is_regular_slice = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> step(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                size_t remaining_offset = relative_offset;

                𐤀𐤌 (is_regular_slice) {
                    لكل (size_t i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        𐤀𐤌 (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_slice = خطأ;
                            اكسر;
                        }

                        step[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        𐤀𐤌 (step[i] <= 0) {
                            is_regular_slice = خطأ;
                            اكسر;
                        }

                        begin[i] = static_cast<int64_t>(remaining_offset / view_src_stride[i]);
                        remaining_offset %= view_src_stride[i];

                        𐤀𐤌 (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            استمر;
                        }

                        end[i] = begin[i] + step[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        𐤀𐤌 (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_slice = خطأ;
                            اكسر;
                        }
                    }
                }

                𐤀𐤌 (is_regular_slice && remaining_offset == 0) {
                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->set_friendly_name(view_name);
                    أرجع sliced;
                }

                ثابت size_t elem_stride = view_src_stride.back();
                ثابت منطقي aligned_offset = elem_stride > 0 && relative_offset % elem_stride == 0;

                𐤀𐤌 (aligned_offset) {
                    size_t suffix_start = 0;
                    size_t expected_stride = elem_stride;
                    لكل (صحيح i = static_cast<صحيح>(ndims) - 1; i >= 0; --i) {
                        𐤀𐤌 (view_stride[i] != expected_stride) {
                            suffix_start = static_cast<size_t>(i + 1);
                            اكسر;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    size_t prefix_elems = 1;
                    size_t suffix_elems = 1;
                    لكل (size_t i = 0; i < suffix_start; ++i) {
                        prefix_elems *= view_ggml_shape[i];
                    }
                    لكل (size_t i = suffix_start; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    𐤀𐤌 (prefix_elems > 0 && src_elems % prefix_elems == 0) {
                        ثابت size_t src_tail_elems = src_elems / prefix_elems;
                        ثابت int64_t tail_begin = static_cast<int64_t>(relative_offset / elem_stride);
                        ثابت int64_t tail_end = tail_begin + static_cast<int64_t>(suffix_elems);

                        𐤀𐤌 (tail_begin >= 0 && tail_end <= static_cast<int64_t>(src_tail_elems)) {
                            auto prefix_tail_pattern = build_prefix_tail_reshape_pattern(
                                view_ov_shape, view_ggml_shape, suffix_start, static_cast<int64_t>(src_tail_elems));

                            auto prefix_tail = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {prefix_tail_pattern.size()},
                                                             prefix_tail_pattern),
                                خطأ);

                            ov::Output<ov::Node> selected = prefix_tail;
                            𐤀𐤌 (tail_begin != 0 || tail_end != static_cast<int64_t>(src_tail_elems)) {
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
                                خطأ);
                            reshaped->set_friendly_name(view_name);
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
        auto innermost_offset = context.get_view_input_offset(input_index, 0);
        auto innermost_src_offset = context.get_view_input_src_offset(input_index, 0);
        auto innermost_shape = context.get_view_input_ggml_shape(input_index, 0);
        auto innermost_src_shape = context.get_view_input_src_ggml_shape(input_index, 0);
        𐤀𐤌 (innermost_offset != innermost_src_offset && innermost_shape == innermost_src_shape) {
            size_t root_view_idx = view_input_size - 1;
            auto root_ggml_shape = context.get_view_input_src_ggml_shape(input_index, root_view_idx);
            auto root_stride = context.get_view_input_src_stride(input_index, root_view_idx);
            auto root_offset = context.get_view_input_src_offset(input_index, root_view_idx);
            auto root_ov_shape = context.get_view_input_src_ov_shape(input_index, root_view_idx);
            auto root_name = context.get_view_input_src_name(input_index, root_view_idx);
            auto innermost_stride = context.get_view_input_stride(input_index, 0);
            auto innermost_ov_shape = context.get_view_input_ov_shape(input_index, 0);
            auto innermost_name = context.get_view_input_name(input_index, 0);

            أرجع process_single_view(input, innermost_offset, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, root_offset, root_stride, root_ggml_shape, root_ov_shape,
                                       root_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = input;

    // Process each view in reverse order (from base to current)
    لكل (صحيح view_idx = view_input_size - 1; view_idx >= 0; view_idx--) {
        auto view_offset = context.get_view_input_offset(input_index, view_idx);
        auto view_stride = context.get_view_input_stride(input_index, view_idx);
        auto view_ggml_shape = context.get_view_input_ggml_shape(input_index, view_idx);
        auto view_ov_shape = context.get_view_input_ov_shape(input_index, view_idx);
        auto view_name = context.get_view_input_name(input_index, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        auto view_src_offset = context.get_view_input_src_offset(input_index, view_idx);
        auto view_src_stride = context.get_view_input_src_stride(input_index, view_idx);
        auto view_src_ggml_shape = context.get_view_input_src_ggml_shape(input_index, view_idx);
        auto view_src_ov_shape = context.get_view_input_src_ov_shape(input_index, view_idx);
        auto view_src_name = context.get_view_input_src_name(input_index, view_idx);
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

    أرجع current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
