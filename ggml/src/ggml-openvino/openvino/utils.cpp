#अन्तर्भाव "utils.h"

#अन्तर्भाव <cmath>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <ctime>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/clamp.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/cos.hpp>
#अन्तर्भाव <openvino/op/divide.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/maximum.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <openvino/op/sin.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/split.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>
#अन्तर्भाव <openvino/op/subtract.hpp>
#अन्तर्भाव <openvino/op/transpose.hpp>
#अन्तर्भाव <string>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {

शून्य अङ्कः_inputs_check(ध्रुव ग्रन्थिः_Context & context, size_t min_inputs, size_t max_inputs) {
    auto निवेशः_size = context.get_निवेशः_size();
    FRONT_END_OP_CONVERSION_CHECK(निवेशः_size >= min_inputs, "Got less inputs than expected");
    FRONT_END_OP_CONVERSION_CHECK(निवेशः_size <= max_inputs, "Got more inputs than expected");
}

पूर्णाङ्क non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    पूर्णाङ्क dim = nb.size() - 1;
    size_t bytes = nb[dim];
    अत्थाय (पूर्णाङ्क i = dim; i > 0; i--) {
        bytes *= ne[i];
        सचे (bytes != nb[i - 1]) {
            पटिनिवत्त i;
        }
    }
    पटिनिवत्त 0;
}

std::shared_ptr<ov::Node> get_dimensions(ध्रुव std::shared_ptr<ov::op::v3::ShapeOf> & shape,
                                         ध्रुव std::vector<पूर्णाङ्क> & dims) {
    using नामाकाश ov::op;
    ध्रुव auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ध्रुव auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    पटिनिवत्त std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(ध्रुव std::shared_ptr<ov::Node> & ग्रन्थिः, ध्रुव std::vector<पूर्णाङ्क> & dims) {
    पटिनिवत्त get_dimensions(std::make_shared<ov::op::v3::ShapeOf>(ग्रन्थिः), dims);
}

OutputVector rename_outputs_with_suffix(ध्रुव OutputVector & outputs, ध्रुव std::string & suffix) {
    अत्थाय (ध्रुव auto & output : outputs) {
        auto ग्रन्थिः = output.get_ग्रन्थिः_shared_ptr();
        std::string name = ग्रन्थिः->get_friendly_name();
        name += "_";
        name += suffix;
        ग्रन्थिः->set_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    पटिनिवत्त outputs;
}

नामाकाश {
ov::Output<ov::Node> rope_yarn_ramp_mix(पूर्णाङ्क अङ्कः_elements, ध्रुव भिन्न corr_dims[2], भिन्न ext_factor) {
    std::vector<भिन्न> dim_ids_vec(अङ्कः_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    auto dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) अङ्कः_elements}, dim_ids_vec);
    auto corr_नीचः = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    auto corr_उच्चः = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    auto denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::Subtract>(corr_उच्चः, corr_नीचः),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    auto ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Subtract>(dim_ids, corr_नीचः), denom);
    auto ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    auto one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    auto ramp_inverted = std::make_shared<ov::op::v1::Subtract>(one, ramp_clamped);
    auto ext_factor_ग्रन्थिः = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_factor});
    auto ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_factor_ग्रन्थिः);
    पटिनिवत्त ramp_mix;
}

भिन्न ggml_rope_yarn_corr_dim(पूर्णाङ्क n_dims, पूर्णाङ्क n_ctx_orig, भिन्न n_rot, भिन्न base) {
#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
    पटिनिवत्त n_dims * logf(n_ctx_orig / (n_rot * 2 * (भिन्न) M_PI)) / (2 * logf(base));
}

शून्य ggml_rope_yarn_corr_dims(पूर्णाङ्क n_dims,
                              पूर्णाङ्क n_ctx_orig,
                              भिन्न freq_base,
                              भिन्न beta_fast,
                              भिन्न beta_slow,
                              भिन्न dims[2]) {
    भिन्न start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    भिन्न end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<भिन्न>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight,
                                                           पूर्णाङ्क mode,
                                                           तार्किक stateful,
                                                           int64_t head_dim) {
    constexpr पूर्णाङ्क TYPE_IMROPE = 2;
    constexpr पूर्णाङ्क TYPE_VISION = 3;
    constexpr पूर्णाङ्क TYPE_MROPE = 4;

    ध्रुव तार्किक is_imrope = (mode == TYPE_IMROPE);
    ध्रुव तार्किक is_vision = (mode == TYPE_VISION);
    ध्रुव तार्किक is_mrope = (mode == TYPE_MROPE);
    ध्रुव तार्किक is_multi = is_imrope || is_vision || is_mrope;

    सचे (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } अञ्ञथा सचे (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, सच्च);
        auto pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } अञ्ञथा {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    भिन्न freq_base;
    भिन्न freq_scale;
    भिन्न ext_factor;
    भिन्न attn_factor;
    भिन्न beta_fast;
    भिन्न beta_slow;
    ध्रुव पूर्णाङ्क n_dims = rope_params[1];
    ध्रुव size_t n_dims_half = n_dims >> 1;
    ध्रुव पूर्णाङ्क n_ctx_orig = rope_params[4];
    memcpy(&freq_base, rope_params + 5, परिमाण(भिन्न));
    memcpy(&freq_scale, rope_params + 6, परिमाण(भिन्न));
    memcpy(&ext_factor, rope_params + 7, परिमाण(भिन्न));
    memcpy(&attn_factor, rope_params + 8, परिमाण(भिन्न));
    memcpy(&beta_fast, rope_params + 9, परिमाण(भिन्न));
    memcpy(&beta_slow, rope_params + 10, परिमाण(भिन्न));

    ध्रुव भिन्न theta_scale = powf(freq_base, -2.0f / n_dims);

    Output<Node> theta;
    भिन्न mscale = attn_factor;
    सचे (is_multi) {
        ध्रुव size_t अङ्कः_pairs = is_vision ? (n_dims > 0 ? (size_t) n_dims : (size_t) (head_dim / 2)) : n_dims_half;
        पूर्णाङ्क sections[4];
        memcpy(sections, rope_params + 11, परिमाण(पूर्णाङ्क) * 4);
        पूर्णाङ्क sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        सचे (sect_dims <= 0) {
            sect_dims = अङ्कः_pairs;
        }
        पूर्णाङ्क sec_w = sections[0] + sections[1];
        पूर्णाङ्क sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(अङ्कः_pairs);
        std::vector<भिन्न> factor(अङ्कः_pairs);
        अत्थाय (size_t j = 0; j < अङ्कः_pairs; j++) {
            पूर्णाङ्क sector = j % sect_dims;
            पूर्णाङ्क plane = 0;
            सचे (is_imrope) {
                सचे (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } अञ्ञथा सचे (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } अञ्ञथा सचे (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } अञ्ञथा {
                    plane = 3;
                }
                factor[j] = std::pow(theta_scale, j);
            } अञ्ञथा सचे (is_vision) {
                पूर्णाङ्क p_idx = 0;
                सचे (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } अञ्ञथा सचे (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } अञ्ञथा सचे (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } अञ्ञथा {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                factor[j] = std::pow(theta_scale, p_idx);
            } अञ्ञथा {
                सचे (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } अञ्ञथा सचे (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } अञ्ञथा सचे (sector >= sec_e) {
                    plane = 3;
                } अञ्ञथा {
                    plane = 0;
                }
                factor[j] = std::pow(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        auto gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{अङ्कः_pairs}, gather_indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> factor_ग्रन्थिः =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{अङ्कः_pairs}, factor);
        सचे (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) अङ्कः_pairs}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) अङ्कः_pairs}),
                मिच्छा);
            factor_ग्रन्थिः = std::make_shared<ov::op::v1::Divide>(factor_ग्रन्थिः, rope_factors);
        }
        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, factor_ग्रन्थिः);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        सचे (ext_factor == 0.0f) {
            theta = theta_interp;
        } अञ्ञथा {
            भिन्न corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
            auto ramp_mix = rope_yarn_ramp_mix(अङ्कः_pairs, corr_dims, ext_factor);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            auto one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::Add>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
        }
    } अञ्ञथा {
        std::vector<भिन्न> factor(n_dims_half);
        Output<Node> freq_factors;
        भिन्न corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        factor[0] = 1.0f;
        अत्थाय (size_t i = 1; i < factor.size(); i++) {
            factor[i] = theta_scale * factor[i - 1];
        }
        सचे (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, factor.size()}, factor);
        } अञ्ञथा {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, factor.size()}, factor);
        }
        सचे (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            सचे (stateful) {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {3}, {(int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    मिच्छा);
            } अञ्ञथा {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    मिच्छा);
            }
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_factors);
        }

        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        सचे (ext_factor == 0.0f) {
            theta = theta_interp;
        } अञ्ञथा {
            auto ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_factor);
            Output<Node> one;
            सचे (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } अञ्ञथा {
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

    सचे (mscale != 1.0f) {
        auto mscale_ग्रन्थिः = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_ग्रन्थिः);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_ग्रन्थिः);
    }

    पटिनिवत्त std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_निवेशः(ध्रुव ग्रन्थिः_Context & context, पूर्णाङ्क निवेशः_index, पूर्णाङ्क slice_len, पूर्णाङ्क axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    auto निवेशः = context.get_निवेशः(निवेशः_index);
    auto * op_params = (size_t *) context.get_निवेशः_op_params(निवेशः_index);
    auto src_stride = context.get_निवेशः_stride(निवेशः_index);

    int64_t slice_start = op_params[0] / src_stride[3];
    सचे (slice_len == 0) {
        slice_len = context.get_निवेशः_shape(निवेशः_index)[3].get_length();
    }
    int64_t slice_end = slice_start + slice_len;

    auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_start});
    auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_end});
    auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    सचे (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } अञ्ञथा {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    auto sliced = std::make_shared<ov::op::v8::Slice>(निवेशः, begin, end, stride, axes);
    पटिनिवत्त sliced;
}

ov::Output<ov::Node> process_view_निवेशः_new(ध्रुव ग्रन्थिः_Context & context, पूर्णाङ्क निवेशः_index) {
    auto निवेशः = context.get_निवेशः(निवेशः_index);

    // Check if this input has view inputs
    size_t view_निवेशः_size = context.get_view_निवेशः_size(निवेशः_index);
    सचे (view_निवेशः_size == 0) {
        // No view inputs, return the input as is
        पटिनिवत्त निवेशः;
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
    auto expected_ov_shape = context.get_view_निवेशः_ov_shape(निवेशः_index, 0);
    auto actual_shape = निवेशः.get_partial_shape();
    सचे (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_length() >= actual_shape.rank().get_length()) {
        ध्रुव int64_t n_actual = actual_shape.rank().get_length();
        ध्रुव int64_t shift = expected_ov_shape.rank().get_length() - n_actual;
        तार्किक shapes_match = सच्च;
        अत्थाय (int64_t i = 0; i < shift; ++i) {
            सचे (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_length() != 1) {
                shapes_match = मिच्छा;
                भङ्ग;
            }
        }
        अत्थाय (int64_t i = 0; i < n_actual && shapes_match; ++i) {
            ध्रुव auto & exp = expected_ov_shape[i + shift];
            ध्रुव तार्किक both_dynamic = exp.is_dynamic() && actual_shape[i].is_dynamic();
            ध्रुव तार्किक both_static_equal =
                exp.is_static() && actual_shape[i].is_static() && exp == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            ध्रुव तार्किक expected_dyn_actual_static = exp.is_dynamic() && actual_shape[i].is_static();
            सचे (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_match = मिच्छा;
                भङ्ग;
            }
        }
        सचे (shapes_match) {
            पटिनिवत्त निवेशः;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    सचे (context.is_static() && view_निवेशः_size == 1) {
        auto view_stride_v = context.get_view_निवेशः_stride(निवेशः_index, 0);
        auto view_src_stride_v = context.get_view_निवेशः_src_stride(निवेशः_index, 0);
        auto view_ggml_shape = context.get_view_निवेशः_ggml_shape(निवेशः_index, 0);
        auto view_src_ggml_shape = context.get_view_निवेशः_src_ggml_shape(निवेशः_index, 0);
        auto view_offset = context.get_view_निवेशः_offset(निवेशः_index, 0);
        auto view_src_offset = context.get_view_निवेशः_src_offset(निवेशः_index, 0);

        size_t ndims = view_ggml_shape.size();
        std::vector<पूर्णाङ्क> diff_dims;
        सचे (view_src_ggml_shape.size() == ndims) {
            अत्थाय (size_t i = 0; i < ndims; ++i) {
                सचे (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<पूर्णाङ्क>(i));
                }
            }
        }

        सचे (diff_dims.size() == 1) {
            पूर्णाङ्क split_dim = diff_dims[0];
            int64_t अङ्कः_splits = static_cast<int64_t>(view_src_ggml_shape[split_dim]);
            int64_t chunk_size = static_cast<int64_t>(view_ggml_shape[split_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            सचे (chunk_size == 1 && अङ्कः_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                तार्किक suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                सचे (suffix_ok) {
                    अत्थाय (size_t i = static_cast<size_t>(split_dim) + 1; i < ndims; ++i) {
                        सचे (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = मिच्छा;
                            भङ्ग;
                        }
                    }
                }

                सचे (suffix_ok && view_src_stride_v[split_dim] > 0) {
                    size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
                    int64_t split_index = static_cast<int64_t>(relative_offset / view_src_stride_v[split_dim]);

                    सचे (split_index >= 0 && split_index < अङ्कः_splits) {
                        auto src_ग्रन्थिः = निवेशः.get_ग्रन्थिः_shared_ptr();
                        std::string rt_कुञ्चिका = "split_dim_" + std::to_string(split_dim);
                        auto & rt_info = src_ग्रन्थिः->get_rt_info();

                        सचे (rt_info.find(rt_कुञ्चिका) == rt_info.end()) {
                            auto axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(split_dim)});
                            auto split_ग्रन्थिः =
                                std::make_shared<ov::op::v1::Split>(निवेशः, axis_const, static_cast<size_t>(अङ्कः_splits));
                            split_ग्रन्थिः->set_friendly_name(src_ग्रन्थिः->get_friendly_name() + "_split");
                            rt_info[rt_कुञ्चिका] = split_ग्रन्थिः;
                        }

                        auto split_ग्रन्थिः = rt_info[rt_कुञ्चिका].as<std::shared_ptr<ov::op::v1::Split>>();
                        पटिनिवत्त split_ग्रन्थिः->output(static_cast<size_t>(split_index));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    auto process_single_view =
        [](ov::Output<ov::Node> current, size_t view_offset, ध्रुव std::vector<size_t> & view_stride,
           ध्रुव ov::Shape & view_ggml_shape, ध्रुव ov::PartialShape & view_ov_shape, ध्रुव std::string & view_name,
           size_t view_src_offset, ध्रुव std::vector<size_t> & view_src_stride, ध्रुव ov::Shape & view_src_ggml_shape,
           ध्रुव ov::PartialShape & view_src_ov_shape, ध्रुव std::string & view_src_name) -> ov::Output<ov::Node> {
        // Stateful execution drops a leading size-1 axis, so `current` can be one rank lower
        // than the ggml shape metadata (view_stride/view_ggml_shape, always GGML_MAX_DIMS
        // entries) assumes. Shift any axis index built from that metadata down by the
        // difference before using it as an OV Slice axis.
        ध्रुव auto current_rank = current.get_partial_shape().rank();
        ध्रुव पूर्णाङ्क axis_shift = current_rank.is_static() ?
                                    static_cast<पूर्णाङ्क>(view_stride.size()) - static_cast<पूर्णाङ्क>(current_rank.get_length()) :
                                    0;

        auto build_reshape_pattern = [](ध्रुव ov::PartialShape & target_ov_shape,
                                        ध्रुव ov::Shape & target_ggml_shape) -> std::vector<int64_t> {
            ध्रुव size_t ndims = target_ggml_shape.size();
            std::vector<int64_t> reshape_pattern(ndims);
            size_t dynamic_dims = 0;

            सचे (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(ndims)) {
                अत्थाय (size_t i = 0; i < ndims; ++i) {
                    सचे (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } अञ्ञथा {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } अञ्ञथा {
                dynamic_dims = 2;
            }

            सचे (dynamic_dims > 1) {
                अत्थाय (size_t i = 0; i < ndims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            पटिनिवत्त reshape_pattern;
        };

        auto build_prefix_tail_reshape_pattern = [](ध्रुव ov::PartialShape & target_ov_shape,
                                                    ध्रुव ov::Shape & target_ggml_shape, size_t prefix_dims,
                                                    int64_t tail_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_pattern(prefix_dims + 1);
            size_t dynamic_dims = 0;

            सचे (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(target_ggml_shape.size())) {
                अत्थाय (size_t i = 0; i < prefix_dims; ++i) {
                    सचे (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } अञ्ञथा {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } अञ्ञथा {
                dynamic_dims = 2;
            }

            सचे (dynamic_dims > 1) {
                अत्थाय (size_t i = 0; i < prefix_dims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            reshape_pattern[prefix_dims] = tail_dim;
            पटिनिवत्त reshape_pattern;
        };

        तार्किक same_stride = view_stride.size() == view_src_stride.size();
        सचे (same_stride) {
            अत्थाय (size_t i = 0; i < view_stride.size(); ++i) {
                सचे (view_stride[i] != view_src_stride[i]) {
                    same_stride = मिच्छा;
                    भङ्ग;
                }
            }
        }

        तार्किक same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        सचे (same_ggml_shape) {
            अत्थाय (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                सचे (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = मिच्छा;
                    भङ्ग;
                }
            }
        }

        सचे (same_stride && same_ggml_shape) {
            पटिनिवत्त current;
        }

        सचे (same_stride) {
            ध्रुव size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
            ध्रुव size_t ndims = view_stride.size();

            std::vector<पूर्णाङ्क> diff_dims;
            सचे (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                अत्थाय (size_t i = 0; i < ndims; ++i) {
                    सचे (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<पूर्णाङ्क>(i));
                    }
                }
            }

            सचे (diff_dims.size() == 1) {
                ध्रुव पूर्णाङ्क slice_dim = diff_dims[0];
                ध्रुव int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                सचे (view_stride[slice_dim] > 0 && relative_offset % view_stride[slice_dim] == 0) {
                    ध्रुव int64_t begin_val = static_cast<int64_t>((relative_offset / view_stride[slice_dim]) %
                                                                   static_cast<size_t>(dim_size));
                    ध्रुव int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);

                    सचे (begin_val >= 0 && end_val <= dim_size) {
                        auto sliced = std::make_shared<ov::op::v8::Slice>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim - axis_shift}));

                        सचे (view_ov_shape.is_static()) {
                            auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                                मिच्छा);
                            reshaped->set_friendly_name(view_name);
                            पटिनिवत्त reshaped;
                        }

                        sliced->set_friendly_name(view_name);
                        पटिनिवत्त sliced;
                    }
                }

                int64_t tail_src_elems = 1;
                int64_t tail_dst_elems = 1;
                अत्थाय (size_t i = slice_dim; i < ndims; ++i) {
                    tail_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    tail_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                ध्रुव size_t elem_stride = view_stride[ndims - 1];
                int64_t tail_begin = 0;
                सचे (elem_stride > 0) {
                    tail_begin =
                        static_cast<int64_t>((relative_offset / elem_stride) % static_cast<size_t>(tail_src_elems));
                }
                ध्रुव int64_t tail_end = tail_begin + tail_dst_elems;

                सचे (tail_begin >= 0 && tail_end <= tail_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(slice_dim);
                    अत्थाय (पूर्णाङ्क i = 0; i < slice_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(tail_src_elems);
                    ध्रुव size_t flat_ndims = flat_shape.size();

                    auto flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), मिच्छा);

                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_end}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

                    सचे (view_ov_shape.is_static()) {
                        auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                            मिच्छा);
                        reshaped->set_friendly_name(view_name);
                        पटिनिवत्त reshaped;
                    }

                    sliced->set_friendly_name(view_name);
                    पटिनिवत्त sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> step(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            size_t remaining_offset = relative_offset;
            अत्थाय (size_t i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                सचे (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(remaining_offset / view_stride[i]);
                    remaining_offset %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            तार्किक in_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            सचे (in_bounds) {
                अत्थाय (size_t i = 0; i < ndims; ++i) {
                    सचे (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        in_bounds = मिच्छा;
                        भङ्ग;
                    }
                }
            }

            सचे (in_bounds && remaining_offset == 0) {
                auto sliced = std::make_shared<ov::op::v8::Slice>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->set_friendly_name(view_name);
                पटिनिवत्त sliced;
            }
        } अञ्ञथा {
            तार्किक same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            ध्रुव size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;

            सचे (same_rank) {
                ध्रुव size_t ndims = view_ggml_shape.size();
                std::vector<पूर्णाङ्क> diff_dims;
                अत्थाय (size_t i = 0; i < ndims; ++i) {
                    सचे (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<पूर्णाङ्क>(i));
                    }
                }

                सचे (diff_dims.size() == 1) {
                    ध्रुव size_t slice_dim = static_cast<size_t>(diff_dims[0]);
                    तार्किक suffix_stride_match = सच्च;
                    अत्थाय (size_t i = slice_dim + 1; i < ndims; ++i) {
                        सचे (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_match = मिच्छा;
                            भङ्ग;
                        }
                    }

                    सचे (suffix_stride_match && view_src_stride[slice_dim] > 0 &&
                        relative_offset % view_src_stride[slice_dim] == 0) {
                        ध्रुव int64_t begin_val = static_cast<int64_t>(relative_offset / view_src_stride[slice_dim]);
                        ध्रुव int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);
                        ध्रुव int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                        सचे (begin_val >= 0 && end_val <= dim_size) {
                            auto sliced = std::make_shared<ov::op::v8::Slice>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(slice_dim)}));
                            sliced->set_friendly_name(view_name);
                            पटिनिवत्त sliced;
                        }
                    }
                }
            }

            size_t view_elems = 1;
            size_t src_elems = 1;
            सचे (same_rank) {
                अत्थाय (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            तार्किक same_अङ्कः_elements = same_rank && view_elems == src_elems;

            सचे (same_rank && relative_offset == 0 && same_अङ्कः_elements) {
                auto reshape_pattern = build_reshape_pattern(view_ov_shape, view_ggml_shape);

                auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_pattern.size()}, reshape_pattern),
                    मिच्छा);
                reshaped->set_friendly_name(view_name);
                पटिनिवत्त reshaped;
            }

            सचे (same_rank) {
                ध्रुव size_t ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                तार्किक is_regular_slice = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> step(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                size_t remaining_offset = relative_offset;

                सचे (is_regular_slice) {
                    अत्थाय (size_t i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        सचे (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_slice = मिच्छा;
                            भङ्ग;
                        }

                        step[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        सचे (step[i] <= 0) {
                            is_regular_slice = मिच्छा;
                            भङ्ग;
                        }

                        begin[i] = static_cast<int64_t>(remaining_offset / view_src_stride[i]);
                        remaining_offset %= view_src_stride[i];

                        सचे (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            अनुवर्त;
                        }

                        end[i] = begin[i] + step[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        सचे (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_slice = मिच्छा;
                            भङ्ग;
                        }
                    }
                }

                सचे (is_regular_slice && remaining_offset == 0) {
                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->set_friendly_name(view_name);
                    पटिनिवत्त sliced;
                }

                ध्रुव size_t elem_stride = view_src_stride.back();
                ध्रुव तार्किक aligned_offset = elem_stride > 0 && relative_offset % elem_stride == 0;

                सचे (aligned_offset) {
                    size_t suffix_start = 0;
                    size_t expected_stride = elem_stride;
                    अत्थाय (पूर्णाङ्क i = static_cast<पूर्णाङ्क>(ndims) - 1; i >= 0; --i) {
                        सचे (view_stride[i] != expected_stride) {
                            suffix_start = static_cast<size_t>(i + 1);
                            भङ्ग;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    size_t prefix_elems = 1;
                    size_t suffix_elems = 1;
                    अत्थाय (size_t i = 0; i < suffix_start; ++i) {
                        prefix_elems *= view_ggml_shape[i];
                    }
                    अत्थाय (size_t i = suffix_start; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    सचे (prefix_elems > 0 && src_elems % prefix_elems == 0) {
                        ध्रुव size_t src_tail_elems = src_elems / prefix_elems;
                        ध्रुव int64_t tail_begin = static_cast<int64_t>(relative_offset / elem_stride);
                        ध्रुव int64_t tail_end = tail_begin + static_cast<int64_t>(suffix_elems);

                        सचे (tail_begin >= 0 && tail_end <= static_cast<int64_t>(src_tail_elems)) {
                            auto prefix_tail_pattern = build_prefix_tail_reshape_pattern(
                                view_ov_shape, view_ggml_shape, suffix_start, static_cast<int64_t>(src_tail_elems));

                            auto prefix_tail = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {prefix_tail_pattern.size()},
                                                             prefix_tail_pattern),
                                मिच्छा);

                            ov::Output<ov::Node> selected = prefix_tail;
                            सचे (tail_begin != 0 || tail_end != static_cast<int64_t>(src_tail_elems)) {
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
                                मिच्छा);
                            reshaped->set_friendly_name(view_name);
                            पटिनिवत्त reshaped;
                        }
                    }
                }
            }

            पटिनिवत्त current;
        }

        (शून्य) view_name;
        (शून्य) view_src_ov_shape;
        (शून्य) view_src_name;

        पटिनिवत्त current;
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
        auto innermost_offset = context.get_view_निवेशः_offset(निवेशः_index, 0);
        auto innermost_src_offset = context.get_view_निवेशः_src_offset(निवेशः_index, 0);
        auto innermost_shape = context.get_view_निवेशः_ggml_shape(निवेशः_index, 0);
        auto innermost_src_shape = context.get_view_निवेशः_src_ggml_shape(निवेशः_index, 0);
        सचे (innermost_offset != innermost_src_offset && innermost_shape == innermost_src_shape) {
            size_t मूलम्_view_idx = view_निवेशः_size - 1;
            auto मूलम्_ggml_shape = context.get_view_निवेशः_src_ggml_shape(निवेशः_index, मूलम्_view_idx);
            auto मूलम्_stride = context.get_view_निवेशः_src_stride(निवेशः_index, मूलम्_view_idx);
            auto मूलम्_offset = context.get_view_निवेशः_src_offset(निवेशः_index, मूलम्_view_idx);
            auto मूलम्_ov_shape = context.get_view_निवेशः_src_ov_shape(निवेशः_index, मूलम्_view_idx);
            auto मूलम्_name = context.get_view_निवेशः_src_name(निवेशः_index, मूलम्_view_idx);
            auto innermost_stride = context.get_view_निवेशः_stride(निवेशः_index, 0);
            auto innermost_ov_shape = context.get_view_निवेशः_ov_shape(निवेशः_index, 0);
            auto innermost_name = context.get_view_निवेशः_name(निवेशः_index, 0);

            पटिनिवत्त process_single_view(निवेशः, innermost_offset, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, मूलम्_offset, मूलम्_stride, मूलम्_ggml_shape, मूलम्_ov_shape,
                                       मूलम्_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = निवेशः;

    // Process each view in reverse order (from base to current)
    अत्थाय (पूर्णाङ्क view_idx = view_निवेशः_size - 1; view_idx >= 0; view_idx--) {
        auto view_offset = context.get_view_निवेशः_offset(निवेशः_index, view_idx);
        auto view_stride = context.get_view_निवेशः_stride(निवेशः_index, view_idx);
        auto view_ggml_shape = context.get_view_निवेशः_ggml_shape(निवेशः_index, view_idx);
        auto view_ov_shape = context.get_view_निवेशः_ov_shape(निवेशः_index, view_idx);
        auto view_name = context.get_view_निवेशः_name(निवेशः_index, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        auto view_src_offset = context.get_view_निवेशः_src_offset(निवेशः_index, view_idx);
        auto view_src_stride = context.get_view_निवेशः_src_stride(निवेशः_index, view_idx);
        auto view_src_ggml_shape = context.get_view_निवेशः_src_ggml_shape(निवेशः_index, view_idx);
        auto view_src_ov_shape = context.get_view_निवेशः_src_ov_shape(निवेशः_index, view_idx);
        auto view_src_name = context.get_view_निवेशः_src_name(निवेशः_index, view_idx);
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

    पटिनिवत्त current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
