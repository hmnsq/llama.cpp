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

शून्य अङ्क_inputs_check(अचर गाँठो_Context & context, माप_प्रकार min_inputs, माप_प्रकार max_inputs) {
    स्वतः आगत_size = context.get_आगत_size();
    FRONT_END_OP_CONVERSION_CHECK(आगत_size >= min_inputs, "Got less inputs than expected");
    FRONT_END_OP_CONVERSION_CHECK(आगत_size <= max_inputs, "Got more inputs than expected");
}

पूर्णाङ्क non_cont_dim(std::vector<माप_प्रकार> ne, std::vector<माप_प्रकार> nb) {
    पूर्णाङ्क dim = nb.size() - 1;
    माप_प्रकार bytes = nb[dim];
    लागि (पूर्णाङ्क i = dim; i > 0; i--) {
        bytes *= ne[i];
        यदि (bytes != nb[i - 1]) {
            फर्काऊ i;
        }
    }
    फर्काऊ 0;
}

std::shared_ptr<ov::Node> get_dimensions(अचर std::shared_ptr<ov::op::v3::ShapeOf> & shape,
                                         अचर std::vector<पूर्णाङ्क> & dims) {
    using नामाकाश ov::op;
    अचर स्वतः zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    अचर स्वतः dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    फर्काऊ std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(अचर std::shared_ptr<ov::Node> & गाँठो, अचर std::vector<पूर्णाङ्क> & dims) {
    फर्काऊ get_dimensions(std::make_shared<ov::op::v3::ShapeOf>(गाँठो), dims);
}

OutputVector rename_outputs_with_suffix(अचर OutputVector & outputs, अचर std::string & suffix) {
    लागि (अचर स्वतः & output : outputs) {
        स्वतः गाँठो = output.get_गाँठो_shared_ptr();
        std::string name = गाँठो->get_friendly_name();
        name += "_";
        name += suffix;
        गाँठो->set_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    फर्काऊ outputs;
}

नामाकाश {
ov::Output<ov::Node> rope_yarn_ramp_mix(पूर्णाङ्क अङ्क_elements, अचर भिन्न corr_dims[2], भिन्न ext_factor) {
    std::vector<भिन्न> dim_ids_vec(अङ्क_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    स्वतः dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (माप_प्रकार) अङ्क_elements}, dim_ids_vec);
    स्वतः corr_तल्लो = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    स्वतः corr_माथिल्लो = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    स्वतः denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::Subtract>(corr_माथिल्लो, corr_तल्लो),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    स्वतः ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Subtract>(dim_ids, corr_तल्लो), denom);
    स्वतः ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    स्वतः one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    स्वतः ramp_inverted = std::make_shared<ov::op::v1::Subtract>(one, ramp_clamped);
    स्वतः ext_factor_गाँठो = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_factor});
    स्वतः ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_factor_गाँठो);
    फर्काऊ ramp_mix;
}

भिन्न ggml_rope_yarn_corr_dim(पूर्णाङ्क n_dims, पूर्णाङ्क n_ctx_orig, भिन्न n_rot, भिन्न base) {
#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
    फर्काऊ n_dims * logf(n_ctx_orig / (n_rot * 2 * (भिन्न) M_PI)) / (2 * logf(base));
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

    अचर तार्किक is_imrope = (mode == TYPE_IMROPE);
    अचर तार्किक is_vision = (mode == TYPE_VISION);
    अचर तार्किक is_mrope = (mode == TYPE_MROPE);
    अचर तार्किक is_multi = is_imrope || is_vision || is_mrope;

    यदि (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        स्वतः pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } अन्यथा यदि (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        स्वतः pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, सत्य);
        स्वतः pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } अन्यथा {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        स्वतः pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    भिन्न freq_base;
    भिन्न freq_scale;
    भिन्न ext_factor;
    भिन्न attn_factor;
    भिन्न beta_fast;
    भिन्न beta_slow;
    अचर पूर्णाङ्क n_dims = rope_params[1];
    अचर माप_प्रकार n_dims_half = n_dims >> 1;
    अचर पूर्णाङ्क n_ctx_orig = rope_params[4];
    स_नकल(&freq_base, rope_params + 5, परिमाण(भिन्न));
    स_नकल(&freq_scale, rope_params + 6, परिमाण(भिन्न));
    स_नकल(&ext_factor, rope_params + 7, परिमाण(भिन्न));
    स_नकल(&attn_factor, rope_params + 8, परिमाण(भिन्न));
    स_नकल(&beta_fast, rope_params + 9, परिमाण(भिन्न));
    स_नकल(&beta_slow, rope_params + 10, परिमाण(भिन्न));

    अचर भिन्न theta_scale = powf(freq_base, -2.0f / n_dims);

    Output<Node> theta;
    भिन्न mscale = attn_factor;
    यदि (is_multi) {
        अचर माप_प्रकार अङ्क_pairs = is_vision ? (n_dims > 0 ? (माप_प्रकार) n_dims : (माप_प्रकार) (head_dim / 2)) : n_dims_half;
        पूर्णाङ्क sections[4];
        स_नकल(sections, rope_params + 11, परिमाण(पूर्णाङ्क) * 4);
        पूर्णाङ्क sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        यदि (sect_dims <= 0) {
            sect_dims = अङ्क_pairs;
        }
        पूर्णाङ्क sec_w = sections[0] + sections[1];
        पूर्णाङ्क sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(अङ्क_pairs);
        std::vector<भिन्न> factor(अङ्क_pairs);
        लागि (माप_प्रकार j = 0; j < अङ्क_pairs; j++) {
            पूर्णाङ्क sector = j % sect_dims;
            पूर्णाङ्क plane = 0;
            यदि (is_imrope) {
                यदि (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } अन्यथा यदि (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } अन्यथा यदि (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } अन्यथा {
                    plane = 3;
                }
                factor[j] = std::घात(theta_scale, j);
            } अन्यथा यदि (is_vision) {
                पूर्णाङ्क p_idx = 0;
                यदि (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } अन्यथा यदि (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } अन्यथा यदि (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } अन्यथा {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                factor[j] = std::घात(theta_scale, p_idx);
            } अन्यथा {
                यदि (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } अन्यथा यदि (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } अन्यथा यदि (sector >= sec_e) {
                    plane = 3;
                } अन्यथा {
                    plane = 0;
                }
                factor[j] = std::घात(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        स्वतः gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{अङ्क_pairs}, gather_indices);
        स्वतः gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> factor_गाँठो =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{अङ्क_pairs}, factor);
        यदि (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) अङ्क_pairs}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) अङ्क_pairs}),
                असत्य);
            factor_गाँठो = std::make_shared<ov::op::v1::Divide>(factor_गाँठो, rope_factors);
        }
        स्वतः theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, factor_गाँठो);
        स्वतः theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        यदि (ext_factor == 0.0f) {
            theta = theta_interp;
        } अन्यथा {
            भिन्न corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
            स्वतः ramp_mix = rope_yarn_ramp_mix(अङ्क_pairs, corr_dims, ext_factor);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            स्वतः one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::Add>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
        }
    } अन्यथा {
        std::vector<भिन्न> factor(n_dims_half);
        Output<Node> freq_factors;
        भिन्न corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        factor[0] = 1.0f;
        लागि (माप_प्रकार i = 1; i < factor.size(); i++) {
            factor[i] = theta_scale * factor[i - 1];
        }
        यदि (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, factor.size()}, factor);
        } अन्यथा {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, factor.size()}, factor);
        }
        यदि (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            यदि (stateful) {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {3}, {(int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    असत्य);
            } अन्यथा {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    असत्य);
            }
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_factors);
        }

        स्वतः theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        स्वतः theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        यदि (ext_factor == 0.0f) {
            theta = theta_interp;
        } अन्यथा {
            स्वतः ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_factor);
            Output<Node> one;
            यदि (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } अन्यथा {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            }
            स्वतः one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);

            theta =
                std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                                                  std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
        }
    }

    Output<Node> cos_theta = std::make_shared<ov::op::v0::Cos>(theta);
    Output<Node> sin_theta = std::make_shared<ov::op::v0::Sin>(theta);

    यदि (mscale != 1.0f) {
        स्वतः mscale_गाँठो = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_गाँठो);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_गाँठो);
    }

    फर्काऊ std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_आगत(अचर गाँठो_Context & context, पूर्णाङ्क आगत_index, पूर्णाङ्क slice_len, पूर्णाङ्क axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    स्वतः आगत = context.get_आगत(आगत_index);
    स्वतः * op_params = (माप_प्रकार *) context.get_आगत_op_params(आगत_index);
    स्वतः src_stride = context.get_आगत_stride(आगत_index);

    int64_t slice_start = op_params[0] / src_stride[3];
    यदि (slice_len == 0) {
        slice_len = context.get_आगत_shape(आगत_index)[3].get_length();
    }
    int64_t slice_end = slice_start + slice_len;

    स्वतः begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_start});
    स्वतः end = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_end});
    स्वतः stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    यदि (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } अन्यथा {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    स्वतः sliced = std::make_shared<ov::op::v8::Slice>(आगत, begin, end, stride, axes);
    फर्काऊ sliced;
}

ov::Output<ov::Node> process_view_आगत_new(अचर गाँठो_Context & context, पूर्णाङ्क आगत_index) {
    स्वतः आगत = context.get_आगत(आगत_index);

    // Check if this input has view inputs
    माप_प्रकार view_आगत_size = context.get_view_आगत_size(आगत_index);
    यदि (view_आगत_size == 0) {
        // No view inputs, return the input as is
        फर्काऊ आगत;
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
    स्वतः expected_ov_shape = context.get_view_आगत_ov_shape(आगत_index, 0);
    स्वतः actual_shape = आगत.get_partial_shape();
    यदि (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_length() >= actual_shape.rank().get_length()) {
        अचर int64_t n_actual = actual_shape.rank().get_length();
        अचर int64_t shift = expected_ov_shape.rank().get_length() - n_actual;
        तार्किक shapes_match = सत्य;
        लागि (int64_t i = 0; i < shift; ++i) {
            यदि (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_length() != 1) {
                shapes_match = असत्य;
                रोक;
            }
        }
        लागि (int64_t i = 0; i < n_actual && shapes_match; ++i) {
            अचर स्वतः & exp = expected_ov_shape[i + shift];
            अचर तार्किक both_dynamic = exp.is_dynamic() && actual_shape[i].is_dynamic();
            अचर तार्किक both_static_equal =
                exp.is_static() && actual_shape[i].is_static() && exp == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            अचर तार्किक expected_dyn_actual_static = exp.is_dynamic() && actual_shape[i].is_static();
            यदि (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_match = असत्य;
                रोक;
            }
        }
        यदि (shapes_match) {
            फर्काऊ आगत;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    यदि (context.is_static() && view_आगत_size == 1) {
        स्वतः view_stride_v = context.get_view_आगत_stride(आगत_index, 0);
        स्वतः view_src_stride_v = context.get_view_आगत_src_stride(आगत_index, 0);
        स्वतः view_ggml_shape = context.get_view_आगत_ggml_shape(आगत_index, 0);
        स्वतः view_src_ggml_shape = context.get_view_आगत_src_ggml_shape(आगत_index, 0);
        स्वतः view_offset = context.get_view_आगत_offset(आगत_index, 0);
        स्वतः view_src_offset = context.get_view_आगत_src_offset(आगत_index, 0);

        माप_प्रकार ndims = view_ggml_shape.size();
        std::vector<पूर्णाङ्क> diff_dims;
        यदि (view_src_ggml_shape.size() == ndims) {
            लागि (माप_प्रकार i = 0; i < ndims; ++i) {
                यदि (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<पूर्णाङ्क>(i));
                }
            }
        }

        यदि (diff_dims.size() == 1) {
            पूर्णाङ्क split_dim = diff_dims[0];
            int64_t अङ्क_splits = static_cast<int64_t>(view_src_ggml_shape[split_dim]);
            int64_t chunk_size = static_cast<int64_t>(view_ggml_shape[split_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            यदि (chunk_size == 1 && अङ्क_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                तार्किक suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                यदि (suffix_ok) {
                    लागि (माप_प्रकार i = static_cast<माप_प्रकार>(split_dim) + 1; i < ndims; ++i) {
                        यदि (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = असत्य;
                            रोक;
                        }
                    }
                }

                यदि (suffix_ok && view_src_stride_v[split_dim] > 0) {
                    माप_प्रकार relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
                    int64_t split_index = static_cast<int64_t>(relative_offset / view_src_stride_v[split_dim]);

                    यदि (split_index >= 0 && split_index < अङ्क_splits) {
                        स्वतः src_गाँठो = आगत.get_गाँठो_shared_ptr();
                        std::string rt_साँचो = "split_dim_" + std::to_string(split_dim);
                        स्वतः & rt_info = src_गाँठो->get_rt_info();

                        यदि (rt_info.find(rt_साँचो) == rt_info.end()) {
                            स्वतः axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(split_dim)});
                            स्वतः split_गाँठो =
                                std::make_shared<ov::op::v1::Split>(आगत, axis_const, static_cast<माप_प्रकार>(अङ्क_splits));
                            split_गाँठो->set_friendly_name(src_गाँठो->get_friendly_name() + "_split");
                            rt_info[rt_साँचो] = split_गाँठो;
                        }

                        स्वतः split_गाँठो = rt_info[rt_साँचो].as<std::shared_ptr<ov::op::v1::Split>>();
                        फर्काऊ split_गाँठो->output(static_cast<माप_प्रकार>(split_index));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    स्वतः process_single_view =
        [](ov::Output<ov::Node> current, माप_प्रकार view_offset, अचर std::vector<माप_प्रकार> & view_stride,
           अचर ov::Shape & view_ggml_shape, अचर ov::PartialShape & view_ov_shape, अचर std::string & view_name,
           माप_प्रकार view_src_offset, अचर std::vector<माप_प्रकार> & view_src_stride, अचर ov::Shape & view_src_ggml_shape,
           अचर ov::PartialShape & view_src_ov_shape, अचर std::string & view_src_name) -> ov::Output<ov::Node> {
        // Stateful execution drops a leading size-1 axis, so `current` can be one rank lower
        // than the ggml shape metadata (view_stride/view_ggml_shape, always GGML_MAX_DIMS
        // entries) assumes. Shift any axis index built from that metadata down by the
        // difference before using it as an OV Slice axis.
        अचर स्वतः current_rank = current.get_partial_shape().rank();
        अचर पूर्णाङ्क axis_shift = current_rank.is_static() ?
                                    static_cast<पूर्णाङ्क>(view_stride.size()) - static_cast<पूर्णाङ्क>(current_rank.get_length()) :
                                    0;

        स्वतः build_reshape_pattern = [](अचर ov::PartialShape & target_ov_shape,
                                        अचर ov::Shape & target_ggml_shape) -> std::vector<int64_t> {
            अचर माप_प्रकार ndims = target_ggml_shape.size();
            std::vector<int64_t> reshape_pattern(ndims);
            माप_प्रकार dynamic_dims = 0;

            यदि (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(ndims)) {
                लागि (माप_प्रकार i = 0; i < ndims; ++i) {
                    यदि (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } अन्यथा {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } अन्यथा {
                dynamic_dims = 2;
            }

            यदि (dynamic_dims > 1) {
                लागि (माप_प्रकार i = 0; i < ndims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            फर्काऊ reshape_pattern;
        };

        स्वतः build_prefix_tail_reshape_pattern = [](अचर ov::PartialShape & target_ov_shape,
                                                    अचर ov::Shape & target_ggml_shape, माप_प्रकार prefix_dims,
                                                    int64_t tail_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_pattern(prefix_dims + 1);
            माप_प्रकार dynamic_dims = 0;

            यदि (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(target_ggml_shape.size())) {
                लागि (माप_प्रकार i = 0; i < prefix_dims; ++i) {
                    यदि (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } अन्यथा {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } अन्यथा {
                dynamic_dims = 2;
            }

            यदि (dynamic_dims > 1) {
                लागि (माप_प्रकार i = 0; i < prefix_dims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            reshape_pattern[prefix_dims] = tail_dim;
            फर्काऊ reshape_pattern;
        };

        तार्किक same_stride = view_stride.size() == view_src_stride.size();
        यदि (same_stride) {
            लागि (माप_प्रकार i = 0; i < view_stride.size(); ++i) {
                यदि (view_stride[i] != view_src_stride[i]) {
                    same_stride = असत्य;
                    रोक;
                }
            }
        }

        तार्किक same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        यदि (same_ggml_shape) {
            लागि (माप_प्रकार i = 0; i < view_ggml_shape.size(); ++i) {
                यदि (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = असत्य;
                    रोक;
                }
            }
        }

        यदि (same_stride && same_ggml_shape) {
            फर्काऊ current;
        }

        यदि (same_stride) {
            अचर माप_प्रकार relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
            अचर माप_प्रकार ndims = view_stride.size();

            std::vector<पूर्णाङ्क> diff_dims;
            यदि (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                लागि (माप_प्रकार i = 0; i < ndims; ++i) {
                    यदि (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<पूर्णाङ्क>(i));
                    }
                }
            }

            यदि (diff_dims.size() == 1) {
                अचर पूर्णाङ्क slice_dim = diff_dims[0];
                अचर int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                यदि (view_stride[slice_dim] > 0 && relative_offset % view_stride[slice_dim] == 0) {
                    अचर int64_t begin_val = static_cast<int64_t>((relative_offset / view_stride[slice_dim]) %
                                                                   static_cast<माप_प्रकार>(dim_size));
                    अचर int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);

                    यदि (begin_val >= 0 && end_val <= dim_size) {
                        स्वतः sliced = std::make_shared<ov::op::v8::Slice>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim - axis_shift}));

                        यदि (view_ov_shape.is_static()) {
                            स्वतः reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                                असत्य);
                            reshaped->set_friendly_name(view_name);
                            फर्काऊ reshaped;
                        }

                        sliced->set_friendly_name(view_name);
                        फर्काऊ sliced;
                    }
                }

                int64_t tail_src_elems = 1;
                int64_t tail_dst_elems = 1;
                लागि (माप_प्रकार i = slice_dim; i < ndims; ++i) {
                    tail_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    tail_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                अचर माप_प्रकार elem_stride = view_stride[ndims - 1];
                int64_t tail_begin = 0;
                यदि (elem_stride > 0) {
                    tail_begin =
                        static_cast<int64_t>((relative_offset / elem_stride) % static_cast<माप_प्रकार>(tail_src_elems));
                }
                अचर int64_t tail_end = tail_begin + tail_dst_elems;

                यदि (tail_begin >= 0 && tail_end <= tail_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(slice_dim);
                    लागि (पूर्णाङ्क i = 0; i < slice_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(tail_src_elems);
                    अचर माप_प्रकार flat_ndims = flat_shape.size();

                    स्वतः flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), असत्य);

                    स्वतः sliced = std::make_shared<ov::op::v8::Slice>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_end}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

                    यदि (view_ov_shape.is_static()) {
                        स्वतः reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                            असत्य);
                        reshaped->set_friendly_name(view_name);
                        फर्काऊ reshaped;
                    }

                    sliced->set_friendly_name(view_name);
                    फर्काऊ sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> step(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            माप_प्रकार remaining_offset = relative_offset;
            लागि (माप_प्रकार i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                यदि (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(remaining_offset / view_stride[i]);
                    remaining_offset %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            तार्किक in_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            यदि (in_bounds) {
                लागि (माप_प्रकार i = 0; i < ndims; ++i) {
                    यदि (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        in_bounds = असत्य;
                        रोक;
                    }
                }
            }

            यदि (in_bounds && remaining_offset == 0) {
                स्वतः sliced = std::make_shared<ov::op::v8::Slice>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->set_friendly_name(view_name);
                फर्काऊ sliced;
            }
        } अन्यथा {
            तार्किक same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            अचर माप_प्रकार relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;

            यदि (same_rank) {
                अचर माप_प्रकार ndims = view_ggml_shape.size();
                std::vector<पूर्णाङ्क> diff_dims;
                लागि (माप_प्रकार i = 0; i < ndims; ++i) {
                    यदि (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<पूर्णाङ्क>(i));
                    }
                }

                यदि (diff_dims.size() == 1) {
                    अचर माप_प्रकार slice_dim = static_cast<माप_प्रकार>(diff_dims[0]);
                    तार्किक suffix_stride_match = सत्य;
                    लागि (माप_प्रकार i = slice_dim + 1; i < ndims; ++i) {
                        यदि (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_match = असत्य;
                            रोक;
                        }
                    }

                    यदि (suffix_stride_match && view_src_stride[slice_dim] > 0 &&
                        relative_offset % view_src_stride[slice_dim] == 0) {
                        अचर int64_t begin_val = static_cast<int64_t>(relative_offset / view_src_stride[slice_dim]);
                        अचर int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);
                        अचर int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                        यदि (begin_val >= 0 && end_val <= dim_size) {
                            स्वतः sliced = std::make_shared<ov::op::v8::Slice>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(slice_dim)}));
                            sliced->set_friendly_name(view_name);
                            फर्काऊ sliced;
                        }
                    }
                }
            }

            माप_प्रकार view_elems = 1;
            माप_प्रकार src_elems = 1;
            यदि (same_rank) {
                लागि (माप_प्रकार i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            तार्किक same_अङ्क_elements = same_rank && view_elems == src_elems;

            यदि (same_rank && relative_offset == 0 && same_अङ्क_elements) {
                स्वतः reshape_pattern = build_reshape_pattern(view_ov_shape, view_ggml_shape);

                स्वतः reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_pattern.size()}, reshape_pattern),
                    असत्य);
                reshaped->set_friendly_name(view_name);
                फर्काऊ reshaped;
            }

            यदि (same_rank) {
                अचर माप_प्रकार ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                तार्किक is_regular_slice = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> step(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                माप_प्रकार remaining_offset = relative_offset;

                यदि (is_regular_slice) {
                    लागि (माप_प्रकार i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        यदि (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_slice = असत्य;
                            रोक;
                        }

                        step[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        यदि (step[i] <= 0) {
                            is_regular_slice = असत्य;
                            रोक;
                        }

                        begin[i] = static_cast<int64_t>(remaining_offset / view_src_stride[i]);
                        remaining_offset %= view_src_stride[i];

                        यदि (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            जारी;
                        }

                        end[i] = begin[i] + step[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        यदि (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_slice = असत्य;
                            रोक;
                        }
                    }
                }

                यदि (is_regular_slice && remaining_offset == 0) {
                    स्वतः sliced = std::make_shared<ov::op::v8::Slice>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->set_friendly_name(view_name);
                    फर्काऊ sliced;
                }

                अचर माप_प्रकार elem_stride = view_src_stride.back();
                अचर तार्किक aligned_offset = elem_stride > 0 && relative_offset % elem_stride == 0;

                यदि (aligned_offset) {
                    माप_प्रकार suffix_start = 0;
                    माप_प्रकार expected_stride = elem_stride;
                    लागि (पूर्णाङ्क i = static_cast<पूर्णाङ्क>(ndims) - 1; i >= 0; --i) {
                        यदि (view_stride[i] != expected_stride) {
                            suffix_start = static_cast<माप_प्रकार>(i + 1);
                            रोक;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    माप_प्रकार prefix_elems = 1;
                    माप_प्रकार suffix_elems = 1;
                    लागि (माप_प्रकार i = 0; i < suffix_start; ++i) {
                        prefix_elems *= view_ggml_shape[i];
                    }
                    लागि (माप_प्रकार i = suffix_start; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    यदि (prefix_elems > 0 && src_elems % prefix_elems == 0) {
                        अचर माप_प्रकार src_tail_elems = src_elems / prefix_elems;
                        अचर int64_t tail_begin = static_cast<int64_t>(relative_offset / elem_stride);
                        अचर int64_t tail_end = tail_begin + static_cast<int64_t>(suffix_elems);

                        यदि (tail_begin >= 0 && tail_end <= static_cast<int64_t>(src_tail_elems)) {
                            स्वतः prefix_tail_pattern = build_prefix_tail_reshape_pattern(
                                view_ov_shape, view_ggml_shape, suffix_start, static_cast<int64_t>(src_tail_elems));

                            स्वतः prefix_tail = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {prefix_tail_pattern.size()},
                                                             prefix_tail_pattern),
                                असत्य);

                            ov::Output<ov::Node> selected = prefix_tail;
                            यदि (tail_begin != 0 || tail_end != static_cast<int64_t>(src_tail_elems)) {
                                selected = std::make_shared<ov::op::v8::Slice>(
                                    prefix_tail, ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_begin}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_end}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {static_cast<int64_t>(suffix_start)}));
                            }

                            स्वतः reshape_pattern = build_reshape_pattern(view_ov_shape, view_ggml_shape);
                            स्वतः reshaped = std::make_shared<ov::op::v1::Reshape>(
                                selected,
                                ov::op::v0::Constant::create(ov::element::i64, {reshape_pattern.size()},
                                                             reshape_pattern),
                                असत्य);
                            reshaped->set_friendly_name(view_name);
                            फर्काऊ reshaped;
                        }
                    }
                }
            }

            फर्काऊ current;
        }

        (शून्य) view_name;
        (शून्य) view_src_ov_shape;
        (शून्य) view_src_name;

        फर्काऊ current;
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
        स्वतः innermost_offset = context.get_view_आगत_offset(आगत_index, 0);
        स्वतः innermost_src_offset = context.get_view_आगत_src_offset(आगत_index, 0);
        स्वतः innermost_shape = context.get_view_आगत_ggml_shape(आगत_index, 0);
        स्वतः innermost_src_shape = context.get_view_आगत_src_ggml_shape(आगत_index, 0);
        यदि (innermost_offset != innermost_src_offset && innermost_shape == innermost_src_shape) {
            माप_प्रकार जरा_view_idx = view_आगत_size - 1;
            स्वतः जरा_ggml_shape = context.get_view_आगत_src_ggml_shape(आगत_index, जरा_view_idx);
            स्वतः जरा_stride = context.get_view_आगत_src_stride(आगत_index, जरा_view_idx);
            स्वतः जरा_offset = context.get_view_आगत_src_offset(आगत_index, जरा_view_idx);
            स्वतः जरा_ov_shape = context.get_view_आगत_src_ov_shape(आगत_index, जरा_view_idx);
            स्वतः जरा_name = context.get_view_आगत_src_name(आगत_index, जरा_view_idx);
            स्वतः innermost_stride = context.get_view_आगत_stride(आगत_index, 0);
            स्वतः innermost_ov_shape = context.get_view_आगत_ov_shape(आगत_index, 0);
            स्वतः innermost_name = context.get_view_आगत_name(आगत_index, 0);

            फर्काऊ process_single_view(आगत, innermost_offset, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, जरा_offset, जरा_stride, जरा_ggml_shape, जरा_ov_shape,
                                       जरा_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = आगत;

    // Process each view in reverse order (from base to current)
    लागि (पूर्णाङ्क view_idx = view_आगत_size - 1; view_idx >= 0; view_idx--) {
        स्वतः view_offset = context.get_view_आगत_offset(आगत_index, view_idx);
        स्वतः view_stride = context.get_view_आगत_stride(आगत_index, view_idx);
        स्वतः view_ggml_shape = context.get_view_आगत_ggml_shape(आगत_index, view_idx);
        स्वतः view_ov_shape = context.get_view_आगत_ov_shape(आगत_index, view_idx);
        स्वतः view_name = context.get_view_आगत_name(आगत_index, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        स्वतः view_src_offset = context.get_view_आगत_src_offset(आगत_index, view_idx);
        स्वतः view_src_stride = context.get_view_आगत_src_stride(आगत_index, view_idx);
        स्वतः view_src_ggml_shape = context.get_view_आगत_src_ggml_shape(आगत_index, view_idx);
        स्वतः view_src_ov_shape = context.get_view_आगत_src_ov_shape(आगत_index, view_idx);
        स्वतः view_src_name = context.get_view_आगत_src_name(आगत_index, view_idx);
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

    फर्काऊ current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
