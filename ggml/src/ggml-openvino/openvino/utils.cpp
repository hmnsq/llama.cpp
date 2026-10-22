#અન્તર્ભાવ "utils.h"

#અન્તર્ભાવ <cmath>
#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <ctime>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/op/add.hpp>
#અન્તર્ભાવ <openvino/op/clamp.hpp>
#અન્તર્ભાવ <openvino/op/convert.hpp>
#અન્તર્ભાવ <openvino/op/cos.hpp>
#અન્તર્ભાવ <openvino/op/divide.hpp>
#અન્તર્ભાવ <openvino/op/gather.hpp>
#અન્તર્ભાવ <openvino/op/maximum.hpp>
#અન્તર્ભાવ <openvino/op/multiply.hpp>
#અન્તર્ભાવ <openvino/op/reshape.hpp>
#અન્તર્ભાવ <openvino/op/shape_of.hpp>
#અન્તર્ભાવ <openvino/op/sin.hpp>
#અન્તર્ભાવ <openvino/op/slice.hpp>
#અન્તર્ભાવ <openvino/op/split.hpp>
#અન્તર્ભાવ <openvino/op/squeeze.hpp>
#અન્તર્ભાવ <openvino/op/subtract.hpp>
#અન્તર્ભાવ <openvino/op/transpose.hpp>
#અન્તર્ભાવ <string>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {

શૂન્ય અંક_inputs_check(અચલ ગાંઠ_Context & context, size_t min_inputs, size_t max_inputs) {
    auto આગત_size = context.get_આગત_size();
    FRONT_END_OP_CONVERSION_CHECK(આગત_size >= min_inputs, "Got less inputs than expected");
    FRONT_END_OP_CONVERSION_CHECK(આગત_size <= max_inputs, "Got more inputs than expected");
}

પૂર્ણાંક non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    પૂર્ણાંક dim = nb.size() - 1;
    size_t bytes = nb[dim];
    માટે (પૂર્ણાંક i = dim; i > 0; i--) {
        bytes *= ne[i];
        જો (bytes != nb[i - 1]) {
            પાછા i;
        }
    }
    પાછા 0;
}

std::shared_ptr<ov::Node> get_dimensions(અચલ std::shared_ptr<ov::op::v3::ShapeOf> & shape,
                                         અચલ std::vector<પૂર્ણાંક> & dims) {
    using નામાકાશ ov::op;
    અચલ auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    અચલ auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    પાછા std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(અચલ std::shared_ptr<ov::Node> & ગાંઠ, અચલ std::vector<પૂર્ણાંક> & dims) {
    પાછા get_dimensions(std::make_shared<ov::op::v3::ShapeOf>(ગાંઠ), dims);
}

OutputVector rename_outputs_with_suffix(અચલ OutputVector & outputs, અચલ std::string & suffix) {
    માટે (અચલ auto & output : outputs) {
        auto ગાંઠ = output.get_ગાંઠ_shared_ptr();
        std::string name = ગાંઠ->get_friendly_name();
        name += "_";
        name += suffix;
        ગાંઠ->set_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    પાછા outputs;
}

નામાકાશ {
ov::Output<ov::Node> rope_yarn_ramp_mix(પૂર્ણાંક અંક_elements, અચલ દશાંશ corr_dims[2], દશાંશ ext_factor) {
    std::vector<દશાંશ> dim_ids_vec(અંક_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    auto dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) અંક_elements}, dim_ids_vec);
    auto corr_નીચલું = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    auto corr_ઉપલું = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    auto denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::Subtract>(corr_ઉપલું, corr_નીચલું),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    auto ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Subtract>(dim_ids, corr_નીચલું), denom);
    auto ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    auto one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    auto ramp_inverted = std::make_shared<ov::op::v1::Subtract>(one, ramp_clamped);
    auto ext_factor_ગાંઠ = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_factor});
    auto ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_factor_ગાંઠ);
    પાછા ramp_mix;
}

દશાંશ ggml_rope_yarn_corr_dim(પૂર્ણાંક n_dims, પૂર્ણાંક n_ctx_orig, દશાંશ n_rot, દશાંશ base) {
#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
    પાછા n_dims * logf(n_ctx_orig / (n_rot * 2 * (દશાંશ) M_PI)) / (2 * logf(base));
}

શૂન્ય ggml_rope_yarn_corr_dims(પૂર્ણાંક n_dims,
                              પૂર્ણાંક n_ctx_orig,
                              દશાંશ freq_base,
                              દશાંશ beta_fast,
                              દશાંશ beta_slow,
                              દશાંશ dims[2]) {
    દશાંશ start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    દશાંશ end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<દશાંશ>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight,
                                                           પૂર્ણાંક mode,
                                                           તાર્કિક stateful,
                                                           int64_t head_dim) {
    constexpr પૂર્ણાંક TYPE_IMROPE = 2;
    constexpr પૂર્ણાંક TYPE_VISION = 3;
    constexpr પૂર્ણાંક TYPE_MROPE = 4;

    અચલ તાર્કિક is_imrope = (mode == TYPE_IMROPE);
    અચલ તાર્કિક is_vision = (mode == TYPE_VISION);
    અચલ તાર્કિક is_mrope = (mode == TYPE_MROPE);
    અચલ તાર્કિક is_multi = is_imrope || is_vision || is_mrope;

    જો (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } નહીંતર જો (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, સાચું);
        auto pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } નહીંતર {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    દશાંશ freq_base;
    દશાંશ freq_scale;
    દશાંશ ext_factor;
    દશાંશ attn_factor;
    દશાંશ beta_fast;
    દશાંશ beta_slow;
    અચલ પૂર્ણાંક n_dims = rope_params[1];
    અચલ size_t n_dims_half = n_dims >> 1;
    અચલ પૂર્ણાંક n_ctx_orig = rope_params[4];
    memcpy(&freq_base, rope_params + 5, કદ(દશાંશ));
    memcpy(&freq_scale, rope_params + 6, કદ(દશાંશ));
    memcpy(&ext_factor, rope_params + 7, કદ(દશાંશ));
    memcpy(&attn_factor, rope_params + 8, કદ(દશાંશ));
    memcpy(&beta_fast, rope_params + 9, કદ(દશાંશ));
    memcpy(&beta_slow, rope_params + 10, કદ(દશાંશ));

    અચલ દશાંશ theta_scale = powf(freq_base, -2.0f / n_dims);

    Output<Node> theta;
    દશાંશ mscale = attn_factor;
    જો (is_multi) {
        અચલ size_t અંક_pairs = is_vision ? (n_dims > 0 ? (size_t) n_dims : (size_t) (head_dim / 2)) : n_dims_half;
        પૂર્ણાંક sections[4];
        memcpy(sections, rope_params + 11, કદ(પૂર્ણાંક) * 4);
        પૂર્ણાંક sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        જો (sect_dims <= 0) {
            sect_dims = અંક_pairs;
        }
        પૂર્ણાંક sec_w = sections[0] + sections[1];
        પૂર્ણાંક sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(અંક_pairs);
        std::vector<દશાંશ> factor(અંક_pairs);
        માટે (size_t j = 0; j < અંક_pairs; j++) {
            પૂર્ણાંક sector = j % sect_dims;
            પૂર્ણાંક plane = 0;
            જો (is_imrope) {
                જો (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } નહીંતર જો (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } નહીંતર જો (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } નહીંતર {
                    plane = 3;
                }
                factor[j] = std::pow(theta_scale, j);
            } નહીંતર જો (is_vision) {
                પૂર્ણાંક p_idx = 0;
                જો (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } નહીંતર જો (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } નહીંતર જો (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } નહીંતર {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                factor[j] = std::pow(theta_scale, p_idx);
            } નહીંતર {
                જો (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } નહીંતર જો (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } નહીંતર જો (sector >= sec_e) {
                    plane = 3;
                } નહીંતર {
                    plane = 0;
                }
                factor[j] = std::pow(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        auto gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{અંક_pairs}, gather_indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> factor_ગાંઠ =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{અંક_pairs}, factor);
        જો (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) અંક_pairs}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) અંક_pairs}),
                ખોટું);
            factor_ગાંઠ = std::make_shared<ov::op::v1::Divide>(factor_ગાંઠ, rope_factors);
        }
        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, factor_ગાંઠ);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        જો (ext_factor == 0.0f) {
            theta = theta_interp;
        } નહીંતર {
            દશાંશ corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
            auto ramp_mix = rope_yarn_ramp_mix(અંક_pairs, corr_dims, ext_factor);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            auto one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::Add>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
        }
    } નહીંતર {
        std::vector<દશાંશ> factor(n_dims_half);
        Output<Node> freq_factors;
        દશાંશ corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        factor[0] = 1.0f;
        માટે (size_t i = 1; i < factor.size(); i++) {
            factor[i] = theta_scale * factor[i - 1];
        }
        જો (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, factor.size()}, factor);
        } નહીંતર {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, factor.size()}, factor);
        }
        જો (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            જો (stateful) {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {3}, {(int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    ખોટું);
            } નહીંતર {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    ખોટું);
            }
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_factors);
        }

        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        જો (ext_factor == 0.0f) {
            theta = theta_interp;
        } નહીંતર {
            auto ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_factor);
            Output<Node> one;
            જો (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } નહીંતર {
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

    જો (mscale != 1.0f) {
        auto mscale_ગાંઠ = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_ગાંઠ);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_ગાંઠ);
    }

    પાછા std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_આગત(અચલ ગાંઠ_Context & context, પૂર્ણાંક આગત_index, પૂર્ણાંક slice_len, પૂર્ણાંક axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    auto આગત = context.get_આગત(આગત_index);
    auto * op_params = (size_t *) context.get_આગત_op_params(આગત_index);
    auto src_stride = context.get_આગત_stride(આગત_index);

    int64_t slice_start = op_params[0] / src_stride[3];
    જો (slice_len == 0) {
        slice_len = context.get_આગત_shape(આગત_index)[3].get_length();
    }
    int64_t slice_end = slice_start + slice_len;

    auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_start});
    auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_end});
    auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    જો (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } નહીંતર {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    auto sliced = std::make_shared<ov::op::v8::Slice>(આગત, begin, end, stride, axes);
    પાછા sliced;
}

ov::Output<ov::Node> process_view_આગત_new(અચલ ગાંઠ_Context & context, પૂર્ણાંક આગત_index) {
    auto આગત = context.get_આગત(આગત_index);

    // Check if this input has view inputs
    size_t view_આગત_size = context.get_view_આગત_size(આગત_index);
    જો (view_આગત_size == 0) {
        // No view inputs, return the input as is
        પાછા આગત;
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
    auto expected_ov_shape = context.get_view_આગત_ov_shape(આગત_index, 0);
    auto actual_shape = આગત.get_partial_shape();
    જો (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_length() >= actual_shape.rank().get_length()) {
        અચલ int64_t n_actual = actual_shape.rank().get_length();
        અચલ int64_t shift = expected_ov_shape.rank().get_length() - n_actual;
        તાર્કિક shapes_match = સાચું;
        માટે (int64_t i = 0; i < shift; ++i) {
            જો (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_length() != 1) {
                shapes_match = ખોટું;
                તોડો;
            }
        }
        માટે (int64_t i = 0; i < n_actual && shapes_match; ++i) {
            અચલ auto & exp = expected_ov_shape[i + shift];
            અચલ તાર્કિક both_dynamic = exp.is_dynamic() && actual_shape[i].is_dynamic();
            અચલ તાર્કિક both_static_equal =
                exp.is_static() && actual_shape[i].is_static() && exp == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            અચલ તાર્કિક expected_dyn_actual_static = exp.is_dynamic() && actual_shape[i].is_static();
            જો (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_match = ખોટું;
                તોડો;
            }
        }
        જો (shapes_match) {
            પાછા આગત;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    જો (context.is_static() && view_આગત_size == 1) {
        auto view_stride_v = context.get_view_આગત_stride(આગત_index, 0);
        auto view_src_stride_v = context.get_view_આગત_src_stride(આગત_index, 0);
        auto view_ggml_shape = context.get_view_આગત_ggml_shape(આગત_index, 0);
        auto view_src_ggml_shape = context.get_view_આગત_src_ggml_shape(આગત_index, 0);
        auto view_offset = context.get_view_આગત_offset(આગત_index, 0);
        auto view_src_offset = context.get_view_આગત_src_offset(આગત_index, 0);

        size_t ndims = view_ggml_shape.size();
        std::vector<પૂર્ણાંક> diff_dims;
        જો (view_src_ggml_shape.size() == ndims) {
            માટે (size_t i = 0; i < ndims; ++i) {
                જો (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<પૂર્ણાંક>(i));
                }
            }
        }

        જો (diff_dims.size() == 1) {
            પૂર્ણાંક split_dim = diff_dims[0];
            int64_t અંક_splits = static_cast<int64_t>(view_src_ggml_shape[split_dim]);
            int64_t chunk_size = static_cast<int64_t>(view_ggml_shape[split_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            જો (chunk_size == 1 && અંક_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                તાર્કિક suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                જો (suffix_ok) {
                    માટે (size_t i = static_cast<size_t>(split_dim) + 1; i < ndims; ++i) {
                        જો (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = ખોટું;
                            તોડો;
                        }
                    }
                }

                જો (suffix_ok && view_src_stride_v[split_dim] > 0) {
                    size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
                    int64_t split_index = static_cast<int64_t>(relative_offset / view_src_stride_v[split_dim]);

                    જો (split_index >= 0 && split_index < અંક_splits) {
                        auto src_ગાંઠ = આગત.get_ગાંઠ_shared_ptr();
                        std::string rt_ચાવી = "split_dim_" + std::to_string(split_dim);
                        auto & rt_info = src_ગાંઠ->get_rt_info();

                        જો (rt_info.find(rt_ચાવી) == rt_info.end()) {
                            auto axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(split_dim)});
                            auto split_ગાંઠ =
                                std::make_shared<ov::op::v1::Split>(આગત, axis_const, static_cast<size_t>(અંક_splits));
                            split_ગાંઠ->set_friendly_name(src_ગાંઠ->get_friendly_name() + "_split");
                            rt_info[rt_ચાવી] = split_ગાંઠ;
                        }

                        auto split_ગાંઠ = rt_info[rt_ચાવી].as<std::shared_ptr<ov::op::v1::Split>>();
                        પાછા split_ગાંઠ->output(static_cast<size_t>(split_index));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    auto process_single_view =
        [](ov::Output<ov::Node> current, size_t view_offset, અચલ std::vector<size_t> & view_stride,
           અચલ ov::Shape & view_ggml_shape, અચલ ov::PartialShape & view_ov_shape, અચલ std::string & view_name,
           size_t view_src_offset, અચલ std::vector<size_t> & view_src_stride, અચલ ov::Shape & view_src_ggml_shape,
           અચલ ov::PartialShape & view_src_ov_shape, અચલ std::string & view_src_name) -> ov::Output<ov::Node> {
        // Stateful execution drops a leading size-1 axis, so `current` can be one rank lower
        // than the ggml shape metadata (view_stride/view_ggml_shape, always GGML_MAX_DIMS
        // entries) assumes. Shift any axis index built from that metadata down by the
        // difference before using it as an OV Slice axis.
        અચલ auto current_rank = current.get_partial_shape().rank();
        અચલ પૂર્ણાંક axis_shift = current_rank.is_static() ?
                                    static_cast<પૂર્ણાંક>(view_stride.size()) - static_cast<પૂર્ણાંક>(current_rank.get_length()) :
                                    0;

        auto build_reshape_pattern = [](અચલ ov::PartialShape & target_ov_shape,
                                        અચલ ov::Shape & target_ggml_shape) -> std::vector<int64_t> {
            અચલ size_t ndims = target_ggml_shape.size();
            std::vector<int64_t> reshape_pattern(ndims);
            size_t dynamic_dims = 0;

            જો (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(ndims)) {
                માટે (size_t i = 0; i < ndims; ++i) {
                    જો (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } નહીંતર {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } નહીંતર {
                dynamic_dims = 2;
            }

            જો (dynamic_dims > 1) {
                માટે (size_t i = 0; i < ndims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            પાછા reshape_pattern;
        };

        auto build_prefix_tail_reshape_pattern = [](અચલ ov::PartialShape & target_ov_shape,
                                                    અચલ ov::Shape & target_ggml_shape, size_t prefix_dims,
                                                    int64_t tail_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_pattern(prefix_dims + 1);
            size_t dynamic_dims = 0;

            જો (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(target_ggml_shape.size())) {
                માટે (size_t i = 0; i < prefix_dims; ++i) {
                    જો (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } નહીંતર {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } નહીંતર {
                dynamic_dims = 2;
            }

            જો (dynamic_dims > 1) {
                માટે (size_t i = 0; i < prefix_dims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            reshape_pattern[prefix_dims] = tail_dim;
            પાછા reshape_pattern;
        };

        તાર્કિક same_stride = view_stride.size() == view_src_stride.size();
        જો (same_stride) {
            માટે (size_t i = 0; i < view_stride.size(); ++i) {
                જો (view_stride[i] != view_src_stride[i]) {
                    same_stride = ખોટું;
                    તોડો;
                }
            }
        }

        તાર્કિક same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        જો (same_ggml_shape) {
            માટે (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                જો (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = ખોટું;
                    તોડો;
                }
            }
        }

        જો (same_stride && same_ggml_shape) {
            પાછા current;
        }

        જો (same_stride) {
            અચલ size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
            અચલ size_t ndims = view_stride.size();

            std::vector<પૂર્ણાંક> diff_dims;
            જો (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                માટે (size_t i = 0; i < ndims; ++i) {
                    જો (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<પૂર્ણાંક>(i));
                    }
                }
            }

            જો (diff_dims.size() == 1) {
                અચલ પૂર્ણાંક slice_dim = diff_dims[0];
                અચલ int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                જો (view_stride[slice_dim] > 0 && relative_offset % view_stride[slice_dim] == 0) {
                    અચલ int64_t begin_val = static_cast<int64_t>((relative_offset / view_stride[slice_dim]) %
                                                                   static_cast<size_t>(dim_size));
                    અચલ int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);

                    જો (begin_val >= 0 && end_val <= dim_size) {
                        auto sliced = std::make_shared<ov::op::v8::Slice>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim - axis_shift}));

                        જો (view_ov_shape.is_static()) {
                            auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                                ખોટું);
                            reshaped->set_friendly_name(view_name);
                            પાછા reshaped;
                        }

                        sliced->set_friendly_name(view_name);
                        પાછા sliced;
                    }
                }

                int64_t tail_src_elems = 1;
                int64_t tail_dst_elems = 1;
                માટે (size_t i = slice_dim; i < ndims; ++i) {
                    tail_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    tail_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                અચલ size_t elem_stride = view_stride[ndims - 1];
                int64_t tail_begin = 0;
                જો (elem_stride > 0) {
                    tail_begin =
                        static_cast<int64_t>((relative_offset / elem_stride) % static_cast<size_t>(tail_src_elems));
                }
                અચલ int64_t tail_end = tail_begin + tail_dst_elems;

                જો (tail_begin >= 0 && tail_end <= tail_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(slice_dim);
                    માટે (પૂર્ણાંક i = 0; i < slice_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(tail_src_elems);
                    અચલ size_t flat_ndims = flat_shape.size();

                    auto flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), ખોટું);

                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_end}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

                    જો (view_ov_shape.is_static()) {
                        auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                            ખોટું);
                        reshaped->set_friendly_name(view_name);
                        પાછા reshaped;
                    }

                    sliced->set_friendly_name(view_name);
                    પાછા sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> step(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            size_t remaining_offset = relative_offset;
            માટે (size_t i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                જો (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(remaining_offset / view_stride[i]);
                    remaining_offset %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            તાર્કિક in_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            જો (in_bounds) {
                માટે (size_t i = 0; i < ndims; ++i) {
                    જો (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        in_bounds = ખોટું;
                        તોડો;
                    }
                }
            }

            જો (in_bounds && remaining_offset == 0) {
                auto sliced = std::make_shared<ov::op::v8::Slice>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->set_friendly_name(view_name);
                પાછા sliced;
            }
        } નહીંતર {
            તાર્કિક same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            અચલ size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;

            જો (same_rank) {
                અચલ size_t ndims = view_ggml_shape.size();
                std::vector<પૂર્ણાંક> diff_dims;
                માટે (size_t i = 0; i < ndims; ++i) {
                    જો (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<પૂર્ણાંક>(i));
                    }
                }

                જો (diff_dims.size() == 1) {
                    અચલ size_t slice_dim = static_cast<size_t>(diff_dims[0]);
                    તાર્કિક suffix_stride_match = સાચું;
                    માટે (size_t i = slice_dim + 1; i < ndims; ++i) {
                        જો (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_match = ખોટું;
                            તોડો;
                        }
                    }

                    જો (suffix_stride_match && view_src_stride[slice_dim] > 0 &&
                        relative_offset % view_src_stride[slice_dim] == 0) {
                        અચલ int64_t begin_val = static_cast<int64_t>(relative_offset / view_src_stride[slice_dim]);
                        અચલ int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);
                        અચલ int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                        જો (begin_val >= 0 && end_val <= dim_size) {
                            auto sliced = std::make_shared<ov::op::v8::Slice>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(slice_dim)}));
                            sliced->set_friendly_name(view_name);
                            પાછા sliced;
                        }
                    }
                }
            }

            size_t view_elems = 1;
            size_t src_elems = 1;
            જો (same_rank) {
                માટે (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            તાર્કિક same_અંક_elements = same_rank && view_elems == src_elems;

            જો (same_rank && relative_offset == 0 && same_અંક_elements) {
                auto reshape_pattern = build_reshape_pattern(view_ov_shape, view_ggml_shape);

                auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_pattern.size()}, reshape_pattern),
                    ખોટું);
                reshaped->set_friendly_name(view_name);
                પાછા reshaped;
            }

            જો (same_rank) {
                અચલ size_t ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                તાર્કિક is_regular_slice = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> step(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                size_t remaining_offset = relative_offset;

                જો (is_regular_slice) {
                    માટે (size_t i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        જો (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_slice = ખોટું;
                            તોડો;
                        }

                        step[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        જો (step[i] <= 0) {
                            is_regular_slice = ખોટું;
                            તોડો;
                        }

                        begin[i] = static_cast<int64_t>(remaining_offset / view_src_stride[i]);
                        remaining_offset %= view_src_stride[i];

                        જો (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            ચાલુ;
                        }

                        end[i] = begin[i] + step[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        જો (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_slice = ખોટું;
                            તોડો;
                        }
                    }
                }

                જો (is_regular_slice && remaining_offset == 0) {
                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->set_friendly_name(view_name);
                    પાછા sliced;
                }

                અચલ size_t elem_stride = view_src_stride.back();
                અચલ તાર્કિક aligned_offset = elem_stride > 0 && relative_offset % elem_stride == 0;

                જો (aligned_offset) {
                    size_t suffix_start = 0;
                    size_t expected_stride = elem_stride;
                    માટે (પૂર્ણાંક i = static_cast<પૂર્ણાંક>(ndims) - 1; i >= 0; --i) {
                        જો (view_stride[i] != expected_stride) {
                            suffix_start = static_cast<size_t>(i + 1);
                            તોડો;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    size_t prefix_elems = 1;
                    size_t suffix_elems = 1;
                    માટે (size_t i = 0; i < suffix_start; ++i) {
                        prefix_elems *= view_ggml_shape[i];
                    }
                    માટે (size_t i = suffix_start; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    જો (prefix_elems > 0 && src_elems % prefix_elems == 0) {
                        અચલ size_t src_tail_elems = src_elems / prefix_elems;
                        અચલ int64_t tail_begin = static_cast<int64_t>(relative_offset / elem_stride);
                        અચલ int64_t tail_end = tail_begin + static_cast<int64_t>(suffix_elems);

                        જો (tail_begin >= 0 && tail_end <= static_cast<int64_t>(src_tail_elems)) {
                            auto prefix_tail_pattern = build_prefix_tail_reshape_pattern(
                                view_ov_shape, view_ggml_shape, suffix_start, static_cast<int64_t>(src_tail_elems));

                            auto prefix_tail = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {prefix_tail_pattern.size()},
                                                             prefix_tail_pattern),
                                ખોટું);

                            ov::Output<ov::Node> selected = prefix_tail;
                            જો (tail_begin != 0 || tail_end != static_cast<int64_t>(src_tail_elems)) {
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
                                ખોટું);
                            reshaped->set_friendly_name(view_name);
                            પાછા reshaped;
                        }
                    }
                }
            }

            પાછા current;
        }

        (શૂન્ય) view_name;
        (શૂન્ય) view_src_ov_shape;
        (શૂન્ય) view_src_name;

        પાછા current;
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
        auto innermost_offset = context.get_view_આગત_offset(આગત_index, 0);
        auto innermost_src_offset = context.get_view_આગત_src_offset(આગત_index, 0);
        auto innermost_shape = context.get_view_આગત_ggml_shape(આગત_index, 0);
        auto innermost_src_shape = context.get_view_આગત_src_ggml_shape(આગત_index, 0);
        જો (innermost_offset != innermost_src_offset && innermost_shape == innermost_src_shape) {
            size_t મૂળ_view_idx = view_આગત_size - 1;
            auto મૂળ_ggml_shape = context.get_view_આગત_src_ggml_shape(આગત_index, મૂળ_view_idx);
            auto મૂળ_stride = context.get_view_આગત_src_stride(આગત_index, મૂળ_view_idx);
            auto મૂળ_offset = context.get_view_આગત_src_offset(આગત_index, મૂળ_view_idx);
            auto મૂળ_ov_shape = context.get_view_આગત_src_ov_shape(આગત_index, મૂળ_view_idx);
            auto મૂળ_name = context.get_view_આગત_src_name(આગત_index, મૂળ_view_idx);
            auto innermost_stride = context.get_view_આગત_stride(આગત_index, 0);
            auto innermost_ov_shape = context.get_view_આગત_ov_shape(આગત_index, 0);
            auto innermost_name = context.get_view_આગત_name(આગત_index, 0);

            પાછા process_single_view(આગત, innermost_offset, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, મૂળ_offset, મૂળ_stride, મૂળ_ggml_shape, મૂળ_ov_shape,
                                       મૂળ_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = આગત;

    // Process each view in reverse order (from base to current)
    માટે (પૂર્ણાંક view_idx = view_આગત_size - 1; view_idx >= 0; view_idx--) {
        auto view_offset = context.get_view_આગત_offset(આગત_index, view_idx);
        auto view_stride = context.get_view_આગત_stride(આગત_index, view_idx);
        auto view_ggml_shape = context.get_view_આગત_ggml_shape(આગત_index, view_idx);
        auto view_ov_shape = context.get_view_આગત_ov_shape(આગત_index, view_idx);
        auto view_name = context.get_view_આગત_name(આગત_index, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        auto view_src_offset = context.get_view_આગત_src_offset(આગત_index, view_idx);
        auto view_src_stride = context.get_view_આગત_src_stride(આગત_index, view_idx);
        auto view_src_ggml_shape = context.get_view_આગત_src_ggml_shape(આગત_index, view_idx);
        auto view_src_ov_shape = context.get_view_આગત_src_ov_shape(આગત_index, view_idx);
        auto view_src_name = context.get_view_આગત_src_name(આગત_index, view_idx);
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

    પાછા current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
