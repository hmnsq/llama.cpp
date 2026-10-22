#அந்தர்பாவ "utils.h"

#அந்தர்பாவ <cmath>
#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <ctime>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/op/add.hpp>
#அந்தர்பாவ <openvino/op/clamp.hpp>
#அந்தர்பாவ <openvino/op/convert.hpp>
#அந்தர்பாவ <openvino/op/cos.hpp>
#அந்தர்பாவ <openvino/op/divide.hpp>
#அந்தர்பாவ <openvino/op/gather.hpp>
#அந்தர்பாவ <openvino/op/maximum.hpp>
#அந்தர்பாவ <openvino/op/multiply.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/shape_of.hpp>
#அந்தர்பாவ <openvino/op/sin.hpp>
#அந்தர்பாவ <openvino/op/slice.hpp>
#அந்தர்பாவ <openvino/op/split.hpp>
#அந்தர்பாவ <openvino/op/squeeze.hpp>
#அந்தர்பாவ <openvino/op/subtract.hpp>
#அந்தர்பாவ <openvino/op/transpose.hpp>
#அந்தர்பாவ <string>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {

வெறுமை இலக்கம்_inputs_check(மாறிலி முனை_Context & context, size_t min_inputs, size_t max_inputs) {
    auto உள்ளீடு_size = context.get_உள்ளீடு_size();
    FRONT_END_OP_CONVERSION_CHECK(உள்ளீடு_size >= min_inputs, "Got less inputs than expected");
    FRONT_END_OP_CONVERSION_CHECK(உள்ளீடு_size <= max_inputs, "Got more inputs than expected");
}

முழு non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    முழு dim = nb.size() - 1;
    size_t bytes = nb[dim];
    க்கு (முழு i = dim; i > 0; i--) {
        bytes *= ne[i];
        எனில் (bytes != nb[i - 1]) {
            திரும்பு i;
        }
    }
    திரும்பு 0;
}

std::shared_ptr<ov::Node> get_dimensions(மாறிலி std::shared_ptr<ov::op::v3::ShapeOf> & shape,
                                         மாறிலி std::vector<முழு> & dims) {
    using நாமாகாஶ ov::op;
    மாறிலி auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    மாறிலி auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    திரும்பு std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(மாறிலி std::shared_ptr<ov::Node> & முனை, மாறிலி std::vector<முழு> & dims) {
    திரும்பு get_dimensions(std::make_shared<ov::op::v3::ShapeOf>(முனை), dims);
}

OutputVector rename_outputs_with_suffix(மாறிலி OutputVector & outputs, மாறிலி std::string & suffix) {
    க்கு (மாறிலி auto & output : outputs) {
        auto முனை = output.get_முனை_shared_ptr();
        std::string name = முனை->get_friendly_name();
        name += "_";
        name += suffix;
        முனை->set_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    திரும்பு outputs;
}

நாமாகாஶ {
ov::Output<ov::Node> rope_yarn_ramp_mix(முழு இலக்கம்_elements, மாறிலி தசம corr_dims[2], தசம ext_factor) {
    std::vector<தசம> dim_ids_vec(இலக்கம்_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    auto dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) இலக்கம்_elements}, dim_ids_vec);
    auto corr_கீழ் = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    auto corr_உயர்வு = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    auto denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::Subtract>(corr_உயர்வு, corr_கீழ்),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    auto ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Subtract>(dim_ids, corr_கீழ்), denom);
    auto ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    auto one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    auto ramp_inverted = std::make_shared<ov::op::v1::Subtract>(one, ramp_clamped);
    auto ext_factor_முனை = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_factor});
    auto ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_factor_முனை);
    திரும்பு ramp_mix;
}

தசம ggml_rope_yarn_corr_dim(முழு n_dims, முழு n_ctx_orig, தசம n_rot, தசம base) {
#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
    திரும்பு n_dims * logf(n_ctx_orig / (n_rot * 2 * (தசம) M_PI)) / (2 * logf(base));
}

வெறுமை ggml_rope_yarn_corr_dims(முழு n_dims,
                              முழு n_ctx_orig,
                              தசம freq_base,
                              தசம beta_fast,
                              தசம beta_slow,
                              தசம dims[2]) {
    தசம start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    தசம end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<தசம>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight,
                                                           முழு mode,
                                                           தருக்க stateful,
                                                           int64_t head_dim) {
    constexpr முழு TYPE_IMROPE = 2;
    constexpr முழு TYPE_VISION = 3;
    constexpr முழு TYPE_MROPE = 4;

    மாறிலி தருக்க is_imrope = (mode == TYPE_IMROPE);
    மாறிலி தருக்க is_vision = (mode == TYPE_VISION);
    மாறிலி தருக்க is_mrope = (mode == TYPE_MROPE);
    மாறிலி தருக்க is_multi = is_imrope || is_vision || is_mrope;

    எனில் (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } இல்லையெனில் எனில் (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, உண்மை);
        auto pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } இல்லையெனில் {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    தசம freq_base;
    தசம freq_scale;
    தசம ext_factor;
    தசம attn_factor;
    தசம beta_fast;
    தசம beta_slow;
    மாறிலி முழு n_dims = rope_params[1];
    மாறிலி size_t n_dims_half = n_dims >> 1;
    மாறிலி முழு n_ctx_orig = rope_params[4];
    memcpy(&freq_base, rope_params + 5, அளவு(தசம));
    memcpy(&freq_scale, rope_params + 6, அளவு(தசம));
    memcpy(&ext_factor, rope_params + 7, அளவு(தசம));
    memcpy(&attn_factor, rope_params + 8, அளவு(தசம));
    memcpy(&beta_fast, rope_params + 9, அளவு(தசம));
    memcpy(&beta_slow, rope_params + 10, அளவு(தசம));

    மாறிலி தசம theta_scale = powf(freq_base, -2.0f / n_dims);

    Output<Node> theta;
    தசம mscale = attn_factor;
    எனில் (is_multi) {
        மாறிலி size_t இலக்கம்_pairs = is_vision ? (n_dims > 0 ? (size_t) n_dims : (size_t) (head_dim / 2)) : n_dims_half;
        முழு sections[4];
        memcpy(sections, rope_params + 11, அளவு(முழு) * 4);
        முழு sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        எனில் (sect_dims <= 0) {
            sect_dims = இலக்கம்_pairs;
        }
        முழு sec_w = sections[0] + sections[1];
        முழு sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(இலக்கம்_pairs);
        std::vector<தசம> factor(இலக்கம்_pairs);
        க்கு (size_t j = 0; j < இலக்கம்_pairs; j++) {
            முழு sector = j % sect_dims;
            முழு plane = 0;
            எனில் (is_imrope) {
                எனில் (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } இல்லையெனில் எனில் (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } இல்லையெனில் எனில் (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } இல்லையெனில் {
                    plane = 3;
                }
                factor[j] = std::pow(theta_scale, j);
            } இல்லையெனில் எனில் (is_vision) {
                முழு p_idx = 0;
                எனில் (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } இல்லையெனில் எனில் (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } இல்லையெனில் எனில் (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } இல்லையெனில் {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                factor[j] = std::pow(theta_scale, p_idx);
            } இல்லையெனில் {
                எனில் (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } இல்லையெனில் எனில் (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } இல்லையெனில் எனில் (sector >= sec_e) {
                    plane = 3;
                } இல்லையெனில் {
                    plane = 0;
                }
                factor[j] = std::pow(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        auto gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{இலக்கம்_pairs}, gather_indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> factor_முனை =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{இலக்கம்_pairs}, factor);
        எனில் (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) இலக்கம்_pairs}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) இலக்கம்_pairs}),
                பொய்);
            factor_முனை = std::make_shared<ov::op::v1::Divide>(factor_முனை, rope_factors);
        }
        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, factor_முனை);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        எனில் (ext_factor == 0.0f) {
            theta = theta_interp;
        } இல்லையெனில் {
            தசம corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
            auto ramp_mix = rope_yarn_ramp_mix(இலக்கம்_pairs, corr_dims, ext_factor);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            auto one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::Add>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
        }
    } இல்லையெனில் {
        std::vector<தசம> factor(n_dims_half);
        Output<Node> freq_factors;
        தசம corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        factor[0] = 1.0f;
        க்கு (size_t i = 1; i < factor.size(); i++) {
            factor[i] = theta_scale * factor[i - 1];
        }
        எனில் (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, factor.size()}, factor);
        } இல்லையெனில் {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, factor.size()}, factor);
        }
        எனில் (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            எனில் (stateful) {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {3}, {(int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    பொய்);
            } இல்லையெனில் {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    பொய்);
            }
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_factors);
        }

        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        எனில் (ext_factor == 0.0f) {
            theta = theta_interp;
        } இல்லையெனில் {
            auto ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_factor);
            Output<Node> one;
            எனில் (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } இல்லையெனில் {
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

    எனில் (mscale != 1.0f) {
        auto mscale_முனை = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_முனை);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_முனை);
    }

    திரும்பு std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_உள்ளீடு(மாறிலி முனை_Context & context, முழு உள்ளீடு_index, முழு slice_len, முழு axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    auto உள்ளீடு = context.get_உள்ளீடு(உள்ளீடு_index);
    auto * op_params = (size_t *) context.get_உள்ளீடு_op_params(உள்ளீடு_index);
    auto src_stride = context.get_உள்ளீடு_stride(உள்ளீடு_index);

    int64_t slice_start = op_params[0] / src_stride[3];
    எனில் (slice_len == 0) {
        slice_len = context.get_உள்ளீடு_shape(உள்ளீடு_index)[3].get_length();
    }
    int64_t slice_end = slice_start + slice_len;

    auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_start});
    auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_end});
    auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    எனில் (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } இல்லையெனில் {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    auto sliced = std::make_shared<ov::op::v8::Slice>(உள்ளீடு, begin, end, stride, axes);
    திரும்பு sliced;
}

ov::Output<ov::Node> process_view_உள்ளீடு_new(மாறிலி முனை_Context & context, முழு உள்ளீடு_index) {
    auto உள்ளீடு = context.get_உள்ளீடு(உள்ளீடு_index);

    // Check if this input has view inputs
    size_t view_உள்ளீடு_size = context.get_view_உள்ளீடு_size(உள்ளீடு_index);
    எனில் (view_உள்ளீடு_size == 0) {
        // No view inputs, return the input as is
        திரும்பு உள்ளீடு;
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
    auto expected_ov_shape = context.get_view_உள்ளீடு_ov_shape(உள்ளீடு_index, 0);
    auto actual_shape = உள்ளீடு.get_partial_shape();
    எனில் (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_length() >= actual_shape.rank().get_length()) {
        மாறிலி int64_t n_actual = actual_shape.rank().get_length();
        மாறிலி int64_t shift = expected_ov_shape.rank().get_length() - n_actual;
        தருக்க shapes_match = உண்மை;
        க்கு (int64_t i = 0; i < shift; ++i) {
            எனில் (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_length() != 1) {
                shapes_match = பொய்;
                நிறுத்து;
            }
        }
        க்கு (int64_t i = 0; i < n_actual && shapes_match; ++i) {
            மாறிலி auto & exp = expected_ov_shape[i + shift];
            மாறிலி தருக்க both_dynamic = exp.is_dynamic() && actual_shape[i].is_dynamic();
            மாறிலி தருக்க both_static_equal =
                exp.is_static() && actual_shape[i].is_static() && exp == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            மாறிலி தருக்க expected_dyn_actual_static = exp.is_dynamic() && actual_shape[i].is_static();
            எனில் (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_match = பொய்;
                நிறுத்து;
            }
        }
        எனில் (shapes_match) {
            திரும்பு உள்ளீடு;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    எனில் (context.is_static() && view_உள்ளீடு_size == 1) {
        auto view_stride_v = context.get_view_உள்ளீடு_stride(உள்ளீடு_index, 0);
        auto view_src_stride_v = context.get_view_உள்ளீடு_src_stride(உள்ளீடு_index, 0);
        auto view_ggml_shape = context.get_view_உள்ளீடு_ggml_shape(உள்ளீடு_index, 0);
        auto view_src_ggml_shape = context.get_view_உள்ளீடு_src_ggml_shape(உள்ளீடு_index, 0);
        auto view_offset = context.get_view_உள்ளீடு_offset(உள்ளீடு_index, 0);
        auto view_src_offset = context.get_view_உள்ளீடு_src_offset(உள்ளீடு_index, 0);

        size_t ndims = view_ggml_shape.size();
        std::vector<முழு> diff_dims;
        எனில் (view_src_ggml_shape.size() == ndims) {
            க்கு (size_t i = 0; i < ndims; ++i) {
                எனில் (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<முழு>(i));
                }
            }
        }

        எனில் (diff_dims.size() == 1) {
            முழு split_dim = diff_dims[0];
            int64_t இலக்கம்_splits = static_cast<int64_t>(view_src_ggml_shape[split_dim]);
            int64_t chunk_size = static_cast<int64_t>(view_ggml_shape[split_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            எனில் (chunk_size == 1 && இலக்கம்_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                தருக்க suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                எனில் (suffix_ok) {
                    க்கு (size_t i = static_cast<size_t>(split_dim) + 1; i < ndims; ++i) {
                        எனில் (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = பொய்;
                            நிறுத்து;
                        }
                    }
                }

                எனில் (suffix_ok && view_src_stride_v[split_dim] > 0) {
                    size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
                    int64_t split_index = static_cast<int64_t>(relative_offset / view_src_stride_v[split_dim]);

                    எனில் (split_index >= 0 && split_index < இலக்கம்_splits) {
                        auto src_முனை = உள்ளீடு.get_முனை_shared_ptr();
                        std::string rt_சாவி = "split_dim_" + std::to_string(split_dim);
                        auto & rt_info = src_முனை->get_rt_info();

                        எனில் (rt_info.find(rt_சாவி) == rt_info.end()) {
                            auto axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(split_dim)});
                            auto split_முனை =
                                std::make_shared<ov::op::v1::Split>(உள்ளீடு, axis_const, static_cast<size_t>(இலக்கம்_splits));
                            split_முனை->set_friendly_name(src_முனை->get_friendly_name() + "_split");
                            rt_info[rt_சாவி] = split_முனை;
                        }

                        auto split_முனை = rt_info[rt_சாவி].as<std::shared_ptr<ov::op::v1::Split>>();
                        திரும்பு split_முனை->output(static_cast<size_t>(split_index));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    auto process_single_view =
        [](ov::Output<ov::Node> current, size_t view_offset, மாறிலி std::vector<size_t> & view_stride,
           மாறிலி ov::Shape & view_ggml_shape, மாறிலி ov::PartialShape & view_ov_shape, மாறிலி std::string & view_name,
           size_t view_src_offset, மாறிலி std::vector<size_t> & view_src_stride, மாறிலி ov::Shape & view_src_ggml_shape,
           மாறிலி ov::PartialShape & view_src_ov_shape, மாறிலி std::string & view_src_name) -> ov::Output<ov::Node> {
        // Stateful execution drops a leading size-1 axis, so `current` can be one rank lower
        // than the ggml shape metadata (view_stride/view_ggml_shape, always GGML_MAX_DIMS
        // entries) assumes. Shift any axis index built from that metadata down by the
        // difference before using it as an OV Slice axis.
        மாறிலி auto current_rank = current.get_partial_shape().rank();
        மாறிலி முழு axis_shift = current_rank.is_static() ?
                                    static_cast<முழு>(view_stride.size()) - static_cast<முழு>(current_rank.get_length()) :
                                    0;

        auto build_reshape_pattern = [](மாறிலி ov::PartialShape & target_ov_shape,
                                        மாறிலி ov::Shape & target_ggml_shape) -> std::vector<int64_t> {
            மாறிலி size_t ndims = target_ggml_shape.size();
            std::vector<int64_t> reshape_pattern(ndims);
            size_t dynamic_dims = 0;

            எனில் (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(ndims)) {
                க்கு (size_t i = 0; i < ndims; ++i) {
                    எனில் (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } இல்லையெனில் {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } இல்லையெனில் {
                dynamic_dims = 2;
            }

            எனில் (dynamic_dims > 1) {
                க்கு (size_t i = 0; i < ndims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            திரும்பு reshape_pattern;
        };

        auto build_prefix_tail_reshape_pattern = [](மாறிலி ov::PartialShape & target_ov_shape,
                                                    மாறிலி ov::Shape & target_ggml_shape, size_t prefix_dims,
                                                    int64_t tail_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_pattern(prefix_dims + 1);
            size_t dynamic_dims = 0;

            எனில் (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(target_ggml_shape.size())) {
                க்கு (size_t i = 0; i < prefix_dims; ++i) {
                    எனில் (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } இல்லையெனில் {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } இல்லையெனில் {
                dynamic_dims = 2;
            }

            எனில் (dynamic_dims > 1) {
                க்கு (size_t i = 0; i < prefix_dims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            reshape_pattern[prefix_dims] = tail_dim;
            திரும்பு reshape_pattern;
        };

        தருக்க same_stride = view_stride.size() == view_src_stride.size();
        எனில் (same_stride) {
            க்கு (size_t i = 0; i < view_stride.size(); ++i) {
                எனில் (view_stride[i] != view_src_stride[i]) {
                    same_stride = பொய்;
                    நிறுத்து;
                }
            }
        }

        தருக்க same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        எனில் (same_ggml_shape) {
            க்கு (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                எனில் (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = பொய்;
                    நிறுத்து;
                }
            }
        }

        எனில் (same_stride && same_ggml_shape) {
            திரும்பு current;
        }

        எனில் (same_stride) {
            மாறிலி size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
            மாறிலி size_t ndims = view_stride.size();

            std::vector<முழு> diff_dims;
            எனில் (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                க்கு (size_t i = 0; i < ndims; ++i) {
                    எனில் (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<முழு>(i));
                    }
                }
            }

            எனில் (diff_dims.size() == 1) {
                மாறிலி முழு slice_dim = diff_dims[0];
                மாறிலி int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                எனில் (view_stride[slice_dim] > 0 && relative_offset % view_stride[slice_dim] == 0) {
                    மாறிலி int64_t begin_val = static_cast<int64_t>((relative_offset / view_stride[slice_dim]) %
                                                                   static_cast<size_t>(dim_size));
                    மாறிலி int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);

                    எனில் (begin_val >= 0 && end_val <= dim_size) {
                        auto sliced = std::make_shared<ov::op::v8::Slice>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim - axis_shift}));

                        எனில் (view_ov_shape.is_static()) {
                            auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                                பொய்);
                            reshaped->set_friendly_name(view_name);
                            திரும்பு reshaped;
                        }

                        sliced->set_friendly_name(view_name);
                        திரும்பு sliced;
                    }
                }

                int64_t tail_src_elems = 1;
                int64_t tail_dst_elems = 1;
                க்கு (size_t i = slice_dim; i < ndims; ++i) {
                    tail_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    tail_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                மாறிலி size_t elem_stride = view_stride[ndims - 1];
                int64_t tail_begin = 0;
                எனில் (elem_stride > 0) {
                    tail_begin =
                        static_cast<int64_t>((relative_offset / elem_stride) % static_cast<size_t>(tail_src_elems));
                }
                மாறிலி int64_t tail_end = tail_begin + tail_dst_elems;

                எனில் (tail_begin >= 0 && tail_end <= tail_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(slice_dim);
                    க்கு (முழு i = 0; i < slice_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(tail_src_elems);
                    மாறிலி size_t flat_ndims = flat_shape.size();

                    auto flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), பொய்);

                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_end}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

                    எனில் (view_ov_shape.is_static()) {
                        auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                            பொய்);
                        reshaped->set_friendly_name(view_name);
                        திரும்பு reshaped;
                    }

                    sliced->set_friendly_name(view_name);
                    திரும்பு sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> step(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            size_t remaining_offset = relative_offset;
            க்கு (size_t i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                எனில் (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(remaining_offset / view_stride[i]);
                    remaining_offset %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            தருக்க in_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            எனில் (in_bounds) {
                க்கு (size_t i = 0; i < ndims; ++i) {
                    எனில் (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        in_bounds = பொய்;
                        நிறுத்து;
                    }
                }
            }

            எனில் (in_bounds && remaining_offset == 0) {
                auto sliced = std::make_shared<ov::op::v8::Slice>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->set_friendly_name(view_name);
                திரும்பு sliced;
            }
        } இல்லையெனில் {
            தருக்க same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            மாறிலி size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;

            எனில் (same_rank) {
                மாறிலி size_t ndims = view_ggml_shape.size();
                std::vector<முழு> diff_dims;
                க்கு (size_t i = 0; i < ndims; ++i) {
                    எனில் (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<முழு>(i));
                    }
                }

                எனில் (diff_dims.size() == 1) {
                    மாறிலி size_t slice_dim = static_cast<size_t>(diff_dims[0]);
                    தருக்க suffix_stride_match = உண்மை;
                    க்கு (size_t i = slice_dim + 1; i < ndims; ++i) {
                        எனில் (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_match = பொய்;
                            நிறுத்து;
                        }
                    }

                    எனில் (suffix_stride_match && view_src_stride[slice_dim] > 0 &&
                        relative_offset % view_src_stride[slice_dim] == 0) {
                        மாறிலி int64_t begin_val = static_cast<int64_t>(relative_offset / view_src_stride[slice_dim]);
                        மாறிலி int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);
                        மாறிலி int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                        எனில் (begin_val >= 0 && end_val <= dim_size) {
                            auto sliced = std::make_shared<ov::op::v8::Slice>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(slice_dim)}));
                            sliced->set_friendly_name(view_name);
                            திரும்பு sliced;
                        }
                    }
                }
            }

            size_t view_elems = 1;
            size_t src_elems = 1;
            எனில் (same_rank) {
                க்கு (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            தருக்க same_இலக்கம்_elements = same_rank && view_elems == src_elems;

            எனில் (same_rank && relative_offset == 0 && same_இலக்கம்_elements) {
                auto reshape_pattern = build_reshape_pattern(view_ov_shape, view_ggml_shape);

                auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_pattern.size()}, reshape_pattern),
                    பொய்);
                reshaped->set_friendly_name(view_name);
                திரும்பு reshaped;
            }

            எனில் (same_rank) {
                மாறிலி size_t ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                தருக்க is_regular_slice = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> step(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                size_t remaining_offset = relative_offset;

                எனில் (is_regular_slice) {
                    க்கு (size_t i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        எனில் (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_slice = பொய்;
                            நிறுத்து;
                        }

                        step[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        எனில் (step[i] <= 0) {
                            is_regular_slice = பொய்;
                            நிறுத்து;
                        }

                        begin[i] = static_cast<int64_t>(remaining_offset / view_src_stride[i]);
                        remaining_offset %= view_src_stride[i];

                        எனில் (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            தொடர்;
                        }

                        end[i] = begin[i] + step[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        எனில் (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_slice = பொய்;
                            நிறுத்து;
                        }
                    }
                }

                எனில் (is_regular_slice && remaining_offset == 0) {
                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->set_friendly_name(view_name);
                    திரும்பு sliced;
                }

                மாறிலி size_t elem_stride = view_src_stride.back();
                மாறிலி தருக்க aligned_offset = elem_stride > 0 && relative_offset % elem_stride == 0;

                எனில் (aligned_offset) {
                    size_t suffix_start = 0;
                    size_t expected_stride = elem_stride;
                    க்கு (முழு i = static_cast<முழு>(ndims) - 1; i >= 0; --i) {
                        எனில் (view_stride[i] != expected_stride) {
                            suffix_start = static_cast<size_t>(i + 1);
                            நிறுத்து;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    size_t prefix_elems = 1;
                    size_t suffix_elems = 1;
                    க்கு (size_t i = 0; i < suffix_start; ++i) {
                        prefix_elems *= view_ggml_shape[i];
                    }
                    க்கு (size_t i = suffix_start; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    எனில் (prefix_elems > 0 && src_elems % prefix_elems == 0) {
                        மாறிலி size_t src_tail_elems = src_elems / prefix_elems;
                        மாறிலி int64_t tail_begin = static_cast<int64_t>(relative_offset / elem_stride);
                        மாறிலி int64_t tail_end = tail_begin + static_cast<int64_t>(suffix_elems);

                        எனில் (tail_begin >= 0 && tail_end <= static_cast<int64_t>(src_tail_elems)) {
                            auto prefix_tail_pattern = build_prefix_tail_reshape_pattern(
                                view_ov_shape, view_ggml_shape, suffix_start, static_cast<int64_t>(src_tail_elems));

                            auto prefix_tail = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {prefix_tail_pattern.size()},
                                                             prefix_tail_pattern),
                                பொய்);

                            ov::Output<ov::Node> selected = prefix_tail;
                            எனில் (tail_begin != 0 || tail_end != static_cast<int64_t>(src_tail_elems)) {
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
                                பொய்);
                            reshaped->set_friendly_name(view_name);
                            திரும்பு reshaped;
                        }
                    }
                }
            }

            திரும்பு current;
        }

        (வெறுமை) view_name;
        (வெறுமை) view_src_ov_shape;
        (வெறுமை) view_src_name;

        திரும்பு current;
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
        auto innermost_offset = context.get_view_உள்ளீடு_offset(உள்ளீடு_index, 0);
        auto innermost_src_offset = context.get_view_உள்ளீடு_src_offset(உள்ளீடு_index, 0);
        auto innermost_shape = context.get_view_உள்ளீடு_ggml_shape(உள்ளீடு_index, 0);
        auto innermost_src_shape = context.get_view_உள்ளீடு_src_ggml_shape(உள்ளீடு_index, 0);
        எனில் (innermost_offset != innermost_src_offset && innermost_shape == innermost_src_shape) {
            size_t வேர்_view_idx = view_உள்ளீடு_size - 1;
            auto வேர்_ggml_shape = context.get_view_உள்ளீடு_src_ggml_shape(உள்ளீடு_index, வேர்_view_idx);
            auto வேர்_stride = context.get_view_உள்ளீடு_src_stride(உள்ளீடு_index, வேர்_view_idx);
            auto வேர்_offset = context.get_view_உள்ளீடு_src_offset(உள்ளீடு_index, வேர்_view_idx);
            auto வேர்_ov_shape = context.get_view_உள்ளீடு_src_ov_shape(உள்ளீடு_index, வேர்_view_idx);
            auto வேர்_name = context.get_view_உள்ளீடு_src_name(உள்ளீடு_index, வேர்_view_idx);
            auto innermost_stride = context.get_view_உள்ளீடு_stride(உள்ளீடு_index, 0);
            auto innermost_ov_shape = context.get_view_உள்ளீடு_ov_shape(உள்ளீடு_index, 0);
            auto innermost_name = context.get_view_உள்ளீடு_name(உள்ளீடு_index, 0);

            திரும்பு process_single_view(உள்ளீடு, innermost_offset, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, வேர்_offset, வேர்_stride, வேர்_ggml_shape, வேர்_ov_shape,
                                       வேர்_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = உள்ளீடு;

    // Process each view in reverse order (from base to current)
    க்கு (முழு view_idx = view_உள்ளீடு_size - 1; view_idx >= 0; view_idx--) {
        auto view_offset = context.get_view_உள்ளீடு_offset(உள்ளீடு_index, view_idx);
        auto view_stride = context.get_view_உள்ளீடு_stride(உள்ளீடு_index, view_idx);
        auto view_ggml_shape = context.get_view_உள்ளீடு_ggml_shape(உள்ளீடு_index, view_idx);
        auto view_ov_shape = context.get_view_உள்ளீடு_ov_shape(உள்ளீடு_index, view_idx);
        auto view_name = context.get_view_உள்ளீடு_name(உள்ளீடு_index, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        auto view_src_offset = context.get_view_உள்ளீடு_src_offset(உள்ளீடு_index, view_idx);
        auto view_src_stride = context.get_view_உள்ளீடு_src_stride(உள்ளீடு_index, view_idx);
        auto view_src_ggml_shape = context.get_view_உள்ளீடு_src_ggml_shape(உள்ளீடு_index, view_idx);
        auto view_src_ov_shape = context.get_view_உள்ளீடு_src_ov_shape(உள்ளீடு_index, view_idx);
        auto view_src_name = context.get_view_உள்ளீடு_src_name(உள்ளீடு_index, view_idx);
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

    திரும்பு current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
