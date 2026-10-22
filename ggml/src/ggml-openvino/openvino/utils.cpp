#অন্তর্ভাৱ "utils.h"

#অন্তর্ভাৱ <cmath>
#অন্তর্ভাৱ <cstddef>
#অন্তর্ভাৱ <ctime>
#অন্তর্ভাৱ <memory>
#অন্তর্ভাৱ <openvino/op/add.hpp>
#অন্তর্ভাৱ <openvino/op/clamp.hpp>
#অন্তর্ভাৱ <openvino/op/convert.hpp>
#অন্তর্ভাৱ <openvino/op/cos.hpp>
#অন্তর্ভাৱ <openvino/op/divide.hpp>
#অন্তর্ভাৱ <openvino/op/gather.hpp>
#অন্তর্ভাৱ <openvino/op/maximum.hpp>
#অন্তর্ভাৱ <openvino/op/multiply.hpp>
#অন্তর্ভাৱ <openvino/op/reshape.hpp>
#অন্তর্ভাৱ <openvino/op/shape_of.hpp>
#অন্তর্ভাৱ <openvino/op/sin.hpp>
#অন্তর্ভাৱ <openvino/op/slice.hpp>
#অন্তর্ভাৱ <openvino/op/split.hpp>
#অন্তর্ভাৱ <openvino/op/squeeze.hpp>
#অন্তর্ভাৱ <openvino/op/subtract.hpp>
#অন্তর্ভাৱ <openvino/op/transpose.hpp>
#অন্তর্ভাৱ <string>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {

শূন্য অঙ্ক_inputs_check(ধ্রুৱ গ্ৰন্থি_Context & context, size_t min_inputs, size_t max_inputs) {
    auto প্ৰবেশ_size = context.get_প্ৰবেশ_size();
    FRONT_END_OP_CONVERSION_CHECK(প্ৰবেশ_size >= min_inputs, "Got less inputs than expected");
    FRONT_END_OP_CONVERSION_CHECK(প্ৰবেশ_size <= max_inputs, "Got more inputs than expected");
}

পূর্ণাঙ্ক non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    পূর্ণাঙ্ক dim = nb.size() - 1;
    size_t bytes = nb[dim];
    বাবে (পূর্ণাঙ্ক i = dim; i > 0; i--) {
        bytes *= ne[i];
        যদি (bytes != nb[i - 1]) {
            ঘূৰাই i;
        }
    }
    ঘূৰাই 0;
}

std::shared_ptr<ov::Node> get_dimensions(ধ্রুৱ std::shared_ptr<ov::op::v3::ShapeOf> & shape,
                                         ধ্রুৱ std::vector<পূর্ণাঙ্ক> & dims) {
    using নামাকাশ ov::op;
    ধ্রুৱ auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ধ্রুৱ auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    ঘূৰাই std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(ধ্রুৱ std::shared_ptr<ov::Node> & গ্ৰন্থি, ধ্রুৱ std::vector<পূর্ণাঙ্ক> & dims) {
    ঘূৰাই get_dimensions(std::make_shared<ov::op::v3::ShapeOf>(গ্ৰন্থি), dims);
}

OutputVector rename_outputs_with_suffix(ধ্রুৱ OutputVector & outputs, ধ্রুৱ std::string & suffix) {
    বাবে (ধ্রুৱ auto & output : outputs) {
        auto গ্ৰন্থি = output.get_গ্ৰন্থি_shared_ptr();
        std::string name = গ্ৰন্থি->get_friendly_name();
        name += "_";
        name += suffix;
        গ্ৰন্থি->set_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    ঘূৰাই outputs;
}

নামাকাশ {
ov::Output<ov::Node> rope_yarn_ramp_mix(পূর্ণাঙ্ক অঙ্ক_elements, ধ্রুৱ ভিন্ন corr_dims[2], ভিন্ন ext_factor) {
    std::vector<ভিন্ন> dim_ids_vec(অঙ্ক_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    auto dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) অঙ্ক_elements}, dim_ids_vec);
    auto corr_নিচেৰ = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    auto corr_উপৰেৰ = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    auto denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::Subtract>(corr_উপৰেৰ, corr_নিচেৰ),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    auto ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Subtract>(dim_ids, corr_নিচেৰ), denom);
    auto ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    auto one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    auto ramp_inverted = std::make_shared<ov::op::v1::Subtract>(one, ramp_clamped);
    auto ext_factor_গ্ৰন্থি = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_factor});
    auto ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_factor_গ্ৰন্থি);
    ঘূৰাই ramp_mix;
}

ভিন্ন ggml_rope_yarn_corr_dim(পূর্ণাঙ্ক n_dims, পূর্ণাঙ্ক n_ctx_orig, ভিন্ন n_rot, ভিন্ন base) {
#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
    ঘূৰাই n_dims * logf(n_ctx_orig / (n_rot * 2 * (ভিন্ন) M_PI)) / (2 * logf(base));
}

শূন্য ggml_rope_yarn_corr_dims(পূর্ণাঙ্ক n_dims,
                              পূর্ণাঙ্ক n_ctx_orig,
                              ভিন্ন freq_base,
                              ভিন্ন beta_fast,
                              ভিন্ন beta_slow,
                              ভিন্ন dims[2]) {
    ভিন্ন start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    ভিন্ন end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<ভিন্ন>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight,
                                                           পূর্ণাঙ্ক mode,
                                                           তার্কিক stateful,
                                                           int64_t head_dim) {
    constexpr পূর্ণাঙ্ক TYPE_IMROPE = 2;
    constexpr পূর্ণাঙ্ক TYPE_VISION = 3;
    constexpr পূর্ণাঙ্ক TYPE_MROPE = 4;

    ধ্রুৱ তার্কিক is_imrope = (mode == TYPE_IMROPE);
    ধ্রুৱ তার্কিক is_vision = (mode == TYPE_VISION);
    ধ্রুৱ তার্কিক is_mrope = (mode == TYPE_MROPE);
    ধ্রুৱ তার্কিক is_multi = is_imrope || is_vision || is_mrope;

    যদি (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } অন্যথা যদি (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, সঁচা);
        auto pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } অন্যথা {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    ভিন্ন freq_base;
    ভিন্ন freq_scale;
    ভিন্ন ext_factor;
    ভিন্ন attn_factor;
    ভিন্ন beta_fast;
    ভিন্ন beta_slow;
    ধ্রুৱ পূর্ণাঙ্ক n_dims = rope_params[1];
    ধ্রুৱ size_t n_dims_half = n_dims >> 1;
    ধ্রুৱ পূর্ণাঙ্ক n_ctx_orig = rope_params[4];
    memcpy(&freq_base, rope_params + 5, পরিমাণ(ভিন্ন));
    memcpy(&freq_scale, rope_params + 6, পরিমাণ(ভিন্ন));
    memcpy(&ext_factor, rope_params + 7, পরিমাণ(ভিন্ন));
    memcpy(&attn_factor, rope_params + 8, পরিমাণ(ভিন্ন));
    memcpy(&beta_fast, rope_params + 9, পরিমাণ(ভিন্ন));
    memcpy(&beta_slow, rope_params + 10, পরিমাণ(ভিন্ন));

    ধ্রুৱ ভিন্ন theta_scale = powf(freq_base, -2.0f / n_dims);

    Output<Node> theta;
    ভিন্ন mscale = attn_factor;
    যদি (is_multi) {
        ধ্রুৱ size_t অঙ্ক_pairs = is_vision ? (n_dims > 0 ? (size_t) n_dims : (size_t) (head_dim / 2)) : n_dims_half;
        পূর্ণাঙ্ক sections[4];
        memcpy(sections, rope_params + 11, পরিমাণ(পূর্ণাঙ্ক) * 4);
        পূর্ণাঙ্ক sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        যদি (sect_dims <= 0) {
            sect_dims = অঙ্ক_pairs;
        }
        পূর্ণাঙ্ক sec_w = sections[0] + sections[1];
        পূর্ণাঙ্ক sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(অঙ্ক_pairs);
        std::vector<ভিন্ন> factor(অঙ্ক_pairs);
        বাবে (size_t j = 0; j < অঙ্ক_pairs; j++) {
            পূর্ণাঙ্ক sector = j % sect_dims;
            পূর্ণাঙ্ক plane = 0;
            যদি (is_imrope) {
                যদি (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } অন্যথা যদি (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } অন্যথা যদি (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } অন্যথা {
                    plane = 3;
                }
                factor[j] = std::pow(theta_scale, j);
            } অন্যথা যদি (is_vision) {
                পূর্ণাঙ্ক p_idx = 0;
                যদি (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } অন্যথা যদি (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } অন্যথা যদি (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } অন্যথা {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                factor[j] = std::pow(theta_scale, p_idx);
            } অন্যথা {
                যদি (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } অন্যথা যদি (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } অন্যথা যদি (sector >= sec_e) {
                    plane = 3;
                } অন্যথা {
                    plane = 0;
                }
                factor[j] = std::pow(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        auto gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{অঙ্ক_pairs}, gather_indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> factor_গ্ৰন্থি =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{অঙ্ক_pairs}, factor);
        যদি (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) অঙ্ক_pairs}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) অঙ্ক_pairs}),
                মিছা);
            factor_গ্ৰন্থি = std::make_shared<ov::op::v1::Divide>(factor_গ্ৰন্থি, rope_factors);
        }
        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, factor_গ্ৰন্থি);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        যদি (ext_factor == 0.0f) {
            theta = theta_interp;
        } অন্যথা {
            ভিন্ন corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
            auto ramp_mix = rope_yarn_ramp_mix(অঙ্ক_pairs, corr_dims, ext_factor);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            auto one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::Add>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
        }
    } অন্যথা {
        std::vector<ভিন্ন> factor(n_dims_half);
        Output<Node> freq_factors;
        ভিন্ন corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        factor[0] = 1.0f;
        বাবে (size_t i = 1; i < factor.size(); i++) {
            factor[i] = theta_scale * factor[i - 1];
        }
        যদি (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, factor.size()}, factor);
        } অন্যথা {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, factor.size()}, factor);
        }
        যদি (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            যদি (stateful) {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {3}, {(int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    মিছা);
            } অন্যথা {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    মিছা);
            }
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_factors);
        }

        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        যদি (ext_factor == 0.0f) {
            theta = theta_interp;
        } অন্যথা {
            auto ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_factor);
            Output<Node> one;
            যদি (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } অন্যথা {
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

    যদি (mscale != 1.0f) {
        auto mscale_গ্ৰন্থি = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_গ্ৰন্থি);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_গ্ৰন্থি);
    }

    ঘূৰাই std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_প্ৰবেশ(ধ্রুৱ গ্ৰন্থি_Context & context, পূর্ণাঙ্ক প্ৰবেশ_index, পূর্ণাঙ্ক slice_len, পূর্ণাঙ্ক axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    auto প্ৰবেশ = context.get_প্ৰবেশ(প্ৰবেশ_index);
    auto * op_params = (size_t *) context.get_প্ৰবেশ_op_params(প্ৰবেশ_index);
    auto src_stride = context.get_প্ৰবেশ_stride(প্ৰবেশ_index);

    int64_t slice_start = op_params[0] / src_stride[3];
    যদি (slice_len == 0) {
        slice_len = context.get_প্ৰবেশ_shape(প্ৰবেশ_index)[3].get_length();
    }
    int64_t slice_end = slice_start + slice_len;

    auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_start});
    auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_end});
    auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    যদি (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } অন্যথা {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    auto sliced = std::make_shared<ov::op::v8::Slice>(প্ৰবেশ, begin, end, stride, axes);
    ঘূৰাই sliced;
}

ov::Output<ov::Node> process_view_প্ৰবেশ_new(ধ্রুৱ গ্ৰন্থি_Context & context, পূর্ণাঙ্ক প্ৰবেশ_index) {
    auto প্ৰবেশ = context.get_প্ৰবেশ(প্ৰবেশ_index);

    // Check if this input has view inputs
    size_t view_প্ৰবেশ_size = context.get_view_প্ৰবেশ_size(প্ৰবেশ_index);
    যদি (view_প্ৰবেশ_size == 0) {
        // No view inputs, return the input as is
        ঘূৰাই প্ৰবেশ;
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
    auto expected_ov_shape = context.get_view_প্ৰবেশ_ov_shape(প্ৰবেশ_index, 0);
    auto actual_shape = প্ৰবেশ.get_partial_shape();
    যদি (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_length() >= actual_shape.rank().get_length()) {
        ধ্রুৱ int64_t n_actual = actual_shape.rank().get_length();
        ধ্রুৱ int64_t shift = expected_ov_shape.rank().get_length() - n_actual;
        তার্কিক shapes_match = সঁচা;
        বাবে (int64_t i = 0; i < shift; ++i) {
            যদি (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_length() != 1) {
                shapes_match = মিছা;
                ভঙ্গ;
            }
        }
        বাবে (int64_t i = 0; i < n_actual && shapes_match; ++i) {
            ধ্রুৱ auto & exp = expected_ov_shape[i + shift];
            ধ্রুৱ তার্কিক both_dynamic = exp.is_dynamic() && actual_shape[i].is_dynamic();
            ধ্রুৱ তার্কিক both_static_equal =
                exp.is_static() && actual_shape[i].is_static() && exp == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            ধ্রুৱ তার্কিক expected_dyn_actual_static = exp.is_dynamic() && actual_shape[i].is_static();
            যদি (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_match = মিছা;
                ভঙ্গ;
            }
        }
        যদি (shapes_match) {
            ঘূৰাই প্ৰবেশ;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    যদি (context.is_static() && view_প্ৰবেশ_size == 1) {
        auto view_stride_v = context.get_view_প্ৰবেশ_stride(প্ৰবেশ_index, 0);
        auto view_src_stride_v = context.get_view_প্ৰবেশ_src_stride(প্ৰবেশ_index, 0);
        auto view_ggml_shape = context.get_view_প্ৰবেশ_ggml_shape(প্ৰবেশ_index, 0);
        auto view_src_ggml_shape = context.get_view_প্ৰবেশ_src_ggml_shape(প্ৰবেশ_index, 0);
        auto view_offset = context.get_view_প্ৰবেশ_offset(প্ৰবেশ_index, 0);
        auto view_src_offset = context.get_view_প্ৰবেশ_src_offset(প্ৰবেশ_index, 0);

        size_t ndims = view_ggml_shape.size();
        std::vector<পূর্ণাঙ্ক> diff_dims;
        যদি (view_src_ggml_shape.size() == ndims) {
            বাবে (size_t i = 0; i < ndims; ++i) {
                যদি (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<পূর্ণাঙ্ক>(i));
                }
            }
        }

        যদি (diff_dims.size() == 1) {
            পূর্ণাঙ্ক split_dim = diff_dims[0];
            int64_t অঙ্ক_splits = static_cast<int64_t>(view_src_ggml_shape[split_dim]);
            int64_t chunk_size = static_cast<int64_t>(view_ggml_shape[split_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            যদি (chunk_size == 1 && অঙ্ক_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                তার্কিক suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                যদি (suffix_ok) {
                    বাবে (size_t i = static_cast<size_t>(split_dim) + 1; i < ndims; ++i) {
                        যদি (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = মিছা;
                            ভঙ্গ;
                        }
                    }
                }

                যদি (suffix_ok && view_src_stride_v[split_dim] > 0) {
                    size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
                    int64_t split_index = static_cast<int64_t>(relative_offset / view_src_stride_v[split_dim]);

                    যদি (split_index >= 0 && split_index < অঙ্ক_splits) {
                        auto src_গ্ৰন্থি = প্ৰবেশ.get_গ্ৰন্থি_shared_ptr();
                        std::string rt_চাবি = "split_dim_" + std::to_string(split_dim);
                        auto & rt_info = src_গ্ৰন্থি->get_rt_info();

                        যদি (rt_info.find(rt_চাবি) == rt_info.end()) {
                            auto axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(split_dim)});
                            auto split_গ্ৰন্থি =
                                std::make_shared<ov::op::v1::Split>(প্ৰবেশ, axis_const, static_cast<size_t>(অঙ্ক_splits));
                            split_গ্ৰন্থি->set_friendly_name(src_গ্ৰন্থি->get_friendly_name() + "_split");
                            rt_info[rt_চাবি] = split_গ্ৰন্থি;
                        }

                        auto split_গ্ৰন্থি = rt_info[rt_চাবি].as<std::shared_ptr<ov::op::v1::Split>>();
                        ঘূৰাই split_গ্ৰন্থি->output(static_cast<size_t>(split_index));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    auto process_single_view =
        [](ov::Output<ov::Node> current, size_t view_offset, ধ্রুৱ std::vector<size_t> & view_stride,
           ধ্রুৱ ov::Shape & view_ggml_shape, ধ্রুৱ ov::PartialShape & view_ov_shape, ধ্রুৱ std::string & view_name,
           size_t view_src_offset, ধ্রুৱ std::vector<size_t> & view_src_stride, ধ্রুৱ ov::Shape & view_src_ggml_shape,
           ধ্রুৱ ov::PartialShape & view_src_ov_shape, ধ্রুৱ std::string & view_src_name) -> ov::Output<ov::Node> {
        // Stateful execution drops a leading size-1 axis, so `current` can be one rank lower
        // than the ggml shape metadata (view_stride/view_ggml_shape, always GGML_MAX_DIMS
        // entries) assumes. Shift any axis index built from that metadata down by the
        // difference before using it as an OV Slice axis.
        ধ্রুৱ auto current_rank = current.get_partial_shape().rank();
        ধ্রুৱ পূর্ণাঙ্ক axis_shift = current_rank.is_static() ?
                                    static_cast<পূর্ণাঙ্ক>(view_stride.size()) - static_cast<পূর্ণাঙ্ক>(current_rank.get_length()) :
                                    0;

        auto build_reshape_pattern = [](ধ্রুৱ ov::PartialShape & target_ov_shape,
                                        ধ্রুৱ ov::Shape & target_ggml_shape) -> std::vector<int64_t> {
            ধ্রুৱ size_t ndims = target_ggml_shape.size();
            std::vector<int64_t> reshape_pattern(ndims);
            size_t dynamic_dims = 0;

            যদি (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(ndims)) {
                বাবে (size_t i = 0; i < ndims; ++i) {
                    যদি (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } অন্যথা {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } অন্যথা {
                dynamic_dims = 2;
            }

            যদি (dynamic_dims > 1) {
                বাবে (size_t i = 0; i < ndims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            ঘূৰাই reshape_pattern;
        };

        auto build_prefix_tail_reshape_pattern = [](ধ্রুৱ ov::PartialShape & target_ov_shape,
                                                    ধ্রুৱ ov::Shape & target_ggml_shape, size_t prefix_dims,
                                                    int64_t tail_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_pattern(prefix_dims + 1);
            size_t dynamic_dims = 0;

            যদি (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(target_ggml_shape.size())) {
                বাবে (size_t i = 0; i < prefix_dims; ++i) {
                    যদি (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } অন্যথা {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } অন্যথা {
                dynamic_dims = 2;
            }

            যদি (dynamic_dims > 1) {
                বাবে (size_t i = 0; i < prefix_dims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            reshape_pattern[prefix_dims] = tail_dim;
            ঘূৰাই reshape_pattern;
        };

        তার্কিক same_stride = view_stride.size() == view_src_stride.size();
        যদি (same_stride) {
            বাবে (size_t i = 0; i < view_stride.size(); ++i) {
                যদি (view_stride[i] != view_src_stride[i]) {
                    same_stride = মিছা;
                    ভঙ্গ;
                }
            }
        }

        তার্কিক same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        যদি (same_ggml_shape) {
            বাবে (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                যদি (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = মিছা;
                    ভঙ্গ;
                }
            }
        }

        যদি (same_stride && same_ggml_shape) {
            ঘূৰাই current;
        }

        যদি (same_stride) {
            ধ্রুৱ size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
            ধ্রুৱ size_t ndims = view_stride.size();

            std::vector<পূর্ণাঙ্ক> diff_dims;
            যদি (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                বাবে (size_t i = 0; i < ndims; ++i) {
                    যদি (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<পূর্ণাঙ্ক>(i));
                    }
                }
            }

            যদি (diff_dims.size() == 1) {
                ধ্রুৱ পূর্ণাঙ্ক slice_dim = diff_dims[0];
                ধ্রুৱ int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                যদি (view_stride[slice_dim] > 0 && relative_offset % view_stride[slice_dim] == 0) {
                    ধ্রুৱ int64_t begin_val = static_cast<int64_t>((relative_offset / view_stride[slice_dim]) %
                                                                   static_cast<size_t>(dim_size));
                    ধ্রুৱ int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);

                    যদি (begin_val >= 0 && end_val <= dim_size) {
                        auto sliced = std::make_shared<ov::op::v8::Slice>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim - axis_shift}));

                        যদি (view_ov_shape.is_static()) {
                            auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                                মিছা);
                            reshaped->set_friendly_name(view_name);
                            ঘূৰাই reshaped;
                        }

                        sliced->set_friendly_name(view_name);
                        ঘূৰাই sliced;
                    }
                }

                int64_t tail_src_elems = 1;
                int64_t tail_dst_elems = 1;
                বাবে (size_t i = slice_dim; i < ndims; ++i) {
                    tail_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    tail_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                ধ্রুৱ size_t elem_stride = view_stride[ndims - 1];
                int64_t tail_begin = 0;
                যদি (elem_stride > 0) {
                    tail_begin =
                        static_cast<int64_t>((relative_offset / elem_stride) % static_cast<size_t>(tail_src_elems));
                }
                ধ্রুৱ int64_t tail_end = tail_begin + tail_dst_elems;

                যদি (tail_begin >= 0 && tail_end <= tail_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(slice_dim);
                    বাবে (পূর্ণাঙ্ক i = 0; i < slice_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(tail_src_elems);
                    ধ্রুৱ size_t flat_ndims = flat_shape.size();

                    auto flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), মিছা);

                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_end}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

                    যদি (view_ov_shape.is_static()) {
                        auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                            মিছা);
                        reshaped->set_friendly_name(view_name);
                        ঘূৰাই reshaped;
                    }

                    sliced->set_friendly_name(view_name);
                    ঘূৰাই sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> step(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            size_t remaining_offset = relative_offset;
            বাবে (size_t i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                যদি (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(remaining_offset / view_stride[i]);
                    remaining_offset %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            তার্কিক in_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            যদি (in_bounds) {
                বাবে (size_t i = 0; i < ndims; ++i) {
                    যদি (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        in_bounds = মিছা;
                        ভঙ্গ;
                    }
                }
            }

            যদি (in_bounds && remaining_offset == 0) {
                auto sliced = std::make_shared<ov::op::v8::Slice>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->set_friendly_name(view_name);
                ঘূৰাই sliced;
            }
        } অন্যথা {
            তার্কিক same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            ধ্রুৱ size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;

            যদি (same_rank) {
                ধ্রুৱ size_t ndims = view_ggml_shape.size();
                std::vector<পূর্ণাঙ্ক> diff_dims;
                বাবে (size_t i = 0; i < ndims; ++i) {
                    যদি (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<পূর্ণাঙ্ক>(i));
                    }
                }

                যদি (diff_dims.size() == 1) {
                    ধ্রুৱ size_t slice_dim = static_cast<size_t>(diff_dims[0]);
                    তার্কিক suffix_stride_match = সঁচা;
                    বাবে (size_t i = slice_dim + 1; i < ndims; ++i) {
                        যদি (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_match = মিছা;
                            ভঙ্গ;
                        }
                    }

                    যদি (suffix_stride_match && view_src_stride[slice_dim] > 0 &&
                        relative_offset % view_src_stride[slice_dim] == 0) {
                        ধ্রুৱ int64_t begin_val = static_cast<int64_t>(relative_offset / view_src_stride[slice_dim]);
                        ধ্রুৱ int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);
                        ধ্রুৱ int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                        যদি (begin_val >= 0 && end_val <= dim_size) {
                            auto sliced = std::make_shared<ov::op::v8::Slice>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(slice_dim)}));
                            sliced->set_friendly_name(view_name);
                            ঘূৰাই sliced;
                        }
                    }
                }
            }

            size_t view_elems = 1;
            size_t src_elems = 1;
            যদি (same_rank) {
                বাবে (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            তার্কিক same_অঙ্ক_elements = same_rank && view_elems == src_elems;

            যদি (same_rank && relative_offset == 0 && same_অঙ্ক_elements) {
                auto reshape_pattern = build_reshape_pattern(view_ov_shape, view_ggml_shape);

                auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_pattern.size()}, reshape_pattern),
                    মিছা);
                reshaped->set_friendly_name(view_name);
                ঘূৰাই reshaped;
            }

            যদি (same_rank) {
                ধ্রুৱ size_t ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                তার্কিক is_regular_slice = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> step(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                size_t remaining_offset = relative_offset;

                যদি (is_regular_slice) {
                    বাবে (size_t i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        যদি (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_slice = মিছা;
                            ভঙ্গ;
                        }

                        step[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        যদি (step[i] <= 0) {
                            is_regular_slice = মিছা;
                            ভঙ্গ;
                        }

                        begin[i] = static_cast<int64_t>(remaining_offset / view_src_stride[i]);
                        remaining_offset %= view_src_stride[i];

                        যদি (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            অনুৱর্ত;
                        }

                        end[i] = begin[i] + step[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        যদি (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_slice = মিছা;
                            ভঙ্গ;
                        }
                    }
                }

                যদি (is_regular_slice && remaining_offset == 0) {
                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->set_friendly_name(view_name);
                    ঘূৰাই sliced;
                }

                ধ্রুৱ size_t elem_stride = view_src_stride.back();
                ধ্রুৱ তার্কিক aligned_offset = elem_stride > 0 && relative_offset % elem_stride == 0;

                যদি (aligned_offset) {
                    size_t suffix_start = 0;
                    size_t expected_stride = elem_stride;
                    বাবে (পূর্ণাঙ্ক i = static_cast<পূর্ণাঙ্ক>(ndims) - 1; i >= 0; --i) {
                        যদি (view_stride[i] != expected_stride) {
                            suffix_start = static_cast<size_t>(i + 1);
                            ভঙ্গ;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    size_t prefix_elems = 1;
                    size_t suffix_elems = 1;
                    বাবে (size_t i = 0; i < suffix_start; ++i) {
                        prefix_elems *= view_ggml_shape[i];
                    }
                    বাবে (size_t i = suffix_start; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    যদি (prefix_elems > 0 && src_elems % prefix_elems == 0) {
                        ধ্রুৱ size_t src_tail_elems = src_elems / prefix_elems;
                        ধ্রুৱ int64_t tail_begin = static_cast<int64_t>(relative_offset / elem_stride);
                        ধ্রুৱ int64_t tail_end = tail_begin + static_cast<int64_t>(suffix_elems);

                        যদি (tail_begin >= 0 && tail_end <= static_cast<int64_t>(src_tail_elems)) {
                            auto prefix_tail_pattern = build_prefix_tail_reshape_pattern(
                                view_ov_shape, view_ggml_shape, suffix_start, static_cast<int64_t>(src_tail_elems));

                            auto prefix_tail = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {prefix_tail_pattern.size()},
                                                             prefix_tail_pattern),
                                মিছা);

                            ov::Output<ov::Node> selected = prefix_tail;
                            যদি (tail_begin != 0 || tail_end != static_cast<int64_t>(src_tail_elems)) {
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
                                মিছা);
                            reshaped->set_friendly_name(view_name);
                            ঘূৰাই reshaped;
                        }
                    }
                }
            }

            ঘূৰাই current;
        }

        (শূন্য) view_name;
        (শূন্য) view_src_ov_shape;
        (শূন্য) view_src_name;

        ঘূৰাই current;
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
        auto innermost_offset = context.get_view_প্ৰবেশ_offset(প্ৰবেশ_index, 0);
        auto innermost_src_offset = context.get_view_প্ৰবেশ_src_offset(প্ৰবেশ_index, 0);
        auto innermost_shape = context.get_view_প্ৰবেশ_ggml_shape(প্ৰবেশ_index, 0);
        auto innermost_src_shape = context.get_view_প্ৰবেশ_src_ggml_shape(প্ৰবেশ_index, 0);
        যদি (innermost_offset != innermost_src_offset && innermost_shape == innermost_src_shape) {
            size_t মূল_view_idx = view_প্ৰবেশ_size - 1;
            auto মূল_ggml_shape = context.get_view_প্ৰবেশ_src_ggml_shape(প্ৰবেশ_index, মূল_view_idx);
            auto মূল_stride = context.get_view_প্ৰবেশ_src_stride(প্ৰবেশ_index, মূল_view_idx);
            auto মূল_offset = context.get_view_প্ৰবেশ_src_offset(প্ৰবেশ_index, মূল_view_idx);
            auto মূল_ov_shape = context.get_view_প্ৰবেশ_src_ov_shape(প্ৰবেশ_index, মূল_view_idx);
            auto মূল_name = context.get_view_প্ৰবেশ_src_name(প্ৰবেশ_index, মূল_view_idx);
            auto innermost_stride = context.get_view_প্ৰবেশ_stride(প্ৰবেশ_index, 0);
            auto innermost_ov_shape = context.get_view_প্ৰবেশ_ov_shape(প্ৰবেশ_index, 0);
            auto innermost_name = context.get_view_প্ৰবেশ_name(প্ৰবেশ_index, 0);

            ঘূৰাই process_single_view(প্ৰবেশ, innermost_offset, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, মূল_offset, মূল_stride, মূল_ggml_shape, মূল_ov_shape,
                                       মূল_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = প্ৰবেশ;

    // Process each view in reverse order (from base to current)
    বাবে (পূর্ণাঙ্ক view_idx = view_প্ৰবেশ_size - 1; view_idx >= 0; view_idx--) {
        auto view_offset = context.get_view_প্ৰবেশ_offset(প্ৰবেশ_index, view_idx);
        auto view_stride = context.get_view_প্ৰবেশ_stride(প্ৰবেশ_index, view_idx);
        auto view_ggml_shape = context.get_view_প্ৰবেশ_ggml_shape(প্ৰবেশ_index, view_idx);
        auto view_ov_shape = context.get_view_প্ৰবেশ_ov_shape(প্ৰবেশ_index, view_idx);
        auto view_name = context.get_view_প্ৰবেশ_name(প্ৰবেশ_index, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        auto view_src_offset = context.get_view_প্ৰবেশ_src_offset(প্ৰবেশ_index, view_idx);
        auto view_src_stride = context.get_view_প্ৰবেশ_src_stride(প্ৰবেশ_index, view_idx);
        auto view_src_ggml_shape = context.get_view_প্ৰবেশ_src_ggml_shape(প্ৰবেশ_index, view_idx);
        auto view_src_ov_shape = context.get_view_প্ৰবেশ_src_ov_shape(প্ৰবেশ_index, view_idx);
        auto view_src_name = context.get_view_প্ৰবেশ_src_name(প্ৰবেশ_index, view_idx);
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

    ঘূৰাই current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
