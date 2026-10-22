#ଅନ୍ତର୍ଭାଵ "utils.h"

#ଅନ୍ତର୍ଭାଵ <cmath>
#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <ctime>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/op/add.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/clamp.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/convert.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/cos.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/divide.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/gather.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/maximum.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/multiply.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/reshape.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/shape_of.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/sin.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/slice.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/split.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/squeeze.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/subtract.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/transpose.hpp>
#ଅନ୍ତର୍ଭାଵ <string>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {

ଶୂନ୍ୟ ଅଙ୍କ_inputs_check(ଅଚଳ ଗଣ୍ଠି_Context & context, size_t min_inputs, size_t max_inputs) {
    auto ନିବେଶ_size = context.get_ନିବେଶ_size();
    FRONT_END_OP_CONVERSION_CHECK(ନିବେଶ_size >= min_inputs, "Got less inputs than expected");
    FRONT_END_OP_CONVERSION_CHECK(ନିବେଶ_size <= max_inputs, "Got more inputs than expected");
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା dim = nb.size() - 1;
    size_t bytes = nb[dim];
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = dim; i > 0; i--) {
        bytes *= ne[i];
        ଯଦି (bytes != nb[i - 1]) {
            ଫେରନ୍ତୁ i;
        }
    }
    ଫେରନ୍ତୁ 0;
}

std::shared_ptr<ov::Node> get_dimensions(ଅଚଳ std::shared_ptr<ov::op::v3::ShapeOf> & shape,
                                         ଅଚଳ std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> & dims) {
    using ନାମାକାଶ ov::op;
    ଅଚଳ auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ଅଚଳ auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    ଫେରନ୍ତୁ std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(ଅଚଳ std::shared_ptr<ov::Node> & ଗଣ୍ଠି, ଅଚଳ std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> & dims) {
    ଫେରନ୍ତୁ get_dimensions(std::make_shared<ov::op::v3::ShapeOf>(ଗଣ୍ଠି), dims);
}

OutputVector rename_outputs_with_suffix(ଅଚଳ OutputVector & outputs, ଅଚଳ std::string & suffix) {
    ପାଇଁ (ଅଚଳ auto & output : outputs) {
        auto ଗଣ୍ଠି = output.get_ଗଣ୍ଠି_shared_ptr();
        std::string name = ଗଣ୍ଠି->get_friendly_name();
        name += "_";
        name += suffix;
        ଗଣ୍ଠି->set_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    ଫେରନ୍ତୁ outputs;
}

ନାମାକାଶ {
ov::Output<ov::Node> rope_yarn_ramp_mix(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_elements, ଅଚଳ ଦଶମିକ corr_dims[2], ଦଶମିକ ext_factor) {
    std::vector<ଦଶମିକ> dim_ids_vec(ଅଙ୍କ_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    auto dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) ଅଙ୍କ_elements}, dim_ids_vec);
    auto corr_ତଳ = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    auto corr_ଉପର = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    auto denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::Subtract>(corr_ଉପର, corr_ତଳ),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    auto ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Subtract>(dim_ids, corr_ତଳ), denom);
    auto ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    auto one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    auto ramp_inverted = std::make_shared<ov::op::v1::Subtract>(one, ramp_clamped);
    auto ext_factor_ଗଣ୍ଠି = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_factor});
    auto ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_factor_ଗଣ୍ଠି);
    ଫେରନ୍ତୁ ramp_mix;
}

ଦଶମିକ ggml_rope_yarn_corr_dim(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_dims, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ctx_orig, ଦଶମିକ n_rot, ଦଶମିକ base) {
#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
    ଫେରନ୍ତୁ n_dims * logf(n_ctx_orig / (n_rot * 2 * (ଦଶମିକ) M_PI)) / (2 * logf(base));
}

ଶୂନ୍ୟ ggml_rope_yarn_corr_dims(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_dims,
                              ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ctx_orig,
                              ଦଶମିକ freq_base,
                              ଦଶମିକ beta_fast,
                              ଦଶମିକ beta_slow,
                              ଦଶମିକ dims[2]) {
    ଦଶମିକ start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    ଦଶମିକ end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<ଦଶମିକ>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight,
                                                           ପୂର୍ଣ୍ଣସଂଖ୍ୟା mode,
                                                           ତାର୍କିକ stateful,
                                                           int64_t head_dim) {
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା TYPE_IMROPE = 2;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା TYPE_VISION = 3;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା TYPE_MROPE = 4;

    ଅଚଳ ତାର୍କିକ is_imrope = (mode == TYPE_IMROPE);
    ଅଚଳ ତାର୍କିକ is_vision = (mode == TYPE_VISION);
    ଅଚଳ ତାର୍କିକ is_mrope = (mode == TYPE_MROPE);
    ଅଚଳ ତାର୍କିକ is_multi = is_imrope || is_vision || is_mrope;

    ଯଦି (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } ଅନ୍ୟଥା ଯଦି (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, ସତ୍ୟ);
        auto pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } ଅନ୍ୟଥା {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    ଦଶମିକ freq_base;
    ଦଶମିକ freq_scale;
    ଦଶମିକ ext_factor;
    ଦଶମିକ attn_factor;
    ଦଶମିକ beta_fast;
    ଦଶମିକ beta_slow;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_dims = rope_params[1];
    ଅଚଳ size_t n_dims_half = n_dims >> 1;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ctx_orig = rope_params[4];
    memcpy(&freq_base, rope_params + 5, ଆକାର(ଦଶମିକ));
    memcpy(&freq_scale, rope_params + 6, ଆକାର(ଦଶମିକ));
    memcpy(&ext_factor, rope_params + 7, ଆକାର(ଦଶମିକ));
    memcpy(&attn_factor, rope_params + 8, ଆକାର(ଦଶମିକ));
    memcpy(&beta_fast, rope_params + 9, ଆକାର(ଦଶମିକ));
    memcpy(&beta_slow, rope_params + 10, ଆକାର(ଦଶମିକ));

    ଅଚଳ ଦଶମିକ theta_scale = powf(freq_base, -2.0f / n_dims);

    Output<Node> theta;
    ଦଶମିକ mscale = attn_factor;
    ଯଦି (is_multi) {
        ଅଚଳ size_t ଅଙ୍କ_pairs = is_vision ? (n_dims > 0 ? (size_t) n_dims : (size_t) (head_dim / 2)) : n_dims_half;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା sections[4];
        memcpy(sections, rope_params + 11, ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା) * 4);
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        ଯଦି (sect_dims <= 0) {
            sect_dims = ଅଙ୍କ_pairs;
        }
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା sec_w = sections[0] + sections[1];
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(ଅଙ୍କ_pairs);
        std::vector<ଦଶମିକ> factor(ଅଙ୍କ_pairs);
        ପାଇଁ (size_t j = 0; j < ଅଙ୍କ_pairs; j++) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା sector = j % sect_dims;
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା plane = 0;
            ଯଦି (is_imrope) {
                ଯଦି (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } ଅନ୍ୟଥା ଯଦି (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } ଅନ୍ୟଥା ଯଦି (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } ଅନ୍ୟଥା {
                    plane = 3;
                }
                factor[j] = std::pow(theta_scale, j);
            } ଅନ୍ୟଥା ଯଦି (is_vision) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା p_idx = 0;
                ଯଦି (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } ଅନ୍ୟଥା ଯଦି (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } ଅନ୍ୟଥା ଯଦି (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } ଅନ୍ୟଥା {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                factor[j] = std::pow(theta_scale, p_idx);
            } ଅନ୍ୟଥା {
                ଯଦି (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } ଅନ୍ୟଥା ଯଦି (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } ଅନ୍ୟଥା ଯଦି (sector >= sec_e) {
                    plane = 3;
                } ଅନ୍ୟଥା {
                    plane = 0;
                }
                factor[j] = std::pow(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        auto gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{ଅଙ୍କ_pairs}, gather_indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> factor_ଗଣ୍ଠି =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{ଅଙ୍କ_pairs}, factor);
        ଯଦି (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) ଅଙ୍କ_pairs}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) ଅଙ୍କ_pairs}),
                ମିଥ୍ୟା);
            factor_ଗଣ୍ଠି = std::make_shared<ov::op::v1::Divide>(factor_ଗଣ୍ଠି, rope_factors);
        }
        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, factor_ଗଣ୍ଠି);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        ଯଦି (ext_factor == 0.0f) {
            theta = theta_interp;
        } ଅନ୍ୟଥା {
            ଦଶମିକ corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
            auto ramp_mix = rope_yarn_ramp_mix(ଅଙ୍କ_pairs, corr_dims, ext_factor);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            auto one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::Add>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
        }
    } ଅନ୍ୟଥା {
        std::vector<ଦଶମିକ> factor(n_dims_half);
        Output<Node> freq_factors;
        ଦଶମିକ corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        factor[0] = 1.0f;
        ପାଇଁ (size_t i = 1; i < factor.size(); i++) {
            factor[i] = theta_scale * factor[i - 1];
        }
        ଯଦି (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, factor.size()}, factor);
        } ଅନ୍ୟଥା {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, factor.size()}, factor);
        }
        ଯଦି (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            ଯଦି (stateful) {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {3}, {(int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    ମିଥ୍ୟା);
            } ଅନ୍ୟଥା {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    ମିଥ୍ୟା);
            }
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_factors);
        }

        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        ଯଦି (ext_factor == 0.0f) {
            theta = theta_interp;
        } ଅନ୍ୟଥା {
            auto ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_factor);
            Output<Node> one;
            ଯଦି (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } ଅନ୍ୟଥା {
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

    ଯଦି (mscale != 1.0f) {
        auto mscale_ଗଣ୍ଠି = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_ଗଣ୍ଠି);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_ଗଣ୍ଠି);
    }

    ଫେରନ୍ତୁ std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_ନିବେଶ(ଅଚଳ ଗଣ୍ଠି_Context & context, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ନିବେଶ_index, ପୂର୍ଣ୍ଣସଂଖ୍ୟା slice_len, ପୂର୍ଣ୍ଣସଂଖ୍ୟା axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    auto ନିବେଶ = context.get_ନିବେଶ(ନିବେଶ_index);
    auto * op_params = (size_t *) context.get_ନିବେଶ_op_params(ନିବେଶ_index);
    auto src_stride = context.get_ନିବେଶ_stride(ନିବେଶ_index);

    int64_t slice_start = op_params[0] / src_stride[3];
    ଯଦି (slice_len == 0) {
        slice_len = context.get_ନିବେଶ_shape(ନିବେଶ_index)[3].get_length();
    }
    int64_t slice_end = slice_start + slice_len;

    auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_start});
    auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_end});
    auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    ଯଦି (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } ଅନ୍ୟଥା {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    auto sliced = std::make_shared<ov::op::v8::Slice>(ନିବେଶ, begin, end, stride, axes);
    ଫେରନ୍ତୁ sliced;
}

ov::Output<ov::Node> process_view_ନିବେଶ_new(ଅଚଳ ଗଣ୍ଠି_Context & context, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ନିବେଶ_index) {
    auto ନିବେଶ = context.get_ନିବେଶ(ନିବେଶ_index);

    // Check if this input has view inputs
    size_t view_ନିବେଶ_size = context.get_view_ନିବେଶ_size(ନିବେଶ_index);
    ଯଦି (view_ନିବେଶ_size == 0) {
        // No view inputs, return the input as is
        ଫେରନ୍ତୁ ନିବେଶ;
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
    auto expected_ov_shape = context.get_view_ନିବେଶ_ov_shape(ନିବେଶ_index, 0);
    auto actual_shape = ନିବେଶ.get_partial_shape();
    ଯଦି (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_length() >= actual_shape.rank().get_length()) {
        ଅଚଳ int64_t n_actual = actual_shape.rank().get_length();
        ଅଚଳ int64_t shift = expected_ov_shape.rank().get_length() - n_actual;
        ତାର୍କିକ shapes_match = ସତ୍ୟ;
        ପାଇଁ (int64_t i = 0; i < shift; ++i) {
            ଯଦି (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_length() != 1) {
                shapes_match = ମିଥ୍ୟା;
                ଭାଙ୍ଗ;
            }
        }
        ପାଇଁ (int64_t i = 0; i < n_actual && shapes_match; ++i) {
            ଅଚଳ auto & exp = expected_ov_shape[i + shift];
            ଅଚଳ ତାର୍କିକ both_dynamic = exp.is_dynamic() && actual_shape[i].is_dynamic();
            ଅଚଳ ତାର୍କିକ both_static_equal =
                exp.is_static() && actual_shape[i].is_static() && exp == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            ଅଚଳ ତାର୍କିକ expected_dyn_actual_static = exp.is_dynamic() && actual_shape[i].is_static();
            ଯଦି (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_match = ମିଥ୍ୟା;
                ଭାଙ୍ଗ;
            }
        }
        ଯଦି (shapes_match) {
            ଫେରନ୍ତୁ ନିବେଶ;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    ଯଦି (context.is_static() && view_ନିବେଶ_size == 1) {
        auto view_stride_v = context.get_view_ନିବେଶ_stride(ନିବେଶ_index, 0);
        auto view_src_stride_v = context.get_view_ନିବେଶ_src_stride(ନିବେଶ_index, 0);
        auto view_ggml_shape = context.get_view_ନିବେଶ_ggml_shape(ନିବେଶ_index, 0);
        auto view_src_ggml_shape = context.get_view_ନିବେଶ_src_ggml_shape(ନିବେଶ_index, 0);
        auto view_offset = context.get_view_ନିବେଶ_offset(ନିବେଶ_index, 0);
        auto view_src_offset = context.get_view_ନିବେଶ_src_offset(ନିବେଶ_index, 0);

        size_t ndims = view_ggml_shape.size();
        std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> diff_dims;
        ଯଦି (view_src_ggml_shape.size() == ndims) {
            ପାଇଁ (size_t i = 0; i < ndims; ++i) {
                ଯଦି (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(i));
                }
            }
        }

        ଯଦି (diff_dims.size() == 1) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା split_dim = diff_dims[0];
            int64_t ଅଙ୍କ_splits = static_cast<int64_t>(view_src_ggml_shape[split_dim]);
            int64_t chunk_size = static_cast<int64_t>(view_ggml_shape[split_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            ଯଦି (chunk_size == 1 && ଅଙ୍କ_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                ତାର୍କିକ suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                ଯଦି (suffix_ok) {
                    ପାଇଁ (size_t i = static_cast<size_t>(split_dim) + 1; i < ndims; ++i) {
                        ଯଦି (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = ମିଥ୍ୟା;
                            ଭାଙ୍ଗ;
                        }
                    }
                }

                ଯଦି (suffix_ok && view_src_stride_v[split_dim] > 0) {
                    size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
                    int64_t split_index = static_cast<int64_t>(relative_offset / view_src_stride_v[split_dim]);

                    ଯଦି (split_index >= 0 && split_index < ଅଙ୍କ_splits) {
                        auto src_ଗଣ୍ଠି = ନିବେଶ.get_ଗଣ୍ଠି_shared_ptr();
                        std::string rt_ଚାବି = "split_dim_" + std::to_string(split_dim);
                        auto & rt_info = src_ଗଣ୍ଠି->get_rt_info();

                        ଯଦି (rt_info.find(rt_ଚାବି) == rt_info.end()) {
                            auto axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(split_dim)});
                            auto split_ଗଣ୍ଠି =
                                std::make_shared<ov::op::v1::Split>(ନିବେଶ, axis_const, static_cast<size_t>(ଅଙ୍କ_splits));
                            split_ଗଣ୍ଠି->set_friendly_name(src_ଗଣ୍ଠି->get_friendly_name() + "_split");
                            rt_info[rt_ଚାବି] = split_ଗଣ୍ଠି;
                        }

                        auto split_ଗଣ୍ଠି = rt_info[rt_ଚାବି].as<std::shared_ptr<ov::op::v1::Split>>();
                        ଫେରନ୍ତୁ split_ଗଣ୍ଠି->output(static_cast<size_t>(split_index));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    auto process_single_view =
        [](ov::Output<ov::Node> current, size_t view_offset, ଅଚଳ std::vector<size_t> & view_stride,
           ଅଚଳ ov::Shape & view_ggml_shape, ଅଚଳ ov::PartialShape & view_ov_shape, ଅଚଳ std::string & view_name,
           size_t view_src_offset, ଅଚଳ std::vector<size_t> & view_src_stride, ଅଚଳ ov::Shape & view_src_ggml_shape,
           ଅଚଳ ov::PartialShape & view_src_ov_shape, ଅଚଳ std::string & view_src_name) -> ov::Output<ov::Node> {
        // Stateful execution drops a leading size-1 axis, so `current` can be one rank lower
        // than the ggml shape metadata (view_stride/view_ggml_shape, always GGML_MAX_DIMS
        // entries) assumes. Shift any axis index built from that metadata down by the
        // difference before using it as an OV Slice axis.
        ଅଚଳ auto current_rank = current.get_partial_shape().rank();
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା axis_shift = current_rank.is_static() ?
                                    static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(view_stride.size()) - static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(current_rank.get_length()) :
                                    0;

        auto build_reshape_pattern = [](ଅଚଳ ov::PartialShape & target_ov_shape,
                                        ଅଚଳ ov::Shape & target_ggml_shape) -> std::vector<int64_t> {
            ଅଚଳ size_t ndims = target_ggml_shape.size();
            std::vector<int64_t> reshape_pattern(ndims);
            size_t dynamic_dims = 0;

            ଯଦି (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(ndims)) {
                ପାଇଁ (size_t i = 0; i < ndims; ++i) {
                    ଯଦି (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } ଅନ୍ୟଥା {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } ଅନ୍ୟଥା {
                dynamic_dims = 2;
            }

            ଯଦି (dynamic_dims > 1) {
                ପାଇଁ (size_t i = 0; i < ndims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            ଫେରନ୍ତୁ reshape_pattern;
        };

        auto build_prefix_tail_reshape_pattern = [](ଅଚଳ ov::PartialShape & target_ov_shape,
                                                    ଅଚଳ ov::Shape & target_ggml_shape, size_t prefix_dims,
                                                    int64_t tail_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_pattern(prefix_dims + 1);
            size_t dynamic_dims = 0;

            ଯଦି (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(target_ggml_shape.size())) {
                ପାଇଁ (size_t i = 0; i < prefix_dims; ++i) {
                    ଯଦି (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } ଅନ୍ୟଥା {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } ଅନ୍ୟଥା {
                dynamic_dims = 2;
            }

            ଯଦି (dynamic_dims > 1) {
                ପାଇଁ (size_t i = 0; i < prefix_dims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            reshape_pattern[prefix_dims] = tail_dim;
            ଫେରନ୍ତୁ reshape_pattern;
        };

        ତାର୍କିକ same_stride = view_stride.size() == view_src_stride.size();
        ଯଦି (same_stride) {
            ପାଇଁ (size_t i = 0; i < view_stride.size(); ++i) {
                ଯଦି (view_stride[i] != view_src_stride[i]) {
                    same_stride = ମିଥ୍ୟା;
                    ଭାଙ୍ଗ;
                }
            }
        }

        ତାର୍କିକ same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        ଯଦି (same_ggml_shape) {
            ପାଇଁ (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                ଯଦି (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = ମିଥ୍ୟା;
                    ଭାଙ୍ଗ;
                }
            }
        }

        ଯଦି (same_stride && same_ggml_shape) {
            ଫେରନ୍ତୁ current;
        }

        ଯଦି (same_stride) {
            ଅଚଳ size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
            ଅଚଳ size_t ndims = view_stride.size();

            std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> diff_dims;
            ଯଦି (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                ପାଇଁ (size_t i = 0; i < ndims; ++i) {
                    ଯଦି (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(i));
                    }
                }
            }

            ଯଦି (diff_dims.size() == 1) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା slice_dim = diff_dims[0];
                ଅଚଳ int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                ଯଦି (view_stride[slice_dim] > 0 && relative_offset % view_stride[slice_dim] == 0) {
                    ଅଚଳ int64_t begin_val = static_cast<int64_t>((relative_offset / view_stride[slice_dim]) %
                                                                   static_cast<size_t>(dim_size));
                    ଅଚଳ int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);

                    ଯଦି (begin_val >= 0 && end_val <= dim_size) {
                        auto sliced = std::make_shared<ov::op::v8::Slice>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim - axis_shift}));

                        ଯଦି (view_ov_shape.is_static()) {
                            auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                                ମିଥ୍ୟା);
                            reshaped->set_friendly_name(view_name);
                            ଫେରନ୍ତୁ reshaped;
                        }

                        sliced->set_friendly_name(view_name);
                        ଫେରନ୍ତୁ sliced;
                    }
                }

                int64_t tail_src_elems = 1;
                int64_t tail_dst_elems = 1;
                ପାଇଁ (size_t i = slice_dim; i < ndims; ++i) {
                    tail_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    tail_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                ଅଚଳ size_t elem_stride = view_stride[ndims - 1];
                int64_t tail_begin = 0;
                ଯଦି (elem_stride > 0) {
                    tail_begin =
                        static_cast<int64_t>((relative_offset / elem_stride) % static_cast<size_t>(tail_src_elems));
                }
                ଅଚଳ int64_t tail_end = tail_begin + tail_dst_elems;

                ଯଦି (tail_begin >= 0 && tail_end <= tail_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(slice_dim);
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < slice_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(tail_src_elems);
                    ଅଚଳ size_t flat_ndims = flat_shape.size();

                    auto flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), ମିଥ୍ୟା);

                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_end}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

                    ଯଦି (view_ov_shape.is_static()) {
                        auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                            ମିଥ୍ୟା);
                        reshaped->set_friendly_name(view_name);
                        ଫେରନ୍ତୁ reshaped;
                    }

                    sliced->set_friendly_name(view_name);
                    ଫେରନ୍ତୁ sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> step(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            size_t remaining_offset = relative_offset;
            ପାଇଁ (size_t i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                ଯଦି (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(remaining_offset / view_stride[i]);
                    remaining_offset %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            ତାର୍କିକ in_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            ଯଦି (in_bounds) {
                ପାଇଁ (size_t i = 0; i < ndims; ++i) {
                    ଯଦି (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        in_bounds = ମିଥ୍ୟା;
                        ଭାଙ୍ଗ;
                    }
                }
            }

            ଯଦି (in_bounds && remaining_offset == 0) {
                auto sliced = std::make_shared<ov::op::v8::Slice>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->set_friendly_name(view_name);
                ଫେରନ୍ତୁ sliced;
            }
        } ଅନ୍ୟଥା {
            ତାର୍କିକ same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            ଅଚଳ size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;

            ଯଦି (same_rank) {
                ଅଚଳ size_t ndims = view_ggml_shape.size();
                std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> diff_dims;
                ପାଇଁ (size_t i = 0; i < ndims; ++i) {
                    ଯଦି (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(i));
                    }
                }

                ଯଦି (diff_dims.size() == 1) {
                    ଅଚଳ size_t slice_dim = static_cast<size_t>(diff_dims[0]);
                    ତାର୍କିକ suffix_stride_match = ସତ୍ୟ;
                    ପାଇଁ (size_t i = slice_dim + 1; i < ndims; ++i) {
                        ଯଦି (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_match = ମିଥ୍ୟା;
                            ଭାଙ୍ଗ;
                        }
                    }

                    ଯଦି (suffix_stride_match && view_src_stride[slice_dim] > 0 &&
                        relative_offset % view_src_stride[slice_dim] == 0) {
                        ଅଚଳ int64_t begin_val = static_cast<int64_t>(relative_offset / view_src_stride[slice_dim]);
                        ଅଚଳ int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);
                        ଅଚଳ int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                        ଯଦି (begin_val >= 0 && end_val <= dim_size) {
                            auto sliced = std::make_shared<ov::op::v8::Slice>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(slice_dim)}));
                            sliced->set_friendly_name(view_name);
                            ଫେରନ୍ତୁ sliced;
                        }
                    }
                }
            }

            size_t view_elems = 1;
            size_t src_elems = 1;
            ଯଦି (same_rank) {
                ପାଇଁ (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            ତାର୍କିକ same_ଅଙ୍କ_elements = same_rank && view_elems == src_elems;

            ଯଦି (same_rank && relative_offset == 0 && same_ଅଙ୍କ_elements) {
                auto reshape_pattern = build_reshape_pattern(view_ov_shape, view_ggml_shape);

                auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_pattern.size()}, reshape_pattern),
                    ମିଥ୍ୟା);
                reshaped->set_friendly_name(view_name);
                ଫେରନ୍ତୁ reshaped;
            }

            ଯଦି (same_rank) {
                ଅଚଳ size_t ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                ତାର୍କିକ is_regular_slice = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> step(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                size_t remaining_offset = relative_offset;

                ଯଦି (is_regular_slice) {
                    ପାଇଁ (size_t i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        ଯଦି (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_slice = ମିଥ୍ୟା;
                            ଭାଙ୍ଗ;
                        }

                        step[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        ଯଦି (step[i] <= 0) {
                            is_regular_slice = ମିଥ୍ୟା;
                            ଭାଙ୍ଗ;
                        }

                        begin[i] = static_cast<int64_t>(remaining_offset / view_src_stride[i]);
                        remaining_offset %= view_src_stride[i];

                        ଯଦି (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            ଜାରି;
                        }

                        end[i] = begin[i] + step[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        ଯଦି (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_slice = ମିଥ୍ୟା;
                            ଭାଙ୍ଗ;
                        }
                    }
                }

                ଯଦି (is_regular_slice && remaining_offset == 0) {
                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->set_friendly_name(view_name);
                    ଫେରନ୍ତୁ sliced;
                }

                ଅଚଳ size_t elem_stride = view_src_stride.back();
                ଅଚଳ ତାର୍କିକ aligned_offset = elem_stride > 0 && relative_offset % elem_stride == 0;

                ଯଦି (aligned_offset) {
                    size_t suffix_start = 0;
                    size_t expected_stride = elem_stride;
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(ndims) - 1; i >= 0; --i) {
                        ଯଦି (view_stride[i] != expected_stride) {
                            suffix_start = static_cast<size_t>(i + 1);
                            ଭାଙ୍ଗ;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    size_t prefix_elems = 1;
                    size_t suffix_elems = 1;
                    ପାଇଁ (size_t i = 0; i < suffix_start; ++i) {
                        prefix_elems *= view_ggml_shape[i];
                    }
                    ପାଇଁ (size_t i = suffix_start; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    ଯଦି (prefix_elems > 0 && src_elems % prefix_elems == 0) {
                        ଅଚଳ size_t src_tail_elems = src_elems / prefix_elems;
                        ଅଚଳ int64_t tail_begin = static_cast<int64_t>(relative_offset / elem_stride);
                        ଅଚଳ int64_t tail_end = tail_begin + static_cast<int64_t>(suffix_elems);

                        ଯଦି (tail_begin >= 0 && tail_end <= static_cast<int64_t>(src_tail_elems)) {
                            auto prefix_tail_pattern = build_prefix_tail_reshape_pattern(
                                view_ov_shape, view_ggml_shape, suffix_start, static_cast<int64_t>(src_tail_elems));

                            auto prefix_tail = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {prefix_tail_pattern.size()},
                                                             prefix_tail_pattern),
                                ମିଥ୍ୟା);

                            ov::Output<ov::Node> selected = prefix_tail;
                            ଯଦି (tail_begin != 0 || tail_end != static_cast<int64_t>(src_tail_elems)) {
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
                                ମିଥ୍ୟା);
                            reshaped->set_friendly_name(view_name);
                            ଫେରନ୍ତୁ reshaped;
                        }
                    }
                }
            }

            ଫେରନ୍ତୁ current;
        }

        (ଶୂନ୍ୟ) view_name;
        (ଶୂନ୍ୟ) view_src_ov_shape;
        (ଶୂନ୍ୟ) view_src_name;

        ଫେରନ୍ତୁ current;
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
        auto innermost_offset = context.get_view_ନିବେଶ_offset(ନିବେଶ_index, 0);
        auto innermost_src_offset = context.get_view_ନିବେଶ_src_offset(ନିବେଶ_index, 0);
        auto innermost_shape = context.get_view_ନିବେଶ_ggml_shape(ନିବେଶ_index, 0);
        auto innermost_src_shape = context.get_view_ନିବେଶ_src_ggml_shape(ନିବେଶ_index, 0);
        ଯଦି (innermost_offset != innermost_src_offset && innermost_shape == innermost_src_shape) {
            size_t ମୂଳ_view_idx = view_ନିବେଶ_size - 1;
            auto ମୂଳ_ggml_shape = context.get_view_ନିବେଶ_src_ggml_shape(ନିବେଶ_index, ମୂଳ_view_idx);
            auto ମୂଳ_stride = context.get_view_ନିବେଶ_src_stride(ନିବେଶ_index, ମୂଳ_view_idx);
            auto ମୂଳ_offset = context.get_view_ନିବେଶ_src_offset(ନିବେଶ_index, ମୂଳ_view_idx);
            auto ମୂଳ_ov_shape = context.get_view_ନିବେଶ_src_ov_shape(ନିବେଶ_index, ମୂଳ_view_idx);
            auto ମୂଳ_name = context.get_view_ନିବେଶ_src_name(ନିବେଶ_index, ମୂଳ_view_idx);
            auto innermost_stride = context.get_view_ନିବେଶ_stride(ନିବେଶ_index, 0);
            auto innermost_ov_shape = context.get_view_ନିବେଶ_ov_shape(ନିବେଶ_index, 0);
            auto innermost_name = context.get_view_ନିବେଶ_name(ନିବେଶ_index, 0);

            ଫେରନ୍ତୁ process_single_view(ନିବେଶ, innermost_offset, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, ମୂଳ_offset, ମୂଳ_stride, ମୂଳ_ggml_shape, ମୂଳ_ov_shape,
                                       ମୂଳ_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = ନିବେଶ;

    // Process each view in reverse order (from base to current)
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା view_idx = view_ନିବେଶ_size - 1; view_idx >= 0; view_idx--) {
        auto view_offset = context.get_view_ନିବେଶ_offset(ନିବେଶ_index, view_idx);
        auto view_stride = context.get_view_ନିବେଶ_stride(ନିବେଶ_index, view_idx);
        auto view_ggml_shape = context.get_view_ନିବେଶ_ggml_shape(ନିବେଶ_index, view_idx);
        auto view_ov_shape = context.get_view_ନିବେଶ_ov_shape(ନିବେଶ_index, view_idx);
        auto view_name = context.get_view_ନିବେଶ_name(ନିବେଶ_index, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        auto view_src_offset = context.get_view_ନିବେଶ_src_offset(ନିବେଶ_index, view_idx);
        auto view_src_stride = context.get_view_ନିବେଶ_src_stride(ନିବେଶ_index, view_idx);
        auto view_src_ggml_shape = context.get_view_ନିବେଶ_src_ggml_shape(ନିବେଶ_index, view_idx);
        auto view_src_ov_shape = context.get_view_ନିବେଶ_src_ov_shape(ନିବେଶ_index, view_idx);
        auto view_src_name = context.get_view_ନିବେଶ_src_name(ନିବେଶ_index, view_idx);
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

    ଫେରନ୍ତୁ current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
