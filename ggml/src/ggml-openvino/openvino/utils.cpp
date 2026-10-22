#ಅನ್ತರ್ಭಾವ "utils.h"

#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <ctime>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/clamp.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/cos.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/divide.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gather.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/maximum.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/sin.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/split.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/squeeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/subtract.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/transpose.hpp>
#ಅನ್ತರ್ಭಾವ <string>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {

ಶೂನ್ಯ ಅಂಕೆ_inputs_check(ಧ್ರುವ ಗಂಟು_Context & context, size_t min_inputs, size_t max_inputs) {
    auto ಒಳಹರಿವು_size = context.get_ಒಳಹರಿವು_size();
    FRONT_END_OP_CONVERSION_CHECK(ಒಳಹರಿವು_size >= min_inputs, "Got less inputs than expected");
    FRONT_END_OP_CONVERSION_CHECK(ಒಳಹರಿವು_size <= max_inputs, "Got more inputs than expected");
}

ಪೂರ್ಣಾಙ್ಕ non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    ಪೂರ್ಣಾಙ್ಕ dim = nb.size() - 1;
    size_t bytes = nb[dim];
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = dim; i > 0; i--) {
        bytes *= ne[i];
        ಯದಿ (bytes != nb[i - 1]) {
            ಪ್ರತ್ಯಾಗಮ i;
        }
    }
    ಪ್ರತ್ಯಾಗಮ 0;
}

std::shared_ptr<ov::Node> get_dimensions(ಧ್ರುವ std::shared_ptr<ov::op::v3::ShapeOf> & shape,
                                         ಧ್ರುವ std::vector<ಪೂರ್ಣಾಙ್ಕ> & dims) {
    using ನಾಮಾಕಾಶ ov::op;
    ಧ್ರುವ auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ಧ್ರುವ auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    ಪ್ರತ್ಯಾಗಮ std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(ಧ್ರುವ std::shared_ptr<ov::Node> & ಗಂಟು, ಧ್ರುವ std::vector<ಪೂರ್ಣಾಙ್ಕ> & dims) {
    ಪ್ರತ್ಯಾಗಮ get_dimensions(std::make_shared<ov::op::v3::ShapeOf>(ಗಂಟು), dims);
}

OutputVector rename_outputs_with_suffix(ಧ್ರುವ OutputVector & outputs, ಧ್ರುವ std::string & suffix) {
    ಪ್ರತಿ (ಧ್ರುವ auto & output : outputs) {
        auto ಗಂಟು = output.get_ಗಂಟು_shared_ptr();
        std::string name = ಗಂಟು->get_friendly_name();
        name += "_";
        name += suffix;
        ಗಂಟು->set_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    ಪ್ರತ್ಯಾಗಮ outputs;
}

ನಾಮಾಕಾಶ {
ov::Output<ov::Node> rope_yarn_ramp_mix(ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_elements, ಧ್ರುವ ಭಿನ್ನ corr_dims[2], ಭಿನ್ನ ext_factor) {
    std::vector<ಭಿನ್ನ> dim_ids_vec(ಅಂಕೆ_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    auto dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) ಅಂಕೆ_elements}, dim_ids_vec);
    auto corr_ಕೆಳಗಿನ = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    auto corr_ಮೇಲಿನ = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    auto denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::Subtract>(corr_ಮೇಲಿನ, corr_ಕೆಳಗಿನ),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    auto ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Subtract>(dim_ids, corr_ಕೆಳಗಿನ), denom);
    auto ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    auto one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    auto ramp_inverted = std::make_shared<ov::op::v1::Subtract>(one, ramp_clamped);
    auto ext_factor_ಗಂಟು = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_factor});
    auto ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_factor_ಗಂಟು);
    ಪ್ರತ್ಯಾಗಮ ramp_mix;
}

ಭಿನ್ನ ggml_rope_yarn_corr_dim(ಪೂರ್ಣಾಙ್ಕ n_dims, ಪೂರ್ಣಾಙ್ಕ n_ctx_orig, ಭಿನ್ನ n_rot, ಭಿನ್ನ base) {
#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
    ಪ್ರತ್ಯಾಗಮ n_dims * logf(n_ctx_orig / (n_rot * 2 * (ಭಿನ್ನ) M_PI)) / (2 * logf(base));
}

ಶೂನ್ಯ ggml_rope_yarn_corr_dims(ಪೂರ್ಣಾಙ್ಕ n_dims,
                              ಪೂರ್ಣಾಙ್ಕ n_ctx_orig,
                              ಭಿನ್ನ freq_base,
                              ಭಿನ್ನ beta_fast,
                              ಭಿನ್ನ beta_slow,
                              ಭಿನ್ನ dims[2]) {
    ಭಿನ್ನ start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    ಭಿನ್ನ end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<ಭಿನ್ನ>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight,
                                                           ಪೂರ್ಣಾಙ್ಕ mode,
                                                           ತಾರ್ಕಿಕ stateful,
                                                           int64_t head_dim) {
    constexpr ಪೂರ್ಣಾಙ್ಕ TYPE_IMROPE = 2;
    constexpr ಪೂರ್ಣಾಙ್ಕ TYPE_VISION = 3;
    constexpr ಪೂರ್ಣಾಙ್ಕ TYPE_MROPE = 4;

    ಧ್ರುವ ತಾರ್ಕಿಕ is_imrope = (mode == TYPE_IMROPE);
    ಧ್ರುವ ತಾರ್ಕಿಕ is_vision = (mode == TYPE_VISION);
    ಧ್ರುವ ತಾರ್ಕಿಕ is_mrope = (mode == TYPE_MROPE);
    ಧ್ರುವ ತಾರ್ಕಿಕ is_multi = is_imrope || is_vision || is_mrope;

    ಯದಿ (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } ಅನ್ಯಥಾ ಯದಿ (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, ಸತ್ಯ);
        auto pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } ಅನ್ಯಥಾ {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    ಭಿನ್ನ freq_base;
    ಭಿನ್ನ freq_scale;
    ಭಿನ್ನ ext_factor;
    ಭಿನ್ನ attn_factor;
    ಭಿನ್ನ beta_fast;
    ಭಿನ್ನ beta_slow;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_dims = rope_params[1];
    ಧ್ರುವ size_t n_dims_half = n_dims >> 1;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_ctx_orig = rope_params[4];
    memcpy(&freq_base, rope_params + 5, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&freq_scale, rope_params + 6, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&ext_factor, rope_params + 7, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&attn_factor, rope_params + 8, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&beta_fast, rope_params + 9, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&beta_slow, rope_params + 10, ಪರಿಮಾಣ(ಭಿನ್ನ));

    ಧ್ರುವ ಭಿನ್ನ theta_scale = powf(freq_base, -2.0f / n_dims);

    Output<Node> theta;
    ಭಿನ್ನ mscale = attn_factor;
    ಯದಿ (is_multi) {
        ಧ್ರುವ size_t ಅಂಕೆ_pairs = is_vision ? (n_dims > 0 ? (size_t) n_dims : (size_t) (head_dim / 2)) : n_dims_half;
        ಪೂರ್ಣಾಙ್ಕ sections[4];
        memcpy(sections, rope_params + 11, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) * 4);
        ಪೂರ್ಣಾಙ್ಕ sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        ಯದಿ (sect_dims <= 0) {
            sect_dims = ಅಂಕೆ_pairs;
        }
        ಪೂರ್ಣಾಙ್ಕ sec_w = sections[0] + sections[1];
        ಪೂರ್ಣಾಙ್ಕ sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(ಅಂಕೆ_pairs);
        std::vector<ಭಿನ್ನ> factor(ಅಂಕೆ_pairs);
        ಪ್ರತಿ (size_t j = 0; j < ಅಂಕೆ_pairs; j++) {
            ಪೂರ್ಣಾಙ್ಕ sector = j % sect_dims;
            ಪೂರ್ಣಾಙ್ಕ plane = 0;
            ಯದಿ (is_imrope) {
                ಯದಿ (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } ಅನ್ಯಥಾ ಯದಿ (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } ಅನ್ಯಥಾ ಯದಿ (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } ಅನ್ಯಥಾ {
                    plane = 3;
                }
                factor[j] = std::pow(theta_scale, j);
            } ಅನ್ಯಥಾ ಯದಿ (is_vision) {
                ಪೂರ್ಣಾಙ್ಕ p_idx = 0;
                ಯದಿ (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } ಅನ್ಯಥಾ ಯದಿ (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } ಅನ್ಯಥಾ ಯದಿ (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } ಅನ್ಯಥಾ {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                factor[j] = std::pow(theta_scale, p_idx);
            } ಅನ್ಯಥಾ {
                ಯದಿ (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } ಅನ್ಯಥಾ ಯದಿ (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } ಅನ್ಯಥಾ ಯದಿ (sector >= sec_e) {
                    plane = 3;
                } ಅನ್ಯಥಾ {
                    plane = 0;
                }
                factor[j] = std::pow(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        auto gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{ಅಂಕೆ_pairs}, gather_indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> factor_ಗಂಟು =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{ಅಂಕೆ_pairs}, factor);
        ಯದಿ (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) ಅಂಕೆ_pairs}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) ಅಂಕೆ_pairs}),
                ಅಸತ್ಯ);
            factor_ಗಂಟು = std::make_shared<ov::op::v1::Divide>(factor_ಗಂಟು, rope_factors);
        }
        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, factor_ಗಂಟು);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        ಯದಿ (ext_factor == 0.0f) {
            theta = theta_interp;
        } ಅನ್ಯಥಾ {
            ಭಿನ್ನ corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
            auto ramp_mix = rope_yarn_ramp_mix(ಅಂಕೆ_pairs, corr_dims, ext_factor);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            auto one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::Add>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
        }
    } ಅನ್ಯಥಾ {
        std::vector<ಭಿನ್ನ> factor(n_dims_half);
        Output<Node> freq_factors;
        ಭಿನ್ನ corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        factor[0] = 1.0f;
        ಪ್ರತಿ (size_t i = 1; i < factor.size(); i++) {
            factor[i] = theta_scale * factor[i - 1];
        }
        ಯದಿ (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, factor.size()}, factor);
        } ಅನ್ಯಥಾ {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, factor.size()}, factor);
        }
        ಯದಿ (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            ಯದಿ (stateful) {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {3}, {(int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    ಅಸತ್ಯ);
            } ಅನ್ಯಥಾ {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    ಅಸತ್ಯ);
            }
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_factors);
        }

        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        ಯದಿ (ext_factor == 0.0f) {
            theta = theta_interp;
        } ಅನ್ಯಥಾ {
            auto ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_factor);
            Output<Node> one;
            ಯದಿ (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } ಅನ್ಯಥಾ {
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

    ಯದಿ (mscale != 1.0f) {
        auto mscale_ಗಂಟು = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_ಗಂಟು);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_ಗಂಟು);
    }

    ಪ್ರತ್ಯಾಗಮ std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_ಒಳಹರಿವು(ಧ್ರುವ ಗಂಟು_Context & context, ಪೂರ್ಣಾಙ್ಕ ಒಳಹರಿವು_index, ಪೂರ್ಣಾಙ್ಕ slice_len, ಪೂರ್ಣಾಙ್ಕ axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    auto ಒಳಹರಿವು = context.get_ಒಳಹರಿವು(ಒಳಹರಿವು_index);
    auto * op_params = (size_t *) context.get_ಒಳಹರಿವು_op_params(ಒಳಹರಿವು_index);
    auto src_stride = context.get_ಒಳಹರಿವು_stride(ಒಳಹರಿವು_index);

    int64_t slice_start = op_params[0] / src_stride[3];
    ಯದಿ (slice_len == 0) {
        slice_len = context.get_ಒಳಹರಿವು_shape(ಒಳಹರಿವು_index)[3].get_length();
    }
    int64_t slice_end = slice_start + slice_len;

    auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_start});
    auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_end});
    auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    ಯದಿ (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } ಅನ್ಯಥಾ {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    auto sliced = std::make_shared<ov::op::v8::Slice>(ಒಳಹರಿವು, begin, end, stride, axes);
    ಪ್ರತ್ಯಾಗಮ sliced;
}

ov::Output<ov::Node> process_view_ಒಳಹರಿವು_new(ಧ್ರುವ ಗಂಟು_Context & context, ಪೂರ್ಣಾಙ್ಕ ಒಳಹರಿವು_index) {
    auto ಒಳಹರಿವು = context.get_ಒಳಹರಿವು(ಒಳಹರಿವು_index);

    // Check if this input has view inputs
    size_t view_ಒಳಹರಿವು_size = context.get_view_ಒಳಹರಿವು_size(ಒಳಹರಿವು_index);
    ಯದಿ (view_ಒಳಹರಿವು_size == 0) {
        // No view inputs, return the input as is
        ಪ್ರತ್ಯಾಗಮ ಒಳಹರಿವು;
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
    auto expected_ov_shape = context.get_view_ಒಳಹರಿವು_ov_shape(ಒಳಹರಿವು_index, 0);
    auto actual_shape = ಒಳಹರಿವು.get_partial_shape();
    ಯದಿ (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_length() >= actual_shape.rank().get_length()) {
        ಧ್ರುವ int64_t n_actual = actual_shape.rank().get_length();
        ಧ್ರುವ int64_t shift = expected_ov_shape.rank().get_length() - n_actual;
        ತಾರ್ಕಿಕ shapes_match = ಸತ್ಯ;
        ಪ್ರತಿ (int64_t i = 0; i < shift; ++i) {
            ಯದಿ (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_length() != 1) {
                shapes_match = ಅಸತ್ಯ;
                ಭಙ್ಗ;
            }
        }
        ಪ್ರತಿ (int64_t i = 0; i < n_actual && shapes_match; ++i) {
            ಧ್ರುವ auto & exp = expected_ov_shape[i + shift];
            ಧ್ರುವ ತಾರ್ಕಿಕ both_dynamic = exp.is_dynamic() && actual_shape[i].is_dynamic();
            ಧ್ರುವ ತಾರ್ಕಿಕ both_static_equal =
                exp.is_static() && actual_shape[i].is_static() && exp == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            ಧ್ರುವ ತಾರ್ಕಿಕ expected_dyn_actual_static = exp.is_dynamic() && actual_shape[i].is_static();
            ಯದಿ (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_match = ಅಸತ್ಯ;
                ಭಙ್ಗ;
            }
        }
        ಯದಿ (shapes_match) {
            ಪ್ರತ್ಯಾಗಮ ಒಳಹರಿವು;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    ಯದಿ (context.is_static() && view_ಒಳಹರಿವು_size == 1) {
        auto view_stride_v = context.get_view_ಒಳಹರಿವು_stride(ಒಳಹರಿವು_index, 0);
        auto view_src_stride_v = context.get_view_ಒಳಹರಿವು_src_stride(ಒಳಹರಿವು_index, 0);
        auto view_ggml_shape = context.get_view_ಒಳಹರಿವು_ggml_shape(ಒಳಹರಿವು_index, 0);
        auto view_src_ggml_shape = context.get_view_ಒಳಹರಿವು_src_ggml_shape(ಒಳಹರಿವು_index, 0);
        auto view_offset = context.get_view_ಒಳಹರಿವು_offset(ಒಳಹರಿವು_index, 0);
        auto view_src_offset = context.get_view_ಒಳಹರಿವು_src_offset(ಒಳಹರಿವು_index, 0);

        size_t ndims = view_ggml_shape.size();
        std::vector<ಪೂರ್ಣಾಙ್ಕ> diff_dims;
        ಯದಿ (view_src_ggml_shape.size() == ndims) {
            ಪ್ರತಿ (size_t i = 0; i < ndims; ++i) {
                ಯದಿ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<ಪೂರ್ಣಾಙ್ಕ>(i));
                }
            }
        }

        ಯದಿ (diff_dims.size() == 1) {
            ಪೂರ್ಣಾಙ್ಕ split_dim = diff_dims[0];
            int64_t ಅಂಕೆ_splits = static_cast<int64_t>(view_src_ggml_shape[split_dim]);
            int64_t chunk_size = static_cast<int64_t>(view_ggml_shape[split_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            ಯದಿ (chunk_size == 1 && ಅಂಕೆ_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                ತಾರ್ಕಿಕ suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                ಯದಿ (suffix_ok) {
                    ಪ್ರತಿ (size_t i = static_cast<size_t>(split_dim) + 1; i < ndims; ++i) {
                        ಯದಿ (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = ಅಸತ್ಯ;
                            ಭಙ್ಗ;
                        }
                    }
                }

                ಯದಿ (suffix_ok && view_src_stride_v[split_dim] > 0) {
                    size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
                    int64_t split_index = static_cast<int64_t>(relative_offset / view_src_stride_v[split_dim]);

                    ಯದಿ (split_index >= 0 && split_index < ಅಂಕೆ_splits) {
                        auto src_ಗಂಟು = ಒಳಹರಿವು.get_ಗಂಟು_shared_ptr();
                        std::string rt_ಕೀಲಿ = "split_dim_" + std::to_string(split_dim);
                        auto & rt_info = src_ಗಂಟು->get_rt_info();

                        ಯದಿ (rt_info.find(rt_ಕೀಲಿ) == rt_info.end()) {
                            auto axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(split_dim)});
                            auto split_ಗಂಟು =
                                std::make_shared<ov::op::v1::Split>(ಒಳಹರಿವು, axis_const, static_cast<size_t>(ಅಂಕೆ_splits));
                            split_ಗಂಟು->set_friendly_name(src_ಗಂಟು->get_friendly_name() + "_split");
                            rt_info[rt_ಕೀಲಿ] = split_ಗಂಟು;
                        }

                        auto split_ಗಂಟು = rt_info[rt_ಕೀಲಿ].as<std::shared_ptr<ov::op::v1::Split>>();
                        ಪ್ರತ್ಯಾಗಮ split_ಗಂಟು->output(static_cast<size_t>(split_index));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    auto process_single_view =
        [](ov::Output<ov::Node> current, size_t view_offset, ಧ್ರುವ std::vector<size_t> & view_stride,
           ಧ್ರುವ ov::Shape & view_ggml_shape, ಧ್ರುವ ov::PartialShape & view_ov_shape, ಧ್ರುವ std::string & view_name,
           size_t view_src_offset, ಧ್ರುವ std::vector<size_t> & view_src_stride, ಧ್ರುವ ov::Shape & view_src_ggml_shape,
           ಧ್ರುವ ov::PartialShape & view_src_ov_shape, ಧ್ರುವ std::string & view_src_name) -> ov::Output<ov::Node> {
        // Stateful execution drops a leading size-1 axis, so `current` can be one rank lower
        // than the ggml shape metadata (view_stride/view_ggml_shape, always GGML_MAX_DIMS
        // entries) assumes. Shift any axis index built from that metadata down by the
        // difference before using it as an OV Slice axis.
        ಧ್ರುವ auto current_rank = current.get_partial_shape().rank();
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ axis_shift = current_rank.is_static() ?
                                    static_cast<ಪೂರ್ಣಾಙ್ಕ>(view_stride.size()) - static_cast<ಪೂರ್ಣಾಙ್ಕ>(current_rank.get_length()) :
                                    0;

        auto build_reshape_pattern = [](ಧ್ರುವ ov::PartialShape & target_ov_shape,
                                        ಧ್ರುವ ov::Shape & target_ggml_shape) -> std::vector<int64_t> {
            ಧ್ರುವ size_t ndims = target_ggml_shape.size();
            std::vector<int64_t> reshape_pattern(ndims);
            size_t dynamic_dims = 0;

            ಯದಿ (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(ndims)) {
                ಪ್ರತಿ (size_t i = 0; i < ndims; ++i) {
                    ಯದಿ (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } ಅನ್ಯಥಾ {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } ಅನ್ಯಥಾ {
                dynamic_dims = 2;
            }

            ಯದಿ (dynamic_dims > 1) {
                ಪ್ರತಿ (size_t i = 0; i < ndims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            ಪ್ರತ್ಯಾಗಮ reshape_pattern;
        };

        auto build_prefix_tail_reshape_pattern = [](ಧ್ರುವ ov::PartialShape & target_ov_shape,
                                                    ಧ್ರುವ ov::Shape & target_ggml_shape, size_t prefix_dims,
                                                    int64_t tail_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_pattern(prefix_dims + 1);
            size_t dynamic_dims = 0;

            ಯದಿ (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(target_ggml_shape.size())) {
                ಪ್ರತಿ (size_t i = 0; i < prefix_dims; ++i) {
                    ಯದಿ (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } ಅನ್ಯಥಾ {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } ಅನ್ಯಥಾ {
                dynamic_dims = 2;
            }

            ಯದಿ (dynamic_dims > 1) {
                ಪ್ರತಿ (size_t i = 0; i < prefix_dims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            reshape_pattern[prefix_dims] = tail_dim;
            ಪ್ರತ್ಯಾಗಮ reshape_pattern;
        };

        ತಾರ್ಕಿಕ same_stride = view_stride.size() == view_src_stride.size();
        ಯದಿ (same_stride) {
            ಪ್ರತಿ (size_t i = 0; i < view_stride.size(); ++i) {
                ಯದಿ (view_stride[i] != view_src_stride[i]) {
                    same_stride = ಅಸತ್ಯ;
                    ಭಙ್ಗ;
                }
            }
        }

        ತಾರ್ಕಿಕ same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        ಯದಿ (same_ggml_shape) {
            ಪ್ರತಿ (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                ಯದಿ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = ಅಸತ್ಯ;
                    ಭಙ್ಗ;
                }
            }
        }

        ಯದಿ (same_stride && same_ggml_shape) {
            ಪ್ರತ್ಯಾಗಮ current;
        }

        ಯದಿ (same_stride) {
            ಧ್ರುವ size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
            ಧ್ರುವ size_t ndims = view_stride.size();

            std::vector<ಪೂರ್ಣಾಙ್ಕ> diff_dims;
            ಯದಿ (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                ಪ್ರತಿ (size_t i = 0; i < ndims; ++i) {
                    ಯದಿ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<ಪೂರ್ಣಾಙ್ಕ>(i));
                    }
                }
            }

            ಯದಿ (diff_dims.size() == 1) {
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ slice_dim = diff_dims[0];
                ಧ್ರುವ int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                ಯದಿ (view_stride[slice_dim] > 0 && relative_offset % view_stride[slice_dim] == 0) {
                    ಧ್ರುವ int64_t begin_val = static_cast<int64_t>((relative_offset / view_stride[slice_dim]) %
                                                                   static_cast<size_t>(dim_size));
                    ಧ್ರುವ int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);

                    ಯದಿ (begin_val >= 0 && end_val <= dim_size) {
                        auto sliced = std::make_shared<ov::op::v8::Slice>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim - axis_shift}));

                        ಯದಿ (view_ov_shape.is_static()) {
                            auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                                ಅಸತ್ಯ);
                            reshaped->set_friendly_name(view_name);
                            ಪ್ರತ್ಯಾಗಮ reshaped;
                        }

                        sliced->set_friendly_name(view_name);
                        ಪ್ರತ್ಯಾಗಮ sliced;
                    }
                }

                int64_t tail_src_elems = 1;
                int64_t tail_dst_elems = 1;
                ಪ್ರತಿ (size_t i = slice_dim; i < ndims; ++i) {
                    tail_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    tail_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                ಧ್ರುವ size_t elem_stride = view_stride[ndims - 1];
                int64_t tail_begin = 0;
                ಯದಿ (elem_stride > 0) {
                    tail_begin =
                        static_cast<int64_t>((relative_offset / elem_stride) % static_cast<size_t>(tail_src_elems));
                }
                ಧ್ರುವ int64_t tail_end = tail_begin + tail_dst_elems;

                ಯದಿ (tail_begin >= 0 && tail_end <= tail_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(slice_dim);
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < slice_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(tail_src_elems);
                    ಧ್ರುವ size_t flat_ndims = flat_shape.size();

                    auto flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), ಅಸತ್ಯ);

                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_end}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

                    ಯದಿ (view_ov_shape.is_static()) {
                        auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                            ಅಸತ್ಯ);
                        reshaped->set_friendly_name(view_name);
                        ಪ್ರತ್ಯಾಗಮ reshaped;
                    }

                    sliced->set_friendly_name(view_name);
                    ಪ್ರತ್ಯಾಗಮ sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> step(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            size_t remaining_offset = relative_offset;
            ಪ್ರತಿ (size_t i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                ಯದಿ (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(remaining_offset / view_stride[i]);
                    remaining_offset %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            ತಾರ್ಕಿಕ in_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            ಯದಿ (in_bounds) {
                ಪ್ರತಿ (size_t i = 0; i < ndims; ++i) {
                    ಯದಿ (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        in_bounds = ಅಸತ್ಯ;
                        ಭಙ್ಗ;
                    }
                }
            }

            ಯದಿ (in_bounds && remaining_offset == 0) {
                auto sliced = std::make_shared<ov::op::v8::Slice>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->set_friendly_name(view_name);
                ಪ್ರತ್ಯಾಗಮ sliced;
            }
        } ಅನ್ಯಥಾ {
            ತಾರ್ಕಿಕ same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            ಧ್ರುವ size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;

            ಯದಿ (same_rank) {
                ಧ್ರುವ size_t ndims = view_ggml_shape.size();
                std::vector<ಪೂರ್ಣಾಙ್ಕ> diff_dims;
                ಪ್ರತಿ (size_t i = 0; i < ndims; ++i) {
                    ಯದಿ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<ಪೂರ್ಣಾಙ್ಕ>(i));
                    }
                }

                ಯದಿ (diff_dims.size() == 1) {
                    ಧ್ರುವ size_t slice_dim = static_cast<size_t>(diff_dims[0]);
                    ತಾರ್ಕಿಕ suffix_stride_match = ಸತ್ಯ;
                    ಪ್ರತಿ (size_t i = slice_dim + 1; i < ndims; ++i) {
                        ಯದಿ (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_match = ಅಸತ್ಯ;
                            ಭಙ್ಗ;
                        }
                    }

                    ಯದಿ (suffix_stride_match && view_src_stride[slice_dim] > 0 &&
                        relative_offset % view_src_stride[slice_dim] == 0) {
                        ಧ್ರುವ int64_t begin_val = static_cast<int64_t>(relative_offset / view_src_stride[slice_dim]);
                        ಧ್ರುವ int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);
                        ಧ್ರುವ int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                        ಯದಿ (begin_val >= 0 && end_val <= dim_size) {
                            auto sliced = std::make_shared<ov::op::v8::Slice>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(slice_dim)}));
                            sliced->set_friendly_name(view_name);
                            ಪ್ರತ್ಯಾಗಮ sliced;
                        }
                    }
                }
            }

            size_t view_elems = 1;
            size_t src_elems = 1;
            ಯದಿ (same_rank) {
                ಪ್ರತಿ (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            ತಾರ್ಕಿಕ same_ಅಂಕೆ_elements = same_rank && view_elems == src_elems;

            ಯದಿ (same_rank && relative_offset == 0 && same_ಅಂಕೆ_elements) {
                auto reshape_pattern = build_reshape_pattern(view_ov_shape, view_ggml_shape);

                auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_pattern.size()}, reshape_pattern),
                    ಅಸತ್ಯ);
                reshaped->set_friendly_name(view_name);
                ಪ್ರತ್ಯಾಗಮ reshaped;
            }

            ಯದಿ (same_rank) {
                ಧ್ರುವ size_t ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                ತಾರ್ಕಿಕ is_regular_slice = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> step(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                size_t remaining_offset = relative_offset;

                ಯದಿ (is_regular_slice) {
                    ಪ್ರತಿ (size_t i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        ಯದಿ (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_slice = ಅಸತ್ಯ;
                            ಭಙ್ಗ;
                        }

                        step[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        ಯದಿ (step[i] <= 0) {
                            is_regular_slice = ಅಸತ್ಯ;
                            ಭಙ್ಗ;
                        }

                        begin[i] = static_cast<int64_t>(remaining_offset / view_src_stride[i]);
                        remaining_offset %= view_src_stride[i];

                        ಯದಿ (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            ಅನುವರ್ತ;
                        }

                        end[i] = begin[i] + step[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        ಯದಿ (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_slice = ಅಸತ್ಯ;
                            ಭಙ್ಗ;
                        }
                    }
                }

                ಯದಿ (is_regular_slice && remaining_offset == 0) {
                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->set_friendly_name(view_name);
                    ಪ್ರತ್ಯಾಗಮ sliced;
                }

                ಧ್ರುವ size_t elem_stride = view_src_stride.back();
                ಧ್ರುವ ತಾರ್ಕಿಕ aligned_offset = elem_stride > 0 && relative_offset % elem_stride == 0;

                ಯದಿ (aligned_offset) {
                    size_t suffix_start = 0;
                    size_t expected_stride = elem_stride;
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = static_cast<ಪೂರ್ಣಾಙ್ಕ>(ndims) - 1; i >= 0; --i) {
                        ಯದಿ (view_stride[i] != expected_stride) {
                            suffix_start = static_cast<size_t>(i + 1);
                            ಭಙ್ಗ;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    size_t prefix_elems = 1;
                    size_t suffix_elems = 1;
                    ಪ್ರತಿ (size_t i = 0; i < suffix_start; ++i) {
                        prefix_elems *= view_ggml_shape[i];
                    }
                    ಪ್ರತಿ (size_t i = suffix_start; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    ಯದಿ (prefix_elems > 0 && src_elems % prefix_elems == 0) {
                        ಧ್ರುವ size_t src_tail_elems = src_elems / prefix_elems;
                        ಧ್ರುವ int64_t tail_begin = static_cast<int64_t>(relative_offset / elem_stride);
                        ಧ್ರುವ int64_t tail_end = tail_begin + static_cast<int64_t>(suffix_elems);

                        ಯದಿ (tail_begin >= 0 && tail_end <= static_cast<int64_t>(src_tail_elems)) {
                            auto prefix_tail_pattern = build_prefix_tail_reshape_pattern(
                                view_ov_shape, view_ggml_shape, suffix_start, static_cast<int64_t>(src_tail_elems));

                            auto prefix_tail = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {prefix_tail_pattern.size()},
                                                             prefix_tail_pattern),
                                ಅಸತ್ಯ);

                            ov::Output<ov::Node> selected = prefix_tail;
                            ಯದಿ (tail_begin != 0 || tail_end != static_cast<int64_t>(src_tail_elems)) {
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
                                ಅಸತ್ಯ);
                            reshaped->set_friendly_name(view_name);
                            ಪ್ರತ್ಯಾಗಮ reshaped;
                        }
                    }
                }
            }

            ಪ್ರತ್ಯಾಗಮ current;
        }

        (ಶೂನ್ಯ) view_name;
        (ಶೂನ್ಯ) view_src_ov_shape;
        (ಶೂನ್ಯ) view_src_name;

        ಪ್ರತ್ಯಾಗಮ current;
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
        auto innermost_offset = context.get_view_ಒಳಹರಿವು_offset(ಒಳಹರಿವು_index, 0);
        auto innermost_src_offset = context.get_view_ಒಳಹರಿವು_src_offset(ಒಳಹರಿವು_index, 0);
        auto innermost_shape = context.get_view_ಒಳಹರಿವು_ggml_shape(ಒಳಹರಿವು_index, 0);
        auto innermost_src_shape = context.get_view_ಒಳಹರಿವು_src_ggml_shape(ಒಳಹರಿವು_index, 0);
        ಯದಿ (innermost_offset != innermost_src_offset && innermost_shape == innermost_src_shape) {
            size_t ಬೇರು_view_idx = view_ಒಳಹರಿವು_size - 1;
            auto ಬೇರು_ggml_shape = context.get_view_ಒಳಹರಿವು_src_ggml_shape(ಒಳಹರಿವು_index, ಬೇರು_view_idx);
            auto ಬೇರು_stride = context.get_view_ಒಳಹರಿವು_src_stride(ಒಳಹರಿವು_index, ಬೇರು_view_idx);
            auto ಬೇರು_offset = context.get_view_ಒಳಹರಿವು_src_offset(ಒಳಹರಿವು_index, ಬೇರು_view_idx);
            auto ಬೇರು_ov_shape = context.get_view_ಒಳಹರಿವು_src_ov_shape(ಒಳಹರಿವು_index, ಬೇರು_view_idx);
            auto ಬೇರು_name = context.get_view_ಒಳಹರಿವು_src_name(ಒಳಹರಿವು_index, ಬೇರು_view_idx);
            auto innermost_stride = context.get_view_ಒಳಹರಿವು_stride(ಒಳಹರಿವು_index, 0);
            auto innermost_ov_shape = context.get_view_ಒಳಹರಿವು_ov_shape(ಒಳಹರಿವು_index, 0);
            auto innermost_name = context.get_view_ಒಳಹರಿವು_name(ಒಳಹರಿವು_index, 0);

            ಪ್ರತ್ಯಾಗಮ process_single_view(ಒಳಹರಿವು, innermost_offset, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, ಬೇರು_offset, ಬೇರು_stride, ಬೇರು_ggml_shape, ಬೇರು_ov_shape,
                                       ಬೇರು_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = ಒಳಹರಿವು;

    // Process each view in reverse order (from base to current)
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ view_idx = view_ಒಳಹರಿವು_size - 1; view_idx >= 0; view_idx--) {
        auto view_offset = context.get_view_ಒಳಹರಿವು_offset(ಒಳಹರಿವು_index, view_idx);
        auto view_stride = context.get_view_ಒಳಹರಿವು_stride(ಒಳಹರಿವು_index, view_idx);
        auto view_ggml_shape = context.get_view_ಒಳಹರಿವು_ggml_shape(ಒಳಹರಿವು_index, view_idx);
        auto view_ov_shape = context.get_view_ಒಳಹರಿವು_ov_shape(ಒಳಹರಿವು_index, view_idx);
        auto view_name = context.get_view_ಒಳಹರಿವು_name(ಒಳಹರಿವು_index, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        auto view_src_offset = context.get_view_ಒಳಹರಿವು_src_offset(ಒಳಹರಿವು_index, view_idx);
        auto view_src_stride = context.get_view_ಒಳಹರಿವು_src_stride(ಒಳಹರಿವು_index, view_idx);
        auto view_src_ggml_shape = context.get_view_ಒಳಹರಿವು_src_ggml_shape(ಒಳಹರಿವು_index, view_idx);
        auto view_src_ov_shape = context.get_view_ಒಳಹರಿವು_src_ov_shape(ಒಳಹರಿವು_index, view_idx);
        auto view_src_name = context.get_view_ಒಳಹರಿವು_src_name(ಒಳಹರಿವು_index, view_idx);
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

    ಪ್ರತ್ಯಾಗಮ current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
