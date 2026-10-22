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

ಶೂನ್ಯ ಅಂಕೆ_inputs_check(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context, size_t min_inputs, size_t max_inputs) {
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_ಗಾತ್ರ = context.get_ಒಳಹರಿವು_ಗಾತ್ರ();
    FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(ಒಳಹರಿವು_ಗಾತ್ರ >= min_inputs, "Got less inputs than expected");
    FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(ಒಳಹರಿವು_ಗಾತ್ರ <= max_inputs, "Got more inputs than expected");
}

ಪೂರ್ಣಾಂಕ non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    ಪೂರ್ಣಾಂಕ dim = nb.size() - 1;
    size_t ಬೈಟುಗಳು = nb[dim];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = dim; i > 0; i--) {
        ಬೈಟುಗಳು *= ne[i];
        ಆದರೆ (ಬೈಟುಗಳು != nb[i - 1]) {
            ಹಿಂತಿರುಗಿ i;
        }
    }
    ಹಿಂತಿರುಗಿ 0;
}

std::shared_ptr<ov::Node> get_dimensions(ಸ್ಥಿರಾಂಕ std::shared_ptr<ov::op::v3::Shape_ದ> & shape,
                                         ಸ್ಥಿರಾಂಕ std::vector<ಪೂರ್ಣಾಂಕ> & dims) {
    ಬಳಸಿ ನಾಮಾಕಾಶ ov::op;
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    ಹಿಂತಿರುಗಿ std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(ಸ್ಥಿರಾಂಕ std::shared_ptr<ov::Node> & ಗಂಟು, ಸ್ಥಿರಾಂಕ std::vector<ಪೂರ್ಣಾಂಕ> & dims) {
    ಹಿಂತಿರುಗಿ get_dimensions(std::make_shared<ov::op::v3::Shape_ದ>(ಗಂಟು), dims);
}

Output_ಸದಿಶ rename_outputs_with_suffix(ಸ್ಥಿರಾಂಕ Output_ಸದಿಶ & outputs, ಸ್ಥಿರಾಂಕ std::string & suffix) {
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & output : outputs) {
        ಸ್ವಯಂಚಾಲಿತ ಗಂಟು = output.get_ಗಂಟು_shared_ptr();
        std::string name = ಗಂಟು->get_friendly_name();
        name += "_";
        name += suffix;
        ಗಂಟು->ಗಣ_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    ಹಿಂತಿರುಗಿ outputs;
}

ನಾಮಾಕಾಶ {
ov::Output<ov::Node> rope_yarn_ramp_mix(ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elements, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ corr_dims[2], ದಶಮಾಂಶ ext_ಗುಣಕ) {
    std::vector<ದಶಮಾಂಶ> dim_ids_vec(ಅಂಕೆ_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    ಸ್ವಯಂಚಾಲಿತ dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) ಅಂಕೆ_elements}, dim_ids_vec);
    ಸ್ವಯಂಚಾಲಿತ corr_ಕೆಳಗಿನ = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    ಸ್ವಯಂಚಾಲಿತ corr_ಮೇಲಿನ = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    ಸ್ವಯಂಚಾಲಿತ denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::ಕಳೆ>(corr_ಮೇಲಿನ, corr_ಕೆಳಗಿನ),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    ಸ್ವಯಂಚಾಲಿತ ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::ಕಳೆ>(dim_ids, corr_ಕೆಳಗಿನ), denom);
    ಸ್ವಯಂಚಾಲಿತ ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    ಸ್ವಯಂಚಾಲಿತ one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    ಸ್ವಯಂಚಾಲಿತ ramp_inverted = std::make_shared<ov::op::v1::ಕಳೆ>(one, ramp_clamped);
    ಸ್ವಯಂಚಾಲಿತ ext_ಗುಣಕ_ಗಂಟು = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_ಗುಣಕ});
    ಸ್ವಯಂಚಾಲಿತ ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_ಗುಣಕ_ಗಂಟು);
    ಹಿಂತಿರುಗಿ ramp_mix;
}

ದಶಮಾಂಶ ggml_rope_yarn_corr_dim(ಪೂರ್ಣಾಂಕ n_dims, ಪೂರ್ಣಾಂಕ n_ctx_orig, ದಶಮಾಂಶ n_rot, ದಶಮಾಂಶ ತಳ) {
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ M_PI
#    define M_PI 3.14159265358979323846
#ಆದರೆ_ಅಂತ್ಯ
    ಹಿಂತಿರುಗಿ n_dims * logf(n_ctx_orig / (n_rot * 2 * (ದಶಮಾಂಶ) M_PI)) / (2 * logf(ತಳ));
}

ಶೂನ್ಯ ggml_rope_yarn_corr_dims(ಪೂರ್ಣಾಂಕ n_dims,
                              ಪೂರ್ಣಾಂಕ n_ctx_orig,
                              ದಶಮಾಂಶ freq_ತಳ,
                              ದಶಮಾಂಶ beta_fast,
                              ದಶಮಾಂಶ beta_slow,
                              ದಶಮಾಂಶ dims[2]) {
    ದಶಮಾಂಶ start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_ತಳ));
    ದಶಮಾಂಶ end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_ತಳ));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<ದಶಮಾಂಶ>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_ತೂಕ,
                                                           ಪೂರ್ಣಾಂಕ mode,
                                                           ತಾರ್ಕಿಕ stateful,
                                                           int64_t ತಲೆ_dim) {
    constexpr ಪೂರ್ಣಾಂಕ TYPE_IMROPE = 2;
    constexpr ಪೂರ್ಣಾಂಕ TYPE_VISION = 3;
    constexpr ಪೂರ್ಣಾಂಕ TYPE_MROPE = 4;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_imrope = (mode == TYPE_IMROPE);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_vision = (mode == TYPE_VISION);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_mrope = (mode == TYPE_MROPE);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_multi = is_imrope || is_vision || is_mrope;

    ಆದರೆ (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        ಸ್ವಯಂಚಾಲಿತ pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        ಸ್ವಯಂಚಾಲಿತ pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, ನಿಜ);
        ಸ್ವಯಂಚಾಲಿತ pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } ಇಲ್ಲದಿದ್ದರೆ {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        ಸ್ವಯಂಚಾಲಿತ pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    ದಶಮಾಂಶ freq_ತಳ;
    ದಶಮಾಂಶ freq_scale;
    ದಶಮಾಂಶ ext_ಗುಣಕ;
    ದಶಮಾಂಶ attn_ಗುಣಕ;
    ದಶಮಾಂಶ beta_fast;
    ದಶಮಾಂಶ beta_slow;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims = rope_params[1];
    ಸ್ಥಿರಾಂಕ size_t n_dims_half = n_dims >> 1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ctx_orig = rope_params[4];
    ಸ್ಮರಣೆ_ನಕಲು(&freq_ತಳ, rope_params + 5, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&freq_scale, rope_params + 6, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&ext_ಗುಣಕ, rope_params + 7, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&attn_ಗುಣಕ, rope_params + 8, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&beta_fast, rope_params + 9, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&beta_slow, rope_params + 10, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta_scale = powf(freq_ತಳ, -2.0f / n_dims);

    Output<Node> theta;
    ದಶಮಾಂಶ mscale = attn_ಗುಣಕ;
    ಆದರೆ (is_multi) {
        ಸ್ಥಿರಾಂಕ size_t ಅಂಕೆ_ಜೋಡಿಗಳು = is_vision ? (n_dims > 0 ? (size_t) n_dims : (size_t) (ತಲೆ_dim / 2)) : n_dims_half;
        ಪೂರ್ಣಾಂಕ sections[4];
        ಸ್ಮರಣೆ_ನಕಲು(sections, rope_params + 11, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * 4);
        ಪೂರ್ಣಾಂಕ sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        ಆದರೆ (sect_dims <= 0) {
            sect_dims = ಅಂಕೆ_ಜೋಡಿಗಳು;
        }
        ಪೂರ್ಣಾಂಕ sec_w = sections[0] + sections[1];
        ಪೂರ್ಣಾಂಕ sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(ಅಂಕೆ_ಜೋಡಿಗಳು);
        std::vector<ದಶಮಾಂಶ> ಗುಣಕ(ಅಂಕೆ_ಜೋಡಿಗಳು);
        ಗಾಗಿ (size_t j = 0; j < ಅಂಕೆ_ಜೋಡಿಗಳು; j++) {
            ಪೂರ್ಣಾಂಕ sector = j % sect_dims;
            ಪೂರ್ಣಾಂಕ plane = 0;
            ಆದರೆ (is_imrope) {
                ಆದರೆ (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    plane = 3;
                }
                ಗುಣಕ[j] = std::ಘಾತ(theta_scale, j);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_vision) {
                ಪೂರ್ಣಾಂಕ p_idx = 0;
                ಆದರೆ (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                ಗುಣಕ[j] = std::ಘಾತ(theta_scale, p_idx);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector >= sec_e) {
                    plane = 3;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    plane = 0;
                }
                ಗುಣಕ[j] = std::ಘಾತ(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        ಸ್ವಯಂಚಾಲಿತ gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{ಅಂಕೆ_ಜೋಡಿಗಳು}, gather_indices);
        ಸ್ವಯಂಚಾಲಿತ gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> ಗುಣಕ_ಗಂಟು =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{ಅಂಕೆ_ಜೋಡಿಗಳು}, ಗುಣಕ);
        ಆದರೆ (rope_freqs_ತೂಕ) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::ತುಂಡು>(
                rope_freqs_ತೂಕ,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) ಅಂಕೆ_ಜೋಡಿಗಳು}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_ತೂಕ->get_output_partial_shape(0).rank().get_ಉದ್ದಳತೆ() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) ಅಂಕೆ_ಜೋಡಿಗಳು}),
                ಸುಳ್ಳು);
            ಗುಣಕ_ಗಂಟು = std::make_shared<ov::op::v1::Divide>(ಗುಣಕ_ಗಂಟು, rope_factors);
        }
        ಸ್ವಯಂಚಾಲಿತ theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, ಗುಣಕ_ಗಂಟು);
        ಸ್ವಯಂಚಾಲಿತ theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        ಆದರೆ (ext_ಗುಣಕ == 0.0f) {
            theta = theta_interp;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ದಶಮಾಂಶ corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ತಳ, beta_fast, beta_slow, corr_dims);
            ಸ್ವಯಂಚಾಲಿತ ramp_mix = rope_yarn_ramp_mix(ಅಂಕೆ_ಜೋಡಿಗಳು, corr_dims, ext_ಗುಣಕ);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            ಸ್ವಯಂಚಾಲಿತ one_minus_ramp = std::make_shared<ov::op::v1::ಕಳೆ>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::ಲಘುಗಣಕ(1.0f / freq_scale));
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        std::vector<ದಶಮಾಂಶ> ಗುಣಕ(n_dims_half);
        Output<Node> freq_factors;
        ದಶಮಾಂಶ corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ತಳ, beta_fast, beta_slow, corr_dims);
        ಗುಣಕ[0] = 1.0f;
        ಗಾಗಿ (size_t i = 1; i < ಗುಣಕ.size(); i++) {
            ಗುಣಕ[i] = theta_scale * ಗುಣಕ[i - 1];
        }
        ಆದರೆ (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, ಗುಣಕ.size()}, ಗುಣಕ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, ಗುಣಕ.size()}, ಗುಣಕ);
        }
        ಆದರೆ (rope_freqs_ತೂಕ) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::ತುಂಡು>(
                rope_freqs_ತೂಕ,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_ತೂಕ->get_output_partial_shape(0).rank().get_ಉದ್ದಳತೆ() - 1}));
            ಆದರೆ (stateful) {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {3}, {(int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    ಸುಳ್ಳು);
            } ಇಲ್ಲದಿದ್ದರೆ {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    ಸುಳ್ಳು);
            }
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_factors);
        }

        ಸ್ವಯಂಚಾಲಿತ theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        ಸ್ವಯಂಚಾಲಿತ theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        ಆದರೆ (ext_ಗುಣಕ == 0.0f) {
            theta = theta_interp;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ವಯಂಚಾಲಿತ ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_ಗುಣಕ);
            Output<Node> one;
            ಆದರೆ (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } ಇಲ್ಲದಿದ್ದರೆ {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            }
            ಸ್ವಯಂಚಾಲಿತ one_minus_ramp = std::make_shared<ov::op::v1::ಕಳೆ>(one, ramp_mix);

            theta =
                std::make_shared<ov::op::v1::ಕೂಡಿಸು>(std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                                                  std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::ಲಘುಗಣಕ(1.0f / freq_scale));
        }
    }

    Output<Node> cos_theta = std::make_shared<ov::op::v0::Cos>(theta);
    Output<Node> sin_theta = std::make_shared<ov::op::v0::Sin>(theta);

    ಆದರೆ (mscale != 1.0f) {
        ಸ್ವಯಂಚಾಲಿತ mscale_ಗಂಟು = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_ಗಂಟು);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_ಗಂಟು);
    }

    ಹಿಂತಿರುಗಿ std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_ಒಳಹರಿವು(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context, ಪೂರ್ಣಾಂಕ ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, ಪೂರ್ಣಾಂಕ ತುಂಡು_len, ಪೂರ್ಣಾಂಕ axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = context.get_ಒಳಹರಿವು(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ);
    ಸ್ವಯಂಚಾಲಿತ * op_params = (size_t *) context.get_ಒಳಹರಿವು_op_params(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ);
    ಸ್ವಯಂಚಾಲಿತ src_stride = context.get_ಒಳಹರಿವು_stride(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ);

    int64_t ತುಂಡು_ಆರಂಭ = op_params[0] / src_stride[3];
    ಆದರೆ (ತುಂಡು_len == 0) {
        ತುಂಡು_len = context.get_ಒಳಹರಿವು_shape(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ)[3].get_ಉದ್ದಳತೆ();
    }
    int64_t ತುಂಡು_ಕೊನೆ = ತುಂಡು_ಆರಂಭ + ತುಂಡು_len;

    ಸ್ವಯಂಚಾಲಿತ begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {ತುಂಡು_ಆರಂಭ});
    ಸ್ವಯಂಚಾಲಿತ end = ov::op::v0::Constant::create(ov::element::i64, {1}, {ತುಂಡು_ಕೊನೆ});
    ಸ್ವಯಂಚಾಲಿತ stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    ಆದರೆ (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } ಇಲ್ಲದಿದ್ದರೆ {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    ಸ್ವಯಂಚಾಲಿತ sliced = std::make_shared<ov::op::v8::ತುಂಡು>(ಒಳಹರಿವು, begin, end, stride, axes);
    ಹಿಂತಿರುಗಿ sliced;
}

ov::Output<ov::Node> process_view_ಒಳಹರಿವು_new(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context, ಪೂರ್ಣಾಂಕ ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ) {
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = context.get_ಒಳಹರಿವು(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ);

    // Check if this input has view inputs
    size_t view_ಒಳಹರಿವು_ಗಾತ್ರ = context.get_view_ಒಳಹರಿವು_ಗಾತ್ರ(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ);
    ಆದರೆ (view_ಒಳಹರಿವು_ಗಾತ್ರ == 0) {
        // No view inputs, return the input as is
        ಹಿಂತಿರುಗಿ ಒಳಹರಿವು;
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
    ಸ್ವಯಂಚಾಲಿತ expected_ov_shape = context.get_view_ಒಳಹರಿವು_ov_shape(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, 0);
    ಸ್ವಯಂಚಾಲಿತ actual_shape = ಒಳಹರಿವು.get_partial_shape();
    ಆದರೆ (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_ಉದ್ದಳತೆ() >= actual_shape.rank().get_ಉದ್ದಳತೆ()) {
        ಸ್ಥಿರಾಂಕ int64_t n_actual = actual_shape.rank().get_ಉದ್ದಳತೆ();
        ಸ್ಥಿರಾಂಕ int64_t ಸರಿಸು = expected_ov_shape.rank().get_ಉದ್ದಳತೆ() - n_actual;
        ತಾರ್ಕಿಕ shapes_ಹೊಂದಿಕೆ = ನಿಜ;
        ಗಾಗಿ (int64_t i = 0; i < ಸರಿಸು; ++i) {
            ಆದರೆ (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_ಉದ್ದಳತೆ() != 1) {
                shapes_ಹೊಂದಿಕೆ = ಸುಳ್ಳು;
                ನಿಲ್ಲಿಸು;
            }
        }
        ಗಾಗಿ (int64_t i = 0; i < n_actual && shapes_ಹೊಂದಿಕೆ; ++i) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಘಾತಾಂಕ = expected_ov_shape[i + ಸರಿಸು];
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ both_dynamic = ಘಾತಾಂಕ.is_dynamic() && actual_shape[i].is_dynamic();
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ both_static_equal =
                ಘಾತಾಂಕ.is_static() && actual_shape[i].is_static() && ಘಾತಾಂಕ == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ expected_dyn_actual_static = ಘಾತಾಂಕ.is_dynamic() && actual_shape[i].is_static();
            ಆದರೆ (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_ಹೊಂದಿಕೆ = ಸುಳ್ಳು;
                ನಿಲ್ಲಿಸು;
            }
        }
        ಆದರೆ (shapes_ಹೊಂದಿಕೆ) {
            ಹಿಂತಿರುಗಿ ಒಳಹರಿವು;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    ಆದರೆ (context.is_static() && view_ಒಳಹರಿವು_ಗಾತ್ರ == 1) {
        ಸ್ವಯಂಚಾಲಿತ view_stride_v = context.get_view_ಒಳಹರಿವು_stride(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, 0);
        ಸ್ವಯಂಚಾಲಿತ view_src_stride_v = context.get_view_ಒಳಹರಿವು_src_stride(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, 0);
        ಸ್ವಯಂಚಾಲಿತ view_ggml_shape = context.get_view_ಒಳಹರಿವು_ggml_shape(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, 0);
        ಸ್ವಯಂಚಾಲಿತ view_src_ggml_shape = context.get_view_ಒಳಹರಿವು_src_ggml_shape(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, 0);
        ಸ್ವಯಂಚಾಲಿತ view_ಸ್ಥಾನಾಂತರ = context.get_view_ಒಳಹರಿವು_ಸ್ಥಾನಾಂತರ(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, 0);
        ಸ್ವಯಂಚಾಲಿತ view_src_ಸ್ಥಾನಾಂತರ = context.get_view_ಒಳಹರಿವು_src_ಸ್ಥಾನಾಂತರ(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, 0);

        size_t ndims = view_ggml_shape.size();
        std::vector<ಪೂರ್ಣಾಂಕ> diff_dims;
        ಆದರೆ (view_src_ggml_shape.size() == ndims) {
            ಗಾಗಿ (size_t i = 0; i < ndims; ++i) {
                ಆದರೆ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<ಪೂರ್ಣಾಂಕ>(i));
                }
            }
        }

        ಆದರೆ (diff_dims.size() == 1) {
            ಪೂರ್ಣಾಂಕ ವಿಭಜಿಸು_dim = diff_dims[0];
            int64_t ಅಂಕೆ_splits = static_cast<int64_t>(view_src_ggml_shape[ವಿಭಜಿಸು_dim]);
            int64_t chunk_ಗಾತ್ರ = static_cast<int64_t>(view_ggml_shape[ವಿಭಜಿಸು_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            ಆದರೆ (chunk_ಗಾತ್ರ == 1 && ಅಂಕೆ_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                ತಾರ್ಕಿಕ suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                ಆದರೆ (suffix_ok) {
                    ಗಾಗಿ (size_t i = static_cast<size_t>(ವಿಭಜಿಸು_dim) + 1; i < ndims; ++i) {
                        ಆದರೆ (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = ಸುಳ್ಳು;
                            ನಿಲ್ಲಿಸು;
                        }
                    }
                }

                ಆದರೆ (suffix_ok && view_src_stride_v[ವಿಭಜಿಸು_dim] > 0) {
                    size_t relative_ಸ್ಥಾನಾಂತರ = view_ಸ್ಥಾನಾಂತರ >= view_src_ಸ್ಥಾನಾಂತರ ? view_ಸ್ಥಾನಾಂತರ - view_src_ಸ್ಥಾನಾಂತರ : 0;
                    int64_t ವಿಭಜಿಸು_ಸೂಚ್ಯಂಕ = static_cast<int64_t>(relative_ಸ್ಥಾನಾಂತರ / view_src_stride_v[ವಿಭಜಿಸು_dim]);

                    ಆದರೆ (ವಿಭಜಿಸು_ಸೂಚ್ಯಂಕ >= 0 && ವಿಭಜಿಸು_ಸೂಚ್ಯಂಕ < ಅಂಕೆ_splits) {
                        ಸ್ವಯಂಚಾಲಿತ src_ಗಂಟು = ಒಳಹರಿವು.get_ಗಂಟು_shared_ptr();
                        std::string rt_ಕೀಲಿ = "split_dim_" + std::to_string(ವಿಭಜಿಸು_dim);
                        ಸ್ವಯಂಚಾಲಿತ & rt_info = src_ಗಂಟು->get_rt_info();

                        ಆದರೆ (rt_info.find(rt_ಕೀಲಿ) == rt_info.end()) {
                            ಸ್ವಯಂಚಾಲಿತ axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(ವಿಭಜಿಸು_dim)});
                            ಸ್ವಯಂಚಾಲಿತ ವಿಭಜಿಸು_ಗಂಟು =
                                std::make_shared<ov::op::v1::ವಿಭಜಿಸು_2>(ಒಳಹರಿವು, axis_const, static_cast<size_t>(ಅಂಕೆ_splits));
                            ವಿಭಜಿಸು_ಗಂಟು->ಗಣ_friendly_name(src_ಗಂಟು->get_friendly_name() + "_split");
                            rt_info[rt_ಕೀಲಿ] = ವಿಭಜಿಸು_ಗಂಟು;
                        }

                        ಸ್ವಯಂಚಾಲಿತ ವಿಭಜಿಸು_ಗಂಟು = rt_info[rt_ಕೀಲಿ].as<std::shared_ptr<ov::op::v1::ವಿಭಜಿಸು_2>>();
                        ಹಿಂತಿರುಗಿ ವಿಭಜಿಸು_ಗಂಟು->output(static_cast<size_t>(ವಿಭಜಿಸು_ಸೂಚ್ಯಂಕ));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    ಸ್ವಯಂಚಾಲಿತ process_single_view =
        [](ov::Output<ov::Node> current, size_t view_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ std::vector<size_t> & view_stride,
           ಸ್ಥಿರಾಂಕ ov::Shape & view_ggml_shape, ಸ್ಥಿರಾಂಕ ov::PartialShape & view_ov_shape, ಸ್ಥಿರಾಂಕ std::string & view_name,
           size_t view_src_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ std::vector<size_t> & view_src_stride, ಸ್ಥಿರಾಂಕ ov::Shape & view_src_ggml_shape,
           ಸ್ಥಿರಾಂಕ ov::PartialShape & view_src_ov_shape, ಸ್ಥಿರಾಂಕ std::string & view_src_name) -> ov::Output<ov::Node> {
        // Stateful execution drops a leading size-1 axis, so `current` can be one rank lower
        // than the ggml shape metadata (view_stride/view_ggml_shape, always GGML_MAX_DIMS
        // entries) assumes. Shift any axis index built from that metadata down by the
        // difference before using it as an OV Slice axis.
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ current_rank = current.get_partial_shape().rank();
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ axis_ಸರಿಸು = current_rank.is_static() ?
                                    static_cast<ಪೂರ್ಣಾಂಕ>(view_stride.size()) - static_cast<ಪೂರ್ಣಾಂಕ>(current_rank.get_ಉದ್ದಳತೆ()) :
                                    0;

        ಸ್ವಯಂಚಾಲಿತ build_reshape_ನಮೂನೆ = [](ಸ್ಥಿರಾಂಕ ov::PartialShape & ಗುರಿ_ov_shape,
                                        ಸ್ಥಿರಾಂಕ ov::Shape & ಗುರಿ_ggml_shape) -> std::vector<int64_t> {
            ಸ್ಥಿರಾಂಕ size_t ndims = ಗುರಿ_ggml_shape.size();
            std::vector<int64_t> reshape_ನಮೂನೆ(ndims);
            size_t dynamic_dims = 0;

            ಆದರೆ (ಗುರಿ_ov_shape.rank().is_static() &&
                ಗುರಿ_ov_shape.rank().get_ಉದ್ದಳತೆ() == static_cast<int64_t>(ndims)) {
                ಗಾಗಿ (size_t i = 0; i < ndims; ++i) {
                    ಆದರೆ (ಗುರಿ_ov_shape[i].is_static()) {
                        reshape_ನಮೂನೆ[i] = ಗುರಿ_ov_shape[i].get_ಉದ್ದಳತೆ();
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        reshape_ನಮೂನೆ[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                dynamic_dims = 2;
            }

            ಆದರೆ (dynamic_dims > 1) {
                ಗಾಗಿ (size_t i = 0; i < ndims; ++i) {
                    reshape_ನಮೂನೆ[i] = static_cast<int64_t>(ಗುರಿ_ggml_shape[i]);
                }
            }

            ಹಿಂತಿರುಗಿ reshape_ನಮೂನೆ;
        };

        ಸ್ವಯಂಚಾಲಿತ build_ಪೂರ್ವಪ್ರತ್ಯಯ_ಬಾಲ_reshape_ನಮೂನೆ = [](ಸ್ಥಿರಾಂಕ ov::PartialShape & ಗುರಿ_ov_shape,
                                                    ಸ್ಥಿರಾಂಕ ov::Shape & ಗುರಿ_ggml_shape, size_t ಪೂರ್ವಪ್ರತ್ಯಯ_dims,
                                                    int64_t ಬಾಲ_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_ನಮೂನೆ(ಪೂರ್ವಪ್ರತ್ಯಯ_dims + 1);
            size_t dynamic_dims = 0;

            ಆದರೆ (ಗುರಿ_ov_shape.rank().is_static() &&
                ಗುರಿ_ov_shape.rank().get_ಉದ್ದಳತೆ() == static_cast<int64_t>(ಗುರಿ_ggml_shape.size())) {
                ಗಾಗಿ (size_t i = 0; i < ಪೂರ್ವಪ್ರತ್ಯಯ_dims; ++i) {
                    ಆದರೆ (ಗುರಿ_ov_shape[i].is_static()) {
                        reshape_ನಮೂನೆ[i] = ಗುರಿ_ov_shape[i].get_ಉದ್ದಳತೆ();
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        reshape_ನಮೂನೆ[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                dynamic_dims = 2;
            }

            ಆದರೆ (dynamic_dims > 1) {
                ಗಾಗಿ (size_t i = 0; i < ಪೂರ್ವಪ್ರತ್ಯಯ_dims; ++i) {
                    reshape_ನಮೂನೆ[i] = static_cast<int64_t>(ಗುರಿ_ggml_shape[i]);
                }
            }

            reshape_ನಮೂನೆ[ಪೂರ್ವಪ್ರತ್ಯಯ_dims] = ಬಾಲ_dim;
            ಹಿಂತಿರುಗಿ reshape_ನಮೂನೆ;
        };

        ತಾರ್ಕಿಕ same_stride = view_stride.size() == view_src_stride.size();
        ಆದರೆ (same_stride) {
            ಗಾಗಿ (size_t i = 0; i < view_stride.size(); ++i) {
                ಆದರೆ (view_stride[i] != view_src_stride[i]) {
                    same_stride = ಸುಳ್ಳು;
                    ನಿಲ್ಲಿಸು;
                }
            }
        }

        ತಾರ್ಕಿಕ same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        ಆದರೆ (same_ggml_shape) {
            ಗಾಗಿ (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                ಆದರೆ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = ಸುಳ್ಳು;
                    ನಿಲ್ಲಿಸು;
                }
            }
        }

        ಆದರೆ (same_stride && same_ggml_shape) {
            ಹಿಂತಿರುಗಿ current;
        }

        ಆದರೆ (same_stride) {
            ಸ್ಥಿರಾಂಕ size_t relative_ಸ್ಥಾನಾಂತರ = view_ಸ್ಥಾನಾಂತರ >= view_src_ಸ್ಥಾನಾಂತರ ? view_ಸ್ಥಾನಾಂತರ - view_src_ಸ್ಥಾನಾಂತರ : 0;
            ಸ್ಥಿರಾಂಕ size_t ndims = view_stride.size();

            std::vector<ಪೂರ್ಣಾಂಕ> diff_dims;
            ಆದರೆ (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                ಗಾಗಿ (size_t i = 0; i < ndims; ++i) {
                    ಆದರೆ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<ಪೂರ್ಣಾಂಕ>(i));
                    }
                }
            }

            ಆದರೆ (diff_dims.size() == 1) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ತುಂಡು_dim = diff_dims[0];
                ಸ್ಥಿರಾಂಕ int64_t dim_ಗಾತ್ರ = static_cast<int64_t>(view_src_ggml_shape[ತುಂಡು_dim]);

                ಆದರೆ (view_stride[ತುಂಡು_dim] > 0 && relative_ಸ್ಥಾನಾಂತರ % view_stride[ತುಂಡು_dim] == 0) {
                    ಸ್ಥಿರಾಂಕ int64_t begin_val = static_cast<int64_t>((relative_ಸ್ಥಾನಾಂತರ / view_stride[ತುಂಡು_dim]) %
                                                                   static_cast<size_t>(dim_ಗಾತ್ರ));
                    ಸ್ಥಿರಾಂಕ int64_t ಕೊನೆ_val = begin_val + static_cast<int64_t>(view_ggml_shape[ತುಂಡು_dim]);

                    ಆದರೆ (begin_val >= 0 && ಕೊನೆ_val <= dim_ಗಾತ್ರ) {
                        ಸ್ವಯಂಚಾಲಿತ sliced = std::make_shared<ov::op::v8::ತುಂಡು>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {ಕೊನೆ_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {ತುಂಡು_dim - axis_ಸರಿಸು}));

                        ಆದರೆ (view_ov_shape.is_static()) {
                            ಸ್ವಯಂಚಾಲಿತ reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.ಗಮ್ಯ_shape()),
                                ಸುಳ್ಳು);
                            reshaped->ಗಣ_friendly_name(view_name);
                            ಹಿಂತಿರುಗಿ reshaped;
                        }

                        sliced->ಗಣ_friendly_name(view_name);
                        ಹಿಂತಿರುಗಿ sliced;
                    }
                }

                int64_t ಬಾಲ_src_elems = 1;
                int64_t ಬಾಲ_dst_elems = 1;
                ಗಾಗಿ (size_t i = ತುಂಡು_dim; i < ndims; ++i) {
                    ಬಾಲ_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    ಬಾಲ_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                ಸ್ಥಿರಾಂಕ size_t elem_stride = view_stride[ndims - 1];
                int64_t ಬಾಲ_begin = 0;
                ಆದರೆ (elem_stride > 0) {
                    ಬಾಲ_begin =
                        static_cast<int64_t>((relative_ಸ್ಥಾನಾಂತರ / elem_stride) % static_cast<size_t>(ಬಾಲ_src_elems));
                }
                ಸ್ಥಿರಾಂಕ int64_t ಬಾಲ_ಕೊನೆ = ಬಾಲ_begin + ಬಾಲ_dst_elems;

                ಆದರೆ (ಬಾಲ_begin >= 0 && ಬಾಲ_ಕೊನೆ <= ಬಾಲ_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(ತುಂಡು_dim);
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ತುಂಡು_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(ಬಾಲ_src_elems);
                    ಸ್ಥಿರಾಂಕ size_t flat_ndims = flat_shape.size();

                    ಸ್ವಯಂಚಾಲಿತ flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), ಸುಳ್ಳು);

                    ಸ್ವಯಂಚಾಲಿತ sliced = std::make_shared<ov::op::v8::ತುಂಡು>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {ಬಾಲ_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {ಬಾಲ_ಕೊನೆ}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {ತುಂಡು_dim}));

                    ಆದರೆ (view_ov_shape.is_static()) {
                        ಸ್ವಯಂಚಾಲಿತ reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.ಗಮ್ಯ_shape()),
                            ಸುಳ್ಳು);
                        reshaped->ಗಣ_friendly_name(view_name);
                        ಹಿಂತಿರುಗಿ reshaped;
                    }

                    sliced->ಗಣ_friendly_name(view_name);
                    ಹಿಂತಿರುಗಿ sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> ಹೆಜ್ಜೆ(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            size_t ಉಳಿದ_ಸ್ಥಾನಾಂತರ = relative_ಸ್ಥಾನಾಂತರ;
            ಗಾಗಿ (size_t i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                ಆದರೆ (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(ಉಳಿದ_ಸ್ಥಾನಾಂತರ / view_stride[i]);
                    ಉಳಿದ_ಸ್ಥಾನಾಂತರ %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            ತಾರ್ಕಿಕ ಒಳ_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            ಆದರೆ (ಒಳ_bounds) {
                ಗಾಗಿ (size_t i = 0; i < ndims; ++i) {
                    ಆದರೆ (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        ಒಳ_bounds = ಸುಳ್ಳು;
                        ನಿಲ್ಲಿಸು;
                    }
                }
            }

            ಆದರೆ (ಒಳ_bounds && ಉಳಿದ_ಸ್ಥಾನಾಂತರ == 0) {
                ಸ್ವಯಂಚಾಲಿತ sliced = std::make_shared<ov::op::v8::ತುಂಡು>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, ಹೆಜ್ಜೆ),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->ಗಣ_friendly_name(view_name);
                ಹಿಂತಿರುಗಿ sliced;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ತಾರ್ಕಿಕ same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            ಸ್ಥಿರಾಂಕ size_t relative_ಸ್ಥಾನಾಂತರ = view_ಸ್ಥಾನಾಂತರ >= view_src_ಸ್ಥಾನಾಂತರ ? view_ಸ್ಥಾನಾಂತರ - view_src_ಸ್ಥಾನಾಂತರ : 0;

            ಆದರೆ (same_rank) {
                ಸ್ಥಿರಾಂಕ size_t ndims = view_ggml_shape.size();
                std::vector<ಪೂರ್ಣಾಂಕ> diff_dims;
                ಗಾಗಿ (size_t i = 0; i < ndims; ++i) {
                    ಆದರೆ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<ಪೂರ್ಣಾಂಕ>(i));
                    }
                }

                ಆದರೆ (diff_dims.size() == 1) {
                    ಸ್ಥಿರಾಂಕ size_t ತುಂಡು_dim = static_cast<size_t>(diff_dims[0]);
                    ತಾರ್ಕಿಕ suffix_stride_ಹೊಂದಿಕೆ = ನಿಜ;
                    ಗಾಗಿ (size_t i = ತುಂಡು_dim + 1; i < ndims; ++i) {
                        ಆದರೆ (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_ಹೊಂದಿಕೆ = ಸುಳ್ಳು;
                            ನಿಲ್ಲಿಸು;
                        }
                    }

                    ಆದರೆ (suffix_stride_ಹೊಂದಿಕೆ && view_src_stride[ತುಂಡು_dim] > 0 &&
                        relative_ಸ್ಥಾನಾಂತರ % view_src_stride[ತುಂಡು_dim] == 0) {
                        ಸ್ಥಿರಾಂಕ int64_t begin_val = static_cast<int64_t>(relative_ಸ್ಥಾನಾಂತರ / view_src_stride[ತುಂಡು_dim]);
                        ಸ್ಥಿರಾಂಕ int64_t ಕೊನೆ_val = begin_val + static_cast<int64_t>(view_ggml_shape[ತುಂಡು_dim]);
                        ಸ್ಥಿರಾಂಕ int64_t dim_ಗಾತ್ರ = static_cast<int64_t>(view_src_ggml_shape[ತುಂಡು_dim]);

                        ಆದರೆ (begin_val >= 0 && ಕೊನೆ_val <= dim_ಗಾತ್ರ) {
                            ಸ್ವಯಂಚಾಲಿತ sliced = std::make_shared<ov::op::v8::ತುಂಡು>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {ಕೊನೆ_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(ತುಂಡು_dim)}));
                            sliced->ಗಣ_friendly_name(view_name);
                            ಹಿಂತಿರುಗಿ sliced;
                        }
                    }
                }
            }

            size_t view_elems = 1;
            size_t src_elems = 1;
            ಆದರೆ (same_rank) {
                ಗಾಗಿ (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            ತಾರ್ಕಿಕ same_ಅಂಕೆ_elements = same_rank && view_elems == src_elems;

            ಆದರೆ (same_rank && relative_ಸ್ಥಾನಾಂತರ == 0 && same_ಅಂಕೆ_elements) {
                ಸ್ವಯಂಚಾಲಿತ reshape_ನಮೂನೆ = build_reshape_ನಮೂನೆ(view_ov_shape, view_ggml_shape);

                ಸ್ವಯಂಚಾಲಿತ reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_ನಮೂನೆ.size()}, reshape_ನಮೂನೆ),
                    ಸುಳ್ಳು);
                reshaped->ಗಣ_friendly_name(view_name);
                ಹಿಂತಿರುಗಿ reshaped;
            }

            ಆದರೆ (same_rank) {
                ಸ್ಥಿರಾಂಕ size_t ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                ತಾರ್ಕಿಕ is_regular_ತುಂಡು = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> ಹೆಜ್ಜೆ(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                size_t ಉಳಿದ_ಸ್ಥಾನಾಂತರ = relative_ಸ್ಥಾನಾಂತರ;

                ಆದರೆ (is_regular_ತುಂಡು) {
                    ಗಾಗಿ (size_t i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        ಆದರೆ (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_ತುಂಡು = ಸುಳ್ಳು;
                            ನಿಲ್ಲಿಸು;
                        }

                        ಹೆಜ್ಜೆ[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        ಆದರೆ (ಹೆಜ್ಜೆ[i] <= 0) {
                            is_regular_ತುಂಡು = ಸುಳ್ಳು;
                            ನಿಲ್ಲಿಸು;
                        }

                        begin[i] = static_cast<int64_t>(ಉಳಿದ_ಸ್ಥಾನಾಂತರ / view_src_stride[i]);
                        ಉಳಿದ_ಸ್ಥಾನಾಂತರ %= view_src_stride[i];

                        ಆದರೆ (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            ಮುಂದುವರಿ;
                        }

                        end[i] = begin[i] + ಹೆಜ್ಜೆ[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        ಆದರೆ (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_ತುಂಡು = ಸುಳ್ಳು;
                            ನಿಲ್ಲಿಸು;
                        }
                    }
                }

                ಆದರೆ (is_regular_ತುಂಡು && ಉಳಿದ_ಸ್ಥಾನಾಂತರ == 0) {
                    ಸ್ವಯಂಚಾಲಿತ sliced = std::make_shared<ov::op::v8::ತುಂಡು>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, ಹೆಜ್ಜೆ),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->ಗಣ_friendly_name(view_name);
                    ಹಿಂತಿರುಗಿ sliced;
                }

                ಸ್ಥಿರಾಂಕ size_t elem_stride = view_src_stride.back();
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ aligned_ಸ್ಥಾನಾಂತರ = elem_stride > 0 && relative_ಸ್ಥಾನಾಂತರ % elem_stride == 0;

                ಆದರೆ (aligned_ಸ್ಥಾನಾಂತರ) {
                    size_t suffix_ಆರಂಭ = 0;
                    size_t expected_stride = elem_stride;
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = static_cast<ಪೂರ್ಣಾಂಕ>(ndims) - 1; i >= 0; --i) {
                        ಆದರೆ (view_stride[i] != expected_stride) {
                            suffix_ಆರಂಭ = static_cast<size_t>(i + 1);
                            ನಿಲ್ಲಿಸು;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    size_t ಪೂರ್ವಪ್ರತ್ಯಯ_elems = 1;
                    size_t suffix_elems = 1;
                    ಗಾಗಿ (size_t i = 0; i < suffix_ಆರಂಭ; ++i) {
                        ಪೂರ್ವಪ್ರತ್ಯಯ_elems *= view_ggml_shape[i];
                    }
                    ಗಾಗಿ (size_t i = suffix_ಆರಂಭ; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    ಆದರೆ (ಪೂರ್ವಪ್ರತ್ಯಯ_elems > 0 && src_elems % ಪೂರ್ವಪ್ರತ್ಯಯ_elems == 0) {
                        ಸ್ಥಿರಾಂಕ size_t src_ಬಾಲ_elems = src_elems / ಪೂರ್ವಪ್ರತ್ಯಯ_elems;
                        ಸ್ಥಿರಾಂಕ int64_t ಬಾಲ_begin = static_cast<int64_t>(relative_ಸ್ಥಾನಾಂತರ / elem_stride);
                        ಸ್ಥಿರಾಂಕ int64_t ಬಾಲ_ಕೊನೆ = ಬಾಲ_begin + static_cast<int64_t>(suffix_elems);

                        ಆದರೆ (ಬಾಲ_begin >= 0 && ಬಾಲ_ಕೊನೆ <= static_cast<int64_t>(src_ಬಾಲ_elems)) {
                            ಸ್ವಯಂಚಾಲಿತ ಪೂರ್ವಪ್ರತ್ಯಯ_ಬಾಲ_ನಮೂನೆ = build_ಪೂರ್ವಪ್ರತ್ಯಯ_ಬಾಲ_reshape_ನಮೂನೆ(
                                view_ov_shape, view_ggml_shape, suffix_ಆರಂಭ, static_cast<int64_t>(src_ಬಾಲ_elems));

                            ಸ್ವಯಂಚಾಲಿತ ಪೂರ್ವಪ್ರತ್ಯಯ_ಬಾಲ = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {ಪೂರ್ವಪ್ರತ್ಯಯ_ಬಾಲ_ನಮೂನೆ.size()},
                                                             ಪೂರ್ವಪ್ರತ್ಯಯ_ಬಾಲ_ನಮೂನೆ),
                                ಸುಳ್ಳು);

                            ov::Output<ov::Node> selected = ಪೂರ್ವಪ್ರತ್ಯಯ_ಬಾಲ;
                            ಆದರೆ (ಬಾಲ_begin != 0 || ಬಾಲ_ಕೊನೆ != static_cast<int64_t>(src_ಬಾಲ_elems)) {
                                selected = std::make_shared<ov::op::v8::ತುಂಡು>(
                                    ಪೂರ್ವಪ್ರತ್ಯಯ_ಬಾಲ, ov::op::v0::Constant::create(ov::element::i64, {1}, {ಬಾಲ_begin}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {ಬಾಲ_ಕೊನೆ}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {static_cast<int64_t>(suffix_ಆರಂಭ)}));
                            }

                            ಸ್ವಯಂಚಾಲಿತ reshape_ನಮೂನೆ = build_reshape_ನಮೂನೆ(view_ov_shape, view_ggml_shape);
                            ಸ್ವಯಂಚಾಲಿತ reshaped = std::make_shared<ov::op::v1::Reshape>(
                                selected,
                                ov::op::v0::Constant::create(ov::element::i64, {reshape_ನಮೂನೆ.size()},
                                                             reshape_ನಮೂನೆ),
                                ಸುಳ್ಳು);
                            reshaped->ಗಣ_friendly_name(view_name);
                            ಹಿಂತಿರುಗಿ reshaped;
                        }
                    }
                }
            }

            ಹಿಂತಿರುಗಿ current;
        }

        (ಶೂನ್ಯ) view_name;
        (ಶೂನ್ಯ) view_src_ov_shape;
        (ಶೂನ್ಯ) view_src_name;

        ಹಿಂತಿರುಗಿ current;
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
        ಸ್ವಯಂಚಾಲಿತ innermost_ಸ್ಥಾನಾಂತರ = context.get_view_ಒಳಹರಿವು_ಸ್ಥಾನಾಂತರ(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, 0);
        ಸ್ವಯಂಚಾಲಿತ innermost_src_ಸ್ಥಾನಾಂತರ = context.get_view_ಒಳಹರಿವು_src_ಸ್ಥಾನಾಂತರ(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, 0);
        ಸ್ವಯಂಚಾಲಿತ innermost_shape = context.get_view_ಒಳಹರಿವು_ggml_shape(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, 0);
        ಸ್ವಯಂಚಾಲಿತ innermost_src_shape = context.get_view_ಒಳಹರಿವು_src_ggml_shape(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, 0);
        ಆದರೆ (innermost_ಸ್ಥಾನಾಂತರ != innermost_src_ಸ್ಥಾನಾಂತರ && innermost_shape == innermost_src_shape) {
            size_t ಬೇರು_view_idx = view_ಒಳಹರಿವು_ಗಾತ್ರ - 1;
            ಸ್ವಯಂಚಾಲಿತ ಬೇರು_ggml_shape = context.get_view_ಒಳಹರಿವು_src_ggml_shape(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, ಬೇರು_view_idx);
            ಸ್ವಯಂಚಾಲಿತ ಬೇರು_stride = context.get_view_ಒಳಹರಿವು_src_stride(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, ಬೇರು_view_idx);
            ಸ್ವಯಂಚಾಲಿತ ಬೇರು_ಸ್ಥಾನಾಂತರ = context.get_view_ಒಳಹರಿವು_src_ಸ್ಥಾನಾಂತರ(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, ಬೇರು_view_idx);
            ಸ್ವಯಂಚಾಲಿತ ಬೇರು_ov_shape = context.get_view_ಒಳಹರಿವು_src_ov_shape(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, ಬೇರು_view_idx);
            ಸ್ವಯಂಚಾಲಿತ ಬೇರು_name = context.get_view_ಒಳಹರಿವು_src_name(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, ಬೇರು_view_idx);
            ಸ್ವಯಂಚಾಲಿತ innermost_stride = context.get_view_ಒಳಹರಿವು_stride(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, 0);
            ಸ್ವಯಂಚಾಲಿತ innermost_ov_shape = context.get_view_ಒಳಹರಿವು_ov_shape(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, 0);
            ಸ್ವಯಂಚಾಲಿತ innermost_name = context.get_view_ಒಳಹರಿವು_name(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, 0);

            ಹಿಂತಿರುಗಿ process_single_view(ಒಳಹರಿವು, innermost_ಸ್ಥಾನಾಂತರ, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, ಬೇರು_ಸ್ಥಾನಾಂತರ, ಬೇರು_stride, ಬೇರು_ggml_shape, ಬೇರು_ov_shape,
                                       ಬೇರು_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = ಒಳಹರಿವು;

    // Process each view in reverse order (from base to current)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ view_idx = view_ಒಳಹರಿವು_ಗಾತ್ರ - 1; view_idx >= 0; view_idx--) {
        ಸ್ವಯಂಚಾಲಿತ view_ಸ್ಥಾನಾಂತರ = context.get_view_ಒಳಹರಿವು_ಸ್ಥಾನಾಂತರ(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, view_idx);
        ಸ್ವಯಂಚಾಲಿತ view_stride = context.get_view_ಒಳಹರಿವು_stride(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, view_idx);
        ಸ್ವಯಂಚಾಲಿತ view_ggml_shape = context.get_view_ಒಳಹರಿವು_ggml_shape(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, view_idx);
        ಸ್ವಯಂಚಾಲಿತ view_ov_shape = context.get_view_ಒಳಹರಿವು_ov_shape(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, view_idx);
        ಸ್ವಯಂಚಾಲಿತ view_name = context.get_view_ಒಳಹರಿವು_name(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        ಸ್ವಯಂಚಾಲಿತ view_src_ಸ್ಥಾನಾಂತರ = context.get_view_ಒಳಹರಿವು_src_ಸ್ಥಾನಾಂತರ(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, view_idx);
        ಸ್ವಯಂಚಾಲಿತ view_src_stride = context.get_view_ಒಳಹರಿವು_src_stride(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, view_idx);
        ಸ್ವಯಂಚಾಲಿತ view_src_ggml_shape = context.get_view_ಒಳಹರಿವು_src_ggml_shape(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, view_idx);
        ಸ್ವಯಂಚಾಲಿತ view_src_ov_shape = context.get_view_ಒಳಹರಿವು_src_ov_shape(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, view_idx);
        ಸ್ವಯಂಚಾಲಿತ view_src_name = context.get_view_ಒಳಹರಿವು_src_name(ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ, view_idx);
        // print source view info
        // std::cout << "View " << view_idx << ": source name = " << view_src_name
        //           << ", source offset = " << view_src_offset << ", source stride = [" << view_src_stride[0] << ","
        //           << view_src_stride[1] << "," << view_src_stride[2] << "," << view_src_stride[3]
        //           << "], source ggml shape = [" << view_src_ggml_shape[0] << "," << view_src_ggml_shape[1] << ","
        //           << view_src_ggml_shape[2] << "," << view_src_ggml_shape[3]
        //           << "], source ov shape = " << view_src_ov_shape << std::endl;

        current = process_single_view(current, view_ಸ್ಥಾನಾಂತರ, view_stride, view_ggml_shape, view_ov_shape, view_name,
                                      view_src_ಸ್ಥಾನಾಂತರ, view_src_stride, view_src_ggml_shape, view_src_ov_shape,
                                      view_src_name);
    }

    ಹಿಂತಿರುಗಿ current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
