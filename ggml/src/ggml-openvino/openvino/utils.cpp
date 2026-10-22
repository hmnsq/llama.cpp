#അന്തര്ഭാവ "utils.h"

#അന്തര്ഭാവ <cmath>
#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <ctime>
#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <openvino/op/add.hpp>
#അന്തര്ഭാവ <openvino/op/clamp.hpp>
#അന്തര്ഭാവ <openvino/op/convert.hpp>
#അന്തര്ഭാവ <openvino/op/cos.hpp>
#അന്തര്ഭാവ <openvino/op/divide.hpp>
#അന്തര്ഭാവ <openvino/op/gather.hpp>
#അന്തര്ഭാവ <openvino/op/maximum.hpp>
#അന്തര്ഭാവ <openvino/op/multiply.hpp>
#അന്തര്ഭാവ <openvino/op/reshape.hpp>
#അന്തര്ഭാവ <openvino/op/shape_of.hpp>
#അന്തര്ഭാവ <openvino/op/sin.hpp>
#അന്തര്ഭാവ <openvino/op/slice.hpp>
#അന്തര്ഭാവ <openvino/op/split.hpp>
#അന്തര്ഭാവ <openvino/op/squeeze.hpp>
#അന്തര്ഭാവ <openvino/op/subtract.hpp>
#അന്തര്ഭാവ <openvino/op/transpose.hpp>
#അന്തര്ഭാവ <string>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {

ശൂന്യം അക്കം_inputs_check(സ്ഥിരം കണ്ണി_Context & context, size_t min_inputs, size_t max_inputs) {
    auto നിവേശം_size = context.get_നിവേശം_size();
    FRONT_END_OP_CONVERSION_CHECK(നിവേശം_size >= min_inputs, "Got less inputs than expected");
    FRONT_END_OP_CONVERSION_CHECK(നിവേശം_size <= max_inputs, "Got more inputs than expected");
}

പൂർണ്ണസംഖ്യ non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    പൂർണ്ണസംഖ്യ dim = nb.size() - 1;
    size_t bytes = nb[dim];
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = dim; i > 0; i--) {
        bytes *= ne[i];
        എങ്കിൽ (bytes != nb[i - 1]) {
            തിരികെ i;
        }
    }
    തിരികെ 0;
}

std::shared_ptr<ov::Node> get_dimensions(സ്ഥിരം std::shared_ptr<ov::op::v3::ShapeOf> & shape,
                                         സ്ഥിരം std::vector<പൂർണ്ണസംഖ്യ> & dims) {
    using നാമാകാശ ov::op;
    സ്ഥിരം auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    സ്ഥിരം auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    തിരികെ std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(സ്ഥിരം std::shared_ptr<ov::Node> & കണ്ണി, സ്ഥിരം std::vector<പൂർണ്ണസംഖ്യ> & dims) {
    തിരികെ get_dimensions(std::make_shared<ov::op::v3::ShapeOf>(കണ്ണി), dims);
}

OutputVector rename_outputs_with_suffix(സ്ഥിരം OutputVector & outputs, സ്ഥിരം std::string & suffix) {
    വേണ്ടി (സ്ഥിരം auto & output : outputs) {
        auto കണ്ണി = output.get_കണ്ണി_shared_ptr();
        std::string name = കണ്ണി->get_friendly_name();
        name += "_";
        name += suffix;
        കണ്ണി->set_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    തിരികെ outputs;
}

നാമാകാശ {
ov::Output<ov::Node> rope_yarn_ramp_mix(പൂർണ്ണസംഖ്യ അക്കം_elements, സ്ഥിരം ദശാംശം corr_dims[2], ദശാംശം ext_factor) {
    std::vector<ദശാംശം> dim_ids_vec(അക്കം_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    auto dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) അക്കം_elements}, dim_ids_vec);
    auto corr_താഴെയുള്ള = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    auto corr_മുകളിലെ = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    auto denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::Subtract>(corr_മുകളിലെ, corr_താഴെയുള്ള),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    auto ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Subtract>(dim_ids, corr_താഴെയുള്ള), denom);
    auto ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    auto one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    auto ramp_inverted = std::make_shared<ov::op::v1::Subtract>(one, ramp_clamped);
    auto ext_factor_കണ്ണി = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_factor});
    auto ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_factor_കണ്ണി);
    തിരികെ ramp_mix;
}

ദശാംശം ggml_rope_yarn_corr_dim(പൂർണ്ണസംഖ്യ n_dims, പൂർണ്ണസംഖ്യ n_ctx_orig, ദശാംശം n_rot, ദശാംശം base) {
#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
    തിരികെ n_dims * logf(n_ctx_orig / (n_rot * 2 * (ദശാംശം) M_PI)) / (2 * logf(base));
}

ശൂന്യം ggml_rope_yarn_corr_dims(പൂർണ്ണസംഖ്യ n_dims,
                              പൂർണ്ണസംഖ്യ n_ctx_orig,
                              ദശാംശം freq_base,
                              ദശാംശം beta_fast,
                              ദശാംശം beta_slow,
                              ദശാംശം dims[2]) {
    ദശാംശം start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    ദശാംശം end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<ദശാംശം>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight,
                                                           പൂർണ്ണസംഖ്യ mode,
                                                           തർക്കം stateful,
                                                           int64_t head_dim) {
    constexpr പൂർണ്ണസംഖ്യ TYPE_IMROPE = 2;
    constexpr പൂർണ്ണസംഖ്യ TYPE_VISION = 3;
    constexpr പൂർണ്ണസംഖ്യ TYPE_MROPE = 4;

    സ്ഥിരം തർക്കം is_imrope = (mode == TYPE_IMROPE);
    സ്ഥിരം തർക്കം is_vision = (mode == TYPE_VISION);
    സ്ഥിരം തർക്കം is_mrope = (mode == TYPE_MROPE);
    സ്ഥിരം തർക്കം is_multi = is_imrope || is_vision || is_mrope;

    എങ്കിൽ (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } അല്ലെങ്കിൽ എങ്കിൽ (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, സത്യം);
        auto pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } അല്ലെങ്കിൽ {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    ദശാംശം freq_base;
    ദശാംശം freq_scale;
    ദശാംശം ext_factor;
    ദശാംശം attn_factor;
    ദശാംശം beta_fast;
    ദശാംശം beta_slow;
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_dims = rope_params[1];
    സ്ഥിരം size_t n_dims_half = n_dims >> 1;
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_ctx_orig = rope_params[4];
    memcpy(&freq_base, rope_params + 5, വലുപ്പം(ദശാംശം));
    memcpy(&freq_scale, rope_params + 6, വലുപ്പം(ദശാംശം));
    memcpy(&ext_factor, rope_params + 7, വലുപ്പം(ദശാംശം));
    memcpy(&attn_factor, rope_params + 8, വലുപ്പം(ദശാംശം));
    memcpy(&beta_fast, rope_params + 9, വലുപ്പം(ദശാംശം));
    memcpy(&beta_slow, rope_params + 10, വലുപ്പം(ദശാംശം));

    സ്ഥിരം ദശാംശം theta_scale = powf(freq_base, -2.0f / n_dims);

    Output<Node> theta;
    ദശാംശം mscale = attn_factor;
    എങ്കിൽ (is_multi) {
        സ്ഥിരം size_t അക്കം_pairs = is_vision ? (n_dims > 0 ? (size_t) n_dims : (size_t) (head_dim / 2)) : n_dims_half;
        പൂർണ്ണസംഖ്യ sections[4];
        memcpy(sections, rope_params + 11, വലുപ്പം(പൂർണ്ണസംഖ്യ) * 4);
        പൂർണ്ണസംഖ്യ sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        എങ്കിൽ (sect_dims <= 0) {
            sect_dims = അക്കം_pairs;
        }
        പൂർണ്ണസംഖ്യ sec_w = sections[0] + sections[1];
        പൂർണ്ണസംഖ്യ sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(അക്കം_pairs);
        std::vector<ദശാംശം> factor(അക്കം_pairs);
        വേണ്ടി (size_t j = 0; j < അക്കം_pairs; j++) {
            പൂർണ്ണസംഖ്യ sector = j % sect_dims;
            പൂർണ്ണസംഖ്യ plane = 0;
            എങ്കിൽ (is_imrope) {
                എങ്കിൽ (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } അല്ലെങ്കിൽ എങ്കിൽ (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } അല്ലെങ്കിൽ എങ്കിൽ (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } അല്ലെങ്കിൽ {
                    plane = 3;
                }
                factor[j] = std::pow(theta_scale, j);
            } അല്ലെങ്കിൽ എങ്കിൽ (is_vision) {
                പൂർണ്ണസംഖ്യ p_idx = 0;
                എങ്കിൽ (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } അല്ലെങ്കിൽ എങ്കിൽ (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } അല്ലെങ്കിൽ എങ്കിൽ (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } അല്ലെങ്കിൽ {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                factor[j] = std::pow(theta_scale, p_idx);
            } അല്ലെങ്കിൽ {
                എങ്കിൽ (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } അല്ലെങ്കിൽ എങ്കിൽ (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } അല്ലെങ്കിൽ എങ്കിൽ (sector >= sec_e) {
                    plane = 3;
                } അല്ലെങ്കിൽ {
                    plane = 0;
                }
                factor[j] = std::pow(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        auto gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{അക്കം_pairs}, gather_indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> factor_കണ്ണി =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{അക്കം_pairs}, factor);
        എങ്കിൽ (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) അക്കം_pairs}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) അക്കം_pairs}),
                അസത്യം);
            factor_കണ്ണി = std::make_shared<ov::op::v1::Divide>(factor_കണ്ണി, rope_factors);
        }
        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, factor_കണ്ണി);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        എങ്കിൽ (ext_factor == 0.0f) {
            theta = theta_interp;
        } അല്ലെങ്കിൽ {
            ദശാംശം corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
            auto ramp_mix = rope_yarn_ramp_mix(അക്കം_pairs, corr_dims, ext_factor);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            auto one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::Add>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
        }
    } അല്ലെങ്കിൽ {
        std::vector<ദശാംശം> factor(n_dims_half);
        Output<Node> freq_factors;
        ദശാംശം corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        factor[0] = 1.0f;
        വേണ്ടി (size_t i = 1; i < factor.size(); i++) {
            factor[i] = theta_scale * factor[i - 1];
        }
        എങ്കിൽ (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, factor.size()}, factor);
        } അല്ലെങ്കിൽ {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, factor.size()}, factor);
        }
        എങ്കിൽ (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_length() - 1}));
            എങ്കിൽ (stateful) {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {3}, {(int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    അസത്യം);
            } അല്ലെങ്കിൽ {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    അസത്യം);
            }
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_factors);
        }

        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        എങ്കിൽ (ext_factor == 0.0f) {
            theta = theta_interp;
        } അല്ലെങ്കിൽ {
            auto ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_factor);
            Output<Node> one;
            എങ്കിൽ (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } അല്ലെങ്കിൽ {
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

    എങ്കിൽ (mscale != 1.0f) {
        auto mscale_കണ്ണി = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_കണ്ണി);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_കണ്ണി);
    }

    തിരികെ std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_നിവേശം(സ്ഥിരം കണ്ണി_Context & context, പൂർണ്ണസംഖ്യ നിവേശം_index, പൂർണ്ണസംഖ്യ slice_len, പൂർണ്ണസംഖ്യ axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    auto നിവേശം = context.get_നിവേശം(നിവേശം_index);
    auto * op_params = (size_t *) context.get_നിവേശം_op_params(നിവേശം_index);
    auto src_stride = context.get_നിവേശം_stride(നിവേശം_index);

    int64_t slice_start = op_params[0] / src_stride[3];
    എങ്കിൽ (slice_len == 0) {
        slice_len = context.get_നിവേശം_shape(നിവേശം_index)[3].get_length();
    }
    int64_t slice_end = slice_start + slice_len;

    auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_start});
    auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_end});
    auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    എങ്കിൽ (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } അല്ലെങ്കിൽ {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    auto sliced = std::make_shared<ov::op::v8::Slice>(നിവേശം, begin, end, stride, axes);
    തിരികെ sliced;
}

ov::Output<ov::Node> process_view_നിവേശം_new(സ്ഥിരം കണ്ണി_Context & context, പൂർണ്ണസംഖ്യ നിവേശം_index) {
    auto നിവേശം = context.get_നിവേശം(നിവേശം_index);

    // Check if this input has view inputs
    size_t view_നിവേശം_size = context.get_view_നിവേശം_size(നിവേശം_index);
    എങ്കിൽ (view_നിവേശം_size == 0) {
        // No view inputs, return the input as is
        തിരികെ നിവേശം;
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
    auto expected_ov_shape = context.get_view_നിവേശം_ov_shape(നിവേശം_index, 0);
    auto actual_shape = നിവേശം.get_partial_shape();
    എങ്കിൽ (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_length() >= actual_shape.rank().get_length()) {
        സ്ഥിരം int64_t n_actual = actual_shape.rank().get_length();
        സ്ഥിരം int64_t shift = expected_ov_shape.rank().get_length() - n_actual;
        തർക്കം shapes_match = സത്യം;
        വേണ്ടി (int64_t i = 0; i < shift; ++i) {
            എങ്കിൽ (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_length() != 1) {
                shapes_match = അസത്യം;
                നിർത്തുക;
            }
        }
        വേണ്ടി (int64_t i = 0; i < n_actual && shapes_match; ++i) {
            സ്ഥിരം auto & exp = expected_ov_shape[i + shift];
            സ്ഥിരം തർക്കം both_dynamic = exp.is_dynamic() && actual_shape[i].is_dynamic();
            സ്ഥിരം തർക്കം both_static_equal =
                exp.is_static() && actual_shape[i].is_static() && exp == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            സ്ഥിരം തർക്കം expected_dyn_actual_static = exp.is_dynamic() && actual_shape[i].is_static();
            എങ്കിൽ (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_match = അസത്യം;
                നിർത്തുക;
            }
        }
        എങ്കിൽ (shapes_match) {
            തിരികെ നിവേശം;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    എങ്കിൽ (context.is_static() && view_നിവേശം_size == 1) {
        auto view_stride_v = context.get_view_നിവേശം_stride(നിവേശം_index, 0);
        auto view_src_stride_v = context.get_view_നിവേശം_src_stride(നിവേശം_index, 0);
        auto view_ggml_shape = context.get_view_നിവേശം_ggml_shape(നിവേശം_index, 0);
        auto view_src_ggml_shape = context.get_view_നിവേശം_src_ggml_shape(നിവേശം_index, 0);
        auto view_offset = context.get_view_നിവേശം_offset(നിവേശം_index, 0);
        auto view_src_offset = context.get_view_നിവേശം_src_offset(നിവേശം_index, 0);

        size_t ndims = view_ggml_shape.size();
        std::vector<പൂർണ്ണസംഖ്യ> diff_dims;
        എങ്കിൽ (view_src_ggml_shape.size() == ndims) {
            വേണ്ടി (size_t i = 0; i < ndims; ++i) {
                എങ്കിൽ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<പൂർണ്ണസംഖ്യ>(i));
                }
            }
        }

        എങ്കിൽ (diff_dims.size() == 1) {
            പൂർണ്ണസംഖ്യ split_dim = diff_dims[0];
            int64_t അക്കം_splits = static_cast<int64_t>(view_src_ggml_shape[split_dim]);
            int64_t chunk_size = static_cast<int64_t>(view_ggml_shape[split_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            എങ്കിൽ (chunk_size == 1 && അക്കം_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                തർക്കം suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                എങ്കിൽ (suffix_ok) {
                    വേണ്ടി (size_t i = static_cast<size_t>(split_dim) + 1; i < ndims; ++i) {
                        എങ്കിൽ (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = അസത്യം;
                            നിർത്തുക;
                        }
                    }
                }

                എങ്കിൽ (suffix_ok && view_src_stride_v[split_dim] > 0) {
                    size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
                    int64_t split_index = static_cast<int64_t>(relative_offset / view_src_stride_v[split_dim]);

                    എങ്കിൽ (split_index >= 0 && split_index < അക്കം_splits) {
                        auto src_കണ്ണി = നിവേശം.get_കണ്ണി_shared_ptr();
                        std::string rt_താക്കോൽ = "split_dim_" + std::to_string(split_dim);
                        auto & rt_info = src_കണ്ണി->get_rt_info();

                        എങ്കിൽ (rt_info.find(rt_താക്കോൽ) == rt_info.end()) {
                            auto axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(split_dim)});
                            auto split_കണ്ണി =
                                std::make_shared<ov::op::v1::Split>(നിവേശം, axis_const, static_cast<size_t>(അക്കം_splits));
                            split_കണ്ണി->set_friendly_name(src_കണ്ണി->get_friendly_name() + "_split");
                            rt_info[rt_താക്കോൽ] = split_കണ്ണി;
                        }

                        auto split_കണ്ണി = rt_info[rt_താക്കോൽ].as<std::shared_ptr<ov::op::v1::Split>>();
                        തിരികെ split_കണ്ണി->output(static_cast<size_t>(split_index));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    auto process_single_view =
        [](ov::Output<ov::Node> current, size_t view_offset, സ്ഥിരം std::vector<size_t> & view_stride,
           സ്ഥിരം ov::Shape & view_ggml_shape, സ്ഥിരം ov::PartialShape & view_ov_shape, സ്ഥിരം std::string & view_name,
           size_t view_src_offset, സ്ഥിരം std::vector<size_t> & view_src_stride, സ്ഥിരം ov::Shape & view_src_ggml_shape,
           സ്ഥിരം ov::PartialShape & view_src_ov_shape, സ്ഥിരം std::string & view_src_name) -> ov::Output<ov::Node> {
        // Stateful execution drops a leading size-1 axis, so `current` can be one rank lower
        // than the ggml shape metadata (view_stride/view_ggml_shape, always GGML_MAX_DIMS
        // entries) assumes. Shift any axis index built from that metadata down by the
        // difference before using it as an OV Slice axis.
        സ്ഥിരം auto current_rank = current.get_partial_shape().rank();
        സ്ഥിരം പൂർണ്ണസംഖ്യ axis_shift = current_rank.is_static() ?
                                    static_cast<പൂർണ്ണസംഖ്യ>(view_stride.size()) - static_cast<പൂർണ്ണസംഖ്യ>(current_rank.get_length()) :
                                    0;

        auto build_reshape_pattern = [](സ്ഥിരം ov::PartialShape & target_ov_shape,
                                        സ്ഥിരം ov::Shape & target_ggml_shape) -> std::vector<int64_t> {
            സ്ഥിരം size_t ndims = target_ggml_shape.size();
            std::vector<int64_t> reshape_pattern(ndims);
            size_t dynamic_dims = 0;

            എങ്കിൽ (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(ndims)) {
                വേണ്ടി (size_t i = 0; i < ndims; ++i) {
                    എങ്കിൽ (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } അല്ലെങ്കിൽ {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } അല്ലെങ്കിൽ {
                dynamic_dims = 2;
            }

            എങ്കിൽ (dynamic_dims > 1) {
                വേണ്ടി (size_t i = 0; i < ndims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            തിരികെ reshape_pattern;
        };

        auto build_prefix_tail_reshape_pattern = [](സ്ഥിരം ov::PartialShape & target_ov_shape,
                                                    സ്ഥിരം ov::Shape & target_ggml_shape, size_t prefix_dims,
                                                    int64_t tail_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_pattern(prefix_dims + 1);
            size_t dynamic_dims = 0;

            എങ്കിൽ (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_length() == static_cast<int64_t>(target_ggml_shape.size())) {
                വേണ്ടി (size_t i = 0; i < prefix_dims; ++i) {
                    എങ്കിൽ (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_length();
                    } അല്ലെങ്കിൽ {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } അല്ലെങ്കിൽ {
                dynamic_dims = 2;
            }

            എങ്കിൽ (dynamic_dims > 1) {
                വേണ്ടി (size_t i = 0; i < prefix_dims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            reshape_pattern[prefix_dims] = tail_dim;
            തിരികെ reshape_pattern;
        };

        തർക്കം same_stride = view_stride.size() == view_src_stride.size();
        എങ്കിൽ (same_stride) {
            വേണ്ടി (size_t i = 0; i < view_stride.size(); ++i) {
                എങ്കിൽ (view_stride[i] != view_src_stride[i]) {
                    same_stride = അസത്യം;
                    നിർത്തുക;
                }
            }
        }

        തർക്കം same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        എങ്കിൽ (same_ggml_shape) {
            വേണ്ടി (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                എങ്കിൽ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = അസത്യം;
                    നിർത്തുക;
                }
            }
        }

        എങ്കിൽ (same_stride && same_ggml_shape) {
            തിരികെ current;
        }

        എങ്കിൽ (same_stride) {
            സ്ഥിരം size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
            സ്ഥിരം size_t ndims = view_stride.size();

            std::vector<പൂർണ്ണസംഖ്യ> diff_dims;
            എങ്കിൽ (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                വേണ്ടി (size_t i = 0; i < ndims; ++i) {
                    എങ്കിൽ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<പൂർണ്ണസംഖ്യ>(i));
                    }
                }
            }

            എങ്കിൽ (diff_dims.size() == 1) {
                സ്ഥിരം പൂർണ്ണസംഖ്യ slice_dim = diff_dims[0];
                സ്ഥിരം int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                എങ്കിൽ (view_stride[slice_dim] > 0 && relative_offset % view_stride[slice_dim] == 0) {
                    സ്ഥിരം int64_t begin_val = static_cast<int64_t>((relative_offset / view_stride[slice_dim]) %
                                                                   static_cast<size_t>(dim_size));
                    സ്ഥിരം int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);

                    എങ്കിൽ (begin_val >= 0 && end_val <= dim_size) {
                        auto sliced = std::make_shared<ov::op::v8::Slice>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim - axis_shift}));

                        എങ്കിൽ (view_ov_shape.is_static()) {
                            auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                                അസത്യം);
                            reshaped->set_friendly_name(view_name);
                            തിരികെ reshaped;
                        }

                        sliced->set_friendly_name(view_name);
                        തിരികെ sliced;
                    }
                }

                int64_t tail_src_elems = 1;
                int64_t tail_dst_elems = 1;
                വേണ്ടി (size_t i = slice_dim; i < ndims; ++i) {
                    tail_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    tail_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                സ്ഥിരം size_t elem_stride = view_stride[ndims - 1];
                int64_t tail_begin = 0;
                എങ്കിൽ (elem_stride > 0) {
                    tail_begin =
                        static_cast<int64_t>((relative_offset / elem_stride) % static_cast<size_t>(tail_src_elems));
                }
                സ്ഥിരം int64_t tail_end = tail_begin + tail_dst_elems;

                എങ്കിൽ (tail_begin >= 0 && tail_end <= tail_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(slice_dim);
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < slice_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(tail_src_elems);
                    സ്ഥിരം size_t flat_ndims = flat_shape.size();

                    auto flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), അസത്യം);

                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {tail_end}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

                    എങ്കിൽ (view_ov_shape.is_static()) {
                        auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                            അസത്യം);
                        reshaped->set_friendly_name(view_name);
                        തിരികെ reshaped;
                    }

                    sliced->set_friendly_name(view_name);
                    തിരികെ sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> step(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            size_t remaining_offset = relative_offset;
            വേണ്ടി (size_t i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                എങ്കിൽ (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(remaining_offset / view_stride[i]);
                    remaining_offset %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            തർക്കം in_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            എങ്കിൽ (in_bounds) {
                വേണ്ടി (size_t i = 0; i < ndims; ++i) {
                    എങ്കിൽ (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        in_bounds = അസത്യം;
                        നിർത്തുക;
                    }
                }
            }

            എങ്കിൽ (in_bounds && remaining_offset == 0) {
                auto sliced = std::make_shared<ov::op::v8::Slice>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->set_friendly_name(view_name);
                തിരികെ sliced;
            }
        } അല്ലെങ്കിൽ {
            തർക്കം same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            സ്ഥിരം size_t relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;

            എങ്കിൽ (same_rank) {
                സ്ഥിരം size_t ndims = view_ggml_shape.size();
                std::vector<പൂർണ്ണസംഖ്യ> diff_dims;
                വേണ്ടി (size_t i = 0; i < ndims; ++i) {
                    എങ്കിൽ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<പൂർണ്ണസംഖ്യ>(i));
                    }
                }

                എങ്കിൽ (diff_dims.size() == 1) {
                    സ്ഥിരം size_t slice_dim = static_cast<size_t>(diff_dims[0]);
                    തർക്കം suffix_stride_match = സത്യം;
                    വേണ്ടി (size_t i = slice_dim + 1; i < ndims; ++i) {
                        എങ്കിൽ (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_match = അസത്യം;
                            നിർത്തുക;
                        }
                    }

                    എങ്കിൽ (suffix_stride_match && view_src_stride[slice_dim] > 0 &&
                        relative_offset % view_src_stride[slice_dim] == 0) {
                        സ്ഥിരം int64_t begin_val = static_cast<int64_t>(relative_offset / view_src_stride[slice_dim]);
                        സ്ഥിരം int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);
                        സ്ഥിരം int64_t dim_size = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                        എങ്കിൽ (begin_val >= 0 && end_val <= dim_size) {
                            auto sliced = std::make_shared<ov::op::v8::Slice>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(slice_dim)}));
                            sliced->set_friendly_name(view_name);
                            തിരികെ sliced;
                        }
                    }
                }
            }

            size_t view_elems = 1;
            size_t src_elems = 1;
            എങ്കിൽ (same_rank) {
                വേണ്ടി (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            തർക്കം same_അക്കം_elements = same_rank && view_elems == src_elems;

            എങ്കിൽ (same_rank && relative_offset == 0 && same_അക്കം_elements) {
                auto reshape_pattern = build_reshape_pattern(view_ov_shape, view_ggml_shape);

                auto reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_pattern.size()}, reshape_pattern),
                    അസത്യം);
                reshaped->set_friendly_name(view_name);
                തിരികെ reshaped;
            }

            എങ്കിൽ (same_rank) {
                സ്ഥിരം size_t ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                തർക്കം is_regular_slice = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> step(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                size_t remaining_offset = relative_offset;

                എങ്കിൽ (is_regular_slice) {
                    വേണ്ടി (size_t i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        എങ്കിൽ (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_slice = അസത്യം;
                            നിർത്തുക;
                        }

                        step[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        എങ്കിൽ (step[i] <= 0) {
                            is_regular_slice = അസത്യം;
                            നിർത്തുക;
                        }

                        begin[i] = static_cast<int64_t>(remaining_offset / view_src_stride[i]);
                        remaining_offset %= view_src_stride[i];

                        എങ്കിൽ (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            തുടരുക;
                        }

                        end[i] = begin[i] + step[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        എങ്കിൽ (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_slice = അസത്യം;
                            നിർത്തുക;
                        }
                    }
                }

                എങ്കിൽ (is_regular_slice && remaining_offset == 0) {
                    auto sliced = std::make_shared<ov::op::v8::Slice>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->set_friendly_name(view_name);
                    തിരികെ sliced;
                }

                സ്ഥിരം size_t elem_stride = view_src_stride.back();
                സ്ഥിരം തർക്കം aligned_offset = elem_stride > 0 && relative_offset % elem_stride == 0;

                എങ്കിൽ (aligned_offset) {
                    size_t suffix_start = 0;
                    size_t expected_stride = elem_stride;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = static_cast<പൂർണ്ണസംഖ്യ>(ndims) - 1; i >= 0; --i) {
                        എങ്കിൽ (view_stride[i] != expected_stride) {
                            suffix_start = static_cast<size_t>(i + 1);
                            നിർത്തുക;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    size_t prefix_elems = 1;
                    size_t suffix_elems = 1;
                    വേണ്ടി (size_t i = 0; i < suffix_start; ++i) {
                        prefix_elems *= view_ggml_shape[i];
                    }
                    വേണ്ടി (size_t i = suffix_start; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    എങ്കിൽ (prefix_elems > 0 && src_elems % prefix_elems == 0) {
                        സ്ഥിരം size_t src_tail_elems = src_elems / prefix_elems;
                        സ്ഥിരം int64_t tail_begin = static_cast<int64_t>(relative_offset / elem_stride);
                        സ്ഥിരം int64_t tail_end = tail_begin + static_cast<int64_t>(suffix_elems);

                        എങ്കിൽ (tail_begin >= 0 && tail_end <= static_cast<int64_t>(src_tail_elems)) {
                            auto prefix_tail_pattern = build_prefix_tail_reshape_pattern(
                                view_ov_shape, view_ggml_shape, suffix_start, static_cast<int64_t>(src_tail_elems));

                            auto prefix_tail = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {prefix_tail_pattern.size()},
                                                             prefix_tail_pattern),
                                അസത്യം);

                            ov::Output<ov::Node> selected = prefix_tail;
                            എങ്കിൽ (tail_begin != 0 || tail_end != static_cast<int64_t>(src_tail_elems)) {
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
                                അസത്യം);
                            reshaped->set_friendly_name(view_name);
                            തിരികെ reshaped;
                        }
                    }
                }
            }

            തിരികെ current;
        }

        (ശൂന്യം) view_name;
        (ശൂന്യം) view_src_ov_shape;
        (ശൂന്യം) view_src_name;

        തിരികെ current;
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
        auto innermost_offset = context.get_view_നിവേശം_offset(നിവേശം_index, 0);
        auto innermost_src_offset = context.get_view_നിവേശം_src_offset(നിവേശം_index, 0);
        auto innermost_shape = context.get_view_നിവേശം_ggml_shape(നിവേശം_index, 0);
        auto innermost_src_shape = context.get_view_നിവേശം_src_ggml_shape(നിവേശം_index, 0);
        എങ്കിൽ (innermost_offset != innermost_src_offset && innermost_shape == innermost_src_shape) {
            size_t വേര്_view_idx = view_നിവേശം_size - 1;
            auto വേര്_ggml_shape = context.get_view_നിവേശം_src_ggml_shape(നിവേശം_index, വേര്_view_idx);
            auto വേര്_stride = context.get_view_നിവേശം_src_stride(നിവേശം_index, വേര്_view_idx);
            auto വേര്_offset = context.get_view_നിവേശം_src_offset(നിവേശം_index, വേര്_view_idx);
            auto വേര്_ov_shape = context.get_view_നിവേശം_src_ov_shape(നിവേശം_index, വേര്_view_idx);
            auto വേര്_name = context.get_view_നിവേശം_src_name(നിവേശം_index, വേര്_view_idx);
            auto innermost_stride = context.get_view_നിവേശം_stride(നിവേശം_index, 0);
            auto innermost_ov_shape = context.get_view_നിവേശം_ov_shape(നിവേശം_index, 0);
            auto innermost_name = context.get_view_നിവേശം_name(നിവേശം_index, 0);

            തിരികെ process_single_view(നിവേശം, innermost_offset, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, വേര്_offset, വേര്_stride, വേര്_ggml_shape, വേര്_ov_shape,
                                       വേര്_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = നിവേശം;

    // Process each view in reverse order (from base to current)
    വേണ്ടി (പൂർണ്ണസംഖ്യ view_idx = view_നിവേശം_size - 1; view_idx >= 0; view_idx--) {
        auto view_offset = context.get_view_നിവേശം_offset(നിവേശം_index, view_idx);
        auto view_stride = context.get_view_നിവേശം_stride(നിവേശം_index, view_idx);
        auto view_ggml_shape = context.get_view_നിവേശം_ggml_shape(നിവേശം_index, view_idx);
        auto view_ov_shape = context.get_view_നിവേശം_ov_shape(നിവേശം_index, view_idx);
        auto view_name = context.get_view_നിവേശം_name(നിവേശം_index, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        auto view_src_offset = context.get_view_നിവേശം_src_offset(നിവേശം_index, view_idx);
        auto view_src_stride = context.get_view_നിവേശം_src_stride(നിവേശം_index, view_idx);
        auto view_src_ggml_shape = context.get_view_നിവേശം_src_ggml_shape(നിവേശം_index, view_idx);
        auto view_src_ov_shape = context.get_view_നിവേശം_src_ov_shape(നിവേശം_index, view_idx);
        auto view_src_name = context.get_view_നിവേശം_src_name(നിവേശം_index, view_idx);
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

    തിരികെ current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
