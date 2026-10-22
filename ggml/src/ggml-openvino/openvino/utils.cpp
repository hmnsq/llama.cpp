#ਅਨ੍ਤਰ੍ਭਾਵ "utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <ctime>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/clamp.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/cos.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/divide.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/gather.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/maximum.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/shape_of.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/sin.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/split.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/squeeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/subtract.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/transpose.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {

ਖਾਲੀ ਅੰਕ_inputs_check(ਅਚਲ ਗੰਢ_Context & context, size_t min_inputs, size_t max_inputs) {
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_ਮਾਪ = context.get_ਨਿਵੇਸ਼_ਮਾਪ();
    FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(ਨਿਵੇਸ਼_ਮਾਪ >= min_inputs, "Got less inputs than expected");
    FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(ਨਿਵੇਸ਼_ਮਾਪ <= max_inputs, "Got more inputs than expected");
}

ਪੂਰਨ non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    ਪੂਰਨ dim = nb.size() - 1;
    size_t ਬਾਈਟਾਂ = nb[dim];
    ਲਈ (ਪੂਰਨ i = dim; i > 0; i--) {
        ਬਾਈਟਾਂ *= ne[i];
        ਜੇ (ਬਾਈਟਾਂ != nb[i - 1]) {
            ਵਾਪਸ i;
        }
    }
    ਵਾਪਸ 0;
}

std::shared_ptr<ov::Node> get_dimensions(ਅਚਲ std::shared_ptr<ov::op::v3::Shape_ਦਾ> & shape,
                                         ਅਚਲ std::vector<ਪੂਰਨ> & dims) {
    ਵਰਤੋਂ ਨਾਮਾਕਾਸ਼ ov::op;
    ਅਚਲ ਸਵੈਚਾਲਿਤ zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ਅਚਲ ਸਵੈਚਾਲਿਤ dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    ਵਾਪਸ std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(ਅਚਲ std::shared_ptr<ov::Node> & ਗੰਢ, ਅਚਲ std::vector<ਪੂਰਨ> & dims) {
    ਵਾਪਸ get_dimensions(std::make_shared<ov::op::v3::Shape_ਦਾ>(ਗੰਢ), dims);
}

Output_ਸਦਿਸ਼ rename_outputs_with_suffix(ਅਚਲ Output_ਸਦਿਸ਼ & outputs, ਅਚਲ std::string & suffix) {
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & output : outputs) {
        ਸਵੈਚਾਲਿਤ ਗੰਢ = output.get_ਗੰਢ_shared_ptr();
        std::string name = ਗੰਢ->get_friendly_name();
        name += "_";
        name += suffix;
        ਗੰਢ->ਸੈੱਟ_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    ਵਾਪਸ outputs;
}

ਨਾਮਾਕਾਸ਼ {
ov::Output<ov::Node> rope_yarn_ramp_mix(ਪੂਰਨ ਅੰਕ_elements, ਅਚਲ ਦਸ਼ਮਲਵ corr_dims[2], ਦਸ਼ਮਲਵ ext_ਗੁਣਕ) {
    std::vector<ਦਸ਼ਮਲਵ> dim_ids_vec(ਅੰਕ_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    ਸਵੈਚਾਲਿਤ dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) ਅੰਕ_elements}, dim_ids_vec);
    ਸਵੈਚਾਲਿਤ corr_ਨੀਵਾਂ = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    ਸਵੈਚਾਲਿਤ corr_ਉੱਚਾ = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    ਸਵੈਚਾਲਿਤ denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::ਘਟਾਓ>(corr_ਉੱਚਾ, corr_ਨੀਵਾਂ),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    ਸਵੈਚਾਲਿਤ ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::ਘਟਾਓ>(dim_ids, corr_ਨੀਵਾਂ), denom);
    ਸਵੈਚਾਲਿਤ ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    ਸਵੈਚਾਲਿਤ one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    ਸਵੈਚਾਲਿਤ ramp_inverted = std::make_shared<ov::op::v1::ਘਟਾਓ>(one, ramp_clamped);
    ਸਵੈਚਾਲਿਤ ext_ਗੁਣਕ_ਗੰਢ = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_ਗੁਣਕ});
    ਸਵੈਚਾਲਿਤ ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_ਗੁਣਕ_ਗੰਢ);
    ਵਾਪਸ ramp_mix;
}

ਦਸ਼ਮਲਵ ggml_rope_yarn_corr_dim(ਪੂਰਨ n_dims, ਪੂਰਨ n_ctx_orig, ਦਸ਼ਮਲਵ n_rot, ਦਸ਼ਮਲਵ ਅਧਾਰ) {
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ M_PI
#    define M_PI 3.14159265358979323846
#ਜੇ_ਅੰਤ
    ਵਾਪਸ n_dims * logf(n_ctx_orig / (n_rot * 2 * (ਦਸ਼ਮਲਵ) M_PI)) / (2 * logf(ਅਧਾਰ));
}

ਖਾਲੀ ggml_rope_yarn_corr_dims(ਪੂਰਨ n_dims,
                              ਪੂਰਨ n_ctx_orig,
                              ਦਸ਼ਮਲਵ freq_ਅਧਾਰ,
                              ਦਸ਼ਮਲਵ beta_fast,
                              ਦਸ਼ਮਲਵ beta_slow,
                              ਦਸ਼ਮਲਵ dims[2]) {
    ਦਸ਼ਮਲਵ start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_ਅਧਾਰ));
    ਦਸ਼ਮਲਵ end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_ਅਧਾਰ));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<ਦਸ਼ਮਲਵ>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_ਭਾਰ,
                                                           ਪੂਰਨ mode,
                                                           ਤਾਰਕਿਕ stateful,
                                                           int64_t ਸਿਰ_dim) {
    constexpr ਪੂਰਨ TYPE_IMROPE = 2;
    constexpr ਪੂਰਨ TYPE_VISION = 3;
    constexpr ਪੂਰਨ TYPE_MROPE = 4;

    ਅਚਲ ਤਾਰਕਿਕ is_imrope = (mode == TYPE_IMROPE);
    ਅਚਲ ਤਾਰਕਿਕ is_vision = (mode == TYPE_VISION);
    ਅਚਲ ਤਾਰਕਿਕ is_mrope = (mode == TYPE_MROPE);
    ਅਚਲ ਤਾਰਕਿਕ is_multi = is_imrope || is_vision || is_mrope;

    ਜੇ (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        ਸਵੈਚਾਲਿਤ pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } ਨਹੀਂਤਾਂ ਜੇ (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        ਸਵੈਚਾਲਿਤ pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, ਸੱਚ);
        ਸਵੈਚਾਲਿਤ pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } ਨਹੀਂਤਾਂ {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        ਸਵੈਚਾਲਿਤ pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    ਦਸ਼ਮਲਵ freq_ਅਧਾਰ;
    ਦਸ਼ਮਲਵ freq_scale;
    ਦਸ਼ਮਲਵ ext_ਗੁਣਕ;
    ਦਸ਼ਮਲਵ attn_ਗੁਣਕ;
    ਦਸ਼ਮਲਵ beta_fast;
    ਦਸ਼ਮਲਵ beta_slow;
    ਅਚਲ ਪੂਰਨ n_dims = rope_params[1];
    ਅਚਲ size_t n_dims_half = n_dims >> 1;
    ਅਚਲ ਪੂਰਨ n_ctx_orig = rope_params[4];
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&freq_ਅਧਾਰ, rope_params + 5, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&freq_scale, rope_params + 6, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ext_ਗੁਣਕ, rope_params + 7, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&attn_ਗੁਣਕ, rope_params + 8, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&beta_fast, rope_params + 9, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&beta_slow, rope_params + 10, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਦਸ਼ਮਲਵ theta_scale = powf(freq_ਅਧਾਰ, -2.0f / n_dims);

    Output<Node> theta;
    ਦਸ਼ਮਲਵ mscale = attn_ਗੁਣਕ;
    ਜੇ (is_multi) {
        ਅਚਲ size_t ਅੰਕ_ਜੋੜੇ = is_vision ? (n_dims > 0 ? (size_t) n_dims : (size_t) (ਸਿਰ_dim / 2)) : n_dims_half;
        ਪੂਰਨ sections[4];
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(sections, rope_params + 11, ਆਕਾਰ(ਪੂਰਨ) * 4);
        ਪੂਰਨ sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        ਜੇ (sect_dims <= 0) {
            sect_dims = ਅੰਕ_ਜੋੜੇ;
        }
        ਪੂਰਨ sec_w = sections[0] + sections[1];
        ਪੂਰਨ sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(ਅੰਕ_ਜੋੜੇ);
        std::vector<ਦਸ਼ਮਲਵ> ਗੁਣਕ(ਅੰਕ_ਜੋੜੇ);
        ਲਈ (size_t j = 0; j < ਅੰਕ_ਜੋੜੇ; j++) {
            ਪੂਰਨ sector = j % sect_dims;
            ਪੂਰਨ plane = 0;
            ਜੇ (is_imrope) {
                ਜੇ (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } ਨਹੀਂਤਾਂ ਜੇ (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } ਨਹੀਂਤਾਂ ਜੇ (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } ਨਹੀਂਤਾਂ {
                    plane = 3;
                }
                ਗੁਣਕ[j] = std::ਘਾਤ(theta_scale, j);
            } ਨਹੀਂਤਾਂ ਜੇ (is_vision) {
                ਪੂਰਨ p_idx = 0;
                ਜੇ (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } ਨਹੀਂਤਾਂ ਜੇ (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } ਨਹੀਂਤਾਂ ਜੇ (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } ਨਹੀਂਤਾਂ {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                ਗੁਣਕ[j] = std::ਘਾਤ(theta_scale, p_idx);
            } ਨਹੀਂਤਾਂ {
                ਜੇ (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } ਨਹੀਂਤਾਂ ਜੇ (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } ਨਹੀਂਤਾਂ ਜੇ (sector >= sec_e) {
                    plane = 3;
                } ਨਹੀਂਤਾਂ {
                    plane = 0;
                }
                ਗੁਣਕ[j] = std::ਘਾਤ(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        ਸਵੈਚਾਲਿਤ gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{ਅੰਕ_ਜੋੜੇ}, gather_indices);
        ਸਵੈਚਾਲਿਤ gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> ਗੁਣਕ_ਗੰਢ =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{ਅੰਕ_ਜੋੜੇ}, ਗੁਣਕ);
        ਜੇ (rope_freqs_ਭਾਰ) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(
                rope_freqs_ਭਾਰ,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) ਅੰਕ_ਜੋੜੇ}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_ਭਾਰ->get_output_partial_shape(0).rank().get_ਲੰਮਾਈ() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) ਅੰਕ_ਜੋੜੇ}),
                ਝੂਠ);
            ਗੁਣਕ_ਗੰਢ = std::make_shared<ov::op::v1::Divide>(ਗੁਣਕ_ਗੰਢ, rope_factors);
        }
        ਸਵੈਚਾਲਿਤ theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, ਗੁਣਕ_ਗੰਢ);
        ਸਵੈਚਾਲਿਤ theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        ਜੇ (ext_ਗੁਣਕ == 0.0f) {
            theta = theta_interp;
        } ਨਹੀਂਤਾਂ {
            ਦਸ਼ਮਲਵ corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ਅਧਾਰ, beta_fast, beta_slow, corr_dims);
            ਸਵੈਚਾਲਿਤ ramp_mix = rope_yarn_ramp_mix(ਅੰਕ_ਜੋੜੇ, corr_dims, ext_ਗੁਣਕ);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            ਸਵੈਚਾਲਿਤ one_minus_ramp = std::make_shared<ov::op::v1::ਘਟਾਓ>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::ਜੋੜੋ>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::ਲਘੂਗਣਕ(1.0f / freq_scale));
        }
    } ਨਹੀਂਤਾਂ {
        std::vector<ਦਸ਼ਮਲਵ> ਗੁਣਕ(n_dims_half);
        Output<Node> freq_factors;
        ਦਸ਼ਮਲਵ corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ਅਧਾਰ, beta_fast, beta_slow, corr_dims);
        ਗੁਣਕ[0] = 1.0f;
        ਲਈ (size_t i = 1; i < ਗੁਣਕ.size(); i++) {
            ਗੁਣਕ[i] = theta_scale * ਗੁਣਕ[i - 1];
        }
        ਜੇ (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, ਗੁਣਕ.size()}, ਗੁਣਕ);
        } ਨਹੀਂਤਾਂ {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, ਗੁਣਕ.size()}, ਗੁਣਕ);
        }
        ਜੇ (rope_freqs_ਭਾਰ) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(
                rope_freqs_ਭਾਰ,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_ਭਾਰ->get_output_partial_shape(0).rank().get_ਲੰਮਾਈ() - 1}));
            ਜੇ (stateful) {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {3}, {(int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    ਝੂਠ);
            } ਨਹੀਂਤਾਂ {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    ਝੂਠ);
            }
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_factors);
        }

        ਸਵੈਚਾਲਿਤ theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        ਸਵੈਚਾਲਿਤ theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        ਜੇ (ext_ਗੁਣਕ == 0.0f) {
            theta = theta_interp;
        } ਨਹੀਂਤਾਂ {
            ਸਵੈਚਾਲਿਤ ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_ਗੁਣਕ);
            Output<Node> one;
            ਜੇ (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } ਨਹੀਂਤਾਂ {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            }
            ਸਵੈਚਾਲਿਤ one_minus_ramp = std::make_shared<ov::op::v1::ਘਟਾਓ>(one, ramp_mix);

            theta =
                std::make_shared<ov::op::v1::ਜੋੜੋ>(std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                                                  std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::ਲਘੂਗਣਕ(1.0f / freq_scale));
        }
    }

    Output<Node> cos_theta = std::make_shared<ov::op::v0::Cos>(theta);
    Output<Node> sin_theta = std::make_shared<ov::op::v0::Sin>(theta);

    ਜੇ (mscale != 1.0f) {
        ਸਵੈਚਾਲਿਤ mscale_ਗੰਢ = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_ਗੰਢ);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_ਗੰਢ);
    }

    ਵਾਪਸ std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_ਨਿਵੇਸ਼(ਅਚਲ ਗੰਢ_Context & context, ਪੂਰਨ ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, ਪੂਰਨ ਟੁਕੜਾ_len, ਪੂਰਨ axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = context.get_ਨਿਵੇਸ਼(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ);
    ਸਵੈਚਾਲਿਤ * op_params = (size_t *) context.get_ਨਿਵੇਸ਼_op_params(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ);
    ਸਵੈਚਾਲਿਤ src_stride = context.get_ਨਿਵੇਸ਼_stride(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ);

    int64_t ਟੁਕੜਾ_ਸ਼ੁਰੂ = op_params[0] / src_stride[3];
    ਜੇ (ਟੁਕੜਾ_len == 0) {
        ਟੁਕੜਾ_len = context.get_ਨਿਵੇਸ਼_shape(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ)[3].get_ਲੰਮਾਈ();
    }
    int64_t ਟੁਕੜਾ_ਸਿਰਾ = ਟੁਕੜਾ_ਸ਼ੁਰੂ + ਟੁਕੜਾ_len;

    ਸਵੈਚਾਲਿਤ begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {ਟੁਕੜਾ_ਸ਼ੁਰੂ});
    ਸਵੈਚਾਲਿਤ end = ov::op::v0::Constant::create(ov::element::i64, {1}, {ਟੁਕੜਾ_ਸਿਰਾ});
    ਸਵੈਚਾਲਿਤ stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    ਜੇ (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } ਨਹੀਂਤਾਂ {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    ਸਵੈਚਾਲਿਤ sliced = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(ਨਿਵੇਸ਼, begin, end, stride, axes);
    ਵਾਪਸ sliced;
}

ov::Output<ov::Node> process_view_ਨਿਵੇਸ਼_new(ਅਚਲ ਗੰਢ_Context & context, ਪੂਰਨ ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ) {
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = context.get_ਨਿਵੇਸ਼(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ);

    // Check if this input has view inputs
    size_t view_ਨਿਵੇਸ਼_ਮਾਪ = context.get_view_ਨਿਵੇਸ਼_ਮਾਪ(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ);
    ਜੇ (view_ਨਿਵੇਸ਼_ਮਾਪ == 0) {
        // No view inputs, return the input as is
        ਵਾਪਸ ਨਿਵੇਸ਼;
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
    ਸਵੈਚਾਲਿਤ expected_ov_shape = context.get_view_ਨਿਵੇਸ਼_ov_shape(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, 0);
    ਸਵੈਚਾਲਿਤ actual_shape = ਨਿਵੇਸ਼.get_partial_shape();
    ਜੇ (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_ਲੰਮਾਈ() >= actual_shape.rank().get_ਲੰਮਾਈ()) {
        ਅਚਲ int64_t n_actual = actual_shape.rank().get_ਲੰਮਾਈ();
        ਅਚਲ int64_t ਖਿਸਕਾਓ = expected_ov_shape.rank().get_ਲੰਮਾਈ() - n_actual;
        ਤਾਰਕਿਕ shapes_ਮੇਲ = ਸੱਚ;
        ਲਈ (int64_t i = 0; i < ਖਿਸਕਾਓ; ++i) {
            ਜੇ (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_ਲੰਮਾਈ() != 1) {
                shapes_ਮੇਲ = ਝੂਠ;
                ਰੋਕੋ;
            }
        }
        ਲਈ (int64_t i = 0; i < n_actual && shapes_ਮੇਲ; ++i) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ & ਘਾਤਾਂਕ = expected_ov_shape[i + ਖਿਸਕਾਓ];
            ਅਚਲ ਤਾਰਕਿਕ both_dynamic = ਘਾਤਾਂਕ.is_dynamic() && actual_shape[i].is_dynamic();
            ਅਚਲ ਤਾਰਕਿਕ both_static_equal =
                ਘਾਤਾਂਕ.is_static() && actual_shape[i].is_static() && ਘਾਤਾਂਕ == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            ਅਚਲ ਤਾਰਕਿਕ expected_dyn_actual_static = ਘਾਤਾਂਕ.is_dynamic() && actual_shape[i].is_static();
            ਜੇ (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_ਮੇਲ = ਝੂਠ;
                ਰੋਕੋ;
            }
        }
        ਜੇ (shapes_ਮੇਲ) {
            ਵਾਪਸ ਨਿਵੇਸ਼;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    ਜੇ (context.is_static() && view_ਨਿਵੇਸ਼_ਮਾਪ == 1) {
        ਸਵੈਚਾਲਿਤ view_stride_v = context.get_view_ਨਿਵੇਸ਼_stride(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, 0);
        ਸਵੈਚਾਲਿਤ view_src_stride_v = context.get_view_ਨਿਵੇਸ਼_src_stride(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, 0);
        ਸਵੈਚਾਲਿਤ view_ggml_shape = context.get_view_ਨਿਵੇਸ਼_ggml_shape(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, 0);
        ਸਵੈਚਾਲਿਤ view_src_ggml_shape = context.get_view_ਨਿਵੇਸ਼_src_ggml_shape(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, 0);
        ਸਵੈਚਾਲਿਤ view_ਵਿੱਥ = context.get_view_ਨਿਵੇਸ਼_ਵਿੱਥ(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, 0);
        ਸਵੈਚਾਲਿਤ view_src_ਵਿੱਥ = context.get_view_ਨਿਵੇਸ਼_src_ਵਿੱਥ(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, 0);

        size_t ndims = view_ggml_shape.size();
        std::vector<ਪੂਰਨ> diff_dims;
        ਜੇ (view_src_ggml_shape.size() == ndims) {
            ਲਈ (size_t i = 0; i < ndims; ++i) {
                ਜੇ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<ਪੂਰਨ>(i));
                }
            }
        }

        ਜੇ (diff_dims.size() == 1) {
            ਪੂਰਨ ਵੰਡੋ_dim = diff_dims[0];
            int64_t ਅੰਕ_splits = static_cast<int64_t>(view_src_ggml_shape[ਵੰਡੋ_dim]);
            int64_t chunk_ਮਾਪ = static_cast<int64_t>(view_ggml_shape[ਵੰਡੋ_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            ਜੇ (chunk_ਮਾਪ == 1 && ਅੰਕ_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                ਤਾਰਕਿਕ suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                ਜੇ (suffix_ok) {
                    ਲਈ (size_t i = static_cast<size_t>(ਵੰਡੋ_dim) + 1; i < ndims; ++i) {
                        ਜੇ (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = ਝੂਠ;
                            ਰੋਕੋ;
                        }
                    }
                }

                ਜੇ (suffix_ok && view_src_stride_v[ਵੰਡੋ_dim] > 0) {
                    size_t relative_ਵਿੱਥ = view_ਵਿੱਥ >= view_src_ਵਿੱਥ ? view_ਵਿੱਥ - view_src_ਵਿੱਥ : 0;
                    int64_t ਵੰਡੋ_ਸੂਚਕਾਂਕ = static_cast<int64_t>(relative_ਵਿੱਥ / view_src_stride_v[ਵੰਡੋ_dim]);

                    ਜੇ (ਵੰਡੋ_ਸੂਚਕਾਂਕ >= 0 && ਵੰਡੋ_ਸੂਚਕਾਂਕ < ਅੰਕ_splits) {
                        ਸਵੈਚਾਲਿਤ src_ਗੰਢ = ਨਿਵੇਸ਼.get_ਗੰਢ_shared_ptr();
                        std::string rt_ਕੁੰਜੀ = "split_dim_" + std::to_string(ਵੰਡੋ_dim);
                        ਸਵੈਚਾਲਿਤ & rt_info = src_ਗੰਢ->get_rt_info();

                        ਜੇ (rt_info.find(rt_ਕੁੰਜੀ) == rt_info.end()) {
                            ਸਵੈਚਾਲਿਤ axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(ਵੰਡੋ_dim)});
                            ਸਵੈਚਾਲਿਤ ਵੰਡੋ_ਗੰਢ =
                                std::make_shared<ov::op::v1::ਵੰਡੋ_2>(ਨਿਵੇਸ਼, axis_const, static_cast<size_t>(ਅੰਕ_splits));
                            ਵੰਡੋ_ਗੰਢ->ਸੈੱਟ_friendly_name(src_ਗੰਢ->get_friendly_name() + "_split");
                            rt_info[rt_ਕੁੰਜੀ] = ਵੰਡੋ_ਗੰਢ;
                        }

                        ਸਵੈਚਾਲਿਤ ਵੰਡੋ_ਗੰਢ = rt_info[rt_ਕੁੰਜੀ].as<std::shared_ptr<ov::op::v1::ਵੰਡੋ_2>>();
                        ਵਾਪਸ ਵੰਡੋ_ਗੰਢ->output(static_cast<size_t>(ਵੰਡੋ_ਸੂਚਕਾਂਕ));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    ਸਵੈਚਾਲਿਤ process_single_view =
        [](ov::Output<ov::Node> current, size_t view_ਵਿੱਥ, ਅਚਲ std::vector<size_t> & view_stride,
           ਅਚਲ ov::Shape & view_ggml_shape, ਅਚਲ ov::PartialShape & view_ov_shape, ਅਚਲ std::string & view_name,
           size_t view_src_ਵਿੱਥ, ਅਚਲ std::vector<size_t> & view_src_stride, ਅਚਲ ov::Shape & view_src_ggml_shape,
           ਅਚਲ ov::PartialShape & view_src_ov_shape, ਅਚਲ std::string & view_src_name) -> ov::Output<ov::Node> {
        // Stateful execution drops a leading size-1 axis, so `current` can be one rank lower
        // than the ggml shape metadata (view_stride/view_ggml_shape, always GGML_MAX_DIMS
        // entries) assumes. Shift any axis index built from that metadata down by the
        // difference before using it as an OV Slice axis.
        ਅਚਲ ਸਵੈਚਾਲਿਤ current_rank = current.get_partial_shape().rank();
        ਅਚਲ ਪੂਰਨ axis_ਖਿਸਕਾਓ = current_rank.is_static() ?
                                    static_cast<ਪੂਰਨ>(view_stride.size()) - static_cast<ਪੂਰਨ>(current_rank.get_ਲੰਮਾਈ()) :
                                    0;

        ਸਵੈਚਾਲਿਤ build_reshape_ਨਮੂਨਾ = [](ਅਚਲ ov::PartialShape & ਨਿਸ਼ਾਨਾ_ov_shape,
                                        ਅਚਲ ov::Shape & ਨਿਸ਼ਾਨਾ_ggml_shape) -> std::vector<int64_t> {
            ਅਚਲ size_t ndims = ਨਿਸ਼ਾਨਾ_ggml_shape.size();
            std::vector<int64_t> reshape_ਨਮੂਨਾ(ndims);
            size_t dynamic_dims = 0;

            ਜੇ (ਨਿਸ਼ਾਨਾ_ov_shape.rank().is_static() &&
                ਨਿਸ਼ਾਨਾ_ov_shape.rank().get_ਲੰਮਾਈ() == static_cast<int64_t>(ndims)) {
                ਲਈ (size_t i = 0; i < ndims; ++i) {
                    ਜੇ (ਨਿਸ਼ਾਨਾ_ov_shape[i].is_static()) {
                        reshape_ਨਮੂਨਾ[i] = ਨਿਸ਼ਾਨਾ_ov_shape[i].get_ਲੰਮਾਈ();
                    } ਨਹੀਂਤਾਂ {
                        reshape_ਨਮੂਨਾ[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } ਨਹੀਂਤਾਂ {
                dynamic_dims = 2;
            }

            ਜੇ (dynamic_dims > 1) {
                ਲਈ (size_t i = 0; i < ndims; ++i) {
                    reshape_ਨਮੂਨਾ[i] = static_cast<int64_t>(ਨਿਸ਼ਾਨਾ_ggml_shape[i]);
                }
            }

            ਵਾਪਸ reshape_ਨਮੂਨਾ;
        };

        ਸਵੈਚਾਲਿਤ build_ਅਗੇਤਰ_ਪੂਛ_reshape_ਨਮੂਨਾ = [](ਅਚਲ ov::PartialShape & ਨਿਸ਼ਾਨਾ_ov_shape,
                                                    ਅਚਲ ov::Shape & ਨਿਸ਼ਾਨਾ_ggml_shape, size_t ਅਗੇਤਰ_dims,
                                                    int64_t ਪੂਛ_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_ਨਮੂਨਾ(ਅਗੇਤਰ_dims + 1);
            size_t dynamic_dims = 0;

            ਜੇ (ਨਿਸ਼ਾਨਾ_ov_shape.rank().is_static() &&
                ਨਿਸ਼ਾਨਾ_ov_shape.rank().get_ਲੰਮਾਈ() == static_cast<int64_t>(ਨਿਸ਼ਾਨਾ_ggml_shape.size())) {
                ਲਈ (size_t i = 0; i < ਅਗੇਤਰ_dims; ++i) {
                    ਜੇ (ਨਿਸ਼ਾਨਾ_ov_shape[i].is_static()) {
                        reshape_ਨਮੂਨਾ[i] = ਨਿਸ਼ਾਨਾ_ov_shape[i].get_ਲੰਮਾਈ();
                    } ਨਹੀਂਤਾਂ {
                        reshape_ਨਮੂਨਾ[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } ਨਹੀਂਤਾਂ {
                dynamic_dims = 2;
            }

            ਜੇ (dynamic_dims > 1) {
                ਲਈ (size_t i = 0; i < ਅਗੇਤਰ_dims; ++i) {
                    reshape_ਨਮੂਨਾ[i] = static_cast<int64_t>(ਨਿਸ਼ਾਨਾ_ggml_shape[i]);
                }
            }

            reshape_ਨਮੂਨਾ[ਅਗੇਤਰ_dims] = ਪੂਛ_dim;
            ਵਾਪਸ reshape_ਨਮੂਨਾ;
        };

        ਤਾਰਕਿਕ same_stride = view_stride.size() == view_src_stride.size();
        ਜੇ (same_stride) {
            ਲਈ (size_t i = 0; i < view_stride.size(); ++i) {
                ਜੇ (view_stride[i] != view_src_stride[i]) {
                    same_stride = ਝੂਠ;
                    ਰੋਕੋ;
                }
            }
        }

        ਤਾਰਕਿਕ same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        ਜੇ (same_ggml_shape) {
            ਲਈ (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                ਜੇ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = ਝੂਠ;
                    ਰੋਕੋ;
                }
            }
        }

        ਜੇ (same_stride && same_ggml_shape) {
            ਵਾਪਸ current;
        }

        ਜੇ (same_stride) {
            ਅਚਲ size_t relative_ਵਿੱਥ = view_ਵਿੱਥ >= view_src_ਵਿੱਥ ? view_ਵਿੱਥ - view_src_ਵਿੱਥ : 0;
            ਅਚਲ size_t ndims = view_stride.size();

            std::vector<ਪੂਰਨ> diff_dims;
            ਜੇ (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                ਲਈ (size_t i = 0; i < ndims; ++i) {
                    ਜੇ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<ਪੂਰਨ>(i));
                    }
                }
            }

            ਜੇ (diff_dims.size() == 1) {
                ਅਚਲ ਪੂਰਨ ਟੁਕੜਾ_dim = diff_dims[0];
                ਅਚਲ int64_t dim_ਮਾਪ = static_cast<int64_t>(view_src_ggml_shape[ਟੁਕੜਾ_dim]);

                ਜੇ (view_stride[ਟੁਕੜਾ_dim] > 0 && relative_ਵਿੱਥ % view_stride[ਟੁਕੜਾ_dim] == 0) {
                    ਅਚਲ int64_t begin_val = static_cast<int64_t>((relative_ਵਿੱਥ / view_stride[ਟੁਕੜਾ_dim]) %
                                                                   static_cast<size_t>(dim_ਮਾਪ));
                    ਅਚਲ int64_t ਸਿਰਾ_val = begin_val + static_cast<int64_t>(view_ggml_shape[ਟੁਕੜਾ_dim]);

                    ਜੇ (begin_val >= 0 && ਸਿਰਾ_val <= dim_ਮਾਪ) {
                        ਸਵੈਚਾਲਿਤ sliced = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {ਸਿਰਾ_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {ਟੁਕੜਾ_dim - axis_ਖਿਸਕਾਓ}));

                        ਜੇ (view_ov_shape.is_static()) {
                            ਸਵੈਚਾਲਿਤ reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.ਮੰਜ਼ਿਲ_shape()),
                                ਝੂਠ);
                            reshaped->ਸੈੱਟ_friendly_name(view_name);
                            ਵਾਪਸ reshaped;
                        }

                        sliced->ਸੈੱਟ_friendly_name(view_name);
                        ਵਾਪਸ sliced;
                    }
                }

                int64_t ਪੂਛ_src_elems = 1;
                int64_t ਪੂਛ_dst_elems = 1;
                ਲਈ (size_t i = ਟੁਕੜਾ_dim; i < ndims; ++i) {
                    ਪੂਛ_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    ਪੂਛ_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                ਅਚਲ size_t elem_stride = view_stride[ndims - 1];
                int64_t ਪੂਛ_begin = 0;
                ਜੇ (elem_stride > 0) {
                    ਪੂਛ_begin =
                        static_cast<int64_t>((relative_ਵਿੱਥ / elem_stride) % static_cast<size_t>(ਪੂਛ_src_elems));
                }
                ਅਚਲ int64_t ਪੂਛ_ਸਿਰਾ = ਪੂਛ_begin + ਪੂਛ_dst_elems;

                ਜੇ (ਪੂਛ_begin >= 0 && ਪੂਛ_ਸਿਰਾ <= ਪੂਛ_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(ਟੁਕੜਾ_dim);
                    ਲਈ (ਪੂਰਨ i = 0; i < ਟੁਕੜਾ_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(ਪੂਛ_src_elems);
                    ਅਚਲ size_t flat_ndims = flat_shape.size();

                    ਸਵੈਚਾਲਿਤ flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), ਝੂਠ);

                    ਸਵੈਚਾਲਿਤ sliced = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {ਪੂਛ_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {ਪੂਛ_ਸਿਰਾ}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {ਟੁਕੜਾ_dim}));

                    ਜੇ (view_ov_shape.is_static()) {
                        ਸਵੈਚਾਲਿਤ reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.ਮੰਜ਼ਿਲ_shape()),
                            ਝੂਠ);
                        reshaped->ਸੈੱਟ_friendly_name(view_name);
                        ਵਾਪਸ reshaped;
                    }

                    sliced->ਸੈੱਟ_friendly_name(view_name);
                    ਵਾਪਸ sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> ਕਦਮ(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            size_t ਬਚਿਆ_ਵਿੱਥ = relative_ਵਿੱਥ;
            ਲਈ (size_t i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                ਜੇ (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(ਬਚਿਆ_ਵਿੱਥ / view_stride[i]);
                    ਬਚਿਆ_ਵਿੱਥ %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            ਤਾਰਕਿਕ ਵਿੱਚ_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            ਜੇ (ਵਿੱਚ_bounds) {
                ਲਈ (size_t i = 0; i < ndims; ++i) {
                    ਜੇ (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        ਵਿੱਚ_bounds = ਝੂਠ;
                        ਰੋਕੋ;
                    }
                }
            }

            ਜੇ (ਵਿੱਚ_bounds && ਬਚਿਆ_ਵਿੱਥ == 0) {
                ਸਵੈਚਾਲਿਤ sliced = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, ਕਦਮ),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->ਸੈੱਟ_friendly_name(view_name);
                ਵਾਪਸ sliced;
            }
        } ਨਹੀਂਤਾਂ {
            ਤਾਰਕਿਕ same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            ਅਚਲ size_t relative_ਵਿੱਥ = view_ਵਿੱਥ >= view_src_ਵਿੱਥ ? view_ਵਿੱਥ - view_src_ਵਿੱਥ : 0;

            ਜੇ (same_rank) {
                ਅਚਲ size_t ndims = view_ggml_shape.size();
                std::vector<ਪੂਰਨ> diff_dims;
                ਲਈ (size_t i = 0; i < ndims; ++i) {
                    ਜੇ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<ਪੂਰਨ>(i));
                    }
                }

                ਜੇ (diff_dims.size() == 1) {
                    ਅਚਲ size_t ਟੁਕੜਾ_dim = static_cast<size_t>(diff_dims[0]);
                    ਤਾਰਕਿਕ suffix_stride_ਮੇਲ = ਸੱਚ;
                    ਲਈ (size_t i = ਟੁਕੜਾ_dim + 1; i < ndims; ++i) {
                        ਜੇ (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_ਮੇਲ = ਝੂਠ;
                            ਰੋਕੋ;
                        }
                    }

                    ਜੇ (suffix_stride_ਮੇਲ && view_src_stride[ਟੁਕੜਾ_dim] > 0 &&
                        relative_ਵਿੱਥ % view_src_stride[ਟੁਕੜਾ_dim] == 0) {
                        ਅਚਲ int64_t begin_val = static_cast<int64_t>(relative_ਵਿੱਥ / view_src_stride[ਟੁਕੜਾ_dim]);
                        ਅਚਲ int64_t ਸਿਰਾ_val = begin_val + static_cast<int64_t>(view_ggml_shape[ਟੁਕੜਾ_dim]);
                        ਅਚਲ int64_t dim_ਮਾਪ = static_cast<int64_t>(view_src_ggml_shape[ਟੁਕੜਾ_dim]);

                        ਜੇ (begin_val >= 0 && ਸਿਰਾ_val <= dim_ਮਾਪ) {
                            ਸਵੈਚਾਲਿਤ sliced = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {ਸਿਰਾ_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(ਟੁਕੜਾ_dim)}));
                            sliced->ਸੈੱਟ_friendly_name(view_name);
                            ਵਾਪਸ sliced;
                        }
                    }
                }
            }

            size_t view_elems = 1;
            size_t src_elems = 1;
            ਜੇ (same_rank) {
                ਲਈ (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            ਤਾਰਕਿਕ same_ਅੰਕ_elements = same_rank && view_elems == src_elems;

            ਜੇ (same_rank && relative_ਵਿੱਥ == 0 && same_ਅੰਕ_elements) {
                ਸਵੈਚਾਲਿਤ reshape_ਨਮੂਨਾ = build_reshape_ਨਮੂਨਾ(view_ov_shape, view_ggml_shape);

                ਸਵੈਚਾਲਿਤ reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_ਨਮੂਨਾ.size()}, reshape_ਨਮੂਨਾ),
                    ਝੂਠ);
                reshaped->ਸੈੱਟ_friendly_name(view_name);
                ਵਾਪਸ reshaped;
            }

            ਜੇ (same_rank) {
                ਅਚਲ size_t ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                ਤਾਰਕਿਕ is_regular_ਟੁਕੜਾ = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> ਕਦਮ(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                size_t ਬਚਿਆ_ਵਿੱਥ = relative_ਵਿੱਥ;

                ਜੇ (is_regular_ਟੁਕੜਾ) {
                    ਲਈ (size_t i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        ਜੇ (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_ਟੁਕੜਾ = ਝੂਠ;
                            ਰੋਕੋ;
                        }

                        ਕਦਮ[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        ਜੇ (ਕਦਮ[i] <= 0) {
                            is_regular_ਟੁਕੜਾ = ਝੂਠ;
                            ਰੋਕੋ;
                        }

                        begin[i] = static_cast<int64_t>(ਬਚਿਆ_ਵਿੱਥ / view_src_stride[i]);
                        ਬਚਿਆ_ਵਿੱਥ %= view_src_stride[i];

                        ਜੇ (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            ਜਾਰੀ;
                        }

                        end[i] = begin[i] + ਕਦਮ[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        ਜੇ (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_ਟੁਕੜਾ = ਝੂਠ;
                            ਰੋਕੋ;
                        }
                    }
                }

                ਜੇ (is_regular_ਟੁਕੜਾ && ਬਚਿਆ_ਵਿੱਥ == 0) {
                    ਸਵੈਚਾਲਿਤ sliced = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, ਕਦਮ),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->ਸੈੱਟ_friendly_name(view_name);
                    ਵਾਪਸ sliced;
                }

                ਅਚਲ size_t elem_stride = view_src_stride.back();
                ਅਚਲ ਤਾਰਕਿਕ aligned_ਵਿੱਥ = elem_stride > 0 && relative_ਵਿੱਥ % elem_stride == 0;

                ਜੇ (aligned_ਵਿੱਥ) {
                    size_t suffix_ਸ਼ੁਰੂ = 0;
                    size_t expected_stride = elem_stride;
                    ਲਈ (ਪੂਰਨ i = static_cast<ਪੂਰਨ>(ndims) - 1; i >= 0; --i) {
                        ਜੇ (view_stride[i] != expected_stride) {
                            suffix_ਸ਼ੁਰੂ = static_cast<size_t>(i + 1);
                            ਰੋਕੋ;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    size_t ਅਗੇਤਰ_elems = 1;
                    size_t suffix_elems = 1;
                    ਲਈ (size_t i = 0; i < suffix_ਸ਼ੁਰੂ; ++i) {
                        ਅਗੇਤਰ_elems *= view_ggml_shape[i];
                    }
                    ਲਈ (size_t i = suffix_ਸ਼ੁਰੂ; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    ਜੇ (ਅਗੇਤਰ_elems > 0 && src_elems % ਅਗੇਤਰ_elems == 0) {
                        ਅਚਲ size_t src_ਪੂਛ_elems = src_elems / ਅਗੇਤਰ_elems;
                        ਅਚਲ int64_t ਪੂਛ_begin = static_cast<int64_t>(relative_ਵਿੱਥ / elem_stride);
                        ਅਚਲ int64_t ਪੂਛ_ਸਿਰਾ = ਪੂਛ_begin + static_cast<int64_t>(suffix_elems);

                        ਜੇ (ਪੂਛ_begin >= 0 && ਪੂਛ_ਸਿਰਾ <= static_cast<int64_t>(src_ਪੂਛ_elems)) {
                            ਸਵੈਚਾਲਿਤ ਅਗੇਤਰ_ਪੂਛ_ਨਮੂਨਾ = build_ਅਗੇਤਰ_ਪੂਛ_reshape_ਨਮੂਨਾ(
                                view_ov_shape, view_ggml_shape, suffix_ਸ਼ੁਰੂ, static_cast<int64_t>(src_ਪੂਛ_elems));

                            ਸਵੈਚਾਲਿਤ ਅਗੇਤਰ_ਪੂਛ = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {ਅਗੇਤਰ_ਪੂਛ_ਨਮੂਨਾ.size()},
                                                             ਅਗੇਤਰ_ਪੂਛ_ਨਮੂਨਾ),
                                ਝੂਠ);

                            ov::Output<ov::Node> selected = ਅਗੇਤਰ_ਪੂਛ;
                            ਜੇ (ਪੂਛ_begin != 0 || ਪੂਛ_ਸਿਰਾ != static_cast<int64_t>(src_ਪੂਛ_elems)) {
                                selected = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(
                                    ਅਗੇਤਰ_ਪੂਛ, ov::op::v0::Constant::create(ov::element::i64, {1}, {ਪੂਛ_begin}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {ਪੂਛ_ਸਿਰਾ}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {static_cast<int64_t>(suffix_ਸ਼ੁਰੂ)}));
                            }

                            ਸਵੈਚਾਲਿਤ reshape_ਨਮੂਨਾ = build_reshape_ਨਮੂਨਾ(view_ov_shape, view_ggml_shape);
                            ਸਵੈਚਾਲਿਤ reshaped = std::make_shared<ov::op::v1::Reshape>(
                                selected,
                                ov::op::v0::Constant::create(ov::element::i64, {reshape_ਨਮੂਨਾ.size()},
                                                             reshape_ਨਮੂਨਾ),
                                ਝੂਠ);
                            reshaped->ਸੈੱਟ_friendly_name(view_name);
                            ਵਾਪਸ reshaped;
                        }
                    }
                }
            }

            ਵਾਪਸ current;
        }

        (ਖਾਲੀ) view_name;
        (ਖਾਲੀ) view_src_ov_shape;
        (ਖਾਲੀ) view_src_name;

        ਵਾਪਸ current;
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
        ਸਵੈਚਾਲਿਤ innermost_ਵਿੱਥ = context.get_view_ਨਿਵੇਸ਼_ਵਿੱਥ(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, 0);
        ਸਵੈਚਾਲਿਤ innermost_src_ਵਿੱਥ = context.get_view_ਨਿਵੇਸ਼_src_ਵਿੱਥ(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, 0);
        ਸਵੈਚਾਲਿਤ innermost_shape = context.get_view_ਨਿਵੇਸ਼_ggml_shape(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, 0);
        ਸਵੈਚਾਲਿਤ innermost_src_shape = context.get_view_ਨਿਵੇਸ਼_src_ggml_shape(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, 0);
        ਜੇ (innermost_ਵਿੱਥ != innermost_src_ਵਿੱਥ && innermost_shape == innermost_src_shape) {
            size_t ਜੜ੍ਹ_view_idx = view_ਨਿਵੇਸ਼_ਮਾਪ - 1;
            ਸਵੈਚਾਲਿਤ ਜੜ੍ਹ_ggml_shape = context.get_view_ਨਿਵੇਸ਼_src_ggml_shape(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, ਜੜ੍ਹ_view_idx);
            ਸਵੈਚਾਲਿਤ ਜੜ੍ਹ_stride = context.get_view_ਨਿਵੇਸ਼_src_stride(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, ਜੜ੍ਹ_view_idx);
            ਸਵੈਚਾਲਿਤ ਜੜ੍ਹ_ਵਿੱਥ = context.get_view_ਨਿਵੇਸ਼_src_ਵਿੱਥ(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, ਜੜ੍ਹ_view_idx);
            ਸਵੈਚਾਲਿਤ ਜੜ੍ਹ_ov_shape = context.get_view_ਨਿਵੇਸ਼_src_ov_shape(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, ਜੜ੍ਹ_view_idx);
            ਸਵੈਚਾਲਿਤ ਜੜ੍ਹ_name = context.get_view_ਨਿਵੇਸ਼_src_name(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, ਜੜ੍ਹ_view_idx);
            ਸਵੈਚਾਲਿਤ innermost_stride = context.get_view_ਨਿਵੇਸ਼_stride(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, 0);
            ਸਵੈਚਾਲਿਤ innermost_ov_shape = context.get_view_ਨਿਵੇਸ਼_ov_shape(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, 0);
            ਸਵੈਚਾਲਿਤ innermost_name = context.get_view_ਨਿਵੇਸ਼_name(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, 0);

            ਵਾਪਸ process_single_view(ਨਿਵੇਸ਼, innermost_ਵਿੱਥ, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, ਜੜ੍ਹ_ਵਿੱਥ, ਜੜ੍ਹ_stride, ਜੜ੍ਹ_ggml_shape, ਜੜ੍ਹ_ov_shape,
                                       ਜੜ੍ਹ_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = ਨਿਵੇਸ਼;

    // Process each view in reverse order (from base to current)
    ਲਈ (ਪੂਰਨ view_idx = view_ਨਿਵੇਸ਼_ਮਾਪ - 1; view_idx >= 0; view_idx--) {
        ਸਵੈਚਾਲਿਤ view_ਵਿੱਥ = context.get_view_ਨਿਵੇਸ਼_ਵਿੱਥ(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, view_idx);
        ਸਵੈਚਾਲਿਤ view_stride = context.get_view_ਨਿਵੇਸ਼_stride(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, view_idx);
        ਸਵੈਚਾਲਿਤ view_ggml_shape = context.get_view_ਨਿਵੇਸ਼_ggml_shape(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, view_idx);
        ਸਵੈਚਾਲਿਤ view_ov_shape = context.get_view_ਨਿਵੇਸ਼_ov_shape(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, view_idx);
        ਸਵੈਚਾਲਿਤ view_name = context.get_view_ਨਿਵੇਸ਼_name(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        ਸਵੈਚਾਲਿਤ view_src_ਵਿੱਥ = context.get_view_ਨਿਵੇਸ਼_src_ਵਿੱਥ(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, view_idx);
        ਸਵੈਚਾਲਿਤ view_src_stride = context.get_view_ਨਿਵੇਸ਼_src_stride(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, view_idx);
        ਸਵੈਚਾਲਿਤ view_src_ggml_shape = context.get_view_ਨਿਵੇਸ਼_src_ggml_shape(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, view_idx);
        ਸਵੈਚਾਲਿਤ view_src_ov_shape = context.get_view_ਨਿਵੇਸ਼_src_ov_shape(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, view_idx);
        ਸਵੈਚਾਲਿਤ view_src_name = context.get_view_ਨਿਵੇਸ਼_src_name(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ, view_idx);
        // print source view info
        // std::cout << "View " << view_idx << ": source name = " << view_src_name
        //           << ", source offset = " << view_src_offset << ", source stride = [" << view_src_stride[0] << ","
        //           << view_src_stride[1] << "," << view_src_stride[2] << "," << view_src_stride[3]
        //           << "], source ggml shape = [" << view_src_ggml_shape[0] << "," << view_src_ggml_shape[1] << ","
        //           << view_src_ggml_shape[2] << "," << view_src_ggml_shape[3]
        //           << "], source ov shape = " << view_src_ov_shape << std::endl;

        current = process_single_view(current, view_ਵਿੱਥ, view_stride, view_ggml_shape, view_ov_shape, view_name,
                                      view_src_ਵਿੱਥ, view_src_stride, view_src_ggml_shape, view_src_ov_shape,
                                      view_src_name);
    }

    ਵਾਪਸ current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
