#ضمن "utils.h"

#ضمن <cmath>
#ضمن <cstddef>
#ضمن <ctime>
#ضمن <memory>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/clamp.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/cos.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/maximum.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/sin.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/split.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/subtract.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <string>

نطاق ov {
نطاق frontend {
نطاق ggml {

فراغ عدد_inputs_check(ثابت گرہ_Context & context, size_t min_inputs, size_t max_inputs) {
    خودکار داخلہ_ناپ = context.get_داخلہ_ناپ();
    FRONT_سرا_OP_CONVERSION_CHECK(داخلہ_ناپ >= min_inputs, "Got less inputs than expected");
    FRONT_سرا_OP_CONVERSION_CHECK(داخلہ_ناپ <= max_inputs, "Got more inputs than expected");
}

صحيح non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    صحيح dim = nb.size() - 1;
    size_t بائٹاں = nb[dim];
    لكل (صحيح i = dim; i > 0; i--) {
        بائٹاں *= ne[i];
        إذا (بائٹاں != nb[i - 1]) {
            أرجع i;
        }
    }
    أرجع 0;
}

std::shared_ptr<ov::Node> get_dimensions(ثابت std::shared_ptr<ov::op::v3::Shape_دا> & shape,
                                         ثابت std::vector<صحيح> & dims) {
    ورتوں نطاق ov::op;
    ثابت خودکار zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ثابت خودکار dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    أرجع std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(ثابت std::shared_ptr<ov::Node> & گرہ, ثابت std::vector<صحيح> & dims) {
    أرجع get_dimensions(std::make_shared<ov::op::v3::Shape_دا>(گرہ), dims);
}

Output_سمتیہ rename_outputs_with_suffix(ثابت Output_سمتیہ & outputs, ثابت std::string & suffix) {
    لكل (ثابت خودکار & output : outputs) {
        خودکار گرہ = output.get_گرہ_shared_ptr();
        std::string name = گرہ->get_friendly_name();
        name += "_";
        name += suffix;
        گرہ->سیٹ_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    أرجع outputs;
}

نطاق {
ov::Output<ov::Node> rope_yarn_ramp_mix(صحيح عدد_elements, ثابت عائم corr_dims[2], عائم ext_ضارب) {
    std::vector<عائم> dim_ids_vec(عدد_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    خودکار dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) عدد_elements}, dim_ids_vec);
    خودکار corr_کم = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    خودکار corr_بلند = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    خودکار denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::گھٹاؤ>(corr_بلند, corr_کم),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    خودکار ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::گھٹاؤ>(dim_ids, corr_کم), denom);
    خودکار ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    خودکار one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    خودکار ramp_inverted = std::make_shared<ov::op::v1::گھٹاؤ>(one, ramp_clamped);
    خودکار ext_ضارب_گرہ = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_ضارب});
    خودکار ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_ضارب_گرہ);
    أرجع ramp_mix;
}

عائم ggml_rope_yarn_corr_dim(صحيح n_dims, صحيح n_ctx_orig, عائم n_rot, عائم ادھار) {
#جے_بنا_تعریف M_PI
#    define M_PI 3.14159265358979323846
#جے_انت
    أرجع n_dims * logf(n_ctx_orig / (n_rot * 2 * (عائم) M_PI)) / (2 * logf(ادھار));
}

فراغ ggml_rope_yarn_corr_dims(صحيح n_dims,
                              صحيح n_ctx_orig,
                              عائم freq_ادھار,
                              عائم beta_fast,
                              عائم beta_slow,
                              عائم dims[2]) {
    عائم start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_ادھار));
    عائم end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_ادھار));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<عائم>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_بھار,
                                                           صحيح mode,
                                                           منطقي stateful,
                                                           int64_t سر_dim) {
    constexpr صحيح TYPE_IMROPE = 2;
    constexpr صحيح TYPE_VISION = 3;
    constexpr صحيح TYPE_MROPE = 4;

    ثابت منطقي is_imrope = (mode == TYPE_IMROPE);
    ثابت منطقي is_vision = (mode == TYPE_VISION);
    ثابت منطقي is_mrope = (mode == TYPE_MROPE);
    ثابت منطقي is_multi = is_imrope || is_vision || is_mrope;

    إذا (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        خودکار pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } وإلا إذا (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        خودکار pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, سچ);
        خودکار pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } وإلا {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        خودکار pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    عائم freq_ادھار;
    عائم freq_scale;
    عائم ext_ضارب;
    عائم attn_ضارب;
    عائم beta_fast;
    عائم beta_slow;
    ثابت صحيح n_dims = rope_params[1];
    ثابت size_t n_dims_half = n_dims >> 1;
    ثابت صحيح n_ctx_orig = rope_params[4];
    حافظہ_نقل(&freq_ادھار, rope_params + 5, حجم(عائم));
    حافظہ_نقل(&freq_scale, rope_params + 6, حجم(عائم));
    حافظہ_نقل(&ext_ضارب, rope_params + 7, حجم(عائم));
    حافظہ_نقل(&attn_ضارب, rope_params + 8, حجم(عائم));
    حافظہ_نقل(&beta_fast, rope_params + 9, حجم(عائم));
    حافظہ_نقل(&beta_slow, rope_params + 10, حجم(عائم));

    ثابت عائم theta_scale = powf(freq_ادھار, -2.0f / n_dims);

    Output<Node> theta;
    عائم mscale = attn_ضارب;
    إذا (is_multi) {
        ثابت size_t عدد_جوڑے = is_vision ? (n_dims > 0 ? (size_t) n_dims : (size_t) (سر_dim / 2)) : n_dims_half;
        صحيح sections[4];
        حافظہ_نقل(sections, rope_params + 11, حجم(صحيح) * 4);
        صحيح sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        إذا (sect_dims <= 0) {
            sect_dims = عدد_جوڑے;
        }
        صحيح sec_w = sections[0] + sections[1];
        صحيح sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(عدد_جوڑے);
        std::vector<عائم> ضارب(عدد_جوڑے);
        لكل (size_t j = 0; j < عدد_جوڑے; j++) {
            صحيح sector = j % sect_dims;
            صحيح plane = 0;
            إذا (is_imrope) {
                إذا (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } وإلا إذا (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } وإلا إذا (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } وإلا {
                    plane = 3;
                }
                ضارب[j] = std::قوت(theta_scale, j);
            } وإلا إذا (is_vision) {
                صحيح p_idx = 0;
                إذا (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } وإلا إذا (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } وإلا إذا (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } وإلا {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                ضارب[j] = std::قوت(theta_scale, p_idx);
            } وإلا {
                إذا (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } وإلا إذا (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } وإلا إذا (sector >= sec_e) {
                    plane = 3;
                } وإلا {
                    plane = 0;
                }
                ضارب[j] = std::قوت(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        خودکار gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{عدد_جوڑے}, gather_indices);
        خودکار gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> ضارب_گرہ =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{عدد_جوڑے}, ضارب);
        إذا (rope_freqs_بھار) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::ٹکڑا>(
                rope_freqs_بھار,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) عدد_جوڑے}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_بھار->get_output_partial_shape(0).rank().get_لمائی() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) عدد_جوڑے}),
                خطأ);
            ضارب_گرہ = std::make_shared<ov::op::v1::Divide>(ضارب_گرہ, rope_factors);
        }
        خودکار theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, ضارب_گرہ);
        خودکار theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        إذا (ext_ضارب == 0.0f) {
            theta = theta_interp;
        } وإلا {
            عائم corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ادھار, beta_fast, beta_slow, corr_dims);
            خودکار ramp_mix = rope_yarn_ramp_mix(عدد_جوڑے, corr_dims, ext_ضارب);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            خودکار one_minus_ramp = std::make_shared<ov::op::v1::گھٹاؤ>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::جوڑو>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::لوگارتھم(1.0f / freq_scale));
        }
    } وإلا {
        std::vector<عائم> ضارب(n_dims_half);
        Output<Node> freq_factors;
        عائم corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ادھار, beta_fast, beta_slow, corr_dims);
        ضارب[0] = 1.0f;
        لكل (size_t i = 1; i < ضارب.size(); i++) {
            ضارب[i] = theta_scale * ضارب[i - 1];
        }
        إذا (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, ضارب.size()}, ضارب);
        } وإلا {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, ضارب.size()}, ضارب);
        }
        إذا (rope_freqs_بھار) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::ٹکڑا>(
                rope_freqs_بھار,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_بھار->get_output_partial_shape(0).rank().get_لمائی() - 1}));
            إذا (stateful) {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {3}, {(int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    خطأ);
            } وإلا {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    خطأ);
            }
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_factors);
        }

        خودکار theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        خودکار theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        إذا (ext_ضارب == 0.0f) {
            theta = theta_interp;
        } وإلا {
            خودکار ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_ضارب);
            Output<Node> one;
            إذا (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } وإلا {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            }
            خودکار one_minus_ramp = std::make_shared<ov::op::v1::گھٹاؤ>(one, ramp_mix);

            theta =
                std::make_shared<ov::op::v1::جوڑو>(std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                                                  std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::لوگارتھم(1.0f / freq_scale));
        }
    }

    Output<Node> cos_theta = std::make_shared<ov::op::v0::Cos>(theta);
    Output<Node> sin_theta = std::make_shared<ov::op::v0::Sin>(theta);

    إذا (mscale != 1.0f) {
        خودکار mscale_گرہ = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_گرہ);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_گرہ);
    }

    أرجع std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_داخلہ(ثابت گرہ_Context & context, صحيح داخلہ_اشاریہ, صحيح ٹکڑا_len, صحيح axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    خودکار داخلہ = context.get_داخلہ(داخلہ_اشاریہ);
    خودکار * op_params = (size_t *) context.get_داخلہ_op_params(داخلہ_اشاریہ);
    خودکار src_stride = context.get_داخلہ_stride(داخلہ_اشاریہ);

    int64_t ٹکڑا_شروع = op_params[0] / src_stride[3];
    إذا (ٹکڑا_len == 0) {
        ٹکڑا_len = context.get_داخلہ_shape(داخلہ_اشاریہ)[3].get_لمائی();
    }
    int64_t ٹکڑا_سرا = ٹکڑا_شروع + ٹکڑا_len;

    خودکار begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {ٹکڑا_شروع});
    خودکار end = ov::op::v0::Constant::create(ov::element::i64, {1}, {ٹکڑا_سرا});
    خودکار stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    إذا (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } وإلا {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    خودکار sliced = std::make_shared<ov::op::v8::ٹکڑا>(داخلہ, begin, end, stride, axes);
    أرجع sliced;
}

ov::Output<ov::Node> process_view_داخلہ_new(ثابت گرہ_Context & context, صحيح داخلہ_اشاریہ) {
    خودکار داخلہ = context.get_داخلہ(داخلہ_اشاریہ);

    // Check if this input has view inputs
    size_t view_داخلہ_ناپ = context.get_view_داخلہ_ناپ(داخلہ_اشاریہ);
    إذا (view_داخلہ_ناپ == 0) {
        // No view inputs, return the input as is
        أرجع داخلہ;
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
    خودکار expected_ov_shape = context.get_view_داخلہ_ov_shape(داخلہ_اشاریہ, 0);
    خودکار actual_shape = داخلہ.get_partial_shape();
    إذا (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_لمائی() >= actual_shape.rank().get_لمائی()) {
        ثابت int64_t n_actual = actual_shape.rank().get_لمائی();
        ثابت int64_t کھسکاؤ = expected_ov_shape.rank().get_لمائی() - n_actual;
        منطقي shapes_میل = سچ;
        لكل (int64_t i = 0; i < کھسکاؤ; ++i) {
            إذا (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_لمائی() != 1) {
                shapes_میل = خطأ;
                اكسر;
            }
        }
        لكل (int64_t i = 0; i < n_actual && shapes_میل; ++i) {
            ثابت خودکار & قوت_نما = expected_ov_shape[i + کھسکاؤ];
            ثابت منطقي both_dynamic = قوت_نما.is_dynamic() && actual_shape[i].is_dynamic();
            ثابت منطقي both_static_equal =
                قوت_نما.is_static() && actual_shape[i].is_static() && قوت_نما == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            ثابت منطقي expected_dyn_actual_static = قوت_نما.is_dynamic() && actual_shape[i].is_static();
            إذا (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_میل = خطأ;
                اكسر;
            }
        }
        إذا (shapes_میل) {
            أرجع داخلہ;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    إذا (context.is_static() && view_داخلہ_ناپ == 1) {
        خودکار view_stride_v = context.get_view_داخلہ_stride(داخلہ_اشاریہ, 0);
        خودکار view_src_stride_v = context.get_view_داخلہ_src_stride(داخلہ_اشاریہ, 0);
        خودکار view_ggml_shape = context.get_view_داخلہ_ggml_shape(داخلہ_اشاریہ, 0);
        خودکار view_src_ggml_shape = context.get_view_داخلہ_src_ggml_shape(داخلہ_اشاریہ, 0);
        خودکار view_وتھ = context.get_view_داخلہ_وتھ(داخلہ_اشاریہ, 0);
        خودکار view_src_وتھ = context.get_view_داخلہ_src_وتھ(داخلہ_اشاریہ, 0);

        size_t ndims = view_ggml_shape.size();
        std::vector<صحيح> diff_dims;
        إذا (view_src_ggml_shape.size() == ndims) {
            لكل (size_t i = 0; i < ndims; ++i) {
                إذا (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<صحيح>(i));
                }
            }
        }

        إذا (diff_dims.size() == 1) {
            صحيح ونڈو_dim = diff_dims[0];
            int64_t عدد_splits = static_cast<int64_t>(view_src_ggml_shape[ونڈو_dim]);
            int64_t chunk_ناپ = static_cast<int64_t>(view_ggml_shape[ونڈو_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            إذا (chunk_ناپ == 1 && عدد_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                منطقي suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                إذا (suffix_ok) {
                    لكل (size_t i = static_cast<size_t>(ونڈو_dim) + 1; i < ndims; ++i) {
                        إذا (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = خطأ;
                            اكسر;
                        }
                    }
                }

                إذا (suffix_ok && view_src_stride_v[ونڈو_dim] > 0) {
                    size_t relative_وتھ = view_وتھ >= view_src_وتھ ? view_وتھ - view_src_وتھ : 0;
                    int64_t ونڈو_اشاریہ = static_cast<int64_t>(relative_وتھ / view_src_stride_v[ونڈو_dim]);

                    إذا (ونڈو_اشاریہ >= 0 && ونڈو_اشاریہ < عدد_splits) {
                        خودکار src_گرہ = داخلہ.get_گرہ_shared_ptr();
                        std::string rt_کلید = "split_dim_" + std::to_string(ونڈو_dim);
                        خودکار & rt_info = src_گرہ->get_rt_info();

                        إذا (rt_info.find(rt_کلید) == rt_info.end()) {
                            خودکار axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(ونڈو_dim)});
                            خودکار ونڈو_گرہ =
                                std::make_shared<ov::op::v1::ونڈو_2>(داخلہ, axis_const, static_cast<size_t>(عدد_splits));
                            ونڈو_گرہ->سیٹ_friendly_name(src_گرہ->get_friendly_name() + "_split");
                            rt_info[rt_کلید] = ونڈو_گرہ;
                        }

                        خودکار ونڈو_گرہ = rt_info[rt_کلید].as<std::shared_ptr<ov::op::v1::ونڈو_2>>();
                        أرجع ونڈو_گرہ->output(static_cast<size_t>(ونڈو_اشاریہ));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    خودکار process_single_view =
        [](ov::Output<ov::Node> current, size_t view_وتھ, ثابت std::vector<size_t> & view_stride,
           ثابت ov::Shape & view_ggml_shape, ثابت ov::PartialShape & view_ov_shape, ثابت std::string & view_name,
           size_t view_src_وتھ, ثابت std::vector<size_t> & view_src_stride, ثابت ov::Shape & view_src_ggml_shape,
           ثابت ov::PartialShape & view_src_ov_shape, ثابت std::string & view_src_name) -> ov::Output<ov::Node> {
        // Stateful execution drops a leading size-1 axis, so `current` can be one rank lower
        // than the ggml shape metadata (view_stride/view_ggml_shape, always GGML_MAX_DIMS
        // entries) assumes. Shift any axis index built from that metadata down by the
        // difference before using it as an OV Slice axis.
        ثابت خودکار current_rank = current.get_partial_shape().rank();
        ثابت صحيح axis_کھسکاؤ = current_rank.is_static() ?
                                    static_cast<صحيح>(view_stride.size()) - static_cast<صحيح>(current_rank.get_لمائی()) :
                                    0;

        خودکار build_reshape_نمونہ = [](ثابت ov::PartialShape & نشانہ_ov_shape,
                                        ثابت ov::Shape & نشانہ_ggml_shape) -> std::vector<int64_t> {
            ثابت size_t ndims = نشانہ_ggml_shape.size();
            std::vector<int64_t> reshape_نمونہ(ndims);
            size_t dynamic_dims = 0;

            إذا (نشانہ_ov_shape.rank().is_static() &&
                نشانہ_ov_shape.rank().get_لمائی() == static_cast<int64_t>(ndims)) {
                لكل (size_t i = 0; i < ndims; ++i) {
                    إذا (نشانہ_ov_shape[i].is_static()) {
                        reshape_نمونہ[i] = نشانہ_ov_shape[i].get_لمائی();
                    } وإلا {
                        reshape_نمونہ[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } وإلا {
                dynamic_dims = 2;
            }

            إذا (dynamic_dims > 1) {
                لكل (size_t i = 0; i < ndims; ++i) {
                    reshape_نمونہ[i] = static_cast<int64_t>(نشانہ_ggml_shape[i]);
                }
            }

            أرجع reshape_نمونہ;
        };

        خودکار build_اگیتر_پوچھ_reshape_نمونہ = [](ثابت ov::PartialShape & نشانہ_ov_shape,
                                                    ثابت ov::Shape & نشانہ_ggml_shape, size_t اگیتر_dims,
                                                    int64_t پوچھ_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_نمونہ(اگیتر_dims + 1);
            size_t dynamic_dims = 0;

            إذا (نشانہ_ov_shape.rank().is_static() &&
                نشانہ_ov_shape.rank().get_لمائی() == static_cast<int64_t>(نشانہ_ggml_shape.size())) {
                لكل (size_t i = 0; i < اگیتر_dims; ++i) {
                    إذا (نشانہ_ov_shape[i].is_static()) {
                        reshape_نمونہ[i] = نشانہ_ov_shape[i].get_لمائی();
                    } وإلا {
                        reshape_نمونہ[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } وإلا {
                dynamic_dims = 2;
            }

            إذا (dynamic_dims > 1) {
                لكل (size_t i = 0; i < اگیتر_dims; ++i) {
                    reshape_نمونہ[i] = static_cast<int64_t>(نشانہ_ggml_shape[i]);
                }
            }

            reshape_نمونہ[اگیتر_dims] = پوچھ_dim;
            أرجع reshape_نمونہ;
        };

        منطقي same_stride = view_stride.size() == view_src_stride.size();
        إذا (same_stride) {
            لكل (size_t i = 0; i < view_stride.size(); ++i) {
                إذا (view_stride[i] != view_src_stride[i]) {
                    same_stride = خطأ;
                    اكسر;
                }
            }
        }

        منطقي same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        إذا (same_ggml_shape) {
            لكل (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                إذا (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = خطأ;
                    اكسر;
                }
            }
        }

        إذا (same_stride && same_ggml_shape) {
            أرجع current;
        }

        إذا (same_stride) {
            ثابت size_t relative_وتھ = view_وتھ >= view_src_وتھ ? view_وتھ - view_src_وتھ : 0;
            ثابت size_t ndims = view_stride.size();

            std::vector<صحيح> diff_dims;
            إذا (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                لكل (size_t i = 0; i < ndims; ++i) {
                    إذا (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<صحيح>(i));
                    }
                }
            }

            إذا (diff_dims.size() == 1) {
                ثابت صحيح ٹکڑا_dim = diff_dims[0];
                ثابت int64_t dim_ناپ = static_cast<int64_t>(view_src_ggml_shape[ٹکڑا_dim]);

                إذا (view_stride[ٹکڑا_dim] > 0 && relative_وتھ % view_stride[ٹکڑا_dim] == 0) {
                    ثابت int64_t begin_val = static_cast<int64_t>((relative_وتھ / view_stride[ٹکڑا_dim]) %
                                                                   static_cast<size_t>(dim_ناپ));
                    ثابت int64_t سرا_val = begin_val + static_cast<int64_t>(view_ggml_shape[ٹکڑا_dim]);

                    إذا (begin_val >= 0 && سرا_val <= dim_ناپ) {
                        خودکار sliced = std::make_shared<ov::op::v8::ٹکڑا>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {سرا_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {ٹکڑا_dim - axis_کھسکاؤ}));

                        إذا (view_ov_shape.is_static()) {
                            خودکار reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.منزل_shape()),
                                خطأ);
                            reshaped->سیٹ_friendly_name(view_name);
                            أرجع reshaped;
                        }

                        sliced->سیٹ_friendly_name(view_name);
                        أرجع sliced;
                    }
                }

                int64_t پوچھ_src_elems = 1;
                int64_t پوچھ_dst_elems = 1;
                لكل (size_t i = ٹکڑا_dim; i < ndims; ++i) {
                    پوچھ_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    پوچھ_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                ثابت size_t elem_stride = view_stride[ndims - 1];
                int64_t پوچھ_begin = 0;
                إذا (elem_stride > 0) {
                    پوچھ_begin =
                        static_cast<int64_t>((relative_وتھ / elem_stride) % static_cast<size_t>(پوچھ_src_elems));
                }
                ثابت int64_t پوچھ_سرا = پوچھ_begin + پوچھ_dst_elems;

                إذا (پوچھ_begin >= 0 && پوچھ_سرا <= پوچھ_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(ٹکڑا_dim);
                    لكل (صحيح i = 0; i < ٹکڑا_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(پوچھ_src_elems);
                    ثابت size_t flat_ndims = flat_shape.size();

                    خودکار flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), خطأ);

                    خودکار sliced = std::make_shared<ov::op::v8::ٹکڑا>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {پوچھ_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {پوچھ_سرا}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {ٹکڑا_dim}));

                    إذا (view_ov_shape.is_static()) {
                        خودکار reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.منزل_shape()),
                            خطأ);
                        reshaped->سیٹ_friendly_name(view_name);
                        أرجع reshaped;
                    }

                    sliced->سیٹ_friendly_name(view_name);
                    أرجع sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> قدم(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            size_t بچیا_وتھ = relative_وتھ;
            لكل (size_t i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                إذا (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(بچیا_وتھ / view_stride[i]);
                    بچیا_وتھ %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            منطقي وچ_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            إذا (وچ_bounds) {
                لكل (size_t i = 0; i < ndims; ++i) {
                    إذا (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        وچ_bounds = خطأ;
                        اكسر;
                    }
                }
            }

            إذا (وچ_bounds && بچیا_وتھ == 0) {
                خودکار sliced = std::make_shared<ov::op::v8::ٹکڑا>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, قدم),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->سیٹ_friendly_name(view_name);
                أرجع sliced;
            }
        } وإلا {
            منطقي same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            ثابت size_t relative_وتھ = view_وتھ >= view_src_وتھ ? view_وتھ - view_src_وتھ : 0;

            إذا (same_rank) {
                ثابت size_t ndims = view_ggml_shape.size();
                std::vector<صحيح> diff_dims;
                لكل (size_t i = 0; i < ndims; ++i) {
                    إذا (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<صحيح>(i));
                    }
                }

                إذا (diff_dims.size() == 1) {
                    ثابت size_t ٹکڑا_dim = static_cast<size_t>(diff_dims[0]);
                    منطقي suffix_stride_میل = سچ;
                    لكل (size_t i = ٹکڑا_dim + 1; i < ndims; ++i) {
                        إذا (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_میل = خطأ;
                            اكسر;
                        }
                    }

                    إذا (suffix_stride_میل && view_src_stride[ٹکڑا_dim] > 0 &&
                        relative_وتھ % view_src_stride[ٹکڑا_dim] == 0) {
                        ثابت int64_t begin_val = static_cast<int64_t>(relative_وتھ / view_src_stride[ٹکڑا_dim]);
                        ثابت int64_t سرا_val = begin_val + static_cast<int64_t>(view_ggml_shape[ٹکڑا_dim]);
                        ثابت int64_t dim_ناپ = static_cast<int64_t>(view_src_ggml_shape[ٹکڑا_dim]);

                        إذا (begin_val >= 0 && سرا_val <= dim_ناپ) {
                            خودکار sliced = std::make_shared<ov::op::v8::ٹکڑا>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {سرا_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(ٹکڑا_dim)}));
                            sliced->سیٹ_friendly_name(view_name);
                            أرجع sliced;
                        }
                    }
                }
            }

            size_t view_elems = 1;
            size_t src_elems = 1;
            إذا (same_rank) {
                لكل (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            منطقي same_عدد_elements = same_rank && view_elems == src_elems;

            إذا (same_rank && relative_وتھ == 0 && same_عدد_elements) {
                خودکار reshape_نمونہ = build_reshape_نمونہ(view_ov_shape, view_ggml_shape);

                خودکار reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_نمونہ.size()}, reshape_نمونہ),
                    خطأ);
                reshaped->سیٹ_friendly_name(view_name);
                أرجع reshaped;
            }

            إذا (same_rank) {
                ثابت size_t ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                منطقي is_regular_ٹکڑا = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> قدم(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                size_t بچیا_وتھ = relative_وتھ;

                إذا (is_regular_ٹکڑا) {
                    لكل (size_t i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        إذا (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_ٹکڑا = خطأ;
                            اكسر;
                        }

                        قدم[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        إذا (قدم[i] <= 0) {
                            is_regular_ٹکڑا = خطأ;
                            اكسر;
                        }

                        begin[i] = static_cast<int64_t>(بچیا_وتھ / view_src_stride[i]);
                        بچیا_وتھ %= view_src_stride[i];

                        إذا (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            استمر;
                        }

                        end[i] = begin[i] + قدم[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        إذا (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_ٹکڑا = خطأ;
                            اكسر;
                        }
                    }
                }

                إذا (is_regular_ٹکڑا && بچیا_وتھ == 0) {
                    خودکار sliced = std::make_shared<ov::op::v8::ٹکڑا>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, قدم),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->سیٹ_friendly_name(view_name);
                    أرجع sliced;
                }

                ثابت size_t elem_stride = view_src_stride.back();
                ثابت منطقي aligned_وتھ = elem_stride > 0 && relative_وتھ % elem_stride == 0;

                إذا (aligned_وتھ) {
                    size_t suffix_شروع = 0;
                    size_t expected_stride = elem_stride;
                    لكل (صحيح i = static_cast<صحيح>(ndims) - 1; i >= 0; --i) {
                        إذا (view_stride[i] != expected_stride) {
                            suffix_شروع = static_cast<size_t>(i + 1);
                            اكسر;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    size_t اگیتر_elems = 1;
                    size_t suffix_elems = 1;
                    لكل (size_t i = 0; i < suffix_شروع; ++i) {
                        اگیتر_elems *= view_ggml_shape[i];
                    }
                    لكل (size_t i = suffix_شروع; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    إذا (اگیتر_elems > 0 && src_elems % اگیتر_elems == 0) {
                        ثابت size_t src_پوچھ_elems = src_elems / اگیتر_elems;
                        ثابت int64_t پوچھ_begin = static_cast<int64_t>(relative_وتھ / elem_stride);
                        ثابت int64_t پوچھ_سرا = پوچھ_begin + static_cast<int64_t>(suffix_elems);

                        إذا (پوچھ_begin >= 0 && پوچھ_سرا <= static_cast<int64_t>(src_پوچھ_elems)) {
                            خودکار اگیتر_پوچھ_نمونہ = build_اگیتر_پوچھ_reshape_نمونہ(
                                view_ov_shape, view_ggml_shape, suffix_شروع, static_cast<int64_t>(src_پوچھ_elems));

                            خودکار اگیتر_پوچھ = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {اگیتر_پوچھ_نمونہ.size()},
                                                             اگیتر_پوچھ_نمونہ),
                                خطأ);

                            ov::Output<ov::Node> selected = اگیتر_پوچھ;
                            إذا (پوچھ_begin != 0 || پوچھ_سرا != static_cast<int64_t>(src_پوچھ_elems)) {
                                selected = std::make_shared<ov::op::v8::ٹکڑا>(
                                    اگیتر_پوچھ, ov::op::v0::Constant::create(ov::element::i64, {1}, {پوچھ_begin}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {پوچھ_سرا}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {static_cast<int64_t>(suffix_شروع)}));
                            }

                            خودکار reshape_نمونہ = build_reshape_نمونہ(view_ov_shape, view_ggml_shape);
                            خودکار reshaped = std::make_shared<ov::op::v1::Reshape>(
                                selected,
                                ov::op::v0::Constant::create(ov::element::i64, {reshape_نمونہ.size()},
                                                             reshape_نمونہ),
                                خطأ);
                            reshaped->سیٹ_friendly_name(view_name);
                            أرجع reshaped;
                        }
                    }
                }
            }

            أرجع current;
        }

        (فراغ) view_name;
        (فراغ) view_src_ov_shape;
        (فراغ) view_src_name;

        أرجع current;
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
        خودکار innermost_وتھ = context.get_view_داخلہ_وتھ(داخلہ_اشاریہ, 0);
        خودکار innermost_src_وتھ = context.get_view_داخلہ_src_وتھ(داخلہ_اشاریہ, 0);
        خودکار innermost_shape = context.get_view_داخلہ_ggml_shape(داخلہ_اشاریہ, 0);
        خودکار innermost_src_shape = context.get_view_داخلہ_src_ggml_shape(داخلہ_اشاریہ, 0);
        إذا (innermost_وتھ != innermost_src_وتھ && innermost_shape == innermost_src_shape) {
            size_t جڑ_view_idx = view_داخلہ_ناپ - 1;
            خودکار جڑ_ggml_shape = context.get_view_داخلہ_src_ggml_shape(داخلہ_اشاریہ, جڑ_view_idx);
            خودکار جڑ_stride = context.get_view_داخلہ_src_stride(داخلہ_اشاریہ, جڑ_view_idx);
            خودکار جڑ_وتھ = context.get_view_داخلہ_src_وتھ(داخلہ_اشاریہ, جڑ_view_idx);
            خودکار جڑ_ov_shape = context.get_view_داخلہ_src_ov_shape(داخلہ_اشاریہ, جڑ_view_idx);
            خودکار جڑ_name = context.get_view_داخلہ_src_name(داخلہ_اشاریہ, جڑ_view_idx);
            خودکار innermost_stride = context.get_view_داخلہ_stride(داخلہ_اشاریہ, 0);
            خودکار innermost_ov_shape = context.get_view_داخلہ_ov_shape(داخلہ_اشاریہ, 0);
            خودکار innermost_name = context.get_view_داخلہ_name(داخلہ_اشاریہ, 0);

            أرجع process_single_view(داخلہ, innermost_وتھ, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, جڑ_وتھ, جڑ_stride, جڑ_ggml_shape, جڑ_ov_shape,
                                       جڑ_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = داخلہ;

    // Process each view in reverse order (from base to current)
    لكل (صحيح view_idx = view_داخلہ_ناپ - 1; view_idx >= 0; view_idx--) {
        خودکار view_وتھ = context.get_view_داخلہ_وتھ(داخلہ_اشاریہ, view_idx);
        خودکار view_stride = context.get_view_داخلہ_stride(داخلہ_اشاریہ, view_idx);
        خودکار view_ggml_shape = context.get_view_داخلہ_ggml_shape(داخلہ_اشاریہ, view_idx);
        خودکار view_ov_shape = context.get_view_داخلہ_ov_shape(داخلہ_اشاریہ, view_idx);
        خودکار view_name = context.get_view_داخلہ_name(داخلہ_اشاریہ, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        خودکار view_src_وتھ = context.get_view_داخلہ_src_وتھ(داخلہ_اشاریہ, view_idx);
        خودکار view_src_stride = context.get_view_داخلہ_src_stride(داخلہ_اشاریہ, view_idx);
        خودکار view_src_ggml_shape = context.get_view_داخلہ_src_ggml_shape(داخلہ_اشاریہ, view_idx);
        خودکار view_src_ov_shape = context.get_view_داخلہ_src_ov_shape(داخلہ_اشاریہ, view_idx);
        خودکار view_src_name = context.get_view_داخلہ_src_name(داخلہ_اشاریہ, view_idx);
        // print source view info
        // std::cout << "View " << view_idx << ": source name = " << view_src_name
        //           << ", source offset = " << view_src_offset << ", source stride = [" << view_src_stride[0] << ","
        //           << view_src_stride[1] << "," << view_src_stride[2] << "," << view_src_stride[3]
        //           << "], source ggml shape = [" << view_src_ggml_shape[0] << "," << view_src_ggml_shape[1] << ","
        //           << view_src_ggml_shape[2] << "," << view_src_ggml_shape[3]
        //           << "], source ov shape = " << view_src_ov_shape << std::endl;

        current = process_single_view(current, view_وتھ, view_stride, view_ggml_shape, view_ov_shape, view_name,
                                      view_src_وتھ, view_src_stride, view_src_ggml_shape, view_src_ov_shape,
                                      view_src_name);
    }

    أرجع current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
