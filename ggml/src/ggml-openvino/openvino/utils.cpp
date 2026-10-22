#समावेश "utils.h"

#समावेश <cmath>
#समावेश <cstddef>
#समावेश <ctime>
#समावेश <memory>
#समावेश <openvino/op/add.hpp>
#समावेश <openvino/op/clamp.hpp>
#समावेश <openvino/op/convert.hpp>
#समावेश <openvino/op/cos.hpp>
#समावेश <openvino/op/divide.hpp>
#समावेश <openvino/op/gather.hpp>
#समावेश <openvino/op/maximum.hpp>
#समावेश <openvino/op/multiply.hpp>
#समावेश <openvino/op/reshape.hpp>
#समावेश <openvino/op/shape_of.hpp>
#समावेश <openvino/op/sin.hpp>
#समावेश <openvino/op/slice.hpp>
#समावेश <openvino/op/split.hpp>
#समावेश <openvino/op/squeeze.hpp>
#समावेश <openvino/op/subtract.hpp>
#समावेश <openvino/op/transpose.hpp>
#समावेश <string>

namespace ov {
namespace frontend {
namespace ggml {

व्योम अंक_inputs_check(स्थिर गाँठ_Context & context, माप_प्रकार min_inputs, माप_प्रकार max_inputs) {
    स्वतः निवेश_आकार = context.get_निवेश_आकार();
    FRONT_END_OP_CONVERSION_CHECK(निवेश_आकार >= min_inputs, "Got less inputs than expected");
    FRONT_END_OP_CONVERSION_CHECK(निवेश_आकार <= max_inputs, "Got more inputs than expected");
}

पूर्णांक non_cont_dim(std::vector<माप_प्रकार> ne, std::vector<माप_प्रकार> nb) {
    पूर्णांक dim = nb.size() - 1;
    माप_प्रकार bytes = nb[dim];
    क्रम (पूर्णांक i = dim; i > 0; i--) {
        bytes *= ne[i];
        अगर (bytes != nb[i - 1]) {
            वापस i;
        }
    }
    वापस 0;
}

std::shared_ptr<ov::Node> get_dimensions(स्थिर std::shared_ptr<ov::op::v3::ShapeOf> & shape,
                                         स्थिर std::vector<पूर्णांक> & dims) {
    using namespace ov::op;
    स्थिर स्वतः zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    स्थिर स्वतः dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    वापस std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(स्थिर std::shared_ptr<ov::Node> & गाँठ, स्थिर std::vector<पूर्णांक> & dims) {
    वापस get_dimensions(std::make_shared<ov::op::v3::ShapeOf>(गाँठ), dims);
}

OutputVector rename_outputs_with_suffix(स्थिर OutputVector & outputs, स्थिर std::string & suffix) {
    क्रम (स्थिर स्वतः & output : outputs) {
        स्वतः गाँठ = output.get_गाँठ_shared_ptr();
        std::string name = गाँठ->get_friendly_name();
        name += "_";
        name += suffix;
        गाँठ->set_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    वापस outputs;
}

namespace {
ov::Output<ov::Node> rope_yarn_ramp_mix(पूर्णांक अंक_elements, स्थिर भग्न corr_dims[2], भग्न ext_factor) {
    std::vector<भग्न> dim_ids_vec(अंक_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    स्वतः dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (माप_प्रकार) अंक_elements}, dim_ids_vec);
    स्वतः corr_निचला = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    स्वतः corr_ऊपरी = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    स्वतः denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::Subtract>(corr_ऊपरी, corr_निचला),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    स्वतः ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Subtract>(dim_ids, corr_निचला), denom);
    स्वतः ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    स्वतः one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    स्वतः ramp_inverted = std::make_shared<ov::op::v1::Subtract>(one, ramp_clamped);
    स्वतः ext_factor_गाँठ = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_factor});
    स्वतः ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_factor_गाँठ);
    वापस ramp_mix;
}

भग्न ggml_rope_yarn_corr_dim(पूर्णांक n_dims, पूर्णांक n_ctx_orig, भग्न n_rot, भग्न base) {
#अगर_अघोषित M_PI
#    define M_PI 3.14159265358979323846
#पूर्ण_अगर
    वापस n_dims * logf(n_ctx_orig / (n_rot * 2 * (भग्न) M_PI)) / (2 * logf(base));
}

व्योम ggml_rope_yarn_corr_dims(पूर्णांक n_dims,
                              पूर्णांक n_ctx_orig,
                              भग्न freq_base,
                              भग्न beta_fast,
                              भग्न beta_slow,
                              भग्न dims[2]) {
    भग्न start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    भग्न end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<भग्न>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight,
                                                           पूर्णांक mode,
                                                           bool stateful,
                                                           int64_t शीर्ष_dim) {
    constexpr पूर्णांक TYPE_IMROPE = 2;
    constexpr पूर्णांक TYPE_VISION = 3;
    constexpr पूर्णांक TYPE_MROPE = 4;

    स्थिर bool is_imrope = (mode == TYPE_IMROPE);
    स्थिर bool is_vision = (mode == TYPE_VISION);
    स्थिर bool is_mrope = (mode == TYPE_MROPE);
    स्थिर bool is_multi = is_imrope || is_vision || is_mrope;

    अगर (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        स्वतः pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } अन्यथा अगर (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        स्वतः pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, true);
        स्वतः pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } अन्यथा {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        स्वतः pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    भग्न freq_base;
    भग्न freq_scale;
    भग्न ext_factor;
    भग्न attn_factor;
    भग्न beta_fast;
    भग्न beta_slow;
    स्थिर पूर्णांक n_dims = rope_params[1];
    स्थिर माप_प्रकार n_dims_half = n_dims >> 1;
    स्थिर पूर्णांक n_ctx_orig = rope_params[4];
    स_नकल(&freq_base, rope_params + 5, माप(भग्न));
    स_नकल(&freq_scale, rope_params + 6, माप(भग्न));
    स_नकल(&ext_factor, rope_params + 7, माप(भग्न));
    स_नकल(&attn_factor, rope_params + 8, माप(भग्न));
    स_नकल(&beta_fast, rope_params + 9, माप(भग्न));
    स_नकल(&beta_slow, rope_params + 10, माप(भग्न));

    स्थिर भग्न theta_scale = powf(freq_base, -2.0f / n_dims);

    Output<Node> theta;
    भग्न mscale = attn_factor;
    अगर (is_multi) {
        स्थिर माप_प्रकार अंक_pairs = is_vision ? (n_dims > 0 ? (माप_प्रकार) n_dims : (माप_प्रकार) (शीर्ष_dim / 2)) : n_dims_half;
        पूर्णांक sections[4];
        स_नकल(sections, rope_params + 11, माप(पूर्णांक) * 4);
        पूर्णांक sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        अगर (sect_dims <= 0) {
            sect_dims = अंक_pairs;
        }
        पूर्णांक sec_w = sections[0] + sections[1];
        पूर्णांक sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(अंक_pairs);
        std::vector<भग्न> factor(अंक_pairs);
        क्रम (माप_प्रकार j = 0; j < अंक_pairs; j++) {
            पूर्णांक sector = j % sect_dims;
            पूर्णांक plane = 0;
            अगर (is_imrope) {
                अगर (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } अन्यथा अगर (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } अन्यथा अगर (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } अन्यथा {
                    plane = 3;
                }
                factor[j] = std::घात(theta_scale, j);
            } अन्यथा अगर (is_vision) {
                पूर्णांक p_idx = 0;
                अगर (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } अन्यथा अगर (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } अन्यथा अगर (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } अन्यथा {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                factor[j] = std::घात(theta_scale, p_idx);
            } अन्यथा {
                अगर (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } अन्यथा अगर (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } अन्यथा अगर (sector >= sec_e) {
                    plane = 3;
                } अन्यथा {
                    plane = 0;
                }
                factor[j] = std::घात(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        स्वतः gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{अंक_pairs}, gather_indices);
        स्वतः gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> factor_गाँठ =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{अंक_pairs}, factor);
        अगर (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) अंक_pairs}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_लंबाई() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) अंक_pairs}),
                false);
            factor_गाँठ = std::make_shared<ov::op::v1::Divide>(factor_गाँठ, rope_factors);
        }
        स्वतः theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, factor_गाँठ);
        स्वतः theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        अगर (ext_factor == 0.0f) {
            theta = theta_interp;
        } अन्यथा {
            भग्न corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
            स्वतः ramp_mix = rope_yarn_ramp_mix(अंक_pairs, corr_dims, ext_factor);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            स्वतः one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::Add>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
        }
    } अन्यथा {
        std::vector<भग्न> factor(n_dims_half);
        Output<Node> freq_factors;
        भग्न corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        factor[0] = 1.0f;
        क्रम (माप_प्रकार i = 1; i < factor.size(); i++) {
            factor[i] = theta_scale * factor[i - 1];
        }
        अगर (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, factor.size()}, factor);
        } अन्यथा {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, factor.size()}, factor);
        }
        अगर (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::Slice>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_लंबाई() - 1}));
            अगर (stateful) {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {3}, {(int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    false);
            } अन्यथा {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    false);
            }
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_factors);
        }

        स्वतः theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        स्वतः theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        अगर (ext_factor == 0.0f) {
            theta = theta_interp;
        } अन्यथा {
            स्वतः ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_factor);
            Output<Node> one;
            अगर (stateful) {
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

    अगर (mscale != 1.0f) {
        स्वतः mscale_गाँठ = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_गाँठ);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_गाँठ);
    }

    वापस std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_निवेश(स्थिर गाँठ_Context & context, पूर्णांक निवेश_सूचकांक, पूर्णांक slice_len, पूर्णांक axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    स्वतः निवेश = context.get_निवेश(निवेश_सूचकांक);
    स्वतः * op_params = (माप_प्रकार *) context.get_निवेश_op_params(निवेश_सूचकांक);
    स्वतः src_stride = context.get_निवेश_stride(निवेश_सूचकांक);

    int64_t slice_start = op_params[0] / src_stride[3];
    अगर (slice_len == 0) {
        slice_len = context.get_निवेश_shape(निवेश_सूचकांक)[3].get_लंबाई();
    }
    int64_t slice_end = slice_start + slice_len;

    स्वतः begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_start});
    स्वतः end = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_end});
    स्वतः stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    अगर (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } अन्यथा {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    स्वतः sliced = std::make_shared<ov::op::v8::Slice>(निवेश, begin, end, stride, axes);
    वापस sliced;
}

ov::Output<ov::Node> process_view_निवेश_new(स्थिर गाँठ_Context & context, पूर्णांक निवेश_सूचकांक) {
    स्वतः निवेश = context.get_निवेश(निवेश_सूचकांक);

    // Check if this input has view inputs
    माप_प्रकार view_निवेश_आकार = context.get_view_निवेश_आकार(निवेश_सूचकांक);
    अगर (view_निवेश_आकार == 0) {
        // No view inputs, return the input as is
        वापस निवेश;
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
    स्वतः expected_ov_shape = context.get_view_निवेश_ov_shape(निवेश_सूचकांक, 0);
    स्वतः actual_shape = निवेश.get_partial_shape();
    अगर (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_लंबाई() >= actual_shape.rank().get_लंबाई()) {
        स्थिर int64_t n_actual = actual_shape.rank().get_लंबाई();
        स्थिर int64_t shift = expected_ov_shape.rank().get_लंबाई() - n_actual;
        bool shapes_match = true;
        क्रम (int64_t i = 0; i < shift; ++i) {
            अगर (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_लंबाई() != 1) {
                shapes_match = false;
                अवरोध;
            }
        }
        क्रम (int64_t i = 0; i < n_actual && shapes_match; ++i) {
            स्थिर स्वतः & exp = expected_ov_shape[i + shift];
            स्थिर bool both_dynamic = exp.is_dynamic() && actual_shape[i].is_dynamic();
            स्थिर bool both_static_equal =
                exp.is_static() && actual_shape[i].is_static() && exp == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            स्थिर bool expected_dyn_actual_static = exp.is_dynamic() && actual_shape[i].is_static();
            अगर (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_match = false;
                अवरोध;
            }
        }
        अगर (shapes_match) {
            वापस निवेश;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    अगर (context.is_static() && view_निवेश_आकार == 1) {
        स्वतः view_stride_v = context.get_view_निवेश_stride(निवेश_सूचकांक, 0);
        स्वतः view_src_stride_v = context.get_view_निवेश_src_stride(निवेश_सूचकांक, 0);
        स्वतः view_ggml_shape = context.get_view_निवेश_ggml_shape(निवेश_सूचकांक, 0);
        स्वतः view_src_ggml_shape = context.get_view_निवेश_src_ggml_shape(निवेश_सूचकांक, 0);
        स्वतः view_offset = context.get_view_निवेश_offset(निवेश_सूचकांक, 0);
        स्वतः view_src_offset = context.get_view_निवेश_src_offset(निवेश_सूचकांक, 0);

        माप_प्रकार ndims = view_ggml_shape.size();
        std::vector<पूर्णांक> diff_dims;
        अगर (view_src_ggml_shape.size() == ndims) {
            क्रम (माप_प्रकार i = 0; i < ndims; ++i) {
                अगर (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<पूर्णांक>(i));
                }
            }
        }

        अगर (diff_dims.size() == 1) {
            पूर्णांक split_dim = diff_dims[0];
            int64_t अंक_splits = static_cast<int64_t>(view_src_ggml_shape[split_dim]);
            int64_t chunk_आकार = static_cast<int64_t>(view_ggml_shape[split_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            अगर (chunk_आकार == 1 && अंक_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                bool suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                अगर (suffix_ok) {
                    क्रम (माप_प्रकार i = static_cast<माप_प्रकार>(split_dim) + 1; i < ndims; ++i) {
                        अगर (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = false;
                            अवरोध;
                        }
                    }
                }

                अगर (suffix_ok && view_src_stride_v[split_dim] > 0) {
                    माप_प्रकार relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
                    int64_t split_सूचकांक = static_cast<int64_t>(relative_offset / view_src_stride_v[split_dim]);

                    अगर (split_सूचकांक >= 0 && split_सूचकांक < अंक_splits) {
                        स्वतः src_गाँठ = निवेश.get_गाँठ_shared_ptr();
                        std::string rt_कुंजी = "split_dim_" + std::to_string(split_dim);
                        स्वतः & rt_info = src_गाँठ->get_rt_info();

                        अगर (rt_info.find(rt_कुंजी) == rt_info.end()) {
                            स्वतः axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(split_dim)});
                            स्वतः split_गाँठ =
                                std::make_shared<ov::op::v1::Split>(निवेश, axis_const, static_cast<माप_प्रकार>(अंक_splits));
                            split_गाँठ->set_friendly_name(src_गाँठ->get_friendly_name() + "_split");
                            rt_info[rt_कुंजी] = split_गाँठ;
                        }

                        स्वतः split_गाँठ = rt_info[rt_कुंजी].as<std::shared_ptr<ov::op::v1::Split>>();
                        वापस split_गाँठ->output(static_cast<माप_प्रकार>(split_सूचकांक));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    स्वतः process_single_view =
        [](ov::Output<ov::Node> current, माप_प्रकार view_offset, स्थिर std::vector<माप_प्रकार> & view_stride,
           स्थिर ov::Shape & view_ggml_shape, स्थिर ov::PartialShape & view_ov_shape, स्थिर std::string & view_name,
           माप_प्रकार view_src_offset, स्थिर std::vector<माप_प्रकार> & view_src_stride, स्थिर ov::Shape & view_src_ggml_shape,
           स्थिर ov::PartialShape & view_src_ov_shape, स्थिर std::string & view_src_name) -> ov::Output<ov::Node> {
        // Stateful execution drops a leading size-1 axis, so `current` can be one rank lower
        // than the ggml shape metadata (view_stride/view_ggml_shape, always GGML_MAX_DIMS
        // entries) assumes. Shift any axis index built from that metadata down by the
        // difference before using it as an OV Slice axis.
        स्थिर स्वतः current_rank = current.get_partial_shape().rank();
        स्थिर पूर्णांक axis_shift = current_rank.is_static() ?
                                    static_cast<पूर्णांक>(view_stride.size()) - static_cast<पूर्णांक>(current_rank.get_लंबाई()) :
                                    0;

        स्वतः build_reshape_pattern = [](स्थिर ov::PartialShape & target_ov_shape,
                                        स्थिर ov::Shape & target_ggml_shape) -> std::vector<int64_t> {
            स्थिर माप_प्रकार ndims = target_ggml_shape.size();
            std::vector<int64_t> reshape_pattern(ndims);
            माप_प्रकार dynamic_dims = 0;

            अगर (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_लंबाई() == static_cast<int64_t>(ndims)) {
                क्रम (माप_प्रकार i = 0; i < ndims; ++i) {
                    अगर (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_लंबाई();
                    } अन्यथा {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } अन्यथा {
                dynamic_dims = 2;
            }

            अगर (dynamic_dims > 1) {
                क्रम (माप_प्रकार i = 0; i < ndims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            वापस reshape_pattern;
        };

        स्वतः build_prefix_पुच्छ_reshape_pattern = [](स्थिर ov::PartialShape & target_ov_shape,
                                                    स्थिर ov::Shape & target_ggml_shape, माप_प्रकार prefix_dims,
                                                    int64_t पुच्छ_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_pattern(prefix_dims + 1);
            माप_प्रकार dynamic_dims = 0;

            अगर (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_लंबाई() == static_cast<int64_t>(target_ggml_shape.size())) {
                क्रम (माप_प्रकार i = 0; i < prefix_dims; ++i) {
                    अगर (target_ov_shape[i].is_static()) {
                        reshape_pattern[i] = target_ov_shape[i].get_लंबाई();
                    } अन्यथा {
                        reshape_pattern[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } अन्यथा {
                dynamic_dims = 2;
            }

            अगर (dynamic_dims > 1) {
                क्रम (माप_प्रकार i = 0; i < prefix_dims; ++i) {
                    reshape_pattern[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            reshape_pattern[prefix_dims] = पुच्छ_dim;
            वापस reshape_pattern;
        };

        bool same_stride = view_stride.size() == view_src_stride.size();
        अगर (same_stride) {
            क्रम (माप_प्रकार i = 0; i < view_stride.size(); ++i) {
                अगर (view_stride[i] != view_src_stride[i]) {
                    same_stride = false;
                    अवरोध;
                }
            }
        }

        bool same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        अगर (same_ggml_shape) {
            क्रम (माप_प्रकार i = 0; i < view_ggml_shape.size(); ++i) {
                अगर (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = false;
                    अवरोध;
                }
            }
        }

        अगर (same_stride && same_ggml_shape) {
            वापस current;
        }

        अगर (same_stride) {
            स्थिर माप_प्रकार relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;
            स्थिर माप_प्रकार ndims = view_stride.size();

            std::vector<पूर्णांक> diff_dims;
            अगर (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                क्रम (माप_प्रकार i = 0; i < ndims; ++i) {
                    अगर (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<पूर्णांक>(i));
                    }
                }
            }

            अगर (diff_dims.size() == 1) {
                स्थिर पूर्णांक slice_dim = diff_dims[0];
                स्थिर int64_t dim_आकार = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                अगर (view_stride[slice_dim] > 0 && relative_offset % view_stride[slice_dim] == 0) {
                    स्थिर int64_t begin_val = static_cast<int64_t>((relative_offset / view_stride[slice_dim]) %
                                                                   static_cast<माप_प्रकार>(dim_आकार));
                    स्थिर int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);

                    अगर (begin_val >= 0 && end_val <= dim_आकार) {
                        स्वतः sliced = std::make_shared<ov::op::v8::Slice>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim - axis_shift}));

                        अगर (view_ov_shape.is_static()) {
                            स्वतः reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                                false);
                            reshaped->set_friendly_name(view_name);
                            वापस reshaped;
                        }

                        sliced->set_friendly_name(view_name);
                        वापस sliced;
                    }
                }

                int64_t पुच्छ_src_elems = 1;
                int64_t पुच्छ_dst_elems = 1;
                क्रम (माप_प्रकार i = slice_dim; i < ndims; ++i) {
                    पुच्छ_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    पुच्छ_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                स्थिर माप_प्रकार elem_stride = view_stride[ndims - 1];
                int64_t पुच्छ_begin = 0;
                अगर (elem_stride > 0) {
                    पुच्छ_begin =
                        static_cast<int64_t>((relative_offset / elem_stride) % static_cast<माप_प्रकार>(पुच्छ_src_elems));
                }
                स्थिर int64_t पुच्छ_end = पुच्छ_begin + पुच्छ_dst_elems;

                अगर (पुच्छ_begin >= 0 && पुच्छ_end <= पुच्छ_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(slice_dim);
                    क्रम (पूर्णांक i = 0; i < slice_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(पुच्छ_src_elems);
                    स्थिर माप_प्रकार flat_ndims = flat_shape.size();

                    स्वतः flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), false);

                    स्वतः sliced = std::make_shared<ov::op::v8::Slice>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {पुच्छ_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {पुच्छ_end}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

                    अगर (view_ov_shape.is_static()) {
                        स्वतः reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.to_shape()),
                            false);
                        reshaped->set_friendly_name(view_name);
                        वापस reshaped;
                    }

                    sliced->set_friendly_name(view_name);
                    वापस sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> step(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            माप_प्रकार remaining_offset = relative_offset;
            क्रम (माप_प्रकार i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                अगर (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(remaining_offset / view_stride[i]);
                    remaining_offset %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            bool in_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            अगर (in_bounds) {
                क्रम (माप_प्रकार i = 0; i < ndims; ++i) {
                    अगर (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        in_bounds = false;
                        अवरोध;
                    }
                }
            }

            अगर (in_bounds && remaining_offset == 0) {
                स्वतः sliced = std::make_shared<ov::op::v8::Slice>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->set_friendly_name(view_name);
                वापस sliced;
            }
        } अन्यथा {
            bool same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            स्थिर माप_प्रकार relative_offset = view_offset >= view_src_offset ? view_offset - view_src_offset : 0;

            अगर (same_rank) {
                स्थिर माप_प्रकार ndims = view_ggml_shape.size();
                std::vector<पूर्णांक> diff_dims;
                क्रम (माप_प्रकार i = 0; i < ndims; ++i) {
                    अगर (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<पूर्णांक>(i));
                    }
                }

                अगर (diff_dims.size() == 1) {
                    स्थिर माप_प्रकार slice_dim = static_cast<माप_प्रकार>(diff_dims[0]);
                    bool suffix_stride_match = true;
                    क्रम (माप_प्रकार i = slice_dim + 1; i < ndims; ++i) {
                        अगर (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_match = false;
                            अवरोध;
                        }
                    }

                    अगर (suffix_stride_match && view_src_stride[slice_dim] > 0 &&
                        relative_offset % view_src_stride[slice_dim] == 0) {
                        स्थिर int64_t begin_val = static_cast<int64_t>(relative_offset / view_src_stride[slice_dim]);
                        स्थिर int64_t end_val = begin_val + static_cast<int64_t>(view_ggml_shape[slice_dim]);
                        स्थिर int64_t dim_आकार = static_cast<int64_t>(view_src_ggml_shape[slice_dim]);

                        अगर (begin_val >= 0 && end_val <= dim_आकार) {
                            स्वतः sliced = std::make_shared<ov::op::v8::Slice>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(slice_dim)}));
                            sliced->set_friendly_name(view_name);
                            वापस sliced;
                        }
                    }
                }
            }

            माप_प्रकार view_elems = 1;
            माप_प्रकार src_elems = 1;
            अगर (same_rank) {
                क्रम (माप_प्रकार i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            bool same_अंक_elements = same_rank && view_elems == src_elems;

            अगर (same_rank && relative_offset == 0 && same_अंक_elements) {
                स्वतः reshape_pattern = build_reshape_pattern(view_ov_shape, view_ggml_shape);

                स्वतः reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_pattern.size()}, reshape_pattern),
                    false);
                reshaped->set_friendly_name(view_name);
                वापस reshaped;
            }

            अगर (same_rank) {
                स्थिर माप_प्रकार ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                bool is_regular_slice = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> step(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                माप_प्रकार remaining_offset = relative_offset;

                अगर (is_regular_slice) {
                    क्रम (माप_प्रकार i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        अगर (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_slice = false;
                            अवरोध;
                        }

                        step[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        अगर (step[i] <= 0) {
                            is_regular_slice = false;
                            अवरोध;
                        }

                        begin[i] = static_cast<int64_t>(remaining_offset / view_src_stride[i]);
                        remaining_offset %= view_src_stride[i];

                        अगर (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            जारी;
                        }

                        end[i] = begin[i] + step[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        अगर (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_slice = false;
                            अवरोध;
                        }
                    }
                }

                अगर (is_regular_slice && remaining_offset == 0) {
                    स्वतः sliced = std::make_shared<ov::op::v8::Slice>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, step),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->set_friendly_name(view_name);
                    वापस sliced;
                }

                स्थिर माप_प्रकार elem_stride = view_src_stride.back();
                स्थिर bool aligned_offset = elem_stride > 0 && relative_offset % elem_stride == 0;

                अगर (aligned_offset) {
                    माप_प्रकार suffix_start = 0;
                    माप_प्रकार expected_stride = elem_stride;
                    क्रम (पूर्णांक i = static_cast<पूर्णांक>(ndims) - 1; i >= 0; --i) {
                        अगर (view_stride[i] != expected_stride) {
                            suffix_start = static_cast<माप_प्रकार>(i + 1);
                            अवरोध;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    माप_प्रकार prefix_elems = 1;
                    माप_प्रकार suffix_elems = 1;
                    क्रम (माप_प्रकार i = 0; i < suffix_start; ++i) {
                        prefix_elems *= view_ggml_shape[i];
                    }
                    क्रम (माप_प्रकार i = suffix_start; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    अगर (prefix_elems > 0 && src_elems % prefix_elems == 0) {
                        स्थिर माप_प्रकार src_पुच्छ_elems = src_elems / prefix_elems;
                        स्थिर int64_t पुच्छ_begin = static_cast<int64_t>(relative_offset / elem_stride);
                        स्थिर int64_t पुच्छ_end = पुच्छ_begin + static_cast<int64_t>(suffix_elems);

                        अगर (पुच्छ_begin >= 0 && पुच्छ_end <= static_cast<int64_t>(src_पुच्छ_elems)) {
                            स्वतः prefix_पुच्छ_pattern = build_prefix_पुच्छ_reshape_pattern(
                                view_ov_shape, view_ggml_shape, suffix_start, static_cast<int64_t>(src_पुच्छ_elems));

                            स्वतः prefix_पुच्छ = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {prefix_पुच्छ_pattern.size()},
                                                             prefix_पुच्छ_pattern),
                                false);

                            ov::Output<ov::Node> selected = prefix_पुच्छ;
                            अगर (पुच्छ_begin != 0 || पुच्छ_end != static_cast<int64_t>(src_पुच्छ_elems)) {
                                selected = std::make_shared<ov::op::v8::Slice>(
                                    prefix_पुच्छ, ov::op::v0::Constant::create(ov::element::i64, {1}, {पुच्छ_begin}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {पुच्छ_end}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {static_cast<int64_t>(suffix_start)}));
                            }

                            स्वतः reshape_pattern = build_reshape_pattern(view_ov_shape, view_ggml_shape);
                            स्वतः reshaped = std::make_shared<ov::op::v1::Reshape>(
                                selected,
                                ov::op::v0::Constant::create(ov::element::i64, {reshape_pattern.size()},
                                                             reshape_pattern),
                                false);
                            reshaped->set_friendly_name(view_name);
                            वापस reshaped;
                        }
                    }
                }
            }

            वापस current;
        }

        (व्योम) view_name;
        (व्योम) view_src_ov_shape;
        (व्योम) view_src_name;

        वापस current;
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
        स्वतः innermost_offset = context.get_view_निवेश_offset(निवेश_सूचकांक, 0);
        स्वतः innermost_src_offset = context.get_view_निवेश_src_offset(निवेश_सूचकांक, 0);
        स्वतः innermost_shape = context.get_view_निवेश_ggml_shape(निवेश_सूचकांक, 0);
        स्वतः innermost_src_shape = context.get_view_निवेश_src_ggml_shape(निवेश_सूचकांक, 0);
        अगर (innermost_offset != innermost_src_offset && innermost_shape == innermost_src_shape) {
            माप_प्रकार मूल_view_idx = view_निवेश_आकार - 1;
            स्वतः मूल_ggml_shape = context.get_view_निवेश_src_ggml_shape(निवेश_सूचकांक, मूल_view_idx);
            स्वतः मूल_stride = context.get_view_निवेश_src_stride(निवेश_सूचकांक, मूल_view_idx);
            स्वतः मूल_offset = context.get_view_निवेश_src_offset(निवेश_सूचकांक, मूल_view_idx);
            स्वतः मूल_ov_shape = context.get_view_निवेश_src_ov_shape(निवेश_सूचकांक, मूल_view_idx);
            स्वतः मूल_name = context.get_view_निवेश_src_name(निवेश_सूचकांक, मूल_view_idx);
            स्वतः innermost_stride = context.get_view_निवेश_stride(निवेश_सूचकांक, 0);
            स्वतः innermost_ov_shape = context.get_view_निवेश_ov_shape(निवेश_सूचकांक, 0);
            स्वतः innermost_name = context.get_view_निवेश_name(निवेश_सूचकांक, 0);

            वापस process_single_view(निवेश, innermost_offset, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, मूल_offset, मूल_stride, मूल_ggml_shape, मूल_ov_shape,
                                       मूल_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = निवेश;

    // Process each view in reverse order (from base to current)
    क्रम (पूर्णांक view_idx = view_निवेश_आकार - 1; view_idx >= 0; view_idx--) {
        स्वतः view_offset = context.get_view_निवेश_offset(निवेश_सूचकांक, view_idx);
        स्वतः view_stride = context.get_view_निवेश_stride(निवेश_सूचकांक, view_idx);
        स्वतः view_ggml_shape = context.get_view_निवेश_ggml_shape(निवेश_सूचकांक, view_idx);
        स्वतः view_ov_shape = context.get_view_निवेश_ov_shape(निवेश_सूचकांक, view_idx);
        स्वतः view_name = context.get_view_निवेश_name(निवेश_सूचकांक, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        स्वतः view_src_offset = context.get_view_निवेश_src_offset(निवेश_सूचकांक, view_idx);
        स्वतः view_src_stride = context.get_view_निवेश_src_stride(निवेश_सूचकांक, view_idx);
        स्वतः view_src_ggml_shape = context.get_view_निवेश_src_ggml_shape(निवेश_सूचकांक, view_idx);
        स्वतः view_src_ov_shape = context.get_view_निवेश_src_ov_shape(निवेश_सूचकांक, view_idx);
        स्वतः view_src_name = context.get_view_निवेश_src_name(निवेश_सूचकांक, view_idx);
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

    वापस current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
