#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <climits>
#ضمن <cstdint>
#ضمن <cstdio>
#ضمن <memory>
#ضمن <numeric>
#ضمن <openvino/frontend/exception.hpp>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/negative.hpp>
#ضمن <openvino/op/range.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/scatter_update.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/subtract.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_سمتیہ translate_cpy(ثابت گرہ_Context & context) {
    خودکار op_case = context.get_op_case();
    خودکار داخلہ_shape = context.get_داخلہ_shape(0);
    خودکار output_shape = context.get_داخلہ_shape(1);

    إذا (op_case == 4) {
        خودکار src = process_view_داخلہ_new(context, 0);
        خودکار ادھار = context.get_داخلہ(1);

        int64_t n_elems = 1;
        لكل (ثابت خودکار & dim : context.get_output_shape().منزل_shape()) {
            n_elems *= static_cast<int64_t>(dim);
        }

        ثابت خودکار output_stride = context.get_output_stride();
        ثابت size_t elem_ناپ = output_stride.خالی_تھاواں() ? context.get_output_type().size() : output_stride.back();
        FRONT_سرا_OP_CONVERSION_CHECK(elem_ناپ > 0, "CPY conv state view update has invalid element size");

        ثابت int64_t begin_val = static_cast<int64_t>(context.get_output_op_وتھ() / elem_ناپ);
        ثابت int64_t سرا_val = begin_val + n_elems;

        خودکار flat_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, -1});
        src = std::make_shared<ov::op::v1::Reshape>(src, flat_shape, خطأ);
        إذا (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        خودکار zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خودکار begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val});
        خودکار end = ov::op::v0::Constant::create(ov::element::i64, {1}, {سرا_val});
        خودکار int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {ودھ_توں_ودھ_عدد});
        خودکار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        خودکار سر_part = std::make_shared<ov::op::v8::ٹکڑا>(ادھار, zero, begin, one, axis);
        خودکار پوچھ_part = std::make_shared<ov::op::v8::ٹکڑا>(ادھار, end, int_max, one, axis);
        خودکار res = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{سر_part, src, پوچھ_part}, 3);
        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    // Recurrent state cache writeback into a slot block of the cache. Where the block starts is a
    // runtime input, so the cached model works for any kv head and active sequence count. The
    // result is the full updated cache.
    // op_case 1: gated-delta-net state, op_case 2: conv state, op_case 3: defrag remainder.
    إذا (op_case == 3) {
        // With -np 1 (and generally whenever there is no defrag remainder) this GET_ROWS gathers
        // zero rows: nothing to write back, and the cache is unchanged. NPU rejects zero-size
        // tensors, so short-circuit instead of building a degenerate Slice/Concat chain.
        منطقي is_خالی_تھاواں = خطأ;
        إذا (داخلہ_shape.rank().is_static()) {
            لكل (ثابت خودکار & d : داخلہ_shape) {
                إذا (d.is_static() && d.get_لمائی() == 0) {
                    is_خالی_تھاواں = سچ;
                    اكسر;
                }
            }
        }
        إذا (is_خالی_تھاواں) {
            أرجع {context.get_داخلہ(1)};
        }
    }
    // op_case 7/8/9 are the single-slot variants; op_case 10 writes native GDN state into a
    // multi-slot cache without rollback snapshots.
    ثابت منطقي single_تھاں_سونپو = op_case >= 7 && op_case <= 9;
    ثابت منطقي سدھا_gdn_state = op_case == 7 || op_case == 10;
    صحيح writeback_case = op_case;
    إذا (op_case == 10) {
        writeback_case = 1;
    } وإلا إذا (single_تھاں_سونپو) {
        writeback_case = op_case - 6;
    }
    ثابت std::string تھاں_begin_name = "rs_slot_begin_" + context.get_name();
    ثابت منطقي ٹکڑا_سونپو = writeback_case >= 1 && writeback_case <= 3 &&
                              (single_تھاں_سونپو || context.has_داخلہ(تھاں_begin_name));
    إذا (ٹکڑا_سونپو) {
        إذا (single_تھاں_سونپو && writeback_case == 3) {
            أرجع {context.get_داخلہ(1)};
        }

        ثابت int64_t تھاں_axis = 2;
        خودکار zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خودکار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {ودھ_توں_ودھ_عدد});
        خودکار axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {تھاں_axis});
        خودکار feature = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                    std::vector<int64_t>{1, 1, -1, output_shape[3].get_لمائی()});

        ov::Output<ov::Node> src;
        ov::Output<ov::Node> begin;
        إذا (!single_تھاں_سونپو) {
            begin = context.get_داخلہ(تھاں_begin_name);
        }
        خودکار ادھار = context.get_داخلہ(1);
        إذا (writeback_case == 1) {
            إذا (سدھا_gdn_state) {
                // Non-rollback GDN publishes state directly as [active_slots, heads, value_dim,
                // key_dim]. Flatten each active slot before replacing or updating the cache.
                src = std::make_shared<ov::op::v1::Reshape>(context.get_داخلہ(0), feature, خطأ);
            } وإلا {
                // Multi-slot rollback still consumes GGML's packed [attention | state snapshots]
                // layout. Slice the state block using the runtime source offset.
                خودکار src_begin = context.get_داخلہ("rs_src_begin_" + context.get_name());
                خودکار state_part =
                    std::make_shared<ov::op::v8::ٹکڑا>(context.get_داخلہ(0), src_begin, int_max, one, axis);
                src = std::make_shared<ov::op::v1::Reshape>(state_part, feature, خطأ);
            }
        } وإلا إذا (writeback_case == 2) {
            // conv_input is [previous conv state | new tokens]; the snapshot is the conv_kernel_size - 1
            // columns ending at the last *valid* token. Gather (rather than Slice) keeps the output
            // shape static even though the window start is a runtime value.
            خودکار کھڑکی_ناپ = (int64_t) داخلہ_shape[3].get_لمائی();
            ov::Output<ov::Node> کھڑکی;
            خودکار خانہ_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            ثابت std::string src_begin_name = "rs_src_begin_" + context.get_name();
            إذا (context.has_داخلہ(src_begin_name)) {
                خودکار src_begin = context.get_داخلہ(src_begin_name);
                خودکار src_سرا = std::make_shared<ov::op::v1::جوڑو>(
                    src_begin, ov::op::v0::Constant::create(ov::element::i64, {1}, {کھڑکی_ناپ}));
                کھڑکی = std::make_shared<ov::op::v8::ٹکڑا>(context.get_داخلہ(0), src_begin, src_سرا, one, خانہ_axis);
            } وإلا إذا (context.has_داخلہ("chunk_valid_len")) {
                std::vector<int64_t> offsets(کھڑکی_ناپ);
                std::iota(offsets.begin(), offsets.end(), 0);
                خودکار indices = std::make_shared<ov::op::v1::جوڑو>(
                    ov::op::v0::Constant::create(ov::element::i64, {(size_t) کھڑکی_ناپ}, offsets),
                    context.get_داخلہ("chunk_valid_len"));
                کھڑکی = std::make_shared<ov::op::v8::Gather>(context.get_داخلہ(0), indices, خانہ_axis);
            } وإلا {
                خودکار کھڑکی_begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {-کھڑکی_ناپ});
                کھڑکی =
                    std::make_shared<ov::op::v8::ٹکڑا>(context.get_داخلہ(0), کھڑکی_begin, int_max, one, خانہ_axis);
            }
            ثابت خودکار ادھار_shape = ادھار.get_partial_shape();
            FRONT_سرا_OP_CONVERSION_CHECK(ادھار_shape.rank().is_static() && ادھار_shape.rank().get_لمائی() == 4,
                                          "CPY conv state cache update requires rank-4 base cache");
            FRONT_سرا_OP_CONVERSION_CHECK(ادھار_shape[3].is_static(),
                                          "CPY conv state cache update requires static feature size");
            FRONT_سرا_OP_CONVERSION_CHECK(داخلہ_shape.rank().is_static() && داخلہ_shape.rank().get_لمائی() == 4 &&
                                              داخلہ_shape[2].is_static() && داخلہ_shape[3].is_static(),
                                          "CPY conv state cache update requires static source feature view");

            ثابت int64_t بھریا_feature_ناپ = ادھار_shape[3].get_لمائی();
            ثابت int64_t update_feature_ناپ = داخلہ_shape[2].get_لمائی() * داخلہ_shape[3].get_لمائی();
            ثابت خودکار output_stride = context.get_output_stride();
            ثابت size_t elem_ناپ = output_stride.خالی_تھاواں() ? context.get_output_type().size() : output_stride.back();
            FRONT_سرا_OP_CONVERSION_CHECK(elem_ناپ > 0,
                                          "CPY conv state cache update has invalid element size");
            ثابت int64_t feature_begin = static_cast<int64_t>(context.get_output_op_وتھ() / elem_ناپ) %
                                          بھریا_feature_ناپ;
            ثابت int64_t feature_سرا = feature_begin + update_feature_ناپ;
            FRONT_سرا_OP_CONVERSION_CHECK(feature_begin >= 0 && feature_سرا <= بھریا_feature_ناپ,
                                          "CPY conv state cache update feature range is out of bounds");

            خودکار partial_feature = ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, update_feature_ناپ});
            src = std::make_shared<ov::op::v1::Reshape>(کھڑکی, partial_feature, خطأ);
            إذا (src.get_element_type() != context.get_output_type()) {
                src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
            }

            إذا (single_تھاں_سونپو) {
                أرجع rename_outputs_with_suffix({src}, context.get_name());
            }

            خودکار src_len = std::make_shared<ov::op::v8::Gather>(
                std::make_shared<ov::op::v3::Shape_دا>(src, ov::element::i64), axis,
                ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
            خودکار تھاں_سرا = std::make_shared<ov::op::v1::جوڑو>(begin, src_len);
            خودکار سرگرم_slots = std::make_shared<ov::op::v8::ٹکڑا>(ادھار, begin, تھاں_سرا, one, axis);

            خودکار feature_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            خودکار feature_begin_گرہ = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_begin});
            خودکار feature_سرا_گرہ = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_سرا});
            خودکار feature_سر = std::make_shared<ov::op::v8::ٹکڑا>(سرگرم_slots, zero, feature_begin_گرہ, one,
                                                                    feature_axis);
            خودکار feature_پوچھ = std::make_shared<ov::op::v8::ٹکڑا>(سرگرم_slots, feature_سرا_گرہ, int_max, one,
                                                                    feature_axis);
            src = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{feature_سر, src, feature_پوچھ}, 3);
        } وإلا {
            // op_case 3: gathered remainder rows already have the cache slot layout [1, 1, extra, feature]
            src = context.get_داخلہ(0);
        }

        إذا (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        إذا (single_تھاں_سونپو) {
            أرجع rename_outputs_with_suffix({src}, context.get_name());
        }

        خودکار src_len =
            std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::Shape_دا>(src, ov::element::i64), axis,
                                                 ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        خودکار end = std::make_shared<ov::op::v1::جوڑو>(begin, src_len);
        خودکار سر_part = std::make_shared<ov::op::v8::ٹکڑا>(ادھار, zero, begin, one, axis);
        خودکار پوچھ_part = std::make_shared<ov::op::v8::ٹکڑا>(ادھار, end, int_max, one, axis);
        خودکار res = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{سر_part, src, پوچھ_part}, تھاں_axis);
        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    خودکار داخلہ = process_view_داخلہ_new(context, 0);

    إذا (op_case == 5 || op_case == 6) {
        خودکار داخلہ_shape = context.get_داخلہ_shape(0);
        خودکار output_shape = context.get_output_shape();
        خودکار dst_ggml_shape = context.get_view_داخلہ_ggml_shape(1, 0);
        خودکار dst_stride = context.get_view_داخلہ_stride(1, 0);
        size_t وتھ_بائٹاں = context.get_view_داخلہ_وتھ(1, 0);
        خودکار n_state = (int64_t) context.get_داخلہ_shape(0)[3].get_لمائی();
        خودکار n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
        خودکار kv_buf = context.get_داخلہ(1);  // shape {1,1,1,N}

        Output<Node> token_len_فی_seq;
        Output<Node> n_write_dyn;
        إذا (context.has_داخلہ("token_len_per_seq")) {
            token_len_فی_seq = context.get_داخلہ("token_len_per_seq");
            n_write_dyn = std::make_shared<ov::op::v1::Multiply>(token_len_فی_seq, n_state_c);
        } وإلا {
            n_write_dyn = ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) dst_ggml_shape[3]});
        }
        size_t elem_ناپ = dst_stride[3];
        FRONT_سرا_OP_CONVERSION_CHECK(elem_ناپ > 0, "CPY KV cache view update has invalid element size");
        int64_t شروع_elem = (int64_t) (وتھ_بائٹاں / elem_ناپ);
        // op_case 5: decoder self-attention – write offset advances each step.
        // op_case 6: encoder self-attn or cross-attn – offset fixed at compile time.
        ثابت منطقي is_decoder_self_attn = (op_case == 5);
        خودکار ones_c = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{1, 1, 1});
        خودکار new_shape = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{ones_c, n_write_dyn}, 0);

        خودکار reshaped = std::make_shared<ov::op::v1::Reshape>(داخلہ, new_shape, خطأ);
        خودکار انکڑے = std::make_shared<ov::op::v0::Convert>(reshaped, context.get_output_type());
        // Indices [start_elem .. start_elem + n_write) on axis 3 of {1,1,1,N}
        // For decoder self-attention the write offset advances each step, so compute it
        // dynamically from the model inputs: start = (attention_size - token_len_per_seq) * n_state.
        // For encoder self-attn and cross-attn the offset is fixed at graph-compile time.
        ov::Output<ov::Node> start;
        إذا (is_decoder_self_attn && context.has_داخلہ("attention_size") && context.has_داخلہ("token_len_per_seq")) {
            خودکار attention_ناپ_وچ = context.get_داخلہ("attention_size");
            خودکار token_len_وچ = context.get_داخلہ("token_len_per_seq");
            خودکار past_ٹوکن = std::make_shared<ov::op::v1::گھٹاؤ>(attention_ناپ_وچ, token_len_وچ);
            خودکار new_شروع = std::make_shared<ov::op::v1::Multiply>(past_ٹوکن, n_state_c);
            start = std::make_shared<ov::op::v1::جوڑو>(
                new_شروع, ov::op::v0::Constant::create(ov::element::i64, {1}, {شروع_elem}));
        } وإلا {
            start = ov::op::v0::Constant::create(ov::element::i64, {1}, {شروع_elem});
        }
        خودکار شروع_squeezed = std::make_shared<ov::op::v0::Squeeze>(start);
        خودکار end = std::make_shared<ov::op::v1::جوڑو>(شروع_squeezed, n_write_dyn);
        خودکار سرا_squeezed = std::make_shared<ov::op::v0::Squeeze>(end);
        خودکار قدم = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار قدم_squeezed = std::make_shared<ov::op::v0::Squeeze>(قدم);
        خودکار indices =
            std::make_shared<ov::op::v4::Range>(شروع_squeezed, سرا_squeezed, قدم_squeezed, ov::element::i64);
        خودکار axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        خودکار kv_updated = std::make_shared<ov::op::v3::ScatterUpdate>(kv_buf, indices, انکڑے, axis);
        أرجع rename_outputs_with_suffix({kv_updated}, context.get_name());
    }

    إذا (داخلہ_shape != output_shape) {
        خودکار new_shape = ov::op::v0::Constant::create(
            ov::element::i64, {static_cast<size_t>(output_shape.rank().get_لمائی())}, output_shape.منزل_shape());
        داخلہ = std::make_shared<ov::op::v1::Reshape>(داخلہ, new_shape, خطأ);
    }

    ov::Output<Node> res;
    إذا (context.get_داخلہ_type(0) != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(داخلہ, context.get_output_type());
    } وإلا {
        res = داخلہ;
    }

    إذا (res.get_گرہ_shared_ptr() == context.get_داخلہ(0).get_گرہ_shared_ptr()) {
        أرجع {res};
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
