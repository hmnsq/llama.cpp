#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <climits>
#شامل <cstdint>
#شامل <cstdio>
#شامل <memory>
#شامل <numeric>
#شامل <openvino/frontend/exception.hpp>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/concat.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/gather.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/negative.hpp>
#شامل <openvino/op/range.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/scatter_update.hpp>
#شامل <openvino/op/shape_of.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/squeeze.hpp>
#شامل <openvino/op/subtract.hpp>
#شامل <vector>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_cpy(ثابت گرہ_Context & context) {
    خودکار op_case = context.get_op_case();
    خودکار داخلہ_shape = context.get_داخلہ_shape(0);
    خودکار output_shape = context.get_داخلہ_shape(1);

    اگر (op_case == 4) {
        خودکار src = process_view_داخلہ_new(context, 0);
        خودکار بنیاد = context.get_داخلہ(1);

        int64_t n_elems = 1;
        برائے (ثابت خودکار & dim : context.get_output_shape().مقصد_shape()) {
            n_elems *= static_cast<int64_t>(dim);
        }

        ثابت خودکار output_stride = context.get_output_stride();
        ثابت size_t elem_سائز = output_stride.خالی_خانے() ? context.get_output_type().size() : output_stride.back();
        FRONT_انتہا_OP_CONVERSION_CHECK(elem_سائز > 0, "CPY conv state view update has invalid element size");

        ثابت int64_t begin_val = static_cast<int64_t>(context.get_output_op_انحراف() / elem_سائز);
        ثابت int64_t انتہا_val = begin_val + n_elems;

        خودکار flat_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, -1});
        src = std::make_shared<ov::op::v1::Reshape>(src, flat_shape, جھوٹ);
        اگر (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        خودکار zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خودکار begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val});
        خودکار end = ov::op::v0::Constant::create(ov::element::i64, {1}, {انتہا_val});
        خودکار int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {زیادہ_سے_زیادہ_عدد});
        خودکار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        خودکار سر_part = std::make_shared<ov::op::v8::قاش>(بنیاد, zero, begin, one, axis);
        خودکار دم_part = std::make_shared<ov::op::v8::قاش>(بنیاد, end, int_max, one, axis);
        خودکار res = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{سر_part, src, دم_part}, 3);
        واپس rename_outputs_with_suffix({res}, context.get_name());
    }

    // Recurrent state cache writeback into a slot block of the cache. Where the block starts is a
    // runtime input, so the cached model works for any kv head and active sequence count. The
    // result is the full updated cache.
    // op_case 1: gated-delta-net state, op_case 2: conv state, op_case 3: defrag remainder.
    اگر (op_case == 3) {
        // With -np 1 (and generally whenever there is no defrag remainder) this GET_ROWS gathers
        // zero rows: nothing to write back, and the cache is unchanged. NPU rejects zero-size
        // tensors, so short-circuit instead of building a degenerate Slice/Concat chain.
        منطقی is_خالی_خانے = جھوٹ;
        اگر (داخلہ_shape.rank().is_static()) {
            برائے (ثابت خودکار & d : داخلہ_shape) {
                اگر (d.is_static() && d.get_لمبائی() == 0) {
                    is_خالی_خانے = سچ;
                    توڑو;
                }
            }
        }
        اگر (is_خالی_خانے) {
            واپس {context.get_داخلہ(1)};
        }
    }
    // op_case 7/8/9 are the single-slot variants; op_case 10 writes native GDN state into a
    // multi-slot cache without rollback snapshots.
    ثابت منطقی single_خانہ_تفویض = op_case >= 7 && op_case <= 9;
    ثابت منطقی براہ_راست_gdn_state = op_case == 7 || op_case == 10;
    صحیح writeback_case = op_case;
    اگر (op_case == 10) {
        writeback_case = 1;
    } ورنہ اگر (single_خانہ_تفویض) {
        writeback_case = op_case - 6;
    }
    ثابت std::string خانہ_begin_name = "rs_slot_begin_" + context.get_name();
    ثابت منطقی قاش_تفویض = writeback_case >= 1 && writeback_case <= 3 &&
                              (single_خانہ_تفویض || context.has_داخلہ(خانہ_begin_name));
    اگر (قاش_تفویض) {
        اگر (single_خانہ_تفویض && writeback_case == 3) {
            واپس {context.get_داخلہ(1)};
        }

        ثابت int64_t خانہ_axis = 2;
        خودکار zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خودکار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {زیادہ_سے_زیادہ_عدد});
        خودکار axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {خانہ_axis});
        خودکار feature = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                    std::vector<int64_t>{1, 1, -1, output_shape[3].get_لمبائی()});

        ov::Output<ov::Node> src;
        ov::Output<ov::Node> begin;
        اگر (!single_خانہ_تفویض) {
            begin = context.get_داخلہ(خانہ_begin_name);
        }
        خودکار بنیاد = context.get_داخلہ(1);
        اگر (writeback_case == 1) {
            اگر (براہ_راست_gdn_state) {
                // Non-rollback GDN publishes state directly as [active_slots, heads, value_dim,
                // key_dim]. Flatten each active slot before replacing or updating the cache.
                src = std::make_shared<ov::op::v1::Reshape>(context.get_داخلہ(0), feature, جھوٹ);
            } ورنہ {
                // Multi-slot rollback still consumes GGML's packed [attention | state snapshots]
                // layout. Slice the state block using the runtime source offset.
                خودکار src_begin = context.get_داخلہ("rs_src_begin_" + context.get_name());
                خودکار state_part =
                    std::make_shared<ov::op::v8::قاش>(context.get_داخلہ(0), src_begin, int_max, one, axis);
                src = std::make_shared<ov::op::v1::Reshape>(state_part, feature, جھوٹ);
            }
        } ورنہ اگر (writeback_case == 2) {
            // conv_input is [previous conv state | new tokens]; the snapshot is the conv_kernel_size - 1
            // columns ending at the last *valid* token. Gather (rather than Slice) keeps the output
            // shape static even though the window start is a runtime value.
            خودکار کھڑکی_سائز = (int64_t) داخلہ_shape[3].get_لمبائی();
            ov::Output<ov::Node> کھڑکی;
            خودکار ستون_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            ثابت std::string src_begin_name = "rs_src_begin_" + context.get_name();
            اگر (context.has_داخلہ(src_begin_name)) {
                خودکار src_begin = context.get_داخلہ(src_begin_name);
                خودکار src_انتہا = std::make_shared<ov::op::v1::جوڑو>(
                    src_begin, ov::op::v0::Constant::create(ov::element::i64, {1}, {کھڑکی_سائز}));
                کھڑکی = std::make_shared<ov::op::v8::قاش>(context.get_داخلہ(0), src_begin, src_انتہا, one, ستون_axis);
            } ورنہ اگر (context.has_داخلہ("chunk_valid_len")) {
                std::vector<int64_t> offsets(کھڑکی_سائز);
                std::iota(offsets.begin(), offsets.end(), 0);
                خودکار indices = std::make_shared<ov::op::v1::جوڑو>(
                    ov::op::v0::Constant::create(ov::element::i64, {(size_t) کھڑکی_سائز}, offsets),
                    context.get_داخلہ("chunk_valid_len"));
                کھڑکی = std::make_shared<ov::op::v8::Gather>(context.get_داخلہ(0), indices, ستون_axis);
            } ورنہ {
                خودکار کھڑکی_begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {-کھڑکی_سائز});
                کھڑکی =
                    std::make_shared<ov::op::v8::قاش>(context.get_داخلہ(0), کھڑکی_begin, int_max, one, ستون_axis);
            }
            ثابت خودکار بنیاد_shape = بنیاد.get_partial_shape();
            FRONT_انتہا_OP_CONVERSION_CHECK(بنیاد_shape.rank().is_static() && بنیاد_shape.rank().get_لمبائی() == 4,
                                          "CPY conv state cache update requires rank-4 base cache");
            FRONT_انتہا_OP_CONVERSION_CHECK(بنیاد_shape[3].is_static(),
                                          "CPY conv state cache update requires static feature size");
            FRONT_انتہا_OP_CONVERSION_CHECK(داخلہ_shape.rank().is_static() && داخلہ_shape.rank().get_لمبائی() == 4 &&
                                              داخلہ_shape[2].is_static() && داخلہ_shape[3].is_static(),
                                          "CPY conv state cache update requires static source feature view");

            ثابت int64_t بھرا_feature_سائز = بنیاد_shape[3].get_لمبائی();
            ثابت int64_t update_feature_سائز = داخلہ_shape[2].get_لمبائی() * داخلہ_shape[3].get_لمبائی();
            ثابت خودکار output_stride = context.get_output_stride();
            ثابت size_t elem_سائز = output_stride.خالی_خانے() ? context.get_output_type().size() : output_stride.back();
            FRONT_انتہا_OP_CONVERSION_CHECK(elem_سائز > 0,
                                          "CPY conv state cache update has invalid element size");
            ثابت int64_t feature_begin = static_cast<int64_t>(context.get_output_op_انحراف() / elem_سائز) %
                                          بھرا_feature_سائز;
            ثابت int64_t feature_انتہا = feature_begin + update_feature_سائز;
            FRONT_انتہا_OP_CONVERSION_CHECK(feature_begin >= 0 && feature_انتہا <= بھرا_feature_سائز,
                                          "CPY conv state cache update feature range is out of bounds");

            خودکار partial_feature = ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, update_feature_سائز});
            src = std::make_shared<ov::op::v1::Reshape>(کھڑکی, partial_feature, جھوٹ);
            اگر (src.get_element_type() != context.get_output_type()) {
                src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
            }

            اگر (single_خانہ_تفویض) {
                واپس rename_outputs_with_suffix({src}, context.get_name());
            }

            خودکار src_len = std::make_shared<ov::op::v8::Gather>(
                std::make_shared<ov::op::v3::Shape_کا>(src, ov::element::i64), axis,
                ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
            خودکار خانہ_انتہا = std::make_shared<ov::op::v1::جوڑو>(begin, src_len);
            خودکار فعال_slots = std::make_shared<ov::op::v8::قاش>(بنیاد, begin, خانہ_انتہا, one, axis);

            خودکار feature_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            خودکار feature_begin_گرہ = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_begin});
            خودکار feature_انتہا_گرہ = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_انتہا});
            خودکار feature_سر = std::make_shared<ov::op::v8::قاش>(فعال_slots, zero, feature_begin_گرہ, one,
                                                                    feature_axis);
            خودکار feature_دم = std::make_shared<ov::op::v8::قاش>(فعال_slots, feature_انتہا_گرہ, int_max, one,
                                                                    feature_axis);
            src = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{feature_سر, src, feature_دم}, 3);
        } ورنہ {
            // op_case 3: gathered remainder rows already have the cache slot layout [1, 1, extra, feature]
            src = context.get_داخلہ(0);
        }

        اگر (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        اگر (single_خانہ_تفویض) {
            واپس rename_outputs_with_suffix({src}, context.get_name());
        }

        خودکار src_len =
            std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::Shape_کا>(src, ov::element::i64), axis,
                                                 ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        خودکار end = std::make_shared<ov::op::v1::جوڑو>(begin, src_len);
        خودکار سر_part = std::make_shared<ov::op::v8::قاش>(بنیاد, zero, begin, one, axis);
        خودکار دم_part = std::make_shared<ov::op::v8::قاش>(بنیاد, end, int_max, one, axis);
        خودکار res = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{سر_part, src, دم_part}, خانہ_axis);
        واپس rename_outputs_with_suffix({res}, context.get_name());
    }

    خودکار داخلہ = process_view_داخلہ_new(context, 0);

    اگر (op_case == 5 || op_case == 6) {
        خودکار داخلہ_shape = context.get_داخلہ_shape(0);
        خودکار output_shape = context.get_output_shape();
        خودکار dst_ggml_shape = context.get_view_داخلہ_ggml_shape(1, 0);
        خودکار dst_stride = context.get_view_داخلہ_stride(1, 0);
        size_t انحراف_بائٹیں = context.get_view_داخلہ_انحراف(1, 0);
        خودکار n_state = (int64_t) context.get_داخلہ_shape(0)[3].get_لمبائی();
        خودکار n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
        خودکار kv_buf = context.get_داخلہ(1);  // shape {1,1,1,N}

        Output<Node> token_len_فی_seq;
        Output<Node> n_write_dyn;
        اگر (context.has_داخلہ("token_len_per_seq")) {
            token_len_فی_seq = context.get_داخلہ("token_len_per_seq");
            n_write_dyn = std::make_shared<ov::op::v1::Multiply>(token_len_فی_seq, n_state_c);
        } ورنہ {
            n_write_dyn = ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) dst_ggml_shape[3]});
        }
        size_t elem_سائز = dst_stride[3];
        FRONT_انتہا_OP_CONVERSION_CHECK(elem_سائز > 0, "CPY KV cache view update has invalid element size");
        int64_t آغاز_elem = (int64_t) (انحراف_بائٹیں / elem_سائز);
        // op_case 5: decoder self-attention – write offset advances each step.
        // op_case 6: encoder self-attn or cross-attn – offset fixed at compile time.
        ثابت منطقی is_decoder_self_attn = (op_case == 5);
        خودکار ones_c = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{1, 1, 1});
        خودکار new_shape = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{ones_c, n_write_dyn}, 0);

        خودکار reshaped = std::make_shared<ov::op::v1::Reshape>(داخلہ, new_shape, جھوٹ);
        خودکار ڈیٹا = std::make_shared<ov::op::v0::Convert>(reshaped, context.get_output_type());
        // Indices [start_elem .. start_elem + n_write) on axis 3 of {1,1,1,N}
        // For decoder self-attention the write offset advances each step, so compute it
        // dynamically from the model inputs: start = (attention_size - token_len_per_seq) * n_state.
        // For encoder self-attn and cross-attn the offset is fixed at graph-compile time.
        ov::Output<ov::Node> start;
        اگر (is_decoder_self_attn && context.has_داخلہ("attention_size") && context.has_داخلہ("token_len_per_seq")) {
            خودکار attention_سائز_اندر = context.get_داخلہ("attention_size");
            خودکار token_len_اندر = context.get_داخلہ("token_len_per_seq");
            خودکار past_علامات = std::make_shared<ov::op::v1::تفریق>(attention_سائز_اندر, token_len_اندر);
            خودکار new_آغاز = std::make_shared<ov::op::v1::Multiply>(past_علامات, n_state_c);
            start = std::make_shared<ov::op::v1::جوڑو>(
                new_آغاز, ov::op::v0::Constant::create(ov::element::i64, {1}, {آغاز_elem}));
        } ورنہ {
            start = ov::op::v0::Constant::create(ov::element::i64, {1}, {آغاز_elem});
        }
        خودکار آغاز_squeezed = std::make_shared<ov::op::v0::Squeeze>(start);
        خودکار end = std::make_shared<ov::op::v1::جوڑو>(آغاز_squeezed, n_write_dyn);
        خودکار انتہا_squeezed = std::make_shared<ov::op::v0::Squeeze>(end);
        خودکار قدم = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار قدم_squeezed = std::make_shared<ov::op::v0::Squeeze>(قدم);
        خودکار indices =
            std::make_shared<ov::op::v4::Range>(آغاز_squeezed, انتہا_squeezed, قدم_squeezed, ov::element::i64);
        خودکار axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        خودکار kv_updated = std::make_shared<ov::op::v3::ScatterUpdate>(kv_buf, indices, ڈیٹا, axis);
        واپس rename_outputs_with_suffix({kv_updated}, context.get_name());
    }

    اگر (داخلہ_shape != output_shape) {
        خودکار new_shape = ov::op::v0::Constant::create(
            ov::element::i64, {static_cast<size_t>(output_shape.rank().get_لمبائی())}, output_shape.مقصد_shape());
        داخلہ = std::make_shared<ov::op::v1::Reshape>(داخلہ, new_shape, جھوٹ);
    }

    ov::Output<Node> res;
    اگر (context.get_داخلہ_type(0) != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(داخلہ, context.get_output_type());
    } ورنہ {
        res = داخلہ;
    }

    اگر (res.get_گرہ_shared_ptr() == context.get_داخلہ(0).get_گرہ_shared_ptr()) {
        واپس {res};
    }

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
