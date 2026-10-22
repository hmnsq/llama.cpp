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

Output_وکتور translate_cpy(ثابت گره_Context & context) {
    خودکار op_case = context.get_op_case();
    خودکار درون_داد_shape = context.get_درون_داد_shape(0);
    خودکار output_shape = context.get_درون_داد_shape(1);

    إذا (op_case == 4) {
        خودکار src = process_view_درون_داد_new(context, 0);
        خودکار پایه = context.get_درون_داد(1);

        int64_t n_elems = 1;
        لكل (ثابت خودکار & dim : context.get_output_shape().مقصد_shape()) {
            n_elems *= static_cast<int64_t>(dim);
        }

        ثابت خودکار output_stride = context.get_output_stride();
        ثابت size_t elem_اندازه = output_stride.خالی() ? context.get_output_type().size() : output_stride.back();
        FRONT_انتها_OP_CONVERSION_CHECK(elem_اندازه > 0, "CPY conv state view update has invalid element size");

        ثابت int64_t begin_val = static_cast<int64_t>(context.get_output_op_جابه_جایی() / elem_اندازه);
        ثابت int64_t انتها_val = begin_val + n_elems;

        خودکار flat_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, -1});
        src = std::make_shared<ov::op::v1::Reshape>(src, flat_shape, خطأ);
        إذا (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        خودکار zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خودکار begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val});
        خودکار end = ov::op::v0::Constant::create(ov::element::i64, {1}, {انتها_val});
        خودکار int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {بیشترین_صحیح});
        خودکار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        خودکار سر_part = std::make_shared<ov::op::v8::برش>(پایه, zero, begin, one, axis);
        خودکار دم_part = std::make_shared<ov::op::v8::برش>(پایه, end, int_max, one, axis);
        خودکار res = std::make_shared<ov::op::v0::Concat>(ov::Output_وکتور{سر_part, src, دم_part}, 3);
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
        منطقي is_خالی = خطأ;
        إذا (درون_داد_shape.rank().is_static()) {
            لكل (ثابت خودکار & d : درون_داد_shape) {
                إذا (d.is_static() && d.get_درازی() == 0) {
                    is_خالی = درست;
                    اكسر;
                }
            }
        }
        إذا (is_خالی) {
            أرجع {context.get_درون_داد(1)};
        }
    }
    // op_case 7/8/9 are the single-slot variants; op_case 10 writes native GDN state into a
    // multi-slot cache without rollback snapshots.
    ثابت منطقي single_جایگاه_تخصیص_بده = op_case >= 7 && op_case <= 9;
    ثابت منطقي مستقیم_gdn_state = op_case == 7 || op_case == 10;
    صحيح writeback_case = op_case;
    إذا (op_case == 10) {
        writeback_case = 1;
    } وإلا إذا (single_جایگاه_تخصیص_بده) {
        writeback_case = op_case - 6;
    }
    ثابت std::string جایگاه_begin_name = "rs_slot_begin_" + context.get_name();
    ثابت منطقي برش_تخصیص_بده = writeback_case >= 1 && writeback_case <= 3 &&
                              (single_جایگاه_تخصیص_بده || context.has_درون_داد(جایگاه_begin_name));
    إذا (برش_تخصیص_بده) {
        إذا (single_جایگاه_تخصیص_بده && writeback_case == 3) {
            أرجع {context.get_درون_داد(1)};
        }

        ثابت int64_t جایگاه_axis = 2;
        خودکار zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خودکار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {بیشترین_صحیح});
        خودکار axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {جایگاه_axis});
        خودکار feature = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                    std::vector<int64_t>{1, 1, -1, output_shape[3].get_درازی()});

        ov::Output<ov::Node> src;
        ov::Output<ov::Node> begin;
        إذا (!single_جایگاه_تخصیص_بده) {
            begin = context.get_درون_داد(جایگاه_begin_name);
        }
        خودکار پایه = context.get_درون_داد(1);
        إذا (writeback_case == 1) {
            إذا (مستقیم_gdn_state) {
                // Non-rollback GDN publishes state directly as [active_slots, heads, value_dim,
                // key_dim]. Flatten each active slot before replacing or updating the cache.
                src = std::make_shared<ov::op::v1::Reshape>(context.get_درون_داد(0), feature, خطأ);
            } وإلا {
                // Multi-slot rollback still consumes GGML's packed [attention | state snapshots]
                // layout. Slice the state block using the runtime source offset.
                خودکار src_begin = context.get_درون_داد("rs_src_begin_" + context.get_name());
                خودکار state_part =
                    std::make_shared<ov::op::v8::برش>(context.get_درون_داد(0), src_begin, int_max, one, axis);
                src = std::make_shared<ov::op::v1::Reshape>(state_part, feature, خطأ);
            }
        } وإلا إذا (writeback_case == 2) {
            // conv_input is [previous conv state | new tokens]; the snapshot is the conv_kernel_size - 1
            // columns ending at the last *valid* token. Gather (rather than Slice) keeps the output
            // shape static even though the window start is a runtime value.
            خودکار پنجره_اندازه = (int64_t) درون_داد_shape[3].get_درازی();
            ov::Output<ov::Node> پنجره;
            خودکار ستون_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            ثابت std::string src_begin_name = "rs_src_begin_" + context.get_name();
            إذا (context.has_درون_داد(src_begin_name)) {
                خودکار src_begin = context.get_درون_داد(src_begin_name);
                خودکار src_انتها = std::make_shared<ov::op::v1::جمع_کن>(
                    src_begin, ov::op::v0::Constant::create(ov::element::i64, {1}, {پنجره_اندازه}));
                پنجره = std::make_shared<ov::op::v8::برش>(context.get_درون_داد(0), src_begin, src_انتها, one, ستون_axis);
            } وإلا إذا (context.has_درون_داد("chunk_valid_len")) {
                std::vector<int64_t> offsets(پنجره_اندازه);
                std::iota(offsets.begin(), offsets.end(), 0);
                خودکار indices = std::make_shared<ov::op::v1::جمع_کن>(
                    ov::op::v0::Constant::create(ov::element::i64, {(size_t) پنجره_اندازه}, offsets),
                    context.get_درون_داد("chunk_valid_len"));
                پنجره = std::make_shared<ov::op::v8::Gather>(context.get_درون_داد(0), indices, ستون_axis);
            } وإلا {
                خودکار پنجره_begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {-پنجره_اندازه});
                پنجره =
                    std::make_shared<ov::op::v8::برش>(context.get_درون_داد(0), پنجره_begin, int_max, one, ستون_axis);
            }
            ثابت خودکار پایه_shape = پایه.get_partial_shape();
            FRONT_انتها_OP_CONVERSION_CHECK(پایه_shape.rank().is_static() && پایه_shape.rank().get_درازی() == 4,
                                          "CPY conv state cache update requires rank-4 base cache");
            FRONT_انتها_OP_CONVERSION_CHECK(پایه_shape[3].is_static(),
                                          "CPY conv state cache update requires static feature size");
            FRONT_انتها_OP_CONVERSION_CHECK(درون_داد_shape.rank().is_static() && درون_داد_shape.rank().get_درازی() == 4 &&
                                              درون_داد_shape[2].is_static() && درون_داد_shape[3].is_static(),
                                          "CPY conv state cache update requires static source feature view");

            ثابت int64_t پر_feature_اندازه = پایه_shape[3].get_درازی();
            ثابت int64_t update_feature_اندازه = درون_داد_shape[2].get_درازی() * درون_داد_shape[3].get_درازی();
            ثابت خودکار output_stride = context.get_output_stride();
            ثابت size_t elem_اندازه = output_stride.خالی() ? context.get_output_type().size() : output_stride.back();
            FRONT_انتها_OP_CONVERSION_CHECK(elem_اندازه > 0,
                                          "CPY conv state cache update has invalid element size");
            ثابت int64_t feature_begin = static_cast<int64_t>(context.get_output_op_جابه_جایی() / elem_اندازه) %
                                          پر_feature_اندازه;
            ثابت int64_t feature_انتها = feature_begin + update_feature_اندازه;
            FRONT_انتها_OP_CONVERSION_CHECK(feature_begin >= 0 && feature_انتها <= پر_feature_اندازه,
                                          "CPY conv state cache update feature range is out of bounds");

            خودکار partial_feature = ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, update_feature_اندازه});
            src = std::make_shared<ov::op::v1::Reshape>(پنجره, partial_feature, خطأ);
            إذا (src.get_element_type() != context.get_output_type()) {
                src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
            }

            إذا (single_جایگاه_تخصیص_بده) {
                أرجع rename_outputs_with_suffix({src}, context.get_name());
            }

            خودکار src_len = std::make_shared<ov::op::v8::Gather>(
                std::make_shared<ov::op::v3::Shape_از>(src, ov::element::i64), axis,
                ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
            خودکار جایگاه_انتها = std::make_shared<ov::op::v1::جمع_کن>(begin, src_len);
            خودکار فعال_slots = std::make_shared<ov::op::v8::برش>(پایه, begin, جایگاه_انتها, one, axis);

            خودکار feature_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            خودکار feature_begin_گره = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_begin});
            خودکار feature_انتها_گره = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_انتها});
            خودکار feature_سر = std::make_shared<ov::op::v8::برش>(فعال_slots, zero, feature_begin_گره, one,
                                                                    feature_axis);
            خودکار feature_دم = std::make_shared<ov::op::v8::برش>(فعال_slots, feature_انتها_گره, int_max, one,
                                                                    feature_axis);
            src = std::make_shared<ov::op::v0::Concat>(ov::Output_وکتور{feature_سر, src, feature_دم}, 3);
        } وإلا {
            // op_case 3: gathered remainder rows already have the cache slot layout [1, 1, extra, feature]
            src = context.get_درون_داد(0);
        }

        إذا (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        إذا (single_جایگاه_تخصیص_بده) {
            أرجع rename_outputs_with_suffix({src}, context.get_name());
        }

        خودکار src_len =
            std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::Shape_از>(src, ov::element::i64), axis,
                                                 ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        خودکار end = std::make_shared<ov::op::v1::جمع_کن>(begin, src_len);
        خودکار سر_part = std::make_shared<ov::op::v8::برش>(پایه, zero, begin, one, axis);
        خودکار دم_part = std::make_shared<ov::op::v8::برش>(پایه, end, int_max, one, axis);
        خودکار res = std::make_shared<ov::op::v0::Concat>(ov::Output_وکتور{سر_part, src, دم_part}, جایگاه_axis);
        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    خودکار درون_داد = process_view_درون_داد_new(context, 0);

    إذا (op_case == 5 || op_case == 6) {
        خودکار درون_داد_shape = context.get_درون_داد_shape(0);
        خودکار output_shape = context.get_output_shape();
        خودکار dst_ggml_shape = context.get_view_درون_داد_ggml_shape(1, 0);
        خودکار dst_stride = context.get_view_درون_داد_stride(1, 0);
        size_t جابه_جایی_بایتها = context.get_view_درون_داد_جابه_جایی(1, 0);
        خودکار n_state = (int64_t) context.get_درون_داد_shape(0)[3].get_درازی();
        خودکار n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
        خودکار kv_buf = context.get_درون_داد(1);  // shape {1,1,1,N}

        Output<Node> token_len_در_هر_seq;
        Output<Node> n_write_dyn;
        إذا (context.has_درون_داد("token_len_per_seq")) {
            token_len_در_هر_seq = context.get_درون_داد("token_len_per_seq");
            n_write_dyn = std::make_shared<ov::op::v1::Multiply>(token_len_در_هر_seq, n_state_c);
        } وإلا {
            n_write_dyn = ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) dst_ggml_shape[3]});
        }
        size_t elem_اندازه = dst_stride[3];
        FRONT_انتها_OP_CONVERSION_CHECK(elem_اندازه > 0, "CPY KV cache view update has invalid element size");
        int64_t شروع_elem = (int64_t) (جابه_جایی_بایتها / elem_اندازه);
        // op_case 5: decoder self-attention – write offset advances each step.
        // op_case 6: encoder self-attn or cross-attn – offset fixed at compile time.
        ثابت منطقي is_decoder_self_attn = (op_case == 5);
        خودکار ones_c = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{1, 1, 1});
        خودکار new_shape = std::make_shared<ov::op::v0::Concat>(ov::Output_وکتور{ones_c, n_write_dyn}, 0);

        خودکار reshaped = std::make_shared<ov::op::v1::Reshape>(درون_داد, new_shape, خطأ);
        خودکار داده = std::make_shared<ov::op::v0::Convert>(reshaped, context.get_output_type());
        // Indices [start_elem .. start_elem + n_write) on axis 3 of {1,1,1,N}
        // For decoder self-attention the write offset advances each step, so compute it
        // dynamically from the model inputs: start = (attention_size - token_len_per_seq) * n_state.
        // For encoder self-attn and cross-attn the offset is fixed at graph-compile time.
        ov::Output<ov::Node> start;
        إذا (is_decoder_self_attn && context.has_درون_داد("attention_size") && context.has_درون_داد("token_len_per_seq")) {
            خودکار attention_اندازه_درون = context.get_درون_داد("attention_size");
            خودکار token_len_درون = context.get_درون_داد("token_len_per_seq");
            خودکار past_نشانهها = std::make_shared<ov::op::v1::تفریق>(attention_اندازه_درون, token_len_درون);
            خودکار new_شروع = std::make_shared<ov::op::v1::Multiply>(past_نشانهها, n_state_c);
            start = std::make_shared<ov::op::v1::جمع_کن>(
                new_شروع, ov::op::v0::Constant::create(ov::element::i64, {1}, {شروع_elem}));
        } وإلا {
            start = ov::op::v0::Constant::create(ov::element::i64, {1}, {شروع_elem});
        }
        خودکار شروع_squeezed = std::make_shared<ov::op::v0::Squeeze>(start);
        خودکار end = std::make_shared<ov::op::v1::جمع_کن>(شروع_squeezed, n_write_dyn);
        خودکار انتها_squeezed = std::make_shared<ov::op::v0::Squeeze>(end);
        خودکار گام = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار گام_squeezed = std::make_shared<ov::op::v0::Squeeze>(گام);
        خودکار indices =
            std::make_shared<ov::op::v4::Range>(شروع_squeezed, انتها_squeezed, گام_squeezed, ov::element::i64);
        خودکار axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        خودکار kv_updated = std::make_shared<ov::op::v3::ScatterUpdate>(kv_buf, indices, داده, axis);
        أرجع rename_outputs_with_suffix({kv_updated}, context.get_name());
    }

    إذا (درون_داد_shape != output_shape) {
        خودکار new_shape = ov::op::v0::Constant::create(
            ov::element::i64, {static_cast<size_t>(output_shape.rank().get_درازی())}, output_shape.مقصد_shape());
        درون_داد = std::make_shared<ov::op::v1::Reshape>(درون_داد, new_shape, خطأ);
    }

    ov::Output<Node> res;
    إذا (context.get_درون_داد_type(0) != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(درون_داد, context.get_output_type());
    } وإلا {
        res = درون_داد;
    }

    إذا (res.get_گره_shared_ptr() == context.get_درون_داد(0).get_گره_shared_ptr()) {
        أرجع {res};
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
