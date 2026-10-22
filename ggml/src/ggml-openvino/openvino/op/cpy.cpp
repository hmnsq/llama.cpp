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

Output_ويڪٽر translate_cpy(ثابت ڳنڍ_Context & context) {
    خودڪار op_case = context.get_op_case();
    خودڪار داخلا_shape = context.get_داخلا_shape(0);
    خودڪار output_shape = context.get_داخلا_shape(1);

    إذا (op_case == 4) {
        خودڪار src = process_view_داخلا_new(context, 0);
        خودڪار بنياد = context.get_داخلا(1);

        int64_t n_elems = 1;
        لكل (ثابت خودڪار & dim : context.get_output_shape().منزل_shape()) {
            n_elems *= static_cast<int64_t>(dim);
        }

        ثابت خودڪار output_stride = context.get_output_stride();
        ثابت size_t elem_سائيز = output_stride.خالي_جايون() ? context.get_output_type().size() : output_stride.back();
        FRONT_پڇاڙي_OP_CONVERSION_CHECK(elem_سائيز > 0, "CPY conv state view update has invalid element size");

        ثابت int64_t begin_val = static_cast<int64_t>(context.get_output_op_وٿي() / elem_سائيز);
        ثابت int64_t پڇاڙي_val = begin_val + n_elems;

        خودڪار flat_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, -1});
        src = std::make_shared<ov::op::v1::Reshape>(src, flat_shape, خطأ);
        إذا (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        خودڪار zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خودڪار begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val});
        خودڪار end = ov::op::v0::Constant::create(ov::element::i64, {1}, {پڇاڙي_val});
        خودڪار int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {وڌ_کان_وڌ_عدد});
        خودڪار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودڪار axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        خودڪار سر_part = std::make_shared<ov::op::v8::ٽڪرو>(بنياد, zero, begin, one, axis);
        خودڪار پڇ_part = std::make_shared<ov::op::v8::ٽڪرو>(بنياد, end, int_max, one, axis);
        خودڪار res = std::make_shared<ov::op::v0::Concat>(ov::Output_ويڪٽر{سر_part, src, پڇ_part}, 3);
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
        منطقي is_خالي_جايون = خطأ;
        إذا (داخلا_shape.rank().is_static()) {
            لكل (ثابت خودڪار & d : داخلا_shape) {
                إذا (d.is_static() && d.get_ڊيگهائي() == 0) {
                    is_خالي_جايون = سچ;
                    اكسر;
                }
            }
        }
        إذا (is_خالي_جايون) {
            أرجع {context.get_داخلا(1)};
        }
    }
    // op_case 7/8/9 are the single-slot variants; op_case 10 writes native GDN state into a
    // multi-slot cache without rollback snapshots.
    ثابت منطقي single_جڳهه_سونپيو = op_case >= 7 && op_case <= 9;
    ثابت منطقي سڌو_gdn_state = op_case == 7 || op_case == 10;
    صحيح writeback_case = op_case;
    إذا (op_case == 10) {
        writeback_case = 1;
    } وإلا إذا (single_جڳهه_سونپيو) {
        writeback_case = op_case - 6;
    }
    ثابت std::string جڳهه_begin_name = "rs_slot_begin_" + context.get_name();
    ثابت منطقي ٽڪرو_سونپيو = writeback_case >= 1 && writeback_case <= 3 &&
                              (single_جڳهه_سونپيو || context.has_داخلا(جڳهه_begin_name));
    إذا (ٽڪرو_سونپيو) {
        إذا (single_جڳهه_سونپيو && writeback_case == 3) {
            أرجع {context.get_داخلا(1)};
        }

        ثابت int64_t جڳهه_axis = 2;
        خودڪار zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خودڪار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودڪار int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {وڌ_کان_وڌ_عدد});
        خودڪار axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {جڳهه_axis});
        خودڪار feature = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                    std::vector<int64_t>{1, 1, -1, output_shape[3].get_ڊيگهائي()});

        ov::Output<ov::Node> src;
        ov::Output<ov::Node> begin;
        إذا (!single_جڳهه_سونپيو) {
            begin = context.get_داخلا(جڳهه_begin_name);
        }
        خودڪار بنياد = context.get_داخلا(1);
        إذا (writeback_case == 1) {
            إذا (سڌو_gdn_state) {
                // Non-rollback GDN publishes state directly as [active_slots, heads, value_dim,
                // key_dim]. Flatten each active slot before replacing or updating the cache.
                src = std::make_shared<ov::op::v1::Reshape>(context.get_داخلا(0), feature, خطأ);
            } وإلا {
                // Multi-slot rollback still consumes GGML's packed [attention | state snapshots]
                // layout. Slice the state block using the runtime source offset.
                خودڪار src_begin = context.get_داخلا("rs_src_begin_" + context.get_name());
                خودڪار state_part =
                    std::make_shared<ov::op::v8::ٽڪرو>(context.get_داخلا(0), src_begin, int_max, one, axis);
                src = std::make_shared<ov::op::v1::Reshape>(state_part, feature, خطأ);
            }
        } وإلا إذا (writeback_case == 2) {
            // conv_input is [previous conv state | new tokens]; the snapshot is the conv_kernel_size - 1
            // columns ending at the last *valid* token. Gather (rather than Slice) keeps the output
            // shape static even though the window start is a runtime value.
            خودڪار دري_سائيز = (int64_t) داخلا_shape[3].get_ڊيگهائي();
            ov::Output<ov::Node> دري;
            خودڪار کانو_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            ثابت std::string src_begin_name = "rs_src_begin_" + context.get_name();
            إذا (context.has_داخلا(src_begin_name)) {
                خودڪار src_begin = context.get_داخلا(src_begin_name);
                خودڪار src_پڇاڙي = std::make_shared<ov::op::v1::جوڙيو>(
                    src_begin, ov::op::v0::Constant::create(ov::element::i64, {1}, {دري_سائيز}));
                دري = std::make_shared<ov::op::v8::ٽڪرو>(context.get_داخلا(0), src_begin, src_پڇاڙي, one, کانو_axis);
            } وإلا إذا (context.has_داخلا("chunk_valid_len")) {
                std::vector<int64_t> offsets(دري_سائيز);
                std::iota(offsets.begin(), offsets.end(), 0);
                خودڪار indices = std::make_shared<ov::op::v1::جوڙيو>(
                    ov::op::v0::Constant::create(ov::element::i64, {(size_t) دري_سائيز}, offsets),
                    context.get_داخلا("chunk_valid_len"));
                دري = std::make_shared<ov::op::v8::Gather>(context.get_داخلا(0), indices, کانو_axis);
            } وإلا {
                خودڪار دري_begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {-دري_سائيز});
                دري =
                    std::make_shared<ov::op::v8::ٽڪرو>(context.get_داخلا(0), دري_begin, int_max, one, کانو_axis);
            }
            ثابت خودڪار بنياد_shape = بنياد.get_partial_shape();
            FRONT_پڇاڙي_OP_CONVERSION_CHECK(بنياد_shape.rank().is_static() && بنياد_shape.rank().get_ڊيگهائي() == 4,
                                          "CPY conv state cache update requires rank-4 base cache");
            FRONT_پڇاڙي_OP_CONVERSION_CHECK(بنياد_shape[3].is_static(),
                                          "CPY conv state cache update requires static feature size");
            FRONT_پڇاڙي_OP_CONVERSION_CHECK(داخلا_shape.rank().is_static() && داخلا_shape.rank().get_ڊيگهائي() == 4 &&
                                              داخلا_shape[2].is_static() && داخلا_shape[3].is_static(),
                                          "CPY conv state cache update requires static source feature view");

            ثابت int64_t ڀريل_feature_سائيز = بنياد_shape[3].get_ڊيگهائي();
            ثابت int64_t update_feature_سائيز = داخلا_shape[2].get_ڊيگهائي() * داخلا_shape[3].get_ڊيگهائي();
            ثابت خودڪار output_stride = context.get_output_stride();
            ثابت size_t elem_سائيز = output_stride.خالي_جايون() ? context.get_output_type().size() : output_stride.back();
            FRONT_پڇاڙي_OP_CONVERSION_CHECK(elem_سائيز > 0,
                                          "CPY conv state cache update has invalid element size");
            ثابت int64_t feature_begin = static_cast<int64_t>(context.get_output_op_وٿي() / elem_سائيز) %
                                          ڀريل_feature_سائيز;
            ثابت int64_t feature_پڇاڙي = feature_begin + update_feature_سائيز;
            FRONT_پڇاڙي_OP_CONVERSION_CHECK(feature_begin >= 0 && feature_پڇاڙي <= ڀريل_feature_سائيز,
                                          "CPY conv state cache update feature range is out of bounds");

            خودڪار partial_feature = ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, update_feature_سائيز});
            src = std::make_shared<ov::op::v1::Reshape>(دري, partial_feature, خطأ);
            إذا (src.get_element_type() != context.get_output_type()) {
                src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
            }

            إذا (single_جڳهه_سونپيو) {
                أرجع rename_outputs_with_suffix({src}, context.get_name());
            }

            خودڪار src_len = std::make_shared<ov::op::v8::Gather>(
                std::make_shared<ov::op::v3::Shape_جو>(src, ov::element::i64), axis,
                ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
            خودڪار جڳهه_پڇاڙي = std::make_shared<ov::op::v1::جوڙيو>(begin, src_len);
            خودڪار سرگرم_slots = std::make_shared<ov::op::v8::ٽڪرو>(بنياد, begin, جڳهه_پڇاڙي, one, axis);

            خودڪار feature_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            خودڪار feature_begin_ڳنڍ = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_begin});
            خودڪار feature_پڇاڙي_ڳنڍ = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_پڇاڙي});
            خودڪار feature_سر = std::make_shared<ov::op::v8::ٽڪرو>(سرگرم_slots, zero, feature_begin_ڳنڍ, one,
                                                                    feature_axis);
            خودڪار feature_پڇ = std::make_shared<ov::op::v8::ٽڪرو>(سرگرم_slots, feature_پڇاڙي_ڳنڍ, int_max, one,
                                                                    feature_axis);
            src = std::make_shared<ov::op::v0::Concat>(ov::Output_ويڪٽر{feature_سر, src, feature_پڇ}, 3);
        } وإلا {
            // op_case 3: gathered remainder rows already have the cache slot layout [1, 1, extra, feature]
            src = context.get_داخلا(0);
        }

        إذا (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        إذا (single_جڳهه_سونپيو) {
            أرجع rename_outputs_with_suffix({src}, context.get_name());
        }

        خودڪار src_len =
            std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::Shape_جو>(src, ov::element::i64), axis,
                                                 ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        خودڪار end = std::make_shared<ov::op::v1::جوڙيو>(begin, src_len);
        خودڪار سر_part = std::make_shared<ov::op::v8::ٽڪرو>(بنياد, zero, begin, one, axis);
        خودڪار پڇ_part = std::make_shared<ov::op::v8::ٽڪرو>(بنياد, end, int_max, one, axis);
        خودڪار res = std::make_shared<ov::op::v0::Concat>(ov::Output_ويڪٽر{سر_part, src, پڇ_part}, جڳهه_axis);
        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);

    إذا (op_case == 5 || op_case == 6) {
        خودڪار داخلا_shape = context.get_داخلا_shape(0);
        خودڪار output_shape = context.get_output_shape();
        خودڪار dst_ggml_shape = context.get_view_داخلا_ggml_shape(1, 0);
        خودڪار dst_stride = context.get_view_داخلا_stride(1, 0);
        size_t وٿي_بائيٽون = context.get_view_داخلا_وٿي(1, 0);
        خودڪار n_state = (int64_t) context.get_داخلا_shape(0)[3].get_ڊيگهائي();
        خودڪار n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
        خودڪار kv_buf = context.get_داخلا(1);  // shape {1,1,1,N}

        Output<Node> token_len_في_seq;
        Output<Node> n_write_dyn;
        إذا (context.has_داخلا("token_len_per_seq")) {
            token_len_في_seq = context.get_داخلا("token_len_per_seq");
            n_write_dyn = std::make_shared<ov::op::v1::Multiply>(token_len_في_seq, n_state_c);
        } وإلا {
            n_write_dyn = ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) dst_ggml_shape[3]});
        }
        size_t elem_سائيز = dst_stride[3];
        FRONT_پڇاڙي_OP_CONVERSION_CHECK(elem_سائيز > 0, "CPY KV cache view update has invalid element size");
        int64_t شروعات_elem = (int64_t) (وٿي_بائيٽون / elem_سائيز);
        // op_case 5: decoder self-attention – write offset advances each step.
        // op_case 6: encoder self-attn or cross-attn – offset fixed at compile time.
        ثابت منطقي is_decoder_self_attn = (op_case == 5);
        خودڪار ones_c = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{1, 1, 1});
        خودڪار new_shape = std::make_shared<ov::op::v0::Concat>(ov::Output_ويڪٽر{ones_c, n_write_dyn}, 0);

        خودڪار reshaped = std::make_shared<ov::op::v1::Reshape>(داخلا_2, new_shape, خطأ);
        خودڪار معلومات = std::make_shared<ov::op::v0::Convert>(reshaped, context.get_output_type());
        // Indices [start_elem .. start_elem + n_write) on axis 3 of {1,1,1,N}
        // For decoder self-attention the write offset advances each step, so compute it
        // dynamically from the model inputs: start = (attention_size - token_len_per_seq) * n_state.
        // For encoder self-attn and cross-attn the offset is fixed at graph-compile time.
        ov::Output<ov::Node> start;
        إذا (is_decoder_self_attn && context.has_داخلا("attention_size") && context.has_داخلا("token_len_per_seq")) {
            خودڪار attention_سائيز_اندريون = context.get_داخلا("attention_size");
            خودڪار token_len_اندريون = context.get_داخلا("token_len_per_seq");
            خودڪار past_ٽوڪن = std::make_shared<ov::op::v1::ڪٽيو>(attention_سائيز_اندريون, token_len_اندريون);
            خودڪار new_شروعات = std::make_shared<ov::op::v1::Multiply>(past_ٽوڪن, n_state_c);
            start = std::make_shared<ov::op::v1::جوڙيو>(
                new_شروعات, ov::op::v0::Constant::create(ov::element::i64, {1}, {شروعات_elem}));
        } وإلا {
            start = ov::op::v0::Constant::create(ov::element::i64, {1}, {شروعات_elem});
        }
        خودڪار شروعات_squeezed = std::make_shared<ov::op::v0::Squeeze>(start);
        خودڪار end = std::make_shared<ov::op::v1::جوڙيو>(شروعات_squeezed, n_write_dyn);
        خودڪار پڇاڙي_squeezed = std::make_shared<ov::op::v0::Squeeze>(end);
        خودڪار قدم = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودڪار قدم_squeezed = std::make_shared<ov::op::v0::Squeeze>(قدم);
        خودڪار indices =
            std::make_shared<ov::op::v4::Range>(شروعات_squeezed, پڇاڙي_squeezed, قدم_squeezed, ov::element::i64);
        خودڪار axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        خودڪار kv_updated = std::make_shared<ov::op::v3::ScatterUpdate>(kv_buf, indices, معلومات, axis);
        أرجع rename_outputs_with_suffix({kv_updated}, context.get_name());
    }

    إذا (داخلا_shape != output_shape) {
        خودڪار new_shape = ov::op::v0::Constant::create(
            ov::element::i64, {static_cast<size_t>(output_shape.rank().get_ڊيگهائي())}, output_shape.منزل_shape());
        داخلا_2 = std::make_shared<ov::op::v1::Reshape>(داخلا_2, new_shape, خطأ);
    }

    ov::Output<Node> res;
    إذا (context.get_داخلا_type(0) != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(داخلا_2, context.get_output_type());
    } وإلا {
        res = داخلا_2;
    }

    إذا (res.get_ڳنڍ_shared_ptr() == context.get_داخلا(0).get_ڳنڍ_shared_ptr()) {
        أرجع {res};
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
