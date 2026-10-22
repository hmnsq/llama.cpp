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

Output_ویکټور translate_cpy(ثابت غوټه_Context & context) {
    خپلکاری op_case = context.get_op_case();
    خپلکاری ننوت_shape = context.get_ننوت_shape(0);
    خپلکاری output_shape = context.get_ننوت_shape(1);

    إذا (op_case == 4) {
        خپلکاری src = process_view_ننوت_new(context, 0);
        خپلکاری بنسټ = context.get_ننوت(1);

        int64_t n_elems = 1;
        لكل (ثابت خپلکاری & dim : context.get_output_shape().منزل_shape()) {
            n_elems *= static_cast<int64_t>(dim);
        }

        ثابت خپلکاری output_stride = context.get_output_stride();
        ثابت size_t elem_اندازه = output_stride.تشه() ? context.get_output_type().size() : output_stride.back();
        FRONT_پای_ټکی_OP_CONVERSION_CHECK(elem_اندازه > 0, "CPY conv state view update has invalid element size");

        ثابت int64_t begin_val = static_cast<int64_t>(context.get_output_op_دننه_واټن() / elem_اندازه);
        ثابت int64_t پای_ټکی_val = begin_val + n_elems;

        خپلکاری flat_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, -1});
        src = std::make_shared<ov::op::v1::Reshape>(src, flat_shape, خطأ);
        إذا (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        خپلکاری zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خپلکاری begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val});
        خپلکاری end = ov::op::v0::Constant::create(ov::element::i64, {1}, {پای_ټکی_val});
        خپلکاری int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {لوی_عدد});
        خپلکاری one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خپلکاری axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        خپلکاری سر_part = std::make_shared<ov::op::v8::ټوټه>(بنسټ, zero, begin, one, axis);
        خپلکاری لکۍ_part = std::make_shared<ov::op::v8::ټوټه>(بنسټ, end, int_max, one, axis);
        خپلکاری res = std::make_shared<ov::op::v0::Concat>(ov::Output_ویکټور{سر_part, src, لکۍ_part}, 3);
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
        منطقي is_تشه = خطأ;
        إذا (ننوت_shape.rank().is_static()) {
            لكل (ثابت خپلکاری & d : ننوت_shape) {
                إذا (d.is_static() && d.get_اوږدتیا() == 0) {
                    is_تشه = ریښتیا;
                    اكسر;
                }
            }
        }
        إذا (is_تشه) {
            أرجع {context.get_ننوت(1)};
        }
    }
    // op_case 7/8/9 are the single-slot variants; op_case 10 writes native GDN state into a
    // multi-slot cache without rollback snapshots.
    ثابت منطقي single_خونه_ټاکل = op_case >= 7 && op_case <= 9;
    ثابت منطقي مستقیم_gdn_state = op_case == 7 || op_case == 10;
    صحيح writeback_case = op_case;
    إذا (op_case == 10) {
        writeback_case = 1;
    } وإلا إذا (single_خونه_ټاکل) {
        writeback_case = op_case - 6;
    }
    ثابت std::string خونه_begin_name = "rs_slot_begin_" + context.get_name();
    ثابت منطقي ټوټه_ټاکل = writeback_case >= 1 && writeback_case <= 3 &&
                              (single_خونه_ټاکل || context.has_ننوت(خونه_begin_name));
    إذا (ټوټه_ټاکل) {
        إذا (single_خونه_ټاکل && writeback_case == 3) {
            أرجع {context.get_ننوت(1)};
        }

        ثابت int64_t خونه_axis = 2;
        خپلکاری zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خپلکاری one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خپلکاری int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {لوی_عدد});
        خپلکاری axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {خونه_axis});
        خپلکاری feature = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                    std::vector<int64_t>{1, 1, -1, output_shape[3].get_اوږدتیا()});

        ov::Output<ov::Node> src;
        ov::Output<ov::Node> begin;
        إذا (!single_خونه_ټاکل) {
            begin = context.get_ننوت(خونه_begin_name);
        }
        خپلکاری بنسټ = context.get_ننوت(1);
        إذا (writeback_case == 1) {
            إذا (مستقیم_gdn_state) {
                // Non-rollback GDN publishes state directly as [active_slots, heads, value_dim,
                // key_dim]. Flatten each active slot before replacing or updating the cache.
                src = std::make_shared<ov::op::v1::Reshape>(context.get_ننوت(0), feature, خطأ);
            } وإلا {
                // Multi-slot rollback still consumes GGML's packed [attention | state snapshots]
                // layout. Slice the state block using the runtime source offset.
                خپلکاری src_begin = context.get_ننوت("rs_src_begin_" + context.get_name());
                خپلکاری state_part =
                    std::make_shared<ov::op::v8::ټوټه>(context.get_ننوت(0), src_begin, int_max, one, axis);
                src = std::make_shared<ov::op::v1::Reshape>(state_part, feature, خطأ);
            }
        } وإلا إذا (writeback_case == 2) {
            // conv_input is [previous conv state | new tokens]; the snapshot is the conv_kernel_size - 1
            // columns ending at the last *valid* token. Gather (rather than Slice) keeps the output
            // shape static even though the window start is a runtime value.
            خپلکاری کړکۍ_اندازه = (int64_t) ننوت_shape[3].get_اوږدتیا();
            ov::Output<ov::Node> کړکۍ;
            خپلکاری ستون_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            ثابت std::string src_begin_name = "rs_src_begin_" + context.get_name();
            إذا (context.has_ننوت(src_begin_name)) {
                خپلکاری src_begin = context.get_ننوت(src_begin_name);
                خپلکاری src_پای_ټکی = std::make_shared<ov::op::v1::جمع_کړه>(
                    src_begin, ov::op::v0::Constant::create(ov::element::i64, {1}, {کړکۍ_اندازه}));
                کړکۍ = std::make_shared<ov::op::v8::ټوټه>(context.get_ننوت(0), src_begin, src_پای_ټکی, one, ستون_axis);
            } وإلا إذا (context.has_ننوت("chunk_valid_len")) {
                std::vector<int64_t> offsets(کړکۍ_اندازه);
                std::iota(offsets.begin(), offsets.end(), 0);
                خپلکاری indices = std::make_shared<ov::op::v1::جمع_کړه>(
                    ov::op::v0::Constant::create(ov::element::i64, {(size_t) کړکۍ_اندازه}, offsets),
                    context.get_ننوت("chunk_valid_len"));
                کړکۍ = std::make_shared<ov::op::v8::Gather>(context.get_ننوت(0), indices, ستون_axis);
            } وإلا {
                خپلکاری کړکۍ_begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {-کړکۍ_اندازه});
                کړکۍ =
                    std::make_shared<ov::op::v8::ټوټه>(context.get_ننوت(0), کړکۍ_begin, int_max, one, ستون_axis);
            }
            ثابت خپلکاری بنسټ_shape = بنسټ.get_partial_shape();
            FRONT_پای_ټکی_OP_CONVERSION_CHECK(بنسټ_shape.rank().is_static() && بنسټ_shape.rank().get_اوږدتیا() == 4,
                                          "CPY conv state cache update requires rank-4 base cache");
            FRONT_پای_ټکی_OP_CONVERSION_CHECK(بنسټ_shape[3].is_static(),
                                          "CPY conv state cache update requires static feature size");
            FRONT_پای_ټکی_OP_CONVERSION_CHECK(ننوت_shape.rank().is_static() && ننوت_shape.rank().get_اوږدتیا() == 4 &&
                                              ننوت_shape[2].is_static() && ننوت_shape[3].is_static(),
                                          "CPY conv state cache update requires static source feature view");

            ثابت int64_t ډک_feature_اندازه = بنسټ_shape[3].get_اوږدتیا();
            ثابت int64_t update_feature_اندازه = ننوت_shape[2].get_اوږدتیا() * ننوت_shape[3].get_اوږدتیا();
            ثابت خپلکاری output_stride = context.get_output_stride();
            ثابت size_t elem_اندازه = output_stride.تشه() ? context.get_output_type().size() : output_stride.back();
            FRONT_پای_ټکی_OP_CONVERSION_CHECK(elem_اندازه > 0,
                                          "CPY conv state cache update has invalid element size");
            ثابت int64_t feature_begin = static_cast<int64_t>(context.get_output_op_دننه_واټن() / elem_اندازه) %
                                          ډک_feature_اندازه;
            ثابت int64_t feature_پای_ټکی = feature_begin + update_feature_اندازه;
            FRONT_پای_ټکی_OP_CONVERSION_CHECK(feature_begin >= 0 && feature_پای_ټکی <= ډک_feature_اندازه,
                                          "CPY conv state cache update feature range is out of bounds");

            خپلکاری partial_feature = ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, update_feature_اندازه});
            src = std::make_shared<ov::op::v1::Reshape>(کړکۍ, partial_feature, خطأ);
            إذا (src.get_element_type() != context.get_output_type()) {
                src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
            }

            إذا (single_خونه_ټاکل) {
                أرجع rename_outputs_with_suffix({src}, context.get_name());
            }

            خپلکاری src_len = std::make_shared<ov::op::v8::Gather>(
                std::make_shared<ov::op::v3::Shape_د>(src, ov::element::i64), axis,
                ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
            خپلکاری خونه_پای_ټکی = std::make_shared<ov::op::v1::جمع_کړه>(begin, src_len);
            خپلکاری فعال_slots = std::make_shared<ov::op::v8::ټوټه>(بنسټ, begin, خونه_پای_ټکی, one, axis);

            خپلکاری feature_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            خپلکاری feature_begin_غوټه = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_begin});
            خپلکاری feature_پای_ټکی_غوټه = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_پای_ټکی});
            خپلکاری feature_سر = std::make_shared<ov::op::v8::ټوټه>(فعال_slots, zero, feature_begin_غوټه, one,
                                                                    feature_axis);
            خپلکاری feature_لکۍ = std::make_shared<ov::op::v8::ټوټه>(فعال_slots, feature_پای_ټکی_غوټه, int_max, one,
                                                                    feature_axis);
            src = std::make_shared<ov::op::v0::Concat>(ov::Output_ویکټور{feature_سر, src, feature_لکۍ}, 3);
        } وإلا {
            // op_case 3: gathered remainder rows already have the cache slot layout [1, 1, extra, feature]
            src = context.get_ننوت(0);
        }

        إذا (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        إذا (single_خونه_ټاکل) {
            أرجع rename_outputs_with_suffix({src}, context.get_name());
        }

        خپلکاری src_len =
            std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::Shape_د>(src, ov::element::i64), axis,
                                                 ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        خپلکاری end = std::make_shared<ov::op::v1::جمع_کړه>(begin, src_len);
        خپلکاری سر_part = std::make_shared<ov::op::v8::ټوټه>(بنسټ, zero, begin, one, axis);
        خپلکاری لکۍ_part = std::make_shared<ov::op::v8::ټوټه>(بنسټ, end, int_max, one, axis);
        خپلکاری res = std::make_shared<ov::op::v0::Concat>(ov::Output_ویکټور{سر_part, src, لکۍ_part}, خونه_axis);
        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    خپلکاری ننوت = process_view_ننوت_new(context, 0);

    إذا (op_case == 5 || op_case == 6) {
        خپلکاری ننوت_shape = context.get_ننوت_shape(0);
        خپلکاری output_shape = context.get_output_shape();
        خپلکاری dst_ggml_shape = context.get_view_ننوت_ggml_shape(1, 0);
        خپلکاری dst_stride = context.get_view_ننوت_stride(1, 0);
        size_t دننه_واټن_بایټونه = context.get_view_ننوت_دننه_واټن(1, 0);
        خپلکاری n_state = (int64_t) context.get_ننوت_shape(0)[3].get_اوږدتیا();
        خپلکاری n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
        خپلکاری kv_buf = context.get_ننوت(1);  // shape {1,1,1,N}

        Output<Node> token_len_فی_seq;
        Output<Node> n_write_dyn;
        إذا (context.has_ننوت("token_len_per_seq")) {
            token_len_فی_seq = context.get_ننوت("token_len_per_seq");
            n_write_dyn = std::make_shared<ov::op::v1::Multiply>(token_len_فی_seq, n_state_c);
        } وإلا {
            n_write_dyn = ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) dst_ggml_shape[3]});
        }
        size_t elem_اندازه = dst_stride[3];
        FRONT_پای_ټکی_OP_CONVERSION_CHECK(elem_اندازه > 0, "CPY KV cache view update has invalid element size");
        int64_t پیل_elem = (int64_t) (دننه_واټن_بایټونه / elem_اندازه);
        // op_case 5: decoder self-attention – write offset advances each step.
        // op_case 6: encoder self-attn or cross-attn – offset fixed at compile time.
        ثابت منطقي is_decoder_self_attn = (op_case == 5);
        خپلکاری ones_c = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{1, 1, 1});
        خپلکاری new_shape = std::make_shared<ov::op::v0::Concat>(ov::Output_ویکټور{ones_c, n_write_dyn}, 0);

        خپلکاری reshaped = std::make_shared<ov::op::v1::Reshape>(ننوت, new_shape, خطأ);
        خپلکاری معلومات = std::make_shared<ov::op::v0::Convert>(reshaped, context.get_output_type());
        // Indices [start_elem .. start_elem + n_write) on axis 3 of {1,1,1,N}
        // For decoder self-attention the write offset advances each step, so compute it
        // dynamically from the model inputs: start = (attention_size - token_len_per_seq) * n_state.
        // For encoder self-attn and cross-attn the offset is fixed at graph-compile time.
        ov::Output<ov::Node> start;
        إذا (is_decoder_self_attn && context.has_ننوت("attention_size") && context.has_ننوت("token_len_per_seq")) {
            خپلکاری attention_اندازه_دننه = context.get_ننوت("attention_size");
            خپلکاری token_len_دننه = context.get_ننوت("token_len_per_seq");
            خپلکاری past_نښې = std::make_shared<ov::op::v1::کمول>(attention_اندازه_دننه, token_len_دننه);
            خپلکاری new_پیل = std::make_shared<ov::op::v1::Multiply>(past_نښې, n_state_c);
            start = std::make_shared<ov::op::v1::جمع_کړه>(
                new_پیل, ov::op::v0::Constant::create(ov::element::i64, {1}, {پیل_elem}));
        } وإلا {
            start = ov::op::v0::Constant::create(ov::element::i64, {1}, {پیل_elem});
        }
        خپلکاری پیل_squeezed = std::make_shared<ov::op::v0::Squeeze>(start);
        خپلکاری end = std::make_shared<ov::op::v1::جمع_کړه>(پیل_squeezed, n_write_dyn);
        خپلکاری پای_ټکی_squeezed = std::make_shared<ov::op::v0::Squeeze>(end);
        خپلکاری ګام = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خپلکاری ګام_squeezed = std::make_shared<ov::op::v0::Squeeze>(ګام);
        خپلکاری indices =
            std::make_shared<ov::op::v4::Range>(پیل_squeezed, پای_ټکی_squeezed, ګام_squeezed, ov::element::i64);
        خپلکاری axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        خپلکاری kv_updated = std::make_shared<ov::op::v3::ScatterUpdate>(kv_buf, indices, معلومات, axis);
        أرجع rename_outputs_with_suffix({kv_updated}, context.get_name());
    }

    إذا (ننوت_shape != output_shape) {
        خپلکاری new_shape = ov::op::v0::Constant::create(
            ov::element::i64, {static_cast<size_t>(output_shape.rank().get_اوږدتیا())}, output_shape.منزل_shape());
        ننوت = std::make_shared<ov::op::v1::Reshape>(ننوت, new_shape, خطأ);
    }

    ov::Output<Node> res;
    إذا (context.get_ننوت_type(0) != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(ننوت, context.get_output_type());
    } وإلا {
        res = ننوت;
    }

    إذا (res.get_غوټه_shared_ptr() == context.get_ننوت(0).get_غوټه_shared_ptr()) {
        أرجع {res};
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
