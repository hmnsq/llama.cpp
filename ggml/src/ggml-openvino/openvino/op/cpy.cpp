#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <climits>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstdio>
#अन्तर्भाव <memory>
#अन्तर्भाव <numeric>
#अन्तर्भाव <openvino/frontend/exception.hpp>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/negative.hpp>
#अन्तर्भाव <openvino/op/range.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/scatter_update.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>
#अन्तर्भाव <openvino/op/subtract.hpp>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

Output_सदिशम् translate_cpy(ध्रुव ग्रन्थिः_Context & context) {
    स्वचालित op_case = context.get_op_case();
    स्वचालित निवेशः_shape = context.get_निवेशः_shape(0);
    स्वचालित output_shape = context.get_निवेशः_shape(1);

    यदि (op_case == 4) {
        स्वचालित src = process_view_निवेशः_new(context, 0);
        स्वचालित आधारः = context.get_निवेशः(1);

        int64_t n_elems = 1;
        प्रति (ध्रुव स्वचालित & dim : context.get_output_shape().गन्तव्यम्_shape()) {
            n_elems *= static_cast<int64_t>(dim);
        }

        ध्रुव स्वचालित output_stride = context.get_output_stride();
        ध्रुव size_t elem_आकार = output_stride.रिक्तानि() ? context.get_output_type().size() : output_stride.back();
        FRONT_अन्तः_OP_CONVERSION_CHECK(elem_आकार > 0, "CPY conv state view update has invalid element size");

        ध्रुव int64_t begin_val = static_cast<int64_t>(context.get_output_op_विस्थापनम्() / elem_आकार);
        ध्रुव int64_t अन्तः_val = begin_val + n_elems;

        स्वचालित flat_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, -1});
        src = std::make_shared<ov::op::v1::Reshape>(src, flat_shape, असत्य);
        यदि (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        स्वचालित zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        स्वचालित begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val});
        स्वचालित end = ov::op::v0::Constant::create(ov::element::i64, {1}, {अन्तः_val});
        स्वचालित int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {अधिकतम_पूर्णाङ्कः});
        स्वचालित one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वचालित axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        स्वचालित शीर्ष_part = std::make_shared<ov::op::v8::खण्डिका>(आधारः, zero, begin, one, axis);
        स्वचालित पुच्छ_part = std::make_shared<ov::op::v8::खण्डिका>(आधारः, end, int_max, one, axis);
        स्वचालित res = std::make_shared<ov::op::v0::Concat>(ov::Output_सदिशम्{शीर्ष_part, src, पुच्छ_part}, 3);
        प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
    }

    // Recurrent state cache writeback into a slot block of the cache. Where the block starts is a
    // runtime input, so the cached model works for any kv head and active sequence count. The
    // result is the full updated cache.
    // op_case 1: gated-delta-net state, op_case 2: conv state, op_case 3: defrag remainder.
    यदि (op_case == 3) {
        // With -np 1 (and generally whenever there is no defrag remainder) this GET_ROWS gathers
        // zero rows: nothing to write back, and the cache is unchanged. NPU rejects zero-size
        // tensors, so short-circuit instead of building a degenerate Slice/Concat chain.
        तार्किक is_रिक्तानि = असत्य;
        यदि (निवेशः_shape.rank().is_static()) {
            प्रति (ध्रुव स्वचालित & d : निवेशः_shape) {
                यदि (d.is_static() && d.get_दीर्घता() == 0) {
                    is_रिक्तानि = सत्य;
                    भङ्ग;
                }
            }
        }
        यदि (is_रिक्तानि) {
            प्रत्यागम {context.get_निवेशः(1)};
        }
    }
    // op_case 7/8/9 are the single-slot variants; op_case 10 writes native GDN state into a
    // multi-slot cache without rollback snapshots.
    ध्रुव तार्किक single_slot_नियोजय = op_case >= 7 && op_case <= 9;
    ध्रुव तार्किक प्रत्यक्षम्_gdn_state = op_case == 7 || op_case == 10;
    पूर्णाङ्क writeback_case = op_case;
    यदि (op_case == 10) {
        writeback_case = 1;
    } अन्यथा यदि (single_slot_नियोजय) {
        writeback_case = op_case - 6;
    }
    ध्रुव std::string slot_begin_name = "rs_slot_begin_" + context.get_name();
    ध्रुव तार्किक खण्डिका_नियोजय = writeback_case >= 1 && writeback_case <= 3 &&
                              (single_slot_नियोजय || context.has_निवेशः(slot_begin_name));
    यदि (खण्डिका_नियोजय) {
        यदि (single_slot_नियोजय && writeback_case == 3) {
            प्रत्यागम {context.get_निवेशः(1)};
        }

        ध्रुव int64_t slot_axis = 2;
        स्वचालित zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        स्वचालित one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वचालित int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {अधिकतम_पूर्णाङ्कः});
        स्वचालित axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {slot_axis});
        स्वचालित feature = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                    std::vector<int64_t>{1, 1, -1, output_shape[3].get_दीर्घता()});

        ov::Output<ov::Node> src;
        ov::Output<ov::Node> begin;
        यदि (!single_slot_नियोजय) {
            begin = context.get_निवेशः(slot_begin_name);
        }
        स्वचालित आधारः = context.get_निवेशः(1);
        यदि (writeback_case == 1) {
            यदि (प्रत्यक्षम्_gdn_state) {
                // Non-rollback GDN publishes state directly as [active_slots, heads, value_dim,
                // key_dim]. Flatten each active slot before replacing or updating the cache.
                src = std::make_shared<ov::op::v1::Reshape>(context.get_निवेशः(0), feature, असत्य);
            } अन्यथा {
                // Multi-slot rollback still consumes GGML's packed [attention | state snapshots]
                // layout. Slice the state block using the runtime source offset.
                स्वचालित src_begin = context.get_निवेशः("rs_src_begin_" + context.get_name());
                स्वचालित state_part =
                    std::make_shared<ov::op::v8::खण्डिका>(context.get_निवेशः(0), src_begin, int_max, one, axis);
                src = std::make_shared<ov::op::v1::Reshape>(state_part, feature, असत्य);
            }
        } अन्यथा यदि (writeback_case == 2) {
            // conv_input is [previous conv state | new tokens]; the snapshot is the conv_kernel_size - 1
            // columns ending at the last *valid* token. Gather (rather than Slice) keeps the output
            // shape static even though the window start is a runtime value.
            स्वचालित गवाक्षः_आकार = (int64_t) निवेशः_shape[3].get_दीर्घता();
            ov::Output<ov::Node> गवाक्षः;
            स्वचालित col_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            ध्रुव std::string src_begin_name = "rs_src_begin_" + context.get_name();
            यदि (context.has_निवेशः(src_begin_name)) {
                स्वचालित src_begin = context.get_निवेशः(src_begin_name);
                स्वचालित src_अन्तः = std::make_shared<ov::op::v1::सङ्कलय>(
                    src_begin, ov::op::v0::Constant::create(ov::element::i64, {1}, {गवाक्षः_आकार}));
                गवाक्षः = std::make_shared<ov::op::v8::खण्डिका>(context.get_निवेशः(0), src_begin, src_अन्तः, one, col_axis);
            } अन्यथा यदि (context.has_निवेशः("chunk_valid_len")) {
                std::vector<int64_t> offsets(गवाक्षः_आकार);
                std::iota(offsets.begin(), offsets.end(), 0);
                स्वचालित indices = std::make_shared<ov::op::v1::सङ्कलय>(
                    ov::op::v0::Constant::create(ov::element::i64, {(size_t) गवाक्षः_आकार}, offsets),
                    context.get_निवेशः("chunk_valid_len"));
                गवाक्षः = std::make_shared<ov::op::v8::Gather>(context.get_निवेशः(0), indices, col_axis);
            } अन्यथा {
                स्वचालित गवाक्षः_begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {-गवाक्षः_आकार});
                गवाक्षः =
                    std::make_shared<ov::op::v8::खण्डिका>(context.get_निवेशः(0), गवाक्षः_begin, int_max, one, col_axis);
            }
            ध्रुव स्वचालित आधारः_shape = आधारः.get_partial_shape();
            FRONT_अन्तः_OP_CONVERSION_CHECK(आधारः_shape.rank().is_static() && आधारः_shape.rank().get_दीर्घता() == 4,
                                          "CPY conv state cache update requires rank-4 base cache");
            FRONT_अन्तः_OP_CONVERSION_CHECK(आधारः_shape[3].is_static(),
                                          "CPY conv state cache update requires static feature size");
            FRONT_अन्तः_OP_CONVERSION_CHECK(निवेशः_shape.rank().is_static() && निवेशः_shape.rank().get_दीर्घता() == 4 &&
                                              निवेशः_shape[2].is_static() && निवेशः_shape[3].is_static(),
                                          "CPY conv state cache update requires static source feature view");

            ध्रुव int64_t पूर्णम्_feature_आकार = आधारः_shape[3].get_दीर्घता();
            ध्रुव int64_t update_feature_आकार = निवेशः_shape[2].get_दीर्घता() * निवेशः_shape[3].get_दीर्घता();
            ध्रुव स्वचालित output_stride = context.get_output_stride();
            ध्रुव size_t elem_आकार = output_stride.रिक्तानि() ? context.get_output_type().size() : output_stride.back();
            FRONT_अन्तः_OP_CONVERSION_CHECK(elem_आकार > 0,
                                          "CPY conv state cache update has invalid element size");
            ध्रुव int64_t feature_begin = static_cast<int64_t>(context.get_output_op_विस्थापनम्() / elem_आकार) %
                                          पूर्णम्_feature_आकार;
            ध्रुव int64_t feature_अन्तः = feature_begin + update_feature_आकार;
            FRONT_अन्तः_OP_CONVERSION_CHECK(feature_begin >= 0 && feature_अन्तः <= पूर्णम्_feature_आकार,
                                          "CPY conv state cache update feature range is out of bounds");

            स्वचालित partial_feature = ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, update_feature_आकार});
            src = std::make_shared<ov::op::v1::Reshape>(गवाक्षः, partial_feature, असत्य);
            यदि (src.get_element_type() != context.get_output_type()) {
                src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
            }

            यदि (single_slot_नियोजय) {
                प्रत्यागम rename_outputs_with_suffix({src}, context.get_name());
            }

            स्वचालित src_len = std::make_shared<ov::op::v8::Gather>(
                std::make_shared<ov::op::v3::Shape_सम्बन्धि>(src, ov::element::i64), axis,
                ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
            स्वचालित slot_अन्तः = std::make_shared<ov::op::v1::सङ्कलय>(begin, src_len);
            स्वचालित सक्रिय_slots = std::make_shared<ov::op::v8::खण्डिका>(आधारः, begin, slot_अन्तः, one, axis);

            स्वचालित feature_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            स्वचालित feature_begin_ग्रन्थिः = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_begin});
            स्वचालित feature_अन्तः_ग्रन्थिः = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_अन्तः});
            स्वचालित feature_शीर्ष = std::make_shared<ov::op::v8::खण्डिका>(सक्रिय_slots, zero, feature_begin_ग्रन्थिः, one,
                                                                    feature_axis);
            स्वचालित feature_पुच्छ = std::make_shared<ov::op::v8::खण्डिका>(सक्रिय_slots, feature_अन्तः_ग्रन्थिः, int_max, one,
                                                                    feature_axis);
            src = std::make_shared<ov::op::v0::Concat>(ov::Output_सदिशम्{feature_शीर्ष, src, feature_पुच्छ}, 3);
        } अन्यथा {
            // op_case 3: gathered remainder rows already have the cache slot layout [1, 1, extra, feature]
            src = context.get_निवेशः(0);
        }

        यदि (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        यदि (single_slot_नियोजय) {
            प्रत्यागम rename_outputs_with_suffix({src}, context.get_name());
        }

        स्वचालित src_len =
            std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::Shape_सम्बन्धि>(src, ov::element::i64), axis,
                                                 ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        स्वचालित end = std::make_shared<ov::op::v1::सङ्कलय>(begin, src_len);
        स्वचालित शीर्ष_part = std::make_shared<ov::op::v8::खण्डिका>(आधारः, zero, begin, one, axis);
        स्वचालित पुच्छ_part = std::make_shared<ov::op::v8::खण्डिका>(आधारः, end, int_max, one, axis);
        स्वचालित res = std::make_shared<ov::op::v0::Concat>(ov::Output_सदिशम्{शीर्ष_part, src, पुच्छ_part}, slot_axis);
        प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
    }

    स्वचालित निवेशः = process_view_निवेशः_new(context, 0);

    यदि (op_case == 5 || op_case == 6) {
        स्वचालित निवेशः_shape = context.get_निवेशः_shape(0);
        स्वचालित output_shape = context.get_output_shape();
        स्वचालित dst_ggml_shape = context.get_view_निवेशः_ggml_shape(1, 0);
        स्वचालित dst_stride = context.get_view_निवेशः_stride(1, 0);
        size_t विस्थापनम्_अष्टकानि = context.get_view_निवेशः_विस्थापनम्(1, 0);
        स्वचालित n_state = (int64_t) context.get_निवेशः_shape(0)[3].get_दीर्घता();
        स्वचालित n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
        स्वचालित kv_buf = context.get_निवेशः(1);  // shape {1,1,1,N}

        Output<Node> token_len_प्रत्येकम्_seq;
        Output<Node> n_write_dyn;
        यदि (context.has_निवेशः("token_len_per_seq")) {
            token_len_प्रत्येकम्_seq = context.get_निवेशः("token_len_per_seq");
            n_write_dyn = std::make_shared<ov::op::v1::Multiply>(token_len_प्रत्येकम्_seq, n_state_c);
        } अन्यथा {
            n_write_dyn = ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) dst_ggml_shape[3]});
        }
        size_t elem_आकार = dst_stride[3];
        FRONT_अन्तः_OP_CONVERSION_CHECK(elem_आकार > 0, "CPY KV cache view update has invalid element size");
        int64_t start_elem = (int64_t) (विस्थापनम्_अष्टकानि / elem_आकार);
        // op_case 5: decoder self-attention – write offset advances each step.
        // op_case 6: encoder self-attn or cross-attn – offset fixed at compile time.
        ध्रुव तार्किक is_decoder_self_attn = (op_case == 5);
        स्वचालित ones_c = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{1, 1, 1});
        स्वचालित new_shape = std::make_shared<ov::op::v0::Concat>(ov::Output_सदिशम्{ones_c, n_write_dyn}, 0);

        स्वचालित reshaped = std::make_shared<ov::op::v1::Reshape>(निवेशः, new_shape, असत्य);
        स्वचालित दत्त = std::make_shared<ov::op::v0::Convert>(reshaped, context.get_output_type());
        // Indices [start_elem .. start_elem + n_write) on axis 3 of {1,1,1,N}
        // For decoder self-attention the write offset advances each step, so compute it
        // dynamically from the model inputs: start = (attention_size - token_len_per_seq) * n_state.
        // For encoder self-attn and cross-attn the offset is fixed at graph-compile time.
        ov::Output<ov::Node> start;
        यदि (is_decoder_self_attn && context.has_निवेशः("attention_size") && context.has_निवेशः("token_len_per_seq")) {
            स्वचालित attention_आकार_अन्तर् = context.get_निवेशः("attention_size");
            स्वचालित token_len_अन्तर् = context.get_निवेशः("token_len_per_seq");
            स्वचालित past_tokens = std::make_shared<ov::op::v1::व्यवकलय>(attention_आकार_अन्तर्, token_len_अन्तर्);
            स्वचालित new_start = std::make_shared<ov::op::v1::Multiply>(past_tokens, n_state_c);
            start = std::make_shared<ov::op::v1::सङ्कलय>(
                new_start, ov::op::v0::Constant::create(ov::element::i64, {1}, {start_elem}));
        } अन्यथा {
            start = ov::op::v0::Constant::create(ov::element::i64, {1}, {start_elem});
        }
        स्वचालित start_squeezed = std::make_shared<ov::op::v0::Squeeze>(start);
        स्वचालित end = std::make_shared<ov::op::v1::सङ्कलय>(start_squeezed, n_write_dyn);
        स्वचालित अन्तः_squeezed = std::make_shared<ov::op::v0::Squeeze>(end);
        स्वचालित चरणः = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वचालित चरणः_squeezed = std::make_shared<ov::op::v0::Squeeze>(चरणः);
        स्वचालित indices =
            std::make_shared<ov::op::v4::Range>(start_squeezed, अन्तः_squeezed, चरणः_squeezed, ov::element::i64);
        स्वचालित axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        स्वचालित kv_updated = std::make_shared<ov::op::v3::ScatterUpdate>(kv_buf, indices, दत्त, axis);
        प्रत्यागम rename_outputs_with_suffix({kv_updated}, context.get_name());
    }

    यदि (निवेशः_shape != output_shape) {
        स्वचालित new_shape = ov::op::v0::Constant::create(
            ov::element::i64, {static_cast<size_t>(output_shape.rank().get_दीर्घता())}, output_shape.गन्तव्यम्_shape());
        निवेशः = std::make_shared<ov::op::v1::Reshape>(निवेशः, new_shape, असत्य);
    }

    ov::Output<Node> res;
    यदि (context.get_निवेशः_type(0) != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(निवेशः, context.get_output_type());
    } अन्यथा {
        res = निवेशः;
    }

    यदि (res.get_ग्रन्थिः_shared_ptr() == context.get_निवेशः(0).get_ग्रन्थिः_shared_ptr()) {
        प्रत्यागम {res};
    }

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
