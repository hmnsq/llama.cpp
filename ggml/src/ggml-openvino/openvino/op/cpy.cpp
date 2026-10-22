#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <climits>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstdio>
#అన్తర్భావ <memory>
#అన్తర్భావ <numeric>
#అన్తర్భావ <openvino/frontend/exception.hpp>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/concat.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/gather.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/negative.hpp>
#అన్తర్భావ <openvino/op/range.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/scatter_update.hpp>
#అన్తర్భావ <openvino/op/shape_of.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>
#అన్తర్భావ <openvino/op/squeeze.hpp>
#అన్తర్భావ <openvino/op/subtract.hpp>
#అన్తర్భావ <vector>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_cpy(స్థిరరాశి కణుపు_Context & context) {
    స్వయంచాలక op_case = context.get_op_case();
    స్వయంచాలక ఆదానం_shape = context.get_ఆదానం_shape(0);
    స్వయంచాలక output_shape = context.get_ఆదానం_shape(1);

    ఒకవేళ (op_case == 4) {
        స్వయంచాలక src = process_view_ఆదానం_new(context, 0);
        స్వయంచాలక పునాది = context.get_ఆదానం(1);

        int64_t n_elems = 1;
        కొరకు (స్థిరరాశి స్వయంచాలక & dim : context.get_output_shape().గమ్యం_shape()) {
            n_elems *= static_cast<int64_t>(dim);
        }

        స్థిరరాశి స్వయంచాలక output_stride = context.get_output_stride();
        స్థిరరాశి size_t elem_కొలత = output_stride.ఖాళీలు() ? context.get_output_type().size() : output_stride.back();
        FRONT_చివర_OP_CONVERSION_CHECK(elem_కొలత > 0, "CPY conv state view update has invalid element size");

        స్థిరరాశి int64_t begin_val = static_cast<int64_t>(context.get_output_op_స్థానభ్రంశం() / elem_కొలత);
        స్థిరరాశి int64_t చివర_val = begin_val + n_elems;

        స్వయంచాలక flat_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, -1});
        src = std::make_shared<ov::op::v1::Reshape>(src, flat_shape, అబద్ధం);
        ఒకవేళ (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        స్వయంచాలక zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        స్వయంచాలక begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val});
        స్వయంచాలక end = ov::op::v0::Constant::create(ov::element::i64, {1}, {చివర_val});
        స్వయంచాలక int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {గరిష్ఠ_పూర్ణాంకం});
        స్వయంచాలక one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        స్వయంచాలక axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        స్వయంచాలక తల_part = std::make_shared<ov::op::v8::ముక్క>(పునాది, zero, begin, one, axis);
        స్వయంచాలక తోక_part = std::make_shared<ov::op::v8::ముక్క>(పునాది, end, int_max, one, axis);
        స్వయంచాలక res = std::make_shared<ov::op::v0::Concat>(ov::Output_సదిశ{తల_part, src, తోక_part}, 3);
        తిరిగి rename_outputs_with_suffix({res}, context.get_name());
    }

    // Recurrent state cache writeback into a slot block of the cache. Where the block starts is a
    // runtime input, so the cached model works for any kv head and active sequence count. The
    // result is the full updated cache.
    // op_case 1: gated-delta-net state, op_case 2: conv state, op_case 3: defrag remainder.
    ఒకవేళ (op_case == 3) {
        // With -np 1 (and generally whenever there is no defrag remainder) this GET_ROWS gathers
        // zero rows: nothing to write back, and the cache is unchanged. NPU rejects zero-size
        // tensors, so short-circuit instead of building a degenerate Slice/Concat chain.
        తార్కిక is_ఖాళీలు = అబద్ధం;
        ఒకవేళ (ఆదానం_shape.rank().is_static()) {
            కొరకు (స్థిరరాశి స్వయంచాలక & d : ఆదానం_shape) {
                ఒకవేళ (d.is_static() && d.get_నిడివి() == 0) {
                    is_ఖాళీలు = నిజం;
                    ఆపు;
                }
            }
        }
        ఒకవేళ (is_ఖాళీలు) {
            తిరిగి {context.get_ఆదానం(1)};
        }
    }
    // op_case 7/8/9 are the single-slot variants; op_case 10 writes native GDN state into a
    // multi-slot cache without rollback snapshots.
    స్థిరరాశి తార్కిక single_స్థానం_కేటాయించు = op_case >= 7 && op_case <= 9;
    స్థిరరాశి తార్కిక ప్రత్యక్ష_gdn_state = op_case == 7 || op_case == 10;
    పూర్ణాంకం writeback_case = op_case;
    ఒకవేళ (op_case == 10) {
        writeback_case = 1;
    } లేకపోతే ఒకవేళ (single_స్థానం_కేటాయించు) {
        writeback_case = op_case - 6;
    }
    స్థిరరాశి std::string స్థానం_begin_name = "rs_slot_begin_" + context.get_name();
    స్థిరరాశి తార్కిక ముక్క_కేటాయించు = writeback_case >= 1 && writeback_case <= 3 &&
                              (single_స్థానం_కేటాయించు || context.has_ఆదానం(స్థానం_begin_name));
    ఒకవేళ (ముక్క_కేటాయించు) {
        ఒకవేళ (single_స్థానం_కేటాయించు && writeback_case == 3) {
            తిరిగి {context.get_ఆదానం(1)};
        }

        స్థిరరాశి int64_t స్థానం_axis = 2;
        స్వయంచాలక zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        స్వయంచాలక one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        స్వయంచాలక int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {గరిష్ఠ_పూర్ణాంకం});
        స్వయంచాలక axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {స్థానం_axis});
        స్వయంచాలక feature = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                    std::vector<int64_t>{1, 1, -1, output_shape[3].get_నిడివి()});

        ov::Output<ov::Node> src;
        ov::Output<ov::Node> begin;
        ఒకవేళ (!single_స్థానం_కేటాయించు) {
            begin = context.get_ఆదానం(స్థానం_begin_name);
        }
        స్వయంచాలక పునాది = context.get_ఆదానం(1);
        ఒకవేళ (writeback_case == 1) {
            ఒకవేళ (ప్రత్యక్ష_gdn_state) {
                // Non-rollback GDN publishes state directly as [active_slots, heads, value_dim,
                // key_dim]. Flatten each active slot before replacing or updating the cache.
                src = std::make_shared<ov::op::v1::Reshape>(context.get_ఆదానం(0), feature, అబద్ధం);
            } లేకపోతే {
                // Multi-slot rollback still consumes GGML's packed [attention | state snapshots]
                // layout. Slice the state block using the runtime source offset.
                స్వయంచాలక src_begin = context.get_ఆదానం("rs_src_begin_" + context.get_name());
                స్వయంచాలక state_part =
                    std::make_shared<ov::op::v8::ముక్క>(context.get_ఆదానం(0), src_begin, int_max, one, axis);
                src = std::make_shared<ov::op::v1::Reshape>(state_part, feature, అబద్ధం);
            }
        } లేకపోతే ఒకవేళ (writeback_case == 2) {
            // conv_input is [previous conv state | new tokens]; the snapshot is the conv_kernel_size - 1
            // columns ending at the last *valid* token. Gather (rather than Slice) keeps the output
            // shape static even though the window start is a runtime value.
            స్వయంచాలక కిటికీ_కొలత = (int64_t) ఆదానం_shape[3].get_నిడివి();
            ov::Output<ov::Node> కిటికీ;
            స్వయంచాలక నిలువు_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            స్థిరరాశి std::string src_begin_name = "rs_src_begin_" + context.get_name();
            ఒకవేళ (context.has_ఆదానం(src_begin_name)) {
                స్వయంచాలక src_begin = context.get_ఆదానం(src_begin_name);
                స్వయంచాలక src_చివర = std::make_shared<ov::op::v1::కూడు>(
                    src_begin, ov::op::v0::Constant::create(ov::element::i64, {1}, {కిటికీ_కొలత}));
                కిటికీ = std::make_shared<ov::op::v8::ముక్క>(context.get_ఆదానం(0), src_begin, src_చివర, one, నిలువు_axis);
            } లేకపోతే ఒకవేళ (context.has_ఆదానం("chunk_valid_len")) {
                std::vector<int64_t> offsets(కిటికీ_కొలత);
                std::iota(offsets.begin(), offsets.end(), 0);
                స్వయంచాలక indices = std::make_shared<ov::op::v1::కూడు>(
                    ov::op::v0::Constant::create(ov::element::i64, {(size_t) కిటికీ_కొలత}, offsets),
                    context.get_ఆదానం("chunk_valid_len"));
                కిటికీ = std::make_shared<ov::op::v8::Gather>(context.get_ఆదానం(0), indices, నిలువు_axis);
            } లేకపోతే {
                స్వయంచాలక కిటికీ_begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {-కిటికీ_కొలత});
                కిటికీ =
                    std::make_shared<ov::op::v8::ముక్క>(context.get_ఆదానం(0), కిటికీ_begin, int_max, one, నిలువు_axis);
            }
            స్థిరరాశి స్వయంచాలక పునాది_shape = పునాది.get_partial_shape();
            FRONT_చివర_OP_CONVERSION_CHECK(పునాది_shape.rank().is_static() && పునాది_shape.rank().get_నిడివి() == 4,
                                          "CPY conv state cache update requires rank-4 base cache");
            FRONT_చివర_OP_CONVERSION_CHECK(పునాది_shape[3].is_static(),
                                          "CPY conv state cache update requires static feature size");
            FRONT_చివర_OP_CONVERSION_CHECK(ఆదానం_shape.rank().is_static() && ఆదానం_shape.rank().get_నిడివి() == 4 &&
                                              ఆదానం_shape[2].is_static() && ఆదానం_shape[3].is_static(),
                                          "CPY conv state cache update requires static source feature view");

            స్థిరరాశి int64_t నిండిన_feature_కొలత = పునాది_shape[3].get_నిడివి();
            స్థిరరాశి int64_t update_feature_కొలత = ఆదానం_shape[2].get_నిడివి() * ఆదానం_shape[3].get_నిడివి();
            స్థిరరాశి స్వయంచాలక output_stride = context.get_output_stride();
            స్థిరరాశి size_t elem_కొలత = output_stride.ఖాళీలు() ? context.get_output_type().size() : output_stride.back();
            FRONT_చివర_OP_CONVERSION_CHECK(elem_కొలత > 0,
                                          "CPY conv state cache update has invalid element size");
            స్థిరరాశి int64_t feature_begin = static_cast<int64_t>(context.get_output_op_స్థానభ్రంశం() / elem_కొలత) %
                                          నిండిన_feature_కొలత;
            స్థిరరాశి int64_t feature_చివర = feature_begin + update_feature_కొలత;
            FRONT_చివర_OP_CONVERSION_CHECK(feature_begin >= 0 && feature_చివర <= నిండిన_feature_కొలత,
                                          "CPY conv state cache update feature range is out of bounds");

            స్వయంచాలక partial_feature = ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, update_feature_కొలత});
            src = std::make_shared<ov::op::v1::Reshape>(కిటికీ, partial_feature, అబద్ధం);
            ఒకవేళ (src.get_element_type() != context.get_output_type()) {
                src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
            }

            ఒకవేళ (single_స్థానం_కేటాయించు) {
                తిరిగి rename_outputs_with_suffix({src}, context.get_name());
            }

            స్వయంచాలక src_len = std::make_shared<ov::op::v8::Gather>(
                std::make_shared<ov::op::v3::Shape_యొక్క>(src, ov::element::i64), axis,
                ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
            స్వయంచాలక స్థానం_చివర = std::make_shared<ov::op::v1::కూడు>(begin, src_len);
            స్వయంచాలక చురుకైన_slots = std::make_shared<ov::op::v8::ముక్క>(పునాది, begin, స్థానం_చివర, one, axis);

            స్వయంచాలక feature_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            స్వయంచాలక feature_begin_కణుపు = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_begin});
            స్వయంచాలక feature_చివర_కణుపు = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_చివర});
            స్వయంచాలక feature_తల = std::make_shared<ov::op::v8::ముక్క>(చురుకైన_slots, zero, feature_begin_కణుపు, one,
                                                                    feature_axis);
            స్వయంచాలక feature_తోక = std::make_shared<ov::op::v8::ముక్క>(చురుకైన_slots, feature_చివర_కణుపు, int_max, one,
                                                                    feature_axis);
            src = std::make_shared<ov::op::v0::Concat>(ov::Output_సదిశ{feature_తల, src, feature_తోక}, 3);
        } లేకపోతే {
            // op_case 3: gathered remainder rows already have the cache slot layout [1, 1, extra, feature]
            src = context.get_ఆదానం(0);
        }

        ఒకవేళ (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        ఒకవేళ (single_స్థానం_కేటాయించు) {
            తిరిగి rename_outputs_with_suffix({src}, context.get_name());
        }

        స్వయంచాలక src_len =
            std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::Shape_యొక్క>(src, ov::element::i64), axis,
                                                 ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        స్వయంచాలక end = std::make_shared<ov::op::v1::కూడు>(begin, src_len);
        స్వయంచాలక తల_part = std::make_shared<ov::op::v8::ముక్క>(పునాది, zero, begin, one, axis);
        స్వయంచాలక తోక_part = std::make_shared<ov::op::v8::ముక్క>(పునాది, end, int_max, one, axis);
        స్వయంచాలక res = std::make_shared<ov::op::v0::Concat>(ov::Output_సదిశ{తల_part, src, తోక_part}, స్థానం_axis);
        తిరిగి rename_outputs_with_suffix({res}, context.get_name());
    }

    స్వయంచాలక ఆదానం = process_view_ఆదానం_new(context, 0);

    ఒకవేళ (op_case == 5 || op_case == 6) {
        స్వయంచాలక ఆదానం_shape = context.get_ఆదానం_shape(0);
        స్వయంచాలక output_shape = context.get_output_shape();
        స్వయంచాలక dst_ggml_shape = context.get_view_ఆదానం_ggml_shape(1, 0);
        స్వయంచాలక dst_stride = context.get_view_ఆదానం_stride(1, 0);
        size_t స్థానభ్రంశం_బైట్లు = context.get_view_ఆదానం_స్థానభ్రంశం(1, 0);
        స్వయంచాలక n_state = (int64_t) context.get_ఆదానం_shape(0)[3].get_నిడివి();
        స్వయంచాలక n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
        స్వయంచాలక kv_buf = context.get_ఆదానం(1);  // shape {1,1,1,N}

        Output<Node> token_len_చొప్పున_seq;
        Output<Node> n_write_dyn;
        ఒకవేళ (context.has_ఆదానం("token_len_per_seq")) {
            token_len_చొప్పున_seq = context.get_ఆదానం("token_len_per_seq");
            n_write_dyn = std::make_shared<ov::op::v1::Multiply>(token_len_చొప్పున_seq, n_state_c);
        } లేకపోతే {
            n_write_dyn = ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) dst_ggml_shape[3]});
        }
        size_t elem_కొలత = dst_stride[3];
        FRONT_చివర_OP_CONVERSION_CHECK(elem_కొలత > 0, "CPY KV cache view update has invalid element size");
        int64_t ప్రారంభం_elem = (int64_t) (స్థానభ్రంశం_బైట్లు / elem_కొలత);
        // op_case 5: decoder self-attention – write offset advances each step.
        // op_case 6: encoder self-attn or cross-attn – offset fixed at compile time.
        స్థిరరాశి తార్కిక is_decoder_self_attn = (op_case == 5);
        స్వయంచాలక ones_c = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{1, 1, 1});
        స్వయంచాలక new_shape = std::make_shared<ov::op::v0::Concat>(ov::Output_సదిశ{ones_c, n_write_dyn}, 0);

        స్వయంచాలక reshaped = std::make_shared<ov::op::v1::Reshape>(ఆదానం, new_shape, అబద్ధం);
        స్వయంచాలక దత్తాంశం = std::make_shared<ov::op::v0::Convert>(reshaped, context.get_output_type());
        // Indices [start_elem .. start_elem + n_write) on axis 3 of {1,1,1,N}
        // For decoder self-attention the write offset advances each step, so compute it
        // dynamically from the model inputs: start = (attention_size - token_len_per_seq) * n_state.
        // For encoder self-attn and cross-attn the offset is fixed at graph-compile time.
        ov::Output<ov::Node> start;
        ఒకవేళ (is_decoder_self_attn && context.has_ఆదానం("attention_size") && context.has_ఆదానం("token_len_per_seq")) {
            స్వయంచాలక attention_కొలత_లోపల = context.get_ఆదానం("attention_size");
            స్వయంచాలక token_len_లోపల = context.get_ఆదానం("token_len_per_seq");
            స్వయంచాలక past_టోకెన్లు = std::make_shared<ov::op::v1::వ్యవకలనం>(attention_కొలత_లోపల, token_len_లోపల);
            స్వయంచాలక new_ప్రారంభం = std::make_shared<ov::op::v1::Multiply>(past_టోకెన్లు, n_state_c);
            start = std::make_shared<ov::op::v1::కూడు>(
                new_ప్రారంభం, ov::op::v0::Constant::create(ov::element::i64, {1}, {ప్రారంభం_elem}));
        } లేకపోతే {
            start = ov::op::v0::Constant::create(ov::element::i64, {1}, {ప్రారంభం_elem});
        }
        స్వయంచాలక ప్రారంభం_squeezed = std::make_shared<ov::op::v0::Squeeze>(start);
        స్వయంచాలక end = std::make_shared<ov::op::v1::కూడు>(ప్రారంభం_squeezed, n_write_dyn);
        స్వయంచాలక చివర_squeezed = std::make_shared<ov::op::v0::Squeeze>(end);
        స్వయంచాలక అడుగు = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        స్వయంచాలక అడుగు_squeezed = std::make_shared<ov::op::v0::Squeeze>(అడుగు);
        స్వయంచాలక indices =
            std::make_shared<ov::op::v4::Range>(ప్రారంభం_squeezed, చివర_squeezed, అడుగు_squeezed, ov::element::i64);
        స్వయంచాలక axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        స్వయంచాలక kv_updated = std::make_shared<ov::op::v3::ScatterUpdate>(kv_buf, indices, దత్తాంశం, axis);
        తిరిగి rename_outputs_with_suffix({kv_updated}, context.get_name());
    }

    ఒకవేళ (ఆదానం_shape != output_shape) {
        స్వయంచాలక new_shape = ov::op::v0::Constant::create(
            ov::element::i64, {static_cast<size_t>(output_shape.rank().get_నిడివి())}, output_shape.గమ్యం_shape());
        ఆదానం = std::make_shared<ov::op::v1::Reshape>(ఆదానం, new_shape, అబద్ధం);
    }

    ov::Output<Node> res;
    ఒకవేళ (context.get_ఆదానం_type(0) != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(ఆదానం, context.get_output_type());
    } లేకపోతే {
        res = ఆదానం;
    }

    ఒకవేళ (res.get_కణుపు_shared_ptr() == context.get_ఆదానం(0).get_కణుపు_shared_ptr()) {
        తిరిగి {res};
    }

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
