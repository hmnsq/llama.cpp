#అన్తర్భావ "kv_state_seq_axis.h"

#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/core/graph_util.hpp>
#అన్తర్భావ <openvino/op/assign.hpp>
#అన్తర్భావ <openvino/op/concat.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/read_value.hpp>
#అన్తర్భావ <openvino/op/transpose.hpp>
#అన్తర్భావ <vector>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ pass {

నామాకాశ {

స్థిరరాశి std::vector<int64_t> & seq_axis_perm() {
    // [1, seq, n_heads_kv, head_size] <-> [1, n_heads_kv, seq, head_size]
    స్థిర స్థిరరాశి std::vector<int64_t> perm{0, 2, 1, 3};
    తిరిగి perm;
}

// True when the state still has the frontend's stateful KV layout, so the sequence axis
// can be moved: rank 4, batch and both head dims static, and seq the only dynamic dim,
// at dim 1. Any KV head count is fine. With a single head the rewrite is pure metadata
// ([1, seq, 1, head] and [1, 1, seq, head] are the same memory); with several heads it
// also drops the reader-side transpose of the whole accumulated state, which is where
// most of the gain comes from at depth.
తార్కిక can_move_seq_axis(స్థిరరాశి ov::PartialShape & shape) {
    తిరిగి shape.rank().is_static() && shape.rank().get_నిడివి() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> సరిపోలిక_kv_append(స్థిరరాశి std::shared_ptr<ov::op::v6::కేటాయించు_2> & కేటాయించు) {
    స్వయంచాలక concat = ov::as_type_ptr<ov::op::v0::Concat>(కేటాయించు->get_ఆదానం_కణుపు_shared_ptr(0));
    ఒకవేళ (!concat || concat->get_ఆదానం_కొలత() != 2 || concat->get_axis() != 1) {
        తిరిగి శూన్యసూచిక;
    }
    స్వయంచాలక read_విలువ = ov::as_type_ptr<ov::op::v6::Read_విలువ>(concat->get_ఆదానం_కణుపు_shared_ptr(0));
    ఒకవేళ (!read_విలువ || read_విలువ->get_variable() != కేటాయించు->get_variable()) {
        తిరిగి శూన్యసూచిక;
    }
    ఒకవేళ (!can_move_seq_axis(read_విలువ->get_output_partial_shape(0))) {
        తిరిగి శూన్యసూచిక;
    }
    తిరిగి concat;
}

}  // namespace

తార్కిక KVStateSeqAxis::నిరంతరం_on_model(స్థిరరాశి std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::కేటాయించు_2>> assigns;
    కొరకు (స్థిరరాశి స్వయంచాలక & op : model->get_ops()) {
        ఒకవేళ (స్వయంచాలక కేటాయించు = ov::as_type_ptr<ov::op::v6::కేటాయించు_2>(op)) {
            assigns.push_back(కేటాయించు);
        }
    }

    తార్కిక మారింది = అబద్ధం;
    కొరకు (స్థిరరాశి స్వయంచాలక & కేటాయించు : assigns) {
        స్వయంచాలక concat = సరిపోలిక_kv_append(కేటాయించు);
        ఒకవేళ (!concat) {
            కొనసాగు;
        }
        స్వయంచాలక read_విలువ = ov::as_type_ptr<ov::op::v6::Read_విలువ>(concat->get_ఆదానం_కణుపు_shared_ptr(0));

        స్వయంచాలక variable = read_విలువ->get_variable();
        స్వయంచాలక info = variable->get_info();
        స్థిరరాశి స్వయంచాలక & shape = info.దత్తాంశం_shape;
        info.దత్తాంశం_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_విలువ->validate_and_infer_types();

        స్వయంచాలక readers = concat->output(0).get_లక్ష్యం_inputs();

        స్వయంచాలక new_అడ్డువరుసలు = concat->ఆదానం_విలువ(1);
        స్వయంచాలక perm_లోపల = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->సమితి_వాదం(1, std::make_shared<ov::op::v1::Transpose>(new_అడ్డువరుసలు, perm_లోపల));
        concat->సమితి_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        కొరకు (స్థిరరాశి స్వయంచాలక & reader : readers) {
            స్వయంచాలక * కణుపు = reader.get_కణుపు();
            ఒకవేళ (ov::is_type<ov::op::v6::కేటాయించు_2>(కణుపు)) {
                కొనసాగు;
            }
            తార్కిక dropped = అబద్ధం;
            ఒకవేళ (స్వయంచాలక * transpose = ov::as_type<ov::op::v1::Transpose>(కణుపు)) {
                స్వయంచాలక క్రమం = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_ఆదానం_కణుపు_shared_ptr(1));
                ఒకవేళ (క్రమం && క్రమం->cast_సదిశ<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = నిజం;
                }
            }
            ఒకవేళ (!dropped) {
                స్వయంచాలక perm_బహిర్గతం = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_ఆధారం_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_బహిర్గతం));
            }
        }
        మారింది = నిజం;
    }

    తిరిగి మారింది;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
