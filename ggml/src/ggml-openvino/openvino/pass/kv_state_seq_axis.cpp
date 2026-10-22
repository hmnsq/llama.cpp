#અન્તર્ભાવ "kv_state_seq_axis.h"

#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/core/graph_util.hpp>
#અન્તર્ભાવ <openvino/op/assign.hpp>
#અન્તર્ભાવ <openvino/op/concat.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/read_value.hpp>
#અન્તર્ભાવ <openvino/op/transpose.hpp>
#અન્તર્ભાવ <vector>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ pass {

નામાકાશ {

અચલ std::vector<int64_t> & seq_axis_perm() {
    // [1, seq, n_heads_kv, head_size] <-> [1, n_heads_kv, seq, head_size]
    સ્થિર અચલ std::vector<int64_t> perm{0, 2, 1, 3};
    પાછા perm;
}

// True when the state still has the frontend's stateful KV layout, so the sequence axis
// can be moved: rank 4, batch and both head dims static, and seq the only dynamic dim,
// at dim 1. Any KV head count is fine. With a single head the rewrite is pure metadata
// ([1, seq, 1, head] and [1, 1, seq, head] are the same memory); with several heads it
// also drops the reader-side transpose of the whole accumulated state, which is where
// most of the gain comes from at depth.
તાર્કિક can_move_seq_axis(અચલ ov::PartialShape & shape) {
    પાછા shape.rank().is_static() && shape.rank().get_length() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> match_kv_append(અચલ std::shared_ptr<ov::op::v6::Assign> & assign) {
    auto concat = ov::as_type_ptr<ov::op::v0::Concat>(assign->get_આગત_ગાંઠ_shared_ptr(0));
    જો (!concat || concat->get_આગત_size() != 2 || concat->get_axis() != 1) {
        પાછા nullptr;
    }
    auto read_કિંમત = ov::as_type_ptr<ov::op::v6::Read_કિંમત>(concat->get_આગત_ગાંઠ_shared_ptr(0));
    જો (!read_કિંમત || read_કિંમત->get_variable() != assign->get_variable()) {
        પાછા nullptr;
    }
    જો (!can_move_seq_axis(read_કિંમત->get_output_partial_shape(0))) {
        પાછા nullptr;
    }
    પાછા concat;
}

}  // namespace

તાર્કિક KVStateSeqAxis::run_on_model(અચલ std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::Assign>> assigns;
    માટે (અચલ auto & op : model->get_ops()) {
        જો (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(op)) {
            assigns.push_back(assign);
        }
    }

    તાર્કિક changed = ખોટું;
    માટે (અચલ auto & assign : assigns) {
        auto concat = match_kv_append(assign);
        જો (!concat) {
            ચાલુ;
        }
        auto read_કિંમત = ov::as_type_ptr<ov::op::v6::Read_કિંમત>(concat->get_આગત_ગાંઠ_shared_ptr(0));

        auto variable = read_કિંમત->get_variable();
        auto info = variable->get_info();
        અચલ auto & shape = info.data_shape;
        info.data_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_કિંમત->validate_and_infer_types();

        auto readers = concat->output(0).get_target_inputs();

        auto new_rows = concat->આગત_કિંમત(1);
        auto perm_in = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->set_argument(1, std::make_shared<ov::op::v1::Transpose>(new_rows, perm_in));
        concat->set_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        માટે (અચલ auto & reader : readers) {
            auto * ગાંઠ = reader.get_ગાંઠ();
            જો (ov::is_type<ov::op::v6::Assign>(ગાંઠ)) {
                ચાલુ;
            }
            તાર્કિક dropped = ખોટું;
            જો (auto * transpose = ov::as_type<ov::op::v1::Transpose>(ગાંઠ)) {
                auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_આગત_ગાંઠ_shared_ptr(1));
                જો (order && order->cast_vector<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = સાચું;
                }
            }
            જો (!dropped) {
                auto perm_out = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_source_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_out));
            }
        }
        changed = સાચું;
    }

    પાછા changed;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
