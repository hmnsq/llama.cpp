#ضمن "kv_state_seq_axis.h"

#ضمن <memory>
#ضمن <openvino/core/graph_util.hpp>
#ضمن <openvino/op/assign.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/read_value.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

نطاق {

ثابت std::vector<int64_t> & seq_axis_perm() {
    // [1, seq, n_heads_kv, head_size] <-> [1, n_heads_kv, seq, head_size]
    ساكن ثابت std::vector<int64_t> perm{0, 2, 1, 3};
    ܦܢܐ perm;
}

// True when the state still has the frontend's stateful KV layout, so the sequence axis
// can be moved: rank 4, batch and both head dims static, and seq the only dynamic dim,
// at dim 1. Any KV head count is fine. With a single head the rewrite is pure metadata
// ([1, seq, 1, head] and [1, 1, seq, head] are the same memory); with several heads it
// also drops the reader-side transpose of the whole accumulated state, which is where
// most of the gain comes from at depth.
منطقي can_move_seq_axis(ثابت ov::PartialShape & shape) {
    ܦܢܐ shape.rank().is_static() && shape.rank().get_length() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> match_kv_append(ثابت std::shared_ptr<ov::op::v6::Assign> & assign) {
    auto concat = ov::as_type_ptr<ov::op::v0::Concat>(assign->get_input_node_shared_ptr(0));
    ܐܢ (!concat || concat->get_input_size() != 2 || concat->get_axis() != 1) {
        ܦܢܐ nullptr;
    }
    auto read_value = ov::as_type_ptr<ov::op::v6::ReadValue>(concat->get_input_node_shared_ptr(0));
    ܐܢ (!read_value || read_value->get_variable() != assign->get_variable()) {
        ܦܢܐ nullptr;
    }
    ܐܢ (!can_move_seq_axis(read_value->get_output_partial_shape(0))) {
        ܦܢܐ nullptr;
    }
    ܦܢܐ concat;
}

}  // namespace

منطقي KVStateSeqAxis::run_on_model(ثابت std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::Assign>> assigns;
    لكل (ثابت auto & op : model->get_ops()) {
        ܐܢ (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(op)) {
            assigns.push_back(assign);
        }
    }

    منطقي changed = ܕܓܠܐ;
    لكل (ثابت auto & assign : assigns) {
        auto concat = match_kv_append(assign);
        ܐܢ (!concat) {
            استمر;
        }
        auto read_value = ov::as_type_ptr<ov::op::v6::ReadValue>(concat->get_input_node_shared_ptr(0));

        auto variable = read_value->get_variable();
        auto info = variable->get_info();
        ثابت auto & shape = info.data_shape;
        info.data_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_value->validate_and_infer_types();

        auto readers = concat->output(0).get_target_inputs();

        auto new_rows = concat->input_value(1);
        auto perm_in = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->set_argument(1, std::make_shared<ov::op::v1::Transpose>(new_rows, perm_in));
        concat->set_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        لكل (ثابت auto & reader : readers) {
            auto * node = reader.get_node();
            ܐܢ (ov::is_type<ov::op::v6::Assign>(node)) {
                استمر;
            }
            منطقي dropped = ܕܓܠܐ;
            ܐܢ (auto * transpose = ov::as_type<ov::op::v1::Transpose>(node)) {
                auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_input_node_shared_ptr(1));
                ܐܢ (order && order->cast_vector<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = ܫܪܪܐ;
                }
            }
            ܐܢ (!dropped) {
                auto perm_out = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_source_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_out));
            }
        }
        changed = ܫܪܪܐ;
    }

    ܦܢܐ changed;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
