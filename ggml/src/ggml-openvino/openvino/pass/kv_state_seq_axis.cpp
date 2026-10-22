#அந்தர்பாவ "kv_state_seq_axis.h"

#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/core/graph_util.hpp>
#அந்தர்பாவ <openvino/op/assign.hpp>
#அந்தர்பாவ <openvino/op/concat.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/read_value.hpp>
#அந்தர்பாவ <openvino/op/transpose.hpp>
#அந்தர்பாவ <vector>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ pass {

நாமாகாஶ {

மாறிலி std::vector<int64_t> & seq_axis_perm() {
    // [1, seq, n_heads_kv, head_size] <-> [1, n_heads_kv, seq, head_size]
    நிலையான மாறிலி std::vector<int64_t> perm{0, 2, 1, 3};
    திரும்பு perm;
}

// True when the state still has the frontend's stateful KV layout, so the sequence axis
// can be moved: rank 4, batch and both head dims static, and seq the only dynamic dim,
// at dim 1. Any KV head count is fine. With a single head the rewrite is pure metadata
// ([1, seq, 1, head] and [1, 1, seq, head] are the same memory); with several heads it
// also drops the reader-side transpose of the whole accumulated state, which is where
// most of the gain comes from at depth.
தருக்க can_move_seq_axis(மாறிலி ov::PartialShape & shape) {
    திரும்பு shape.rank().is_static() && shape.rank().get_length() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> match_kv_append(மாறிலி std::shared_ptr<ov::op::v6::Assign> & assign) {
    auto concat = ov::as_type_ptr<ov::op::v0::Concat>(assign->get_உள்ளீடு_முனை_shared_ptr(0));
    எனில் (!concat || concat->get_உள்ளீடு_size() != 2 || concat->get_axis() != 1) {
        திரும்பு nullptr;
    }
    auto read_மதிப்பு = ov::as_type_ptr<ov::op::v6::Read_மதிப்பு>(concat->get_உள்ளீடு_முனை_shared_ptr(0));
    எனில் (!read_மதிப்பு || read_மதிப்பு->get_variable() != assign->get_variable()) {
        திரும்பு nullptr;
    }
    எனில் (!can_move_seq_axis(read_மதிப்பு->get_output_partial_shape(0))) {
        திரும்பு nullptr;
    }
    திரும்பு concat;
}

}  // namespace

தருக்க KVStateSeqAxis::run_on_model(மாறிலி std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::Assign>> assigns;
    க்கு (மாறிலி auto & op : model->get_ops()) {
        எனில் (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(op)) {
            assigns.push_back(assign);
        }
    }

    தருக்க changed = பொய்;
    க்கு (மாறிலி auto & assign : assigns) {
        auto concat = match_kv_append(assign);
        எனில் (!concat) {
            தொடர்;
        }
        auto read_மதிப்பு = ov::as_type_ptr<ov::op::v6::Read_மதிப்பு>(concat->get_உள்ளீடு_முனை_shared_ptr(0));

        auto variable = read_மதிப்பு->get_variable();
        auto info = variable->get_info();
        மாறிலி auto & shape = info.data_shape;
        info.data_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_மதிப்பு->validate_and_infer_types();

        auto readers = concat->output(0).get_target_inputs();

        auto new_rows = concat->உள்ளீடு_மதிப்பு(1);
        auto perm_in = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->set_argument(1, std::make_shared<ov::op::v1::Transpose>(new_rows, perm_in));
        concat->set_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        க்கு (மாறிலி auto & reader : readers) {
            auto * முனை = reader.get_முனை();
            எனில் (ov::is_type<ov::op::v6::Assign>(முனை)) {
                தொடர்;
            }
            தருக்க dropped = பொய்;
            எனில் (auto * transpose = ov::as_type<ov::op::v1::Transpose>(முனை)) {
                auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_உள்ளீடு_முனை_shared_ptr(1));
                எனில் (order && order->cast_vector<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = உண்மை;
                }
            }
            எனில் (!dropped) {
                auto perm_out = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_source_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_out));
            }
        }
        changed = உண்மை;
    }

    திரும்பு changed;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
