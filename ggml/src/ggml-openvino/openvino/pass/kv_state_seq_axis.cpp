#ಅನ್ತರ್ಭಾವ "kv_state_seq_axis.h"

#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/core/graph_util.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/assign.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/read_value.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/transpose.hpp>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ pass {

ನಾಮಾಕಾಶ {

ಧ್ರುವ std::vector<int64_t> & seq_axis_perm() {
    // [1, seq, n_heads_kv, head_size] <-> [1, n_heads_kv, seq, head_size]
    ಸ್ಥಿರ ಧ್ರುವ std::vector<int64_t> perm{0, 2, 1, 3};
    ಪ್ರತ್ಯಾಗಮ perm;
}

// True when the state still has the frontend's stateful KV layout, so the sequence axis
// can be moved: rank 4, batch and both head dims static, and seq the only dynamic dim,
// at dim 1. Any KV head count is fine. With a single head the rewrite is pure metadata
// ([1, seq, 1, head] and [1, 1, seq, head] are the same memory); with several heads it
// also drops the reader-side transpose of the whole accumulated state, which is where
// most of the gain comes from at depth.
ತಾರ್ಕಿಕ can_move_seq_axis(ಧ್ರುವ ov::PartialShape & shape) {
    ಪ್ರತ್ಯಾಗಮ shape.rank().is_static() && shape.rank().get_length() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> match_kv_append(ಧ್ರುವ std::shared_ptr<ov::op::v6::Assign> & assign) {
    auto concat = ov::as_type_ptr<ov::op::v0::Concat>(assign->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0));
    ಯದಿ (!concat || concat->get_ಒಳಹರಿವು_size() != 2 || concat->get_axis() != 1) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    auto read_ಮೌಲ್ಯ = ov::as_type_ptr<ov::op::v6::Read_ಮೌಲ್ಯ>(concat->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0));
    ಯದಿ (!read_ಮೌಲ್ಯ || read_ಮೌಲ್ಯ->get_variable() != assign->get_variable()) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಯದಿ (!can_move_seq_axis(read_ಮೌಲ್ಯ->get_output_partial_shape(0))) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಪ್ರತ್ಯಾಗಮ concat;
}

}  // namespace

ತಾರ್ಕಿಕ KVStateSeqAxis::run_on_model(ಧ್ರುವ std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::Assign>> assigns;
    ಪ್ರತಿ (ಧ್ರುವ auto & op : model->get_ops()) {
        ಯದಿ (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(op)) {
            assigns.push_back(assign);
        }
    }

    ತಾರ್ಕಿಕ changed = ಅಸತ್ಯ;
    ಪ್ರತಿ (ಧ್ರುವ auto & assign : assigns) {
        auto concat = match_kv_append(assign);
        ಯದಿ (!concat) {
            ಅನುವರ್ತ;
        }
        auto read_ಮೌಲ್ಯ = ov::as_type_ptr<ov::op::v6::Read_ಮೌಲ್ಯ>(concat->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0));

        auto variable = read_ಮೌಲ್ಯ->get_variable();
        auto info = variable->get_info();
        ಧ್ರುವ auto & shape = info.data_shape;
        info.data_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_ಮೌಲ್ಯ->validate_and_infer_types();

        auto readers = concat->output(0).get_target_inputs();

        auto new_rows = concat->ಒಳಹರಿವು_ಮೌಲ್ಯ(1);
        auto perm_in = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->set_argument(1, std::make_shared<ov::op::v1::Transpose>(new_rows, perm_in));
        concat->set_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        ಪ್ರತಿ (ಧ್ರುವ auto & reader : readers) {
            auto * ಗಂಟು = reader.get_ಗಂಟು();
            ಯದಿ (ov::is_type<ov::op::v6::Assign>(ಗಂಟು)) {
                ಅನುವರ್ತ;
            }
            ತಾರ್ಕಿಕ dropped = ಅಸತ್ಯ;
            ಯದಿ (auto * transpose = ov::as_type<ov::op::v1::Transpose>(ಗಂಟು)) {
                auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(1));
                ಯದಿ (order && order->cast_vector<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = ಸತ್ಯ;
                }
            }
            ಯದಿ (!dropped) {
                auto perm_out = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_source_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_out));
            }
        }
        changed = ಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ changed;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
