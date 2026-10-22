#অন্তর্ভাৱ "kv_state_seq_axis.h"

#অন্তর্ভাৱ <memory>
#অন্তর্ভাৱ <openvino/core/graph_util.hpp>
#অন্তর্ভাৱ <openvino/op/assign.hpp>
#অন্তর্ভাৱ <openvino/op/concat.hpp>
#অন্তর্ভাৱ <openvino/op/constant.hpp>
#অন্তর্ভাৱ <openvino/op/read_value.hpp>
#অন্তর্ভাৱ <openvino/op/transpose.hpp>
#অন্তর্ভাৱ <vector>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ pass {

নামাকাশ {

ধ্রুৱ std::vector<int64_t> & seq_axis_perm() {
    // [1, seq, n_heads_kv, head_size] <-> [1, n_heads_kv, seq, head_size]
    স্থির ধ্রুৱ std::vector<int64_t> perm{0, 2, 1, 3};
    ঘূৰাই perm;
}

// True when the state still has the frontend's stateful KV layout, so the sequence axis
// can be moved: rank 4, batch and both head dims static, and seq the only dynamic dim,
// at dim 1. Any KV head count is fine. With a single head the rewrite is pure metadata
// ([1, seq, 1, head] and [1, 1, seq, head] are the same memory); with several heads it
// also drops the reader-side transpose of the whole accumulated state, which is where
// most of the gain comes from at depth.
তার্কিক can_move_seq_axis(ধ্রুৱ ov::PartialShape & shape) {
    ঘূৰাই shape.rank().is_static() && shape.rank().get_length() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> match_kv_append(ধ্রুৱ std::shared_ptr<ov::op::v6::Assign> & assign) {
    auto concat = ov::as_type_ptr<ov::op::v0::Concat>(assign->get_প্ৰবেশ_গ্ৰন্থি_shared_ptr(0));
    যদি (!concat || concat->get_প্ৰবেশ_size() != 2 || concat->get_axis() != 1) {
        ঘূৰাই nullptr;
    }
    auto read_মান = ov::as_type_ptr<ov::op::v6::Read_মান>(concat->get_প্ৰবেশ_গ্ৰন্থি_shared_ptr(0));
    যদি (!read_মান || read_মান->get_variable() != assign->get_variable()) {
        ঘূৰাই nullptr;
    }
    যদি (!can_move_seq_axis(read_মান->get_output_partial_shape(0))) {
        ঘূৰাই nullptr;
    }
    ঘূৰাই concat;
}

}  // namespace

তার্কিক KVStateSeqAxis::run_on_model(ধ্রুৱ std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::Assign>> assigns;
    বাবে (ধ্রুৱ auto & op : model->get_ops()) {
        যদি (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(op)) {
            assigns.push_back(assign);
        }
    }

    তার্কিক changed = মিছা;
    বাবে (ধ্রুৱ auto & assign : assigns) {
        auto concat = match_kv_append(assign);
        যদি (!concat) {
            অনুৱর্ত;
        }
        auto read_মান = ov::as_type_ptr<ov::op::v6::Read_মান>(concat->get_প্ৰবেশ_গ্ৰন্থি_shared_ptr(0));

        auto variable = read_মান->get_variable();
        auto info = variable->get_info();
        ধ্রুৱ auto & shape = info.data_shape;
        info.data_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_মান->validate_and_infer_types();

        auto readers = concat->output(0).get_target_inputs();

        auto new_rows = concat->প্ৰবেশ_মান(1);
        auto perm_in = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->set_argument(1, std::make_shared<ov::op::v1::Transpose>(new_rows, perm_in));
        concat->set_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        বাবে (ধ্রুৱ auto & reader : readers) {
            auto * গ্ৰন্থি = reader.get_গ্ৰন্থি();
            যদি (ov::is_type<ov::op::v6::Assign>(গ্ৰন্থি)) {
                অনুৱর্ত;
            }
            তার্কিক dropped = মিছা;
            যদি (auto * transpose = ov::as_type<ov::op::v1::Transpose>(গ্ৰন্থি)) {
                auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_প্ৰবেশ_গ্ৰন্থি_shared_ptr(1));
                যদি (order && order->cast_vector<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = সঁচা;
                }
            }
            যদি (!dropped) {
                auto perm_out = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_source_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_out));
            }
        }
        changed = সঁচা;
    }

    ঘূৰাই changed;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
