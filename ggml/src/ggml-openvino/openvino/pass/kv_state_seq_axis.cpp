#अन्तर्भाव "kv_state_seq_axis.h"

#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/graph_util.hpp>
#अन्तर्भाव <openvino/op/assign.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/read_value.hpp>
#अन्तर्भाव <openvino/op/transpose.hpp>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश pass {

नामाकाश {

ध्रुव std::vector<int64_t> & seq_axis_perm() {
    // [1, seq, n_heads_kv, head_size] <-> [1, n_heads_kv, seq, head_size]
    स्थिर ध्रुव std::vector<int64_t> perm{0, 2, 1, 3};
    प्रत्यागम perm;
}

// True when the state still has the frontend's stateful KV layout, so the sequence axis
// can be moved: rank 4, batch and both head dims static, and seq the only dynamic dim,
// at dim 1. Any KV head count is fine. With a single head the rewrite is pure metadata
// ([1, seq, 1, head] and [1, 1, seq, head] are the same memory); with several heads it
// also drops the reader-side transpose of the whole accumulated state, which is where
// most of the gain comes from at depth.
तार्किक can_move_seq_axis(ध्रुव ov::PartialShape & shape) {
    प्रत्यागम shape.rank().is_static() && shape.rank().get_length() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> match_kv_append(ध्रुव std::shared_ptr<ov::op::v6::Assign> & assign) {
    auto concat = ov::as_type_ptr<ov::op::v0::Concat>(assign->get_निवेशः_ग्रन्थिः_shared_ptr(0));
    जइ (!concat || concat->get_निवेशः_size() != 2 || concat->get_axis() != 1) {
        प्रत्यागम nullptr;
    }
    auto read_मानम् = ov::as_type_ptr<ov::op::v6::Read_मानम्>(concat->get_निवेशः_ग्रन्थिः_shared_ptr(0));
    जइ (!read_मानम् || read_मानम्->get_variable() != assign->get_variable()) {
        प्रत्यागम nullptr;
    }
    जइ (!can_move_seq_axis(read_मानम्->get_output_partial_shape(0))) {
        प्रत्यागम nullptr;
    }
    प्रत्यागम concat;
}

}  // namespace

तार्किक KVStateSeqAxis::run_on_model(ध्रुव std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::Assign>> assigns;
    प्रति (ध्रुव auto & op : model->get_ops()) {
        जइ (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(op)) {
            assigns.push_back(assign);
        }
    }

    तार्किक changed = मिच्छ;
    प्रति (ध्रुव auto & assign : assigns) {
        auto concat = match_kv_append(assign);
        जइ (!concat) {
            अनुवर्त;
        }
        auto read_मानम् = ov::as_type_ptr<ov::op::v6::Read_मानम्>(concat->get_निवेशः_ग्रन्थिः_shared_ptr(0));

        auto variable = read_मानम्->get_variable();
        auto info = variable->get_info();
        ध्रुव auto & shape = info.data_shape;
        info.data_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_मानम्->validate_and_infer_types();

        auto readers = concat->output(0).get_target_inputs();

        auto new_rows = concat->निवेशः_मानम्(1);
        auto perm_in = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->set_argument(1, std::make_shared<ov::op::v1::Transpose>(new_rows, perm_in));
        concat->set_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        प्रति (ध्रुव auto & reader : readers) {
            auto * ग्रन्थिः = reader.get_ग्रन्थिः();
            जइ (ov::is_type<ov::op::v6::Assign>(ग्रन्थिः)) {
                अनुवर्त;
            }
            तार्किक dropped = मिच्छ;
            जइ (auto * transpose = ov::as_type<ov::op::v1::Transpose>(ग्रन्थिः)) {
                auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_निवेशः_ग्रन्थिः_shared_ptr(1));
                जइ (order && order->cast_vector<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = सच्च;
                }
            }
            जइ (!dropped) {
                auto perm_out = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_source_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_out));
            }
        }
        changed = सच्च;
    }

    प्रत्यागम changed;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
