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

स्थिरांक std::vector<int64_t> & seq_axis_perm() {
    // [1, seq, n_heads_kv, head_size] <-> [1, n_heads_kv, seq, head_size]
    स्थिर स्थिरांक std::vector<int64_t> perm{0, 2, 1, 3};
    परत perm;
}

// True when the state still has the frontend's stateful KV layout, so the sequence axis
// can be moved: rank 4, batch and both head dims static, and seq the only dynamic dim,
// at dim 1. Any KV head count is fine. With a single head the rewrite is pure metadata
// ([1, seq, 1, head] and [1, 1, seq, head] are the same memory); with several heads it
// also drops the reader-side transpose of the whole accumulated state, which is where
// most of the gain comes from at depth.
तार्किक can_move_seq_axis(स्थिरांक ov::PartialShape & shape) {
    परत shape.rank().is_static() && shape.rank().get_length() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> match_kv_append(स्थिरांक std::shared_ptr<ov::op::v6::Assign> & assign) {
    स्वतः concat = ov::as_type_ptr<ov::op::v0::Concat>(assign->get_आदान_गाठ_shared_ptr(0));
    जर (!concat || concat->get_आदान_size() != 2 || concat->get_axis() != 1) {
        परत nullptr;
    }
    स्वतः read_मूल्य = ov::as_type_ptr<ov::op::v6::Read_मूल्य>(concat->get_आदान_गाठ_shared_ptr(0));
    जर (!read_मूल्य || read_मूल्य->get_variable() != assign->get_variable()) {
        परत nullptr;
    }
    जर (!can_move_seq_axis(read_मूल्य->get_output_partial_shape(0))) {
        परत nullptr;
    }
    परत concat;
}

}  // namespace

तार्किक KVStateSeqAxis::run_on_model(स्थिरांक std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::Assign>> assigns;
    साठी (स्थिरांक स्वतः & op : model->get_ops()) {
        जर (स्वतः assign = ov::as_type_ptr<ov::op::v6::Assign>(op)) {
            assigns.push_back(assign);
        }
    }

    तार्किक changed = असत्य;
    साठी (स्थिरांक स्वतः & assign : assigns) {
        स्वतः concat = match_kv_append(assign);
        जर (!concat) {
            सुरूठेवा;
        }
        स्वतः read_मूल्य = ov::as_type_ptr<ov::op::v6::Read_मूल्य>(concat->get_आदान_गाठ_shared_ptr(0));

        स्वतः variable = read_मूल्य->get_variable();
        स्वतः info = variable->get_info();
        स्थिरांक स्वतः & shape = info.data_shape;
        info.data_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_मूल्य->validate_and_infer_types();

        स्वतः readers = concat->output(0).get_target_inputs();

        स्वतः new_rows = concat->आदान_मूल्य(1);
        स्वतः perm_in = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->set_argument(1, std::make_shared<ov::op::v1::Transpose>(new_rows, perm_in));
        concat->set_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        साठी (स्थिरांक स्वतः & reader : readers) {
            स्वतः * गाठ = reader.get_गाठ();
            जर (ov::is_type<ov::op::v6::Assign>(गाठ)) {
                सुरूठेवा;
            }
            तार्किक dropped = असत्य;
            जर (स्वतः * transpose = ov::as_type<ov::op::v1::Transpose>(गाठ)) {
                स्वतः order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_आदान_गाठ_shared_ptr(1));
                जर (order && order->cast_vector<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = सत्य;
                }
            }
            जर (!dropped) {
                स्वतः perm_out = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_source_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_out));
            }
        }
        changed = सत्य;
    }

    परत changed;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
