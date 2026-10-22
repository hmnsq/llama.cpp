#समावेश "kv_state_seq_axis.h"

#समावेश <memory>
#समावेश <openvino/core/graph_util.hpp>
#समावेश <openvino/op/assign.hpp>
#समावेश <openvino/op/concat.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/read_value.hpp>
#समावेश <openvino/op/transpose.hpp>
#समावेश <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

namespace {

स्थिर std::vector<int64_t> & seq_axis_perm() {
    // [1, seq, n_heads_kv, head_size] <-> [1, n_heads_kv, seq, head_size]
    अटल स्थिर std::vector<int64_t> perm{0, 2, 1, 3};
    वापस perm;
}

// True when the state still has the frontend's stateful KV layout, so the sequence axis
// can be moved: rank 4, batch and both head dims static, and seq the only dynamic dim,
// at dim 1. Any KV head count is fine. With a single head the rewrite is pure metadata
// ([1, seq, 1, head] and [1, 1, seq, head] are the same memory); with several heads it
// also drops the reader-side transpose of the whole accumulated state, which is where
// most of the gain comes from at depth.
bool can_move_seq_axis(स्थिर ov::PartialShape & shape) {
    वापस shape.rank().is_static() && shape.rank().get_लंबाई() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> match_kv_append(स्थिर std::shared_ptr<ov::op::v6::Assign> & assign) {
    स्वतः concat = ov::as_type_ptr<ov::op::v0::Concat>(assign->get_निवेश_गाँठ_shared_ptr(0));
    अगर (!concat || concat->get_निवेश_आकार() != 2 || concat->get_axis() != 1) {
        वापस nullptr;
    }
    स्वतः read_मूल्य = ov::as_type_ptr<ov::op::v6::Read_मूल्य>(concat->get_निवेश_गाँठ_shared_ptr(0));
    अगर (!read_मूल्य || read_मूल्य->get_variable() != assign->get_variable()) {
        वापस nullptr;
    }
    अगर (!can_move_seq_axis(read_मूल्य->get_output_partial_shape(0))) {
        वापस nullptr;
    }
    वापस concat;
}

}  // namespace

bool KVStateSeqAxis::run_on_model(स्थिर std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::Assign>> assigns;
    क्रम (स्थिर स्वतः & op : model->get_ops()) {
        अगर (स्वतः assign = ov::as_type_ptr<ov::op::v6::Assign>(op)) {
            assigns.push_back(assign);
        }
    }

    bool changed = false;
    क्रम (स्थिर स्वतः & assign : assigns) {
        स्वतः concat = match_kv_append(assign);
        अगर (!concat) {
            जारी;
        }
        स्वतः read_मूल्य = ov::as_type_ptr<ov::op::v6::Read_मूल्य>(concat->get_निवेश_गाँठ_shared_ptr(0));

        स्वतः variable = read_मूल्य->get_variable();
        स्वतः info = variable->get_info();
        स्थिर स्वतः & shape = info.तथ्य_shape;
        info.तथ्य_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_मूल्य->validate_and_infer_types();

        स्वतः readers = concat->output(0).get_target_inputs();

        स्वतः new_rows = concat->निवेश_मूल्य(1);
        स्वतः perm_in = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->set_argument(1, std::make_shared<ov::op::v1::Transpose>(new_rows, perm_in));
        concat->set_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        क्रम (स्थिर स्वतः & reader : readers) {
            स्वतः * गाँठ = reader.get_गाँठ();
            अगर (ov::is_type<ov::op::v6::Assign>(गाँठ)) {
                जारी;
            }
            bool dropped = false;
            अगर (स्वतः * transpose = ov::as_type<ov::op::v1::Transpose>(गाँठ)) {
                स्वतः order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_निवेश_गाँठ_shared_ptr(1));
                अगर (order && order->cast_vector<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = true;
                }
            }
            अगर (!dropped) {
                स्वतः perm_out = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_source_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_out));
            }
        }
        changed = true;
    }

    वापस changed;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
