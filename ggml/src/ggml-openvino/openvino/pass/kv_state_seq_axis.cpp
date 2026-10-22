#അന്തര്ഭാവ "kv_state_seq_axis.h"

#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <openvino/core/graph_util.hpp>
#അന്തര്ഭാവ <openvino/op/assign.hpp>
#അന്തര്ഭാവ <openvino/op/concat.hpp>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/read_value.hpp>
#അന്തര്ഭാവ <openvino/op/transpose.hpp>
#അന്തര്ഭാവ <vector>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ pass {

നാമാകാശ {

സ്ഥിരം std::vector<int64_t> & seq_axis_perm() {
    // [1, seq, n_heads_kv, head_size] <-> [1, n_heads_kv, seq, head_size]
    സ്ഥിതം സ്ഥിരം std::vector<int64_t> perm{0, 2, 1, 3};
    തിരികെ perm;
}

// True when the state still has the frontend's stateful KV layout, so the sequence axis
// can be moved: rank 4, batch and both head dims static, and seq the only dynamic dim,
// at dim 1. Any KV head count is fine. With a single head the rewrite is pure metadata
// ([1, seq, 1, head] and [1, 1, seq, head] are the same memory); with several heads it
// also drops the reader-side transpose of the whole accumulated state, which is where
// most of the gain comes from at depth.
തർക്കം can_move_seq_axis(സ്ഥിരം ov::PartialShape & shape) {
    തിരികെ shape.rank().is_static() && shape.rank().get_length() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> match_kv_append(സ്ഥിരം std::shared_ptr<ov::op::v6::Assign> & assign) {
    auto concat = ov::as_type_ptr<ov::op::v0::Concat>(assign->get_നിവേശം_കണ്ണി_shared_ptr(0));
    എങ്കിൽ (!concat || concat->get_നിവേശം_size() != 2 || concat->get_axis() != 1) {
        തിരികെ nullptr;
    }
    auto read_മൂല്യം = ov::as_type_ptr<ov::op::v6::Read_മൂല്യം>(concat->get_നിവേശം_കണ്ണി_shared_ptr(0));
    എങ്കിൽ (!read_മൂല്യം || read_മൂല്യം->get_variable() != assign->get_variable()) {
        തിരികെ nullptr;
    }
    എങ്കിൽ (!can_move_seq_axis(read_മൂല്യം->get_output_partial_shape(0))) {
        തിരികെ nullptr;
    }
    തിരികെ concat;
}

}  // namespace

തർക്കം KVStateSeqAxis::run_on_model(സ്ഥിരം std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::Assign>> assigns;
    വേണ്ടി (സ്ഥിരം auto & op : model->get_ops()) {
        എങ്കിൽ (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(op)) {
            assigns.push_back(assign);
        }
    }

    തർക്കം changed = അസത്യം;
    വേണ്ടി (സ്ഥിരം auto & assign : assigns) {
        auto concat = match_kv_append(assign);
        എങ്കിൽ (!concat) {
            തുടരുക;
        }
        auto read_മൂല്യം = ov::as_type_ptr<ov::op::v6::Read_മൂല്യം>(concat->get_നിവേശം_കണ്ണി_shared_ptr(0));

        auto variable = read_മൂല്യം->get_variable();
        auto info = variable->get_info();
        സ്ഥിരം auto & shape = info.data_shape;
        info.data_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_മൂല്യം->validate_and_infer_types();

        auto readers = concat->output(0).get_target_inputs();

        auto new_rows = concat->നിവേശം_മൂല്യം(1);
        auto perm_in = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->set_argument(1, std::make_shared<ov::op::v1::Transpose>(new_rows, perm_in));
        concat->set_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        വേണ്ടി (സ്ഥിരം auto & reader : readers) {
            auto * കണ്ണി = reader.get_കണ്ണി();
            എങ്കിൽ (ov::is_type<ov::op::v6::Assign>(കണ്ണി)) {
                തുടരുക;
            }
            തർക്കം dropped = അസത്യം;
            എങ്കിൽ (auto * transpose = ov::as_type<ov::op::v1::Transpose>(കണ്ണി)) {
                auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_നിവേശം_കണ്ണി_shared_ptr(1));
                എങ്കിൽ (order && order->cast_vector<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = സത്യം;
                }
            }
            എങ്കിൽ (!dropped) {
                auto perm_out = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_source_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_out));
            }
        }
        changed = സത്യം;
    }

    തിരികെ changed;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
