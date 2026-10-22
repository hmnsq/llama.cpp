#ଅନ୍ତର୍ଭାଵ "kv_state_seq_axis.h"

#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/core/graph_util.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/assign.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/concat.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/read_value.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/transpose.hpp>
#ଅନ୍ତର୍ଭାଵ <vector>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ pass {

ନାମାକାଶ {

ଅଚଳ std::vector<int64_t> & seq_axis_perm() {
    // [1, seq, n_heads_kv, head_size] <-> [1, n_heads_kv, seq, head_size]
    ସ୍ଥିର ଅଚଳ std::vector<int64_t> perm{0, 2, 1, 3};
    ଫେରନ୍ତୁ perm;
}

// True when the state still has the frontend's stateful KV layout, so the sequence axis
// can be moved: rank 4, batch and both head dims static, and seq the only dynamic dim,
// at dim 1. Any KV head count is fine. With a single head the rewrite is pure metadata
// ([1, seq, 1, head] and [1, 1, seq, head] are the same memory); with several heads it
// also drops the reader-side transpose of the whole accumulated state, which is where
// most of the gain comes from at depth.
ତାର୍କିକ can_move_seq_axis(ଅଚଳ ov::PartialShape & shape) {
    ଫେରନ୍ତୁ shape.rank().is_static() && shape.rank().get_length() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> match_kv_append(ଅଚଳ std::shared_ptr<ov::op::v6::Assign> & assign) {
    auto concat = ov::as_type_ptr<ov::op::v0::Concat>(assign->get_ନିବେଶ_ଗଣ୍ଠି_shared_ptr(0));
    ଯଦି (!concat || concat->get_ନିବେଶ_size() != 2 || concat->get_axis() != 1) {
        ଫେରନ୍ତୁ nullptr;
    }
    auto read_ମୂଲ୍ୟ = ov::as_type_ptr<ov::op::v6::Read_ମୂଲ୍ୟ>(concat->get_ନିବେଶ_ଗଣ୍ଠି_shared_ptr(0));
    ଯଦି (!read_ମୂଲ୍ୟ || read_ମୂଲ୍ୟ->get_variable() != assign->get_variable()) {
        ଫେରନ୍ତୁ nullptr;
    }
    ଯଦି (!can_move_seq_axis(read_ମୂଲ୍ୟ->get_output_partial_shape(0))) {
        ଫେରନ୍ତୁ nullptr;
    }
    ଫେରନ୍ତୁ concat;
}

}  // namespace

ତାର୍କିକ KVStateSeqAxis::run_on_model(ଅଚଳ std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::Assign>> assigns;
    ପାଇଁ (ଅଚଳ auto & op : model->get_ops()) {
        ଯଦି (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(op)) {
            assigns.push_back(assign);
        }
    }

    ତାର୍କିକ changed = ମିଥ୍ୟା;
    ପାଇଁ (ଅଚଳ auto & assign : assigns) {
        auto concat = match_kv_append(assign);
        ଯଦି (!concat) {
            ଜାରି;
        }
        auto read_ମୂଲ୍ୟ = ov::as_type_ptr<ov::op::v6::Read_ମୂଲ୍ୟ>(concat->get_ନିବେଶ_ଗଣ୍ଠି_shared_ptr(0));

        auto variable = read_ମୂଲ୍ୟ->get_variable();
        auto info = variable->get_info();
        ଅଚଳ auto & shape = info.data_shape;
        info.data_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_ମୂଲ୍ୟ->validate_and_infer_types();

        auto readers = concat->output(0).get_target_inputs();

        auto new_rows = concat->ନିବେଶ_ମୂଲ୍ୟ(1);
        auto perm_in = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->set_argument(1, std::make_shared<ov::op::v1::Transpose>(new_rows, perm_in));
        concat->set_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        ପାଇଁ (ଅଚଳ auto & reader : readers) {
            auto * ଗଣ୍ଠି = reader.get_ଗଣ୍ଠି();
            ଯଦି (ov::is_type<ov::op::v6::Assign>(ଗଣ୍ଠି)) {
                ଜାରି;
            }
            ତାର୍କିକ dropped = ମିଥ୍ୟା;
            ଯଦି (auto * transpose = ov::as_type<ov::op::v1::Transpose>(ଗଣ୍ଠି)) {
                auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_ନିବେଶ_ଗଣ୍ଠି_shared_ptr(1));
                ଯଦି (order && order->cast_vector<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = ସତ୍ୟ;
                }
            }
            ଯଦି (!dropped) {
                auto perm_out = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_source_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_out));
            }
        }
        changed = ସତ୍ୟ;
    }

    ଫେରନ୍ତୁ changed;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
