#ਅਨ੍ਤਰ੍ਭਾਵ "kv_state_seq_axis.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/graph_util.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/assign.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/concat.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/read_value.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/transpose.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ pass {

ਨਾਮਾਕਾਸ਼ {

ਅਚਲ std::vector<int64_t> & seq_axis_perm() {
    // [1, seq, n_heads_kv, head_size] <-> [1, n_heads_kv, seq, head_size]
    ਸਥਿਰ ਅਚਲ std::vector<int64_t> perm{0, 2, 1, 3};
    ਵਾਪਸ perm;
}

// True when the state still has the frontend's stateful KV layout, so the sequence axis
// can be moved: rank 4, batch and both head dims static, and seq the only dynamic dim,
// at dim 1. Any KV head count is fine. With a single head the rewrite is pure metadata
// ([1, seq, 1, head] and [1, 1, seq, head] are the same memory); with several heads it
// also drops the reader-side transpose of the whole accumulated state, which is where
// most of the gain comes from at depth.
ਤਾਰਕਿਕ can_move_seq_axis(ਅਚਲ ov::PartialShape & shape) {
    ਵਾਪਸ shape.rank().is_static() && shape.rank().get_ਲੰਮਾਈ() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> ਮੇਲ_kv_append(ਅਚਲ std::shared_ptr<ov::op::v6::ਸੌਂਪੋ_2> & ਸੌਂਪੋ) {
    ਸਵੈਚਾਲਿਤ concat = ov::as_type_ptr<ov::op::v0::Concat>(ਸੌਂਪੋ->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(0));
    ਜੇ (!concat || concat->get_ਨਿਵੇਸ਼_ਮਾਪ() != 2 || concat->get_axis() != 1) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਸਵੈਚਾਲਿਤ read_ਮੁੱਲ = ov::as_type_ptr<ov::op::v6::Read_ਮੁੱਲ>(concat->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(0));
    ਜੇ (!read_ਮੁੱਲ || read_ਮੁੱਲ->get_variable() != ਸੌਂਪੋ->get_variable()) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਜੇ (!can_move_seq_axis(read_ਮੁੱਲ->get_output_partial_shape(0))) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਵਾਪਸ concat;
}

}  // namespace

ਤਾਰਕਿਕ KVStateSeqAxis::ਲੜੀਵਾਰ_on_model(ਅਚਲ std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::ਸੌਂਪੋ_2>> assigns;
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & op : model->get_ops()) {
        ਜੇ (ਸਵੈਚਾਲਿਤ ਸੌਂਪੋ = ov::as_type_ptr<ov::op::v6::ਸੌਂਪੋ_2>(op)) {
            assigns.push_back(ਸੌਂਪੋ);
        }
    }

    ਤਾਰਕਿਕ ਬਦਲਿਆ = ਝੂਠ;
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਸੌਂਪੋ : assigns) {
        ਸਵੈਚਾਲਿਤ concat = ਮੇਲ_kv_append(ਸੌਂਪੋ);
        ਜੇ (!concat) {
            ਜਾਰੀ;
        }
        ਸਵੈਚਾਲਿਤ read_ਮੁੱਲ = ov::as_type_ptr<ov::op::v6::Read_ਮੁੱਲ>(concat->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(0));

        ਸਵੈਚਾਲਿਤ variable = read_ਮੁੱਲ->get_variable();
        ਸਵੈਚਾਲਿਤ info = variable->get_info();
        ਅਚਲ ਸਵੈਚਾਲਿਤ & shape = info.ਅੰਕੜੇ_shape;
        info.ਅੰਕੜੇ_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_ਮੁੱਲ->validate_and_infer_types();

        ਸਵੈਚਾਲਿਤ readers = concat->output(0).get_ਨਿਸ਼ਾਨਾ_inputs();

        ਸਵੈਚਾਲਿਤ new_ਪੰਕਤੀਆਂ = concat->ਨਿਵੇਸ਼_ਮੁੱਲ(1);
        ਸਵੈਚਾਲਿਤ perm_ਵਿੱਚ = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->ਸੈੱਟ_ਦਲੀਲ(1, std::make_shared<ov::op::v1::Transpose>(new_ਪੰਕਤੀਆਂ, perm_ਵਿੱਚ));
        concat->ਸੈੱਟ_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & reader : readers) {
            ਸਵੈਚਾਲਿਤ * ਗੰਢ = reader.get_ਗੰਢ();
            ਜੇ (ov::is_type<ov::op::v6::ਸੌਂਪੋ_2>(ਗੰਢ)) {
                ਜਾਰੀ;
            }
            ਤਾਰਕਿਕ dropped = ਝੂਠ;
            ਜੇ (ਸਵੈਚਾਲਿਤ * transpose = ov::as_type<ov::op::v1::Transpose>(ਗੰਢ)) {
                ਸਵੈਚਾਲਿਤ ਕ੍ਰਮ = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(1));
                ਜੇ (ਕ੍ਰਮ && ਕ੍ਰਮ->cast_ਸਦਿਸ਼<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = ਸੱਚ;
                }
            }
            ਜੇ (!dropped) {
                ਸਵੈਚਾਲਿਤ perm_ਬਾਹਰਲਾ = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_ਸਰੋਤ_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_ਬਾਹਰਲਾ));
            }
        }
        ਬਦਲਿਆ = ਸੱਚ;
    }

    ਵਾਪਸ ਬਦਲਿਆ;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
