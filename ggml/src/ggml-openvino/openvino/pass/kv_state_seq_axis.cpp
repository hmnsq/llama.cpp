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

ಸ್ಥಿರಾಂಕ std::vector<int64_t> & seq_axis_perm() {
    // [1, seq, n_heads_kv, head_size] <-> [1, n_heads_kv, seq, head_size]
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<int64_t> perm{0, 2, 1, 3};
    ಹಿಂತಿರುಗಿ perm;
}

// True when the state still has the frontend's stateful KV layout, so the sequence axis
// can be moved: rank 4, batch and both head dims static, and seq the only dynamic dim,
// at dim 1. Any KV head count is fine. With a single head the rewrite is pure metadata
// ([1, seq, 1, head] and [1, 1, seq, head] are the same memory); with several heads it
// also drops the reader-side transpose of the whole accumulated state, which is where
// most of the gain comes from at depth.
ತಾರ್ಕಿಕ can_move_seq_axis(ಸ್ಥಿರಾಂಕ ov::PartialShape & shape) {
    ಹಿಂತಿರುಗಿ shape.rank().is_static() && shape.rank().get_ಉದ್ದಳತೆ() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> ಹೊಂದಿಕೆ_kv_append(ಸ್ಥಿರಾಂಕ std::shared_ptr<ov::op::v6::ನಿಯೋಜಿಸು_2> & ನಿಯೋಜಿಸು) {
    ಸ್ವಯಂಚಾಲಿತ concat = ov::as_type_ptr<ov::op::v0::Concat>(ನಿಯೋಜಿಸು->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0));
    ಆದರೆ (!concat || concat->get_ಒಳಹರಿವು_ಗಾತ್ರ() != 2 || concat->get_axis() != 1) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಸ್ವಯಂಚಾಲಿತ read_ಮೌಲ್ಯ = ov::as_type_ptr<ov::op::v6::Read_ಮೌಲ್ಯ>(concat->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0));
    ಆದರೆ (!read_ಮೌಲ್ಯ || read_ಮೌಲ್ಯ->get_variable() != ನಿಯೋಜಿಸು->get_variable()) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಆದರೆ (!can_move_seq_axis(read_ಮೌಲ್ಯ->get_output_partial_shape(0))) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಹಿಂತಿರುಗಿ concat;
}

}  // namespace

ತಾರ್ಕಿಕ KVStateSeqAxis::ಸತತ_on_model(ಸ್ಥಿರಾಂಕ std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::ನಿಯೋಜಿಸು_2>> assigns;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & op : model->get_ops()) {
        ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ ನಿಯೋಜಿಸು = ov::as_type_ptr<ov::op::v6::ನಿಯೋಜಿಸು_2>(op)) {
            assigns.push_back(ನಿಯೋಜಿಸು);
        }
    }

    ತಾರ್ಕಿಕ ಬದಲಾಯಿತು = ಸುಳ್ಳು;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ನಿಯೋಜಿಸು : assigns) {
        ಸ್ವಯಂಚಾಲಿತ concat = ಹೊಂದಿಕೆ_kv_append(ನಿಯೋಜಿಸು);
        ಆದರೆ (!concat) {
            ಮುಂದುವರಿ;
        }
        ಸ್ವಯಂಚಾಲಿತ read_ಮೌಲ್ಯ = ov::as_type_ptr<ov::op::v6::Read_ಮೌಲ್ಯ>(concat->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0));

        ಸ್ವಯಂಚಾಲಿತ variable = read_ಮೌಲ್ಯ->get_variable();
        ಸ್ವಯಂಚಾಲಿತ info = variable->get_info();
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & shape = info.ದತ್ತಾಂಶ_shape;
        info.ದತ್ತಾಂಶ_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_ಮೌಲ್ಯ->validate_and_infer_types();

        ಸ್ವಯಂಚಾಲಿತ readers = concat->output(0).get_ಗುರಿ_inputs();

        ಸ್ವಯಂಚಾಲಿತ new_ಅಡ್ಡಸಾಲುಗಳು = concat->ಒಳಹರಿವು_ಮೌಲ್ಯ(1);
        ಸ್ವಯಂಚಾಲಿತ perm_ಒಳ = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->ಗಣ_ವಾದ(1, std::make_shared<ov::op::v1::Transpose>(new_ಅಡ್ಡಸಾಲುಗಳು, perm_ಒಳ));
        concat->ಗಣ_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & reader : readers) {
            ಸ್ವಯಂಚಾಲಿತ * ಗಂಟು = reader.get_ಗಂಟು();
            ಆದರೆ (ov::is_type<ov::op::v6::ನಿಯೋಜಿಸು_2>(ಗಂಟು)) {
                ಮುಂದುವರಿ;
            }
            ತಾರ್ಕಿಕ dropped = ಸುಳ್ಳು;
            ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ * transpose = ov::as_type<ov::op::v1::Transpose>(ಗಂಟು)) {
                ಸ್ವಯಂಚಾಲಿತ ಕ್ರಮ = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(1));
                ಆದರೆ (ಕ್ರಮ && ಕ್ರಮ->cast_ಸದಿಶ<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = ನಿಜ;
                }
            }
            ಆದರೆ (!dropped) {
                ಸ್ವಯಂಚಾಲಿತ perm_ಹೊರಹೋಗು = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_ಮೂಲ_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_ಹೊರಹೋಗು));
            }
        }
        ಬದಲಾಯಿತು = ನಿಜ;
    }

    ಹಿಂತಿರುಗಿ ಬದಲಾಯಿತು;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
