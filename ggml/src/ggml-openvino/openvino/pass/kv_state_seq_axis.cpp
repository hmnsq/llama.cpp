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
    प्रत्यागम shape.rank().is_static() && shape.rank().get_दीर्घता() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> मेलनम्_kv_append(ध्रुव std::shared_ptr<ov::op::v6::नियोजय_2> & नियोजय) {
    स्वचालित concat = ov::as_type_ptr<ov::op::v0::Concat>(नियोजय->get_निवेशः_ग्रन्थिः_shared_ptr(0));
    यदि (!concat || concat->get_निवेशः_आकार() != 2 || concat->get_axis() != 1) {
        प्रत्यागम शून्यसूचकः;
    }
    स्वचालित read_मानम् = ov::as_type_ptr<ov::op::v6::Read_मानम्>(concat->get_निवेशः_ग्रन्थिः_shared_ptr(0));
    यदि (!read_मानम् || read_मानम्->get_variable() != नियोजय->get_variable()) {
        प्रत्यागम शून्यसूचकः;
    }
    यदि (!can_move_seq_axis(read_मानम्->get_output_partial_shape(0))) {
        प्रत्यागम शून्यसूचकः;
    }
    प्रत्यागम concat;
}

}  // namespace

तार्किक KVStateSeqAxis::सततम्_on_model(ध्रुव std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::नियोजय_2>> assigns;
    प्रति (ध्रुव स्वचालित & op : model->get_ops()) {
        यदि (स्वचालित नियोजय = ov::as_type_ptr<ov::op::v6::नियोजय_2>(op)) {
            assigns.push_back(नियोजय);
        }
    }

    तार्किक परिवर्तितम् = असत्य;
    प्रति (ध्रुव स्वचालित & नियोजय : assigns) {
        स्वचालित concat = मेलनम्_kv_append(नियोजय);
        यदि (!concat) {
            अनुवर्त;
        }
        स्वचालित read_मानम् = ov::as_type_ptr<ov::op::v6::Read_मानम्>(concat->get_निवेशः_ग्रन्थिः_shared_ptr(0));

        स्वचालित variable = read_मानम्->get_variable();
        स्वचालित info = variable->get_info();
        ध्रुव स्वचालित & shape = info.दत्त_shape;
        info.दत्त_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_मानम्->validate_and_infer_types();

        स्वचालित readers = concat->output(0).get_target_inputs();

        स्वचालित new_पङ्क्तिकाः = concat->निवेशः_मानम्(1);
        स्वचालित perm_अन्तर् = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->समुच्चयः_तर्कः(1, std::make_shared<ov::op::v1::Transpose>(new_पङ्क्तिकाः, perm_अन्तर्));
        concat->समुच्चयः_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        प्रति (ध्रुव स्वचालित & reader : readers) {
            स्वचालित * ग्रन्थिः = reader.get_ग्रन्थिः();
            यदि (ov::is_type<ov::op::v6::नियोजय_2>(ग्रन्थिः)) {
                अनुवर्त;
            }
            तार्किक dropped = असत्य;
            यदि (स्वचालित * transpose = ov::as_type<ov::op::v1::Transpose>(ग्रन्थिः)) {
                स्वचालित क्रमः = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_निवेशः_ग्रन्थिः_shared_ptr(1));
                यदि (क्रमः && क्रमः->cast_सदिशम्<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = सत्य;
                }
            }
            यदि (!dropped) {
                स्वचालित perm_बहिर्गमनम् = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_source_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_बहिर्गमनम्));
            }
        }
        परिवर्तितम् = सत्य;
    }

    प्रत्यागम परिवर्तितम्;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
