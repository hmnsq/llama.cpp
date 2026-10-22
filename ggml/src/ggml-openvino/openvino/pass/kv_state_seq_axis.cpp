#ضمن "kv_state_seq_axis.h"

#ضمن <memory>
#ضمن <openvino/core/graph_util.hpp>
#ضمن <openvino/op/assign.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/read_value.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

نطاق {

ثابت std::vector<int64_t> & seq_axis_perm() {
    // [1, seq, n_heads_kv, head_size] <-> [1, n_heads_kv, seq, head_size]
    ساكن ثابت std::vector<int64_t> perm{0, 2, 1, 3};
    أرجع perm;
}

// True when the state still has the frontend's stateful KV layout, so the sequence axis
// can be moved: rank 4, batch and both head dims static, and seq the only dynamic dim,
// at dim 1. Any KV head count is fine. With a single head the rewrite is pure metadata
// ([1, seq, 1, head] and [1, 1, seq, head] are the same memory); with several heads it
// also drops the reader-side transpose of the whole accumulated state, which is where
// most of the gain comes from at depth.
منطقي can_move_seq_axis(ثابت ov::PartialShape & shape) {
    أرجع shape.rank().is_static() && shape.rank().get_درازی() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> تطابق_kv_append(ثابت std::shared_ptr<ov::op::v6::تخصیص_بده_2> & تخصیص_بده) {
    خودکار concat = ov::as_type_ptr<ov::op::v0::Concat>(تخصیص_بده->get_درون_داد_گره_shared_ptr(0));
    إذا (!concat || concat->get_درون_داد_اندازه() != 2 || concat->get_axis() != 1) {
        أرجع اشاره_تهی;
    }
    خودکار read_مقدار = ov::as_type_ptr<ov::op::v6::Read_مقدار>(concat->get_درون_داد_گره_shared_ptr(0));
    إذا (!read_مقدار || read_مقدار->get_variable() != تخصیص_بده->get_variable()) {
        أرجع اشاره_تهی;
    }
    إذا (!can_move_seq_axis(read_مقدار->get_output_partial_shape(0))) {
        أرجع اشاره_تهی;
    }
    أرجع concat;
}

}  // namespace

منطقي KVStateSeqAxis::پیاپی_on_model(ثابت std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::تخصیص_بده_2>> assigns;
    لكل (ثابت خودکار & op : model->get_ops()) {
        إذا (خودکار تخصیص_بده = ov::as_type_ptr<ov::op::v6::تخصیص_بده_2>(op)) {
            assigns.push_back(تخصیص_بده);
        }
    }

    منطقي تغییریافته = خطأ;
    لكل (ثابت خودکار & تخصیص_بده : assigns) {
        خودکار concat = تطابق_kv_append(تخصیص_بده);
        إذا (!concat) {
            استمر;
        }
        خودکار read_مقدار = ov::as_type_ptr<ov::op::v6::Read_مقدار>(concat->get_درون_داد_گره_shared_ptr(0));

        خودکار variable = read_مقدار->get_variable();
        خودکار info = variable->get_info();
        ثابت خودکار & shape = info.داده_shape;
        info.داده_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_مقدار->validate_and_infer_types();

        خودکار readers = concat->output(0).get_هدف_inputs();

        خودکار new_ردیفها = concat->درون_داد_مقدار(1);
        خودکار perm_درون = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->مجموعه_آرگومان(1, std::make_shared<ov::op::v1::Transpose>(new_ردیفها, perm_درون));
        concat->مجموعه_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        لكل (ثابت خودکار & reader : readers) {
            خودکار * گره = reader.get_گره();
            إذا (ov::is_type<ov::op::v6::تخصیص_بده_2>(گره)) {
                استمر;
            }
            منطقي dropped = خطأ;
            إذا (خودکار * transpose = ov::as_type<ov::op::v1::Transpose>(گره)) {
                خودکار ترتیب = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_درون_داد_گره_shared_ptr(1));
                إذا (ترتیب && ترتیب->cast_وکتور<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = درست;
                }
            }
            إذا (!dropped) {
                خودکار perm_بیرون = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_منبع_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_بیرون));
            }
        }
        تغییریافته = درست;
    }

    أرجع تغییریافته;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
