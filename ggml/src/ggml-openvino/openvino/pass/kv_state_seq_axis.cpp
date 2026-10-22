#شامل "kv_state_seq_axis.h"

#شامل <memory>
#شامل <openvino/core/graph_util.hpp>
#شامل <openvino/op/assign.hpp>
#شامل <openvino/op/concat.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/read_value.hpp>
#شامل <openvino/op/transpose.hpp>
#شامل <vector>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا pass {

نامخلا {

ثابت std::vector<int64_t> & seq_axis_perm() {
    // [1, seq, n_heads_kv, head_size] <-> [1, n_heads_kv, seq, head_size]
    ساکن ثابت std::vector<int64_t> perm{0, 2, 1, 3};
    واپس perm;
}

// True when the state still has the frontend's stateful KV layout, so the sequence axis
// can be moved: rank 4, batch and both head dims static, and seq the only dynamic dim,
// at dim 1. Any KV head count is fine. With a single head the rewrite is pure metadata
// ([1, seq, 1, head] and [1, 1, seq, head] are the same memory); with several heads it
// also drops the reader-side transpose of the whole accumulated state, which is where
// most of the gain comes from at depth.
منطقی can_move_seq_axis(ثابت ov::PartialShape & shape) {
    واپس shape.rank().is_static() && shape.rank().get_لمبائی() == 4 && shape[0].is_static() &&
           shape[1].is_dynamic() && shape[2].is_static() && shape[3].is_static();
}

std::shared_ptr<ov::op::v0::Concat> مطابقت_kv_append(ثابت std::shared_ptr<ov::op::v6::تفویض_2> & تفویض) {
    خودکار concat = ov::as_type_ptr<ov::op::v0::Concat>(تفویض->get_داخلہ_گرہ_shared_ptr(0));
    اگر (!concat || concat->get_داخلہ_سائز() != 2 || concat->get_axis() != 1) {
        واپس خالی_اشاریہ;
    }
    خودکار read_قدر = ov::as_type_ptr<ov::op::v6::Read_قدر>(concat->get_داخلہ_گرہ_shared_ptr(0));
    اگر (!read_قدر || read_قدر->get_variable() != تفویض->get_variable()) {
        واپس خالی_اشاریہ;
    }
    اگر (!can_move_seq_axis(read_قدر->get_output_partial_shape(0))) {
        واپس خالی_اشاریہ;
    }
    واپس concat;
}

}  // namespace

منطقی KVStateSeqAxis::تسلسل_on_model(ثابت std::shared_ptr<ov::Model> & model) {
    std::vector<std::shared_ptr<ov::op::v6::تفویض_2>> assigns;
    برائے (ثابت خودکار & op : model->get_ops()) {
        اگر (خودکار تفویض = ov::as_type_ptr<ov::op::v6::تفویض_2>(op)) {
            assigns.push_back(تفویض);
        }
    }

    منطقی بدلا = جھوٹ;
    برائے (ثابت خودکار & تفویض : assigns) {
        خودکار concat = مطابقت_kv_append(تفویض);
        اگر (!concat) {
            جاری;
        }
        خودکار read_قدر = ov::as_type_ptr<ov::op::v6::Read_قدر>(concat->get_داخلہ_گرہ_shared_ptr(0));

        خودکار variable = read_قدر->get_variable();
        خودکار info = variable->get_info();
        ثابت خودکار & shape = info.ڈیٹا_shape;
        info.ڈیٹا_shape = ov::PartialShape{shape[0], shape[2], shape[1], shape[3]};
        variable->update(info);
        read_قدر->validate_and_infer_types();

        خودکار readers = concat->output(0).get_ہدف_inputs();

        خودکار new_صفیں = concat->داخلہ_قدر(1);
        خودکار perm_اندر = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
        concat->سیٹ_دلیل(1, std::make_shared<ov::op::v1::Transpose>(new_صفیں, perm_اندر));
        concat->سیٹ_axis(2);
        concat->validate_and_infer_types();

        // Readers still expect seq at dim 1. A reader that is itself the inverse
        // Transpose wanted seq at dim 2 all along, so drop it; give anything else the
        // inverse Transpose so its input is unchanged.
        برائے (ثابت خودکار & reader : readers) {
            خودکار * گرہ = reader.get_گرہ();
            اگر (ov::is_type<ov::op::v6::تفویض_2>(گرہ)) {
                جاری;
            }
            منطقی dropped = جھوٹ;
            اگر (خودکار * transpose = ov::as_type<ov::op::v1::Transpose>(گرہ)) {
                خودکار ترتیب = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_داخلہ_گرہ_shared_ptr(1));
                اگر (ترتیب && ترتیب->cast_سمتیہ<int64_t>() == seq_axis_perm()) {
                    ov::replace_output_update_name(transpose->output(0), concat->output(0));
                    dropped = سچ;
                }
            }
            اگر (!dropped) {
                خودکار perm_برآمد = ov::op::v0::Constant::create(ov::element::i64, {4}, seq_axis_perm());
                reader.replace_ماخذ_output(std::make_shared<ov::op::v1::Transpose>(concat->output(0), perm_برآمد));
            }
        }
        بدلا = سچ;
    }

    واپس بدلا;
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
