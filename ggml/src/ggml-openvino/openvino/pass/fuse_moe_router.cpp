#ضمن "fuse_moe_router.h"

#ضمن <openvino/core/graph_util.hpp>
#ضمن <openvino/core/rt_info.hpp>
#ضمن <openvino/op/broadcast.hpp>
#ضمن <openvino/op/clamp.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/matmul.hpp>
#ضمن <openvino/op/reduce_sum.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/softmax.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/tile.hpp>
#ضمن <openvino/op/topk.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

نطاق {

منطقي constant_is(ثابت ov::Output<ov::Node> & output, ثابت std::vector<int64_t> & قدرون) {
    خودڪار ڳنڍ = ov::as_type_ptr<ov::op::v0::Constant>(output.get_ڳنڍ_shared_ptr());
    أرجع ڳنڍ && ڳنڍ->get_element_type().is_integral_عدد() && ڳنڍ->cast_ويڪٽر<int64_t>() == قدرون;
}

}  // namespace

Fuse_Moe_رائوٽر::Fuse_Moe_رائوٽر() {
    استعمال نطاق ov::pass::pattern;
    استعمال نطاق ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    خودڪار logits = wrap_type<v0::MatMul>();
    خودڪار softmax = wrap_type<v8::Softmax>({logits});
    خودڪار topk = wrap_type<v11::چوٽي_K>({softmax, any_داخلا()});
    topk->سيٽ_output_سائيز(2);
    خودڪار ids = wrap_type<v8::ٽڪرو>({topk->output(1), any_داخلا(), any_داخلا(), any_داخلا(), any_داخلا()});
    خودڪار ids_2d = wrap_type<v0::Squeeze>({ids, any_داخلا()});
    خودڪار probs = wrap_type<v1::Reshape>({softmax, any_داخلا()});
    خودڪار معلومات = wrap_type<v0::Squeeze>({probs, any_داخلا()});
    خودڪار معلومات_batch = wrap_type<v8::Gather>({wrap_type<v3::Shape_جو>({معلومات}), any_داخلا(), any_داخلا()});
    خودڪار ids_ڳڻپ = wrap_type<v8::Gather>({wrap_type<v3::Shape_جو>({ids_2d}), any_داخلا(), any_داخلا()});
    خودڪار هدف = wrap_type<v0::Concat>({معلومات_batch, ids_ڳڻپ});
    خودڪار نشر_2 = wrap_type<v3::نشر>({ids_2d, هدف});
    خودڪار gather = wrap_type<v8::Gather>({معلومات, نشر_2, any_داخلا()});
    خودڪار weights_4d = wrap_type<v0::Unsqueeze>({gather, any_داخلا()});
    خودڪار weights = wrap_type<v1::Reshape>({weights_4d, any_داخلا()});
    خودڪار جوڙ = wrap_type<v1::Reduce_جوڙ>({weights, any_داخلا()});
    خودڪار clamp = wrap_type<v0::Clamp>({جوڙ});
    خودڪار tile = wrap_type<v0::Tile>({clamp, any_داخلا()});
    خودڪار norm = wrap_type<v1::Divide>({weights, tile});

    ثابت خودڪار callback = [=](Matcher & m) {
        ثابت خودڪار & pm = m.get_نمونو_قدر_map();
        ثابت خودڪار ڳنڍ = [&](ثابت std::shared_ptr<ov::Node> & p) { أرجع pm.at(p).get_ڳنڍ_shared_ptr(); };
        ثابت خودڪار داخلا_is = [&](ثابت std::shared_ptr<ov::Node> & p, size_t i, ثابت std::vector<int64_t> & v) {
            أرجع constant_is(ڳنڍ(p)->داخلا_قدر(i), v);
        };
        خودڪار mm = ov::as_type_ptr<v0::MatMul>(ڳنڍ(logits));
        خودڪار sm = ov::as_type_ptr<v8::Softmax>(ڳنڍ(softmax));
        خودڪار tk = ov::as_type_ptr<v11::چوٽي_K>(ڳنڍ(topk));
        خودڪار reduce = ov::as_type_ptr<v1::Reduce_جوڙ>(ڳنڍ(جوڙ));
        خودڪار حد_بندي = ov::as_type_ptr<v0::Clamp>(ڳنڍ(clamp));
        ثابت خودڪار shape = mm->get_output_partial_shape(0);
        إذا (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_داخلا_partial_shape(0).rank() != 4 ||
            mm->get_داخلا_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_گهڻائي() != v11::چوٽي_K::گهڻائي_2::MAX ||
            tk->get_ترتيب_ڏيو_type() != v11::چوٽي_K::ترتيب_ڏيو_Type::ترتيب_ڏيو_قدرون || tk->get_stable() ||
            tk->get_اشاريو_element_type() != ov::element::i32 ||
            !tk->output(0).get_هدف_inputs().خالي_جايون()) {
            أرجع خطأ;
        }
        ثابت int64_t experts = shape[3].get_ڊيگهائي();
        خودڪار end = ov::as_type_ptr<v0::Constant>(ڳنڍ(ids)->get_داخلا_ڳنڍ_shared_ptr(2));
        إذا (!end || !end->get_element_type().is_integral_عدد() || ov::shape_سائيز(end->get_shape()) != 1) {
            أرجع خطأ;
        }
        ثابت int64_t k = end->cast_ويڪٽر<int64_t>()[0];
        ثابت خودڪار sorted_shape = tk->get_output_partial_shape(1);
        إذا (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_ڊيگهائي() < k ||
            !داخلا_is(probs, 1, {1, -1, experts, 1}) ||
            !داخلا_is(معلومات, 1, {0}) || !داخلا_is(ids_2d, 1, {0, 1}) ||
            !داخلا_is(weights_4d, 1, {0}) || !داخلا_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(ڳنڍ(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(ڳنڍ(weights))->get_special_zero() ||
            !داخلا_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(ڳنڍ(gather))->get_batch_dims() != 1 ||
            !داخلا_is(معلومات_batch, 1, {0}) || !داخلا_is(معلومات_batch, 2, {0}) ||
            !داخلا_is(ids_ڳڻپ, 1, {1}) || !داخلا_is(ids_ڳڻپ, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(ڳنڍ(معلومات_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(ڳنڍ(ids_ڳڻپ))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(ڳنڍ(هدف))->get_axis() != 0 ||
            ov::as_type_ptr<v3::نشر>(ڳنڍ(نشر_2))->get_نشر_spec().m_type != ov::op::نشر_Type::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!داخلا_is(جوڙ, 1, {-1}) && !داخلا_is(جوڙ, 1, {3})) ||
            !داخلا_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(ڳنڍ(norm))->get_autob().m_type != ov::op::Auto_نشر_Type::NUMPY) {
            أرجع خطأ;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        إذا (!(حد_بندي->get_min() <= 0.5 * مزدوج(k) / experts && حد_بندي->get_max() >= 2.0)) {
            أرجع خطأ;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        لكل (ثابت خودڪار & داخلا_2 : tk->output(1).get_هدف_inputs()) {
            خودڪار ٽڪرو_2 = ov::as_type_ptr<v8::ٽڪرو>(داخلا_2.get_ڳنڍ()->shared_from_this());
            إذا (!ٽڪرو_2 || داخلا_2.get_اشاريو() != 0 || ٽڪرو_2->get_داخلا_سائيز() != 5 ||
                !constant_is(ٽڪرو_2->داخلا_قدر(1), {0}) || !constant_is(ٽڪرو_2->داخلا_قدر(2), {k}) ||
                !constant_is(ٽڪرو_2->داخلا_قدر(3), {1}) ||
                (!constant_is(ٽڪرو_2->داخلا_قدر(4), {3}) && !constant_is(ٽڪرو_2->داخلا_قدر(4), {-1}))) {
                أرجع خطأ;
            }
            slices.push_back(ٽڪرو_2);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        خودڪار axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        خودڪار hidden = std::make_shared<v0::Squeeze>(mm->داخلا_قدر(0), axis0);
        خودڪار routing = std::make_shared<v0::MatMul>(hidden, mm->داخلا_قدر(1), خطأ, mm->get_transpose_b());
        خودڪار probabilities = std::make_shared<v8::Softmax>(routing, -1);
        خودڪار selected = std::make_shared<v11::چوٽي_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::چوٽي_K::گهڻائي_2::MAX, v11::چوٽي_K::ترتيب_ڏيو_Type::ترتيب_ڏيو_قدرون, tk->get_اشاريو_element_type(), خطأ);
        خودڪار ڪل = std::make_shared<v1::Reduce_جوڙ>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), سچ);
        خودڪار normalized = std::make_shared<v1::Divide>(selected->output(0), ڪل);
        خودڪار weights_ٻاهريون = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        خودڪار ids_ٻاهريون = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, ڳنڍ(norm)}, {hidden, routing, probabilities, selected, ڪل, normalized, weights_ٻاهريون, ids_ٻاهريون});
        ov::replace_ڳنڍ(ڳنڍ(norm), weights_ٻاهريون);
        لكل (ثابت خودڪار & ٽڪرو_2 : slices) {
            ov::replace_ڳنڍ(ٽڪرو_2, ids_ٻاهريون);
        }
        أرجع سچ;
    };
    رجسٽر_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
