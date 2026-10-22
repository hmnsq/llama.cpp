#अन्तर्भाव "fuse_moe_router.h"

#अन्तर्भाव <openvino/core/graph_util.hpp>
#अन्तर्भाव <openvino/core/rt_info.hpp>
#अन्तर्भाव <openvino/op/broadcast.hpp>
#अन्तर्भाव <openvino/op/clamp.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/divide.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/matmul.hpp>
#अन्तर्भाव <openvino/op/reduce_sum.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/softmax.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>
#अन्तर्भाव <openvino/op/tile.hpp>
#अन्तर्भाव <openvino/op/topk.hpp>
#अन्तर्भाव <openvino/op/unsqueeze.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/wrap_type.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश pass {

नामाकाश {

तार्किक constant_is(ध्रुव ov::Output<ov::Node> & output, ध्रुव std::vector<int64_t> & मानानि) {
    स्वचालित ग्रन्थिः = ov::as_type_ptr<ov::op::v0::Constant>(output.get_ग्रन्थिः_shared_ptr());
    प्रत्यागम ग्रन्थिः && ग्रन्थिः->get_element_type().is_integral_सङ्ख्या() && ग्रन्थिः->cast_सदिशम्<int64_t>() == मानानि;
}

}  // namespace

Fuse_Moe_मार्गदर्शकः::Fuse_Moe_मार्गदर्शकः() {
    उपयुज्य नामाकाश ov::pass::pattern;
    उपयुज्य नामाकाश ov::op;

    // Match the GGML rank-4 routing chain. The GPU plugin recognizes the resulting
    // Softmax -> TopK -> ReduceSum -> Divide subgraph as MoERouterFused.
    स्वचालित logits = wrap_type<v0::MatMul>();
    स्वचालित softmax = wrap_type<v8::Softmax>({logits});
    स्वचालित topk = wrap_type<v11::शिखरम्_K>({softmax, any_निवेशः()});
    topk->समुच्चयः_output_आकार(2);
    स्वचालित ids = wrap_type<v8::खण्डिका>({topk->output(1), any_निवेशः(), any_निवेशः(), any_निवेशः(), any_निवेशः()});
    स्वचालित ids_2d = wrap_type<v0::Squeeze>({ids, any_निवेशः()});
    स्वचालित probs = wrap_type<v1::Reshape>({softmax, any_निवेशः()});
    स्वचालित दत्त = wrap_type<v0::Squeeze>({probs, any_निवेशः()});
    स्वचालित दत्त_batch = wrap_type<v8::Gather>({wrap_type<v3::Shape_सम्बन्धि>({दत्त}), any_निवेशः(), any_निवेशः()});
    स्वचालित ids_गणनम् = wrap_type<v8::Gather>({wrap_type<v3::Shape_सम्बन्धि>({ids_2d}), any_निवेशः(), any_निवेशः()});
    स्वचालित target = wrap_type<v0::Concat>({दत्त_batch, ids_गणनम्});
    स्वचालित प्रसारणम्_2 = wrap_type<v3::प्रसारणम्>({ids_2d, target});
    स्वचालित gather = wrap_type<v8::Gather>({दत्त, प्रसारणम्_2, any_निवेशः()});
    स्वचालित weights_4d = wrap_type<v0::Unsqueeze>({gather, any_निवेशः()});
    स्वचालित weights = wrap_type<v1::Reshape>({weights_4d, any_निवेशः()});
    स्वचालित योग_2 = wrap_type<v1::Reduce_योग>({weights, any_निवेशः()});
    स्वचालित clamp = wrap_type<v0::Clamp>({योग_2});
    स्वचालित tile = wrap_type<v0::Tile>({clamp, any_निवेशः()});
    स्वचालित norm = wrap_type<v1::Divide>({weights, tile});

    ध्रुव स्वचालित callback = [=](Matcher & m) {
        ध्रुव स्वचालित & pm = m.get_आदर्शः_मानम्_map();
        ध्रुव स्वचालित ग्रन्थिः = [&](ध्रुव std::shared_ptr<ov::Node> & p) { प्रत्यागम pm.at(p).get_ग्रन्थिः_shared_ptr(); };
        ध्रुव स्वचालित निवेशः_is = [&](ध्रुव std::shared_ptr<ov::Node> & p, size_t i, ध्रुव std::vector<int64_t> & v) {
            प्रत्यागम constant_is(ग्रन्थिः(p)->निवेशः_मानम्(i), v);
        };
        स्वचालित mm = ov::as_type_ptr<v0::MatMul>(ग्रन्थिः(logits));
        स्वचालित sm = ov::as_type_ptr<v8::Softmax>(ग्रन्थिः(softmax));
        स्वचालित tk = ov::as_type_ptr<v11::शिखरम्_K>(ग्रन्थिः(topk));
        स्वचालित reduce = ov::as_type_ptr<v1::Reduce_योग>(ग्रन्थिः(योग_2));
        स्वचालित सीमा = ov::as_type_ptr<v0::Clamp>(ग्रन्थिः(clamp));
        ध्रुव स्वचालित shape = mm->get_output_partial_shape(0);
        यदि (shape.rank() != 4 || shape[0] != 1 || shape[1] != 1 || shape[3].is_dynamic() ||
            mm->get_transpose_a() || mm->get_निवेशः_partial_shape(0).rank() != 4 ||
            mm->get_निवेशः_partial_shape(1).rank() != 2 || (sm->get_axis() != -1 && sm->get_axis() != 3) ||
            tk->get_axis() != 3 || tk->get_बहुलकम्() != v11::शिखरम्_K::बहुलकम्_2::MAX ||
            tk->get_क्रमय_type() != v11::शिखरम्_K::क्रमय_Type::क्रमय_मानानि || tk->get_stable() ||
            tk->get_सूचकांक_element_type() != ov::element::i32 ||
            !tk->output(0).get_target_inputs().रिक्तानि()) {
            प्रत्यागम असत्य;
        }
        ध्रुव int64_t experts = shape[3].get_दीर्घता();
        स्वचालित end = ov::as_type_ptr<v0::Constant>(ग्रन्थिः(ids)->get_निवेशः_ग्रन्थिः_shared_ptr(2));
        यदि (!end || !end->get_element_type().is_integral_सङ्ख्या() || ov::shape_आकार(end->get_shape()) != 1) {
            प्रत्यागम असत्य;
        }
        ध्रुव int64_t k = end->cast_सदिशम्<int64_t>()[0];
        ध्रुव स्वचालित sorted_shape = tk->get_output_partial_shape(1);
        यदि (k <= 0 || k > experts || sorted_shape[3].is_dynamic() || sorted_shape[3].get_दीर्घता() < k ||
            !निवेशः_is(probs, 1, {1, -1, experts, 1}) ||
            !निवेशः_is(दत्त, 1, {0}) || !निवेशः_is(ids_2d, 1, {0, 1}) ||
            !निवेशः_is(weights_4d, 1, {0}) || !निवेशः_is(weights, 1, {1, 1, -1, k}) ||
            ov::as_type_ptr<v1::Reshape>(ग्रन्थिः(probs))->get_special_zero() ||
            ov::as_type_ptr<v1::Reshape>(ग्रन्थिः(weights))->get_special_zero() ||
            !निवेशः_is(gather, 2, {1}) || ov::as_type_ptr<v8::Gather>(ग्रन्थिः(gather))->get_batch_dims() != 1 ||
            !निवेशः_is(दत्त_batch, 1, {0}) || !निवेशः_is(दत्त_batch, 2, {0}) ||
            !निवेशः_is(ids_गणनम्, 1, {1}) || !निवेशः_is(ids_गणनम्, 2, {0}) ||
            ov::as_type_ptr<v8::Gather>(ग्रन्थिः(दत्त_batch))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v8::Gather>(ग्रन्थिः(ids_गणनम्))->get_batch_dims() != 0 ||
            ov::as_type_ptr<v0::Concat>(ग्रन्थिः(target))->get_axis() != 0 ||
            ov::as_type_ptr<v3::प्रसारणम्>(ग्रन्थिः(प्रसारणम्_2))->get_प्रसारणम्_spec().m_type != ov::op::प्रसारणम्_Type::BIDIRECTIONAL ||
            !reduce->get_keep_dims() || (!निवेशः_is(योग_2, 1, {-1}) && !निवेशः_is(योग_2, 1, {3})) ||
            !निवेशः_is(tile, 1, {1, 1, 1, k}) ||
            ov::as_type_ptr<v1::Divide>(ग्रन्थिः(norm))->get_autob().m_type != ov::op::Auto_प्रसारणम्_Type::NUMPY) {
            प्रत्यागम असत्य;
        }
        // The top-k softmax sum is at least k / experts. Leave margin for rounding.
        // This guard is only valid for unbiased softmax routing.
        यदि (!(सीमा->get_min() <= 0.5 * द्विगुण(k) / experts && सीमा->get_max() >= 2.0)) {
            प्रत्यागम असत्य;
        }
        std::vector<std::shared_ptr<ov::Node>> slices;
        प्रति (ध्रुव स्वचालित & निवेशः : tk->output(1).get_target_inputs()) {
            स्वचालित खण्डिका_2 = ov::as_type_ptr<v8::खण्डिका>(निवेशः.get_ग्रन्थिः()->shared_from_this());
            यदि (!खण्डिका_2 || निवेशः.get_सूचकांक() != 0 || खण्डिका_2->get_निवेशः_आकार() != 5 ||
                !constant_is(खण्डिका_2->निवेशः_मानम्(1), {0}) || !constant_is(खण्डिका_2->निवेशः_मानम्(2), {k}) ||
                !constant_is(खण्डिका_2->निवेशः_मानम्(3), {1}) ||
                (!constant_is(खण्डिका_2->निवेशः_मानम्(4), {3}) && !constant_is(खण्डिका_2->निवेशः_मानम्(4), {-1}))) {
                प्रत्यागम असत्य;
            }
            slices.push_back(खण्डिका_2);
        }

        // Remove GGML's leading singleton dimension while building the plugin pattern,
        // then restore it on both outputs for the following GGML nodes.
        स्वचालित axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        स्वचालित hidden = std::make_shared<v0::Squeeze>(mm->निवेशः_मानम्(0), axis0);
        स्वचालित routing = std::make_shared<v0::MatMul>(hidden, mm->निवेशः_मानम्(1), असत्य, mm->get_transpose_b());
        स्वचालित probabilities = std::make_shared<v8::Softmax>(routing, -1);
        स्वचालित selected = std::make_shared<v11::शिखरम्_K>(probabilities,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {k}), 2,
            v11::शिखरम्_K::बहुलकम्_2::MAX, v11::शिखरम्_K::क्रमय_Type::क्रमय_मानानि, tk->get_सूचकांक_element_type(), असत्य);
        स्वचालित योग = std::make_shared<v1::Reduce_योग>(selected->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), सत्य);
        स्वचालित normalized = std::make_shared<v1::Divide>(selected->output(0), योग);
        स्वचालित weights_बहिर्गमनम् = std::make_shared<v0::Unsqueeze>(normalized, axis0);
        स्वचालित ids_बहिर्गमनम् = std::make_shared<v0::Unsqueeze>(selected->output(1), axis0);
        ov::copy_runtime_info({mm, sm, tk, ग्रन्थिः(norm)}, {hidden, routing, probabilities, selected, योग, normalized, weights_बहिर्गमनम्, ids_बहिर्गमनम्});
        ov::replace_ग्रन्थिः(ग्रन्थिः(norm), weights_बहिर्गमनम्);
        प्रति (ध्रुव स्वचालित & खण्डिका_2 : slices) {
            ov::replace_ग्रन्थिः(खण्डिका_2, ids_बहिर्गमनम्);
        }
        प्रत्यागम सत्य;
    };
    पञ्जिकाकोशः_matcher(std::make_shared<Matcher>(norm, "ov::frontend::ggml::pass::FuseMoeRouter"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
