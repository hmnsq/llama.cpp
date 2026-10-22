#అన్తర్భావ "fuse_argsort_topk.h"

#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>
#అన్తర్భావ <openvino/op/topk.hpp>
#అన్తర్భావ <openvino/pass/pattern/op/wrap_type.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ pass {

Fuse_Argsort_శిఖరం_K::Fuse_Argsort_శిఖరం_K() {
    // GGML argsort materializes the full ordering before a VIEW keeps the prefix.
    // Let TopK produce only that prefix when all index consumers are such views.
    స్వయంచాలక pattern = ov::pass::pattern::wrap_type<ov::op::v11::శిఖరం_K>();

    స్థిరరాశి స్వయంచాలక callback = [](ov::pass::pattern::Matcher & m) {
        స్వయంచాలక topk = ov::as_type_ptr<ov::op::v11::శిఖరం_K>(m.get_సరిపోలిక_మూలం());
        ఒకవేళ (!topk->output(0).get_లక్ష్యం_inputs().ఖాళీలు() || topk->output(1).get_లక్ష్యం_inputs().ఖాళీలు() ||
            topk->get_క్రమీకరించు_type() != ov::op::v11::శిఖరం_K::క్రమీకరించు_Type::క్రమీకరించు_విలువలు) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి స్వయంచాలక shape = topk->get_output_partial_shape(1);
        ఒకవేళ (shape.rank().is_dynamic()) {
            తిరిగి అబద్ధం;
        }
        స్థిరరాశి int64_t rank = shape.rank().get_నిడివి();
        స్థిరరాశి int64_t axis = topk->get_axis();
        ఒకవేళ (shape[axis].is_dynamic()) {
            తిరిగి అబద్ధం;
        }

        int64_t prefix = -1;
        కొరకు (స్థిరరాశి స్వయంచాలక & ఆదానం : topk->output(1).get_లక్ష్యం_inputs()) {
            స్థిరరాశి స్వయంచాలక * ముక్క_2 = ov::as_type<ov::op::v8::ముక్క>(ఆదానం.get_కణుపు());
            ఒకవేళ (!ముక్క_2 || ఆదానం.get_సూచీ() != 0 || ముక్క_2->get_ఆదానం_కొలత() != 5) {
                తిరిగి అబద్ధం;
            }
            int64_t విలువలు[4];
            కొరకు (size_t i = 0; i < 4; ++i) {
                స్వయంచాలక value = ov::as_type_ptr<ov::op::v0::Constant>(ముక్క_2->get_ఆదానం_కణుపు_shared_ptr(i + 1));
                ఒకవేళ (!value || ov::shape_కొలత(value->get_shape()) != 1) {
                    తిరిగి అబద్ధం;
                }
                విలువలు[i] = value->cast_సదిశ<int64_t>()[0];
            }
            స్థిరరాశి int64_t ముక్క_axis = విలువలు[3] < 0 ? విలువలు[3] + rank : విలువలు[3];
            ఒకవేళ (విలువలు[0] != 0 || విలువలు[2] != 1 || ముక్క_axis != axis || విలువలు[1] <= 0 ||
                విలువలు[1] >= shape[axis].get_నిడివి() || (prefix != -1 && prefix != విలువలు[1])) {
                తిరిగి అబద్ధం;
            }
            prefix = విలువలు[1];
        }

        // ggml_argsort_top_k sorts the full row, then exposes only its first k indices.
        // Keep other TopK forms unchanged because their ordering can be observable.
        topk->సమితి_వాదం(1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {prefix}));
        topk->validate_and_infer_types();
        తిరిగి నిజం;
    };

    రిజిస్టరు_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::FuseArgsortTopK"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
