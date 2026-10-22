#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

Output_सदिशम् translate_view(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);

    यदि (context.get_op_case() == 1) {
        // Static-mode identity pass-through for VIEWs over a GATED_DELTA_NET combined output or
        // the conv_input CONCAT; the consuming op (CPY/RMS_NORM) does its own runtime-correct
        // slicing on the full tensor (see ggml-decoder.cpp compute_op_case, GGML_OP_VIEW).
        प्रत्यागम {context.get_निवेशः(0)};
    }

    यदि (!context.is_static()) {
        // On the stateless/non-static path VIEW is normally a no-op (consumers re-slice).
        // EXCEPTION: the MoE expert aggregation slices each expert plane out of
        // ffn_moe_weighted [n_embd, n_expert_used, n_tokens] with ggml_view_2d and then
        // sums the planes with a chain of ADDs (llama-graph.cpp). Those ADDs read this
        // VIEW node directly from the tensor map and do NOT re-slice, so a no-op here
        // makes every plane the full tensor and the expert sum collapses. Materialize the
        // single-expert slice here. Gated by name (ffn_moe_weighted...view) so it can't
        // affect any other view.
        ध्रुव std::string & vname = context.get_name();
        यदि (vname.find("ffn_moe_weighted") != std::string::npos) {
            स्वचालित src_ps = context.get_निवेशः_shape(0);
            स्वचालित dst_ps = context.get_output_shape();
            यदि (src_ps.rank().is_static() && dst_ps.rank().is_static() && src_ps.rank() == dst_ps.rank() &&
                src_ps.is_static() && dst_ps.is_static()) {
                स्वचालित sst = context.get_निवेशः_stride(0);
                स्वचालित dst = context.get_output_stride();
                size_t voff = context.get_output_op_विस्थापनम्();
                स्वचालित ss = src_ps.गन्तव्यम्_shape();
                स्वचालित dd = dst_ps.गन्तव्यम्_shape();
                ध्रुव size_t nd = ss.size();
                // Stateful models drop the leading size-1 batch dim, so the real OV tensor can
                // be one rank lower than the ggml shape metadata above. Axis indices derived
                // from that metadata must be shifted down by the difference before they are
                // used as OV axes. Bail out if the dims we would drop are not all 1.
                ध्रुव स्वचालित अन्तर्_rank = context.get_निवेशः(0).get_partial_shape().rank();
                ध्रुव पूर्णाङ्क axis_सरणम् =
                    अन्तर्_rank.is_static() ? (पूर्णाङ्क) nd - (पूर्णाङ्क) अन्तर्_rank.get_दीर्घता() : 0;
                तार्किक सरणम्_ok = axis_सरणम् >= 0 && (size_t) axis_सरणम् < nd;
                प्रति (पूर्णाङ्क a = 0; a < axis_सरणम् && सरणम्_ok; ++a) {
                    सरणम्_ok = (ss[a] == 1 && dd[a] == 1);
                }
                यदि (सरणम्_ok && sst.size() == nd && dst.size() == nd) {
                    // Map each dst axis of size>1 to a src axis with equal (size,stride);
                    // the unmatched src axis of size>1 is the indexed expert axis.
                    // dst_to_src[d] records which src axis each dst axis came from, so we can
                    // later pull the dynamic (token) dim from the right source axis at runtime.
                    std::vector<तार्किक> प्रयुक्तम्(nd, असत्य);
                    std::vector<पूर्णाङ्क> dst_गन्तव्यम्_src(nd, -1);
                    तार्किक ok = सत्य;
                    प्रति (size_t d = 0; d < nd; ++d) {
                        यदि (dd[d] == 1) {
                            अनुवर्त;
                        }
                        पूर्णाङ्क प्राप्तम् = -1;
                        प्रति (size_t s = 0; s < nd; ++s) {
                            यदि (!प्रयुक्तम्[s] && ss[s] == dd[d] && sst[s] == dst[d]) {
                                प्राप्तम् = (पूर्णाङ्क) s;
                                भङ्ग;
                            }
                        }
                        यदि (प्राप्तम् < 0) {
                            ok = असत्य;
                            भङ्ग;
                        }
                        प्रयुक्तम्[प्राप्तम्] = सत्य;
                        dst_गन्तव्यम्_src[d] = प्राप्तम्;
                    }
                    पूर्णाङ्क dropped = -1;
                    यदि (ok) {
                        प्रति (size_t s = 0; s < nd; ++s) {
                            यदि (!प्रयुक्तम्[s] && ss[s] > 1) {
                                यदि (dropped >= 0) {
                                    ok = असत्य;
                                    भङ्ग;
                                }
                                dropped = (पूर्णाङ्क) s;
                            }
                        }
                    }
                    यदि (ok && dropped >= 0) {
                        ध्रुव size_t dstr = sst[dropped];
                        ध्रुव int64_t dsz = (int64_t) ss[dropped];
                        यदि (dstr > 0 && voff % dstr == 0) {
                            ध्रुव int64_t sel = (int64_t) (voff / dstr);
                            यदि (sel >= 0 && sel < dsz) {
                                ov::Output<ov::Node> sl = std::make_shared<ov::op::v8::खण्डिका>(
                                    context.get_निवेशः(0),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel + 1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {dropped - axis_सरणम्}));
                                // Build the reshape target from the (concrete) dst shape, but
                                // keep the dynamic token axis dynamic instead of freezing it
                                // to the captured n_tokens. Without this the constant dst
                                // shape bakes in the prefill token count and the static value
                                // flows downstream, turning every later decoder layer static
                                // (the GPU in-place-concat KV-cache bug). The token axis is
                                // PERMUTED between the sliced input and the dst (e.g. input
                                // [1,tok,expert,emb] -> dst [1,1,tok,emb]), so special_zero
                                // (which copies the same-position dim) is not enough: pull the
                                // dynamic dim from the correct SOURCE axis via ShapeOf+Gather
                                // and place it at the dst token position.
                                ध्रुव int32_t dyn = context.get_op_dynamic_dim();  // output ggml axis, -1 if none
                                // still in ggml metadata axis space; get_shape() reverses ggml order
                                पूर्णाङ्क dst_ov_axis = (dyn != -1) ? ((पूर्णाङ्क) nd - 1 - (पूर्णाङ्क) dyn) : -1;
                                पूर्णाङ्क src_ov_axis = (dst_ov_axis >= axis_सरणम् && dst_ov_axis < (पूर्णाङ्क) nd)
                                                      ? dst_गन्तव्यम्_src[dst_ov_axis]
                                                      : -1;
                                यदि (dst_ov_axis >= 0 && src_ov_axis >= axis_सरणम्) {
                                    // target = concat of per-axis scalars; the token axis is a
                                    // runtime Gather of the slice's shape, the rest are constants.
                                    स्वचालित sl_shape = std::make_shared<ov::op::v3::Shape_सम्बन्धि>(sl, ov::element::i64);
                                    स्वचालित tok_dim = std::make_shared<ov::op::v8::Gather>(
                                        sl_shape,
                                        ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                     {src_ov_axis - axis_सरणम्}),
                                        ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
                                    ov::Output_सदिशम् parts;
                                    प्रति (पूर्णाङ्क a = axis_सरणम्; a < (पूर्णाङ्क) nd; ++a) {
                                        यदि (a == dst_ov_axis) {
                                            parts.push_back(tok_dim);
                                        } अन्यथा {
                                            parts.push_back(ov::op::v0::Constant::create(
                                                ov::element::i64, {1}, {(int64_t) dd[a]}));
                                        }
                                    }
                                    स्वचालित dc = std::make_shared<ov::op::v0::Concat>(parts, 0);
                                    स्वचालित rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, असत्य);
                                    प्रत्यागम rename_outputs_with_suffix({rs}, context.get_name());
                                }
                                std::vector<int64_t> dd_ov(dd.begin() + axis_सरणम्, dd.end());
                                स्वचालित dc = ov::op::v0::Constant::create(ov::element::i64, {dd_ov.size()}, dd_ov);
                                स्वचालित rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, असत्य);
                                प्रत्यागम rename_outputs_with_suffix({rs}, context.get_name());
                            }
                        }
                    }
                }
            }
        }
        प्रत्यागम {context.get_निवेशः(0)};
    }

    स्वचालित निवेशः = context.get_निवेशः(0);
    स्वचालित src_shape = context.get_निवेशः_shape(0);
    स्वचालित dst_shape = context.get_output_shape();

    यदि (src_shape.rank().is_dynamic() || dst_shape.rank().is_dynamic()) {
        प्रत्यागम {निवेशः};
    }

    int64_t src_elems = 1;
    int64_t dst_elems = 1;
    प्रति (int64_t i = 0; i < src_shape.rank().get_दीर्घता(); ++i) {
        यदि (src_shape[i].is_dynamic()) {
            प्रत्यागम {निवेशः};
        }
        src_elems *= src_shape[i].get_दीर्घता();
    }
    प्रति (int64_t i = 0; i < dst_shape.rank().get_दीर्घता(); ++i) {
        यदि (dst_shape[i].is_dynamic()) {
            प्रत्यागम {निवेशः};
        }
        dst_elems *= dst_shape[i].get_दीर्घता();
    }

    यदि (dst_elems >= src_elems) {
        प्रत्यागम {निवेशः};
    }

    स्वचालित src_stride = context.get_निवेशः_stride(0);
    स्वचालित dst_stride = context.get_output_stride();
    size_t view_विस्थापनम् = context.get_output_op_विस्थापनम्();

    तार्किक same_stride = (src_stride.size() == dst_stride.size());
    यदि (same_stride) {
        प्रति (size_t i = 0; i < src_stride.size(); ++i) {
            यदि (src_stride[i] != dst_stride[i]) {
                same_stride = असत्य;
                भङ्ग;
            }
        }
    }

    यदि (!same_stride) {
        प्रत्यागम {निवेशः};
    }

    स्वचालित src_ov_shape = src_shape.गन्तव्यम्_shape();
    स्वचालित dst_ov_shape = dst_shape.गन्तव्यम्_shape();
    size_t ndims = src_ov_shape.size();
    यदि (dst_ov_shape.size() != ndims) {
        प्रत्यागम {निवेशः};
    }

    std::vector<पूर्णाङ्क> diff_dims;
    प्रति (size_t i = 0; i < ndims; ++i) {
        यदि (src_ov_shape[i] != dst_ov_shape[i]) {
            diff_dims.push_back(static_cast<पूर्णाङ्क>(i));
        }
    }

    यदि (diff_dims.size() != 1) {
        प्रत्यागम {निवेशः};
    }

    पूर्णाङ्क खण्डिका_dim = diff_dims[0];
    int64_t dim_आकार = static_cast<int64_t>(src_ov_shape[खण्डिका_dim]);

    size_t ov_stride_for_dim = 1;
    प्रति (size_t i = खण्डिका_dim + 1; i < ndims; ++i) {
        ov_stride_for_dim *= src_ov_shape[i];
    }
    size_t elem_आकार = src_stride.back();
    यदि (elem_आकार == 0) {
        elem_आकार = 1;
    }

    int64_t begin_val = 0;
    यदि (ov_stride_for_dim > 0 && elem_आकार > 0) {
        begin_val = static_cast<int64_t>((view_विस्थापनम् / elem_आकार) / ov_stride_for_dim);
    }
    int64_t अन्तः_val = begin_val + static_cast<int64_t>(dst_ov_shape[खण्डिका_dim]);

    यदि (begin_val < 0 || अन्तः_val > dim_आकार) {
        प्रत्यागम {निवेशः};
    }

    स्वचालित sliced =
        std::make_shared<ov::op::v8::खण्डिका>(निवेशः, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {अन्तः_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {खण्डिका_dim}));

    sliced->समुच्चयः_friendly_name(context.get_output_name());
    प्रत्यागम {sliced->output(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
