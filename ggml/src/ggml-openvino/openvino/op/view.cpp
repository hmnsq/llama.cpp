#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <openvino/op/concat.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/gather.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/shape_of.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_view(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);

    ఒకవేళ (context.get_op_case() == 1) {
        // Static-mode identity pass-through for VIEWs over a GATED_DELTA_NET combined output or
        // the conv_input CONCAT; the consuming op (CPY/RMS_NORM) does its own runtime-correct
        // slicing on the full tensor (see ggml-decoder.cpp compute_op_case, GGML_OP_VIEW).
        తిరిగి {context.get_ఆదానం(0)};
    }

    ఒకవేళ (!context.is_static()) {
        // On the stateless/non-static path VIEW is normally a no-op (consumers re-slice).
        // EXCEPTION: the MoE expert aggregation slices each expert plane out of
        // ffn_moe_weighted [n_embd, n_expert_used, n_tokens] with ggml_view_2d and then
        // sums the planes with a chain of ADDs (llama-graph.cpp). Those ADDs read this
        // VIEW node directly from the tensor map and do NOT re-slice, so a no-op here
        // makes every plane the full tensor and the expert sum collapses. Materialize the
        // single-expert slice here. Gated by name (ffn_moe_weighted...view) so it can't
        // affect any other view.
        స్థిరరాశి std::string & vname = context.get_name();
        ఒకవేళ (vname.find("ffn_moe_weighted") != std::string::npos) {
            స్వయంచాలక src_ps = context.get_ఆదానం_shape(0);
            స్వయంచాలక dst_ps = context.get_output_shape();
            ఒకవేళ (src_ps.rank().is_static() && dst_ps.rank().is_static() && src_ps.rank() == dst_ps.rank() &&
                src_ps.is_static() && dst_ps.is_static()) {
                స్వయంచాలక sst = context.get_ఆదానం_stride(0);
                స్వయంచాలక dst = context.get_output_stride();
                size_t voff = context.get_output_op_స్థానభ్రంశం();
                స్వయంచాలక ss = src_ps.గమ్యం_shape();
                స్వయంచాలక dd = dst_ps.గమ్యం_shape();
                స్థిరరాశి size_t nd = ss.size();
                // Stateful models drop the leading size-1 batch dim, so the real OV tensor can
                // be one rank lower than the ggml shape metadata above. Axis indices derived
                // from that metadata must be shifted down by the difference before they are
                // used as OV axes. Bail out if the dims we would drop are not all 1.
                స్థిరరాశి స్వయంచాలక లోపల_rank = context.get_ఆదానం(0).get_partial_shape().rank();
                స్థిరరాశి పూర్ణాంకం axis_జరుపు =
                    లోపల_rank.is_static() ? (పూర్ణాంకం) nd - (పూర్ణాంకం) లోపల_rank.get_నిడివి() : 0;
                తార్కిక జరుపు_ok = axis_జరుపు >= 0 && (size_t) axis_జరుపు < nd;
                కొరకు (పూర్ణాంకం a = 0; a < axis_జరుపు && జరుపు_ok; ++a) {
                    జరుపు_ok = (ss[a] == 1 && dd[a] == 1);
                }
                ఒకవేళ (జరుపు_ok && sst.size() == nd && dst.size() == nd) {
                    // Map each dst axis of size>1 to a src axis with equal (size,stride);
                    // the unmatched src axis of size>1 is the indexed expert axis.
                    // dst_to_src[d] records which src axis each dst axis came from, so we can
                    // later pull the dynamic (token) dim from the right source axis at runtime.
                    std::vector<తార్కిక> వాడిన(nd, అబద్ధం);
                    std::vector<పూర్ణాంకం> dst_గమ్యం_src(nd, -1);
                    తార్కిక ok = నిజం;
                    కొరకు (size_t d = 0; d < nd; ++d) {
                        ఒకవేళ (dd[d] == 1) {
                            కొనసాగు;
                        }
                        పూర్ణాంకం దొరికింది = -1;
                        కొరకు (size_t s = 0; s < nd; ++s) {
                            ఒకవేళ (!వాడిన[s] && ss[s] == dd[d] && sst[s] == dst[d]) {
                                దొరికింది = (పూర్ణాంకం) s;
                                ఆపు;
                            }
                        }
                        ఒకవేళ (దొరికింది < 0) {
                            ok = అబద్ధం;
                            ఆపు;
                        }
                        వాడిన[దొరికింది] = నిజం;
                        dst_గమ్యం_src[d] = దొరికింది;
                    }
                    పూర్ణాంకం dropped = -1;
                    ఒకవేళ (ok) {
                        కొరకు (size_t s = 0; s < nd; ++s) {
                            ఒకవేళ (!వాడిన[s] && ss[s] > 1) {
                                ఒకవేళ (dropped >= 0) {
                                    ok = అబద్ధం;
                                    ఆపు;
                                }
                                dropped = (పూర్ణాంకం) s;
                            }
                        }
                    }
                    ఒకవేళ (ok && dropped >= 0) {
                        స్థిరరాశి size_t dstr = sst[dropped];
                        స్థిరరాశి int64_t dsz = (int64_t) ss[dropped];
                        ఒకవేళ (dstr > 0 && voff % dstr == 0) {
                            స్థిరరాశి int64_t sel = (int64_t) (voff / dstr);
                            ఒకవేళ (sel >= 0 && sel < dsz) {
                                ov::Output<ov::Node> sl = std::make_shared<ov::op::v8::ముక్క>(
                                    context.get_ఆదానం(0),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel + 1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {dropped - axis_జరుపు}));
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
                                స్థిరరాశి int32_t dyn = context.get_op_dynamic_dim();  // output ggml axis, -1 if none
                                // still in ggml metadata axis space; get_shape() reverses ggml order
                                పూర్ణాంకం dst_ov_axis = (dyn != -1) ? ((పూర్ణాంకం) nd - 1 - (పూర్ణాంకం) dyn) : -1;
                                పూర్ణాంకం src_ov_axis = (dst_ov_axis >= axis_జరుపు && dst_ov_axis < (పూర్ణాంకం) nd)
                                                      ? dst_గమ్యం_src[dst_ov_axis]
                                                      : -1;
                                ఒకవేళ (dst_ov_axis >= 0 && src_ov_axis >= axis_జరుపు) {
                                    // target = concat of per-axis scalars; the token axis is a
                                    // runtime Gather of the slice's shape, the rest are constants.
                                    స్వయంచాలక sl_shape = std::make_shared<ov::op::v3::Shape_యొక్క>(sl, ov::element::i64);
                                    స్వయంచాలక tok_dim = std::make_shared<ov::op::v8::Gather>(
                                        sl_shape,
                                        ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                     {src_ov_axis - axis_జరుపు}),
                                        ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
                                    ov::Output_సదిశ parts;
                                    కొరకు (పూర్ణాంకం a = axis_జరుపు; a < (పూర్ణాంకం) nd; ++a) {
                                        ఒకవేళ (a == dst_ov_axis) {
                                            parts.push_back(tok_dim);
                                        } లేకపోతే {
                                            parts.push_back(ov::op::v0::Constant::create(
                                                ov::element::i64, {1}, {(int64_t) dd[a]}));
                                        }
                                    }
                                    స్వయంచాలక dc = std::make_shared<ov::op::v0::Concat>(parts, 0);
                                    స్వయంచాలక rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, అబద్ధం);
                                    తిరిగి rename_outputs_with_suffix({rs}, context.get_name());
                                }
                                std::vector<int64_t> dd_ov(dd.begin() + axis_జరుపు, dd.end());
                                స్వయంచాలక dc = ov::op::v0::Constant::create(ov::element::i64, {dd_ov.size()}, dd_ov);
                                స్వయంచాలక rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, అబద్ధం);
                                తిరిగి rename_outputs_with_suffix({rs}, context.get_name());
                            }
                        }
                    }
                }
            }
        }
        తిరిగి {context.get_ఆదానం(0)};
    }

    స్వయంచాలక ఆదానం = context.get_ఆదానం(0);
    స్వయంచాలక src_shape = context.get_ఆదానం_shape(0);
    స్వయంచాలక dst_shape = context.get_output_shape();

    ఒకవేళ (src_shape.rank().is_dynamic() || dst_shape.rank().is_dynamic()) {
        తిరిగి {ఆదానం};
    }

    int64_t src_elems = 1;
    int64_t dst_elems = 1;
    కొరకు (int64_t i = 0; i < src_shape.rank().get_నిడివి(); ++i) {
        ఒకవేళ (src_shape[i].is_dynamic()) {
            తిరిగి {ఆదానం};
        }
        src_elems *= src_shape[i].get_నిడివి();
    }
    కొరకు (int64_t i = 0; i < dst_shape.rank().get_నిడివి(); ++i) {
        ఒకవేళ (dst_shape[i].is_dynamic()) {
            తిరిగి {ఆదానం};
        }
        dst_elems *= dst_shape[i].get_నిడివి();
    }

    ఒకవేళ (dst_elems >= src_elems) {
        తిరిగి {ఆదానం};
    }

    స్వయంచాలక src_stride = context.get_ఆదానం_stride(0);
    స్వయంచాలక dst_stride = context.get_output_stride();
    size_t view_స్థానభ్రంశం = context.get_output_op_స్థానభ్రంశం();

    తార్కిక same_stride = (src_stride.size() == dst_stride.size());
    ఒకవేళ (same_stride) {
        కొరకు (size_t i = 0; i < src_stride.size(); ++i) {
            ఒకవేళ (src_stride[i] != dst_stride[i]) {
                same_stride = అబద్ధం;
                ఆపు;
            }
        }
    }

    ఒకవేళ (!same_stride) {
        తిరిగి {ఆదానం};
    }

    స్వయంచాలక src_ov_shape = src_shape.గమ్యం_shape();
    స్వయంచాలక dst_ov_shape = dst_shape.గమ్యం_shape();
    size_t ndims = src_ov_shape.size();
    ఒకవేళ (dst_ov_shape.size() != ndims) {
        తిరిగి {ఆదానం};
    }

    std::vector<పూర్ణాంకం> diff_dims;
    కొరకు (size_t i = 0; i < ndims; ++i) {
        ఒకవేళ (src_ov_shape[i] != dst_ov_shape[i]) {
            diff_dims.push_back(static_cast<పూర్ణాంకం>(i));
        }
    }

    ఒకవేళ (diff_dims.size() != 1) {
        తిరిగి {ఆదానం};
    }

    పూర్ణాంకం ముక్క_dim = diff_dims[0];
    int64_t dim_కొలత = static_cast<int64_t>(src_ov_shape[ముక్క_dim]);

    size_t ov_stride_for_dim = 1;
    కొరకు (size_t i = ముక్క_dim + 1; i < ndims; ++i) {
        ov_stride_for_dim *= src_ov_shape[i];
    }
    size_t elem_కొలత = src_stride.back();
    ఒకవేళ (elem_కొలత == 0) {
        elem_కొలత = 1;
    }

    int64_t begin_val = 0;
    ఒకవేళ (ov_stride_for_dim > 0 && elem_కొలత > 0) {
        begin_val = static_cast<int64_t>((view_స్థానభ్రంశం / elem_కొలత) / ov_stride_for_dim);
    }
    int64_t చివర_val = begin_val + static_cast<int64_t>(dst_ov_shape[ముక్క_dim]);

    ఒకవేళ (begin_val < 0 || చివర_val > dim_కొలత) {
        తిరిగి {ఆదానం};
    }

    స్వయంచాలక sliced =
        std::make_shared<ov::op::v8::ముక్క>(ఆదానం, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {చివర_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {ముక్క_dim}));

    sliced->సమితి_friendly_name(context.get_output_name());
    తిరిగి {sliced->output(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
