#pragma once

#अन्तर्भाव "openvino/op/op.hpp"

नामाकाश ov::op::internal {
/// \note GatedDeltaNet op class is under development and subject to change
///
/// \brief Operator performing Gated Delta Net computation
/// \ingroup ov_ops_cpp_api
वर्ग OPENVINO_API GatedDeltaNet : सार्वजनिक ov::op::Op {
सार्वजनिक:
    OPENVINO_OP("GatedDeltaNet")

    GatedDeltaNet() = पूर्वनिर्धारित;
    /// \brief Constructs a GatedDeltaNet operation.
    ///
    /// \param query Query tensor input.
    /// \param key Key tensor input.
    /// \param value Value tensor input.
    /// \param recurrent_state Initial recurrent state tensor.
    /// \param gate Gate tensor controlling state decay/update.
    /// \param beta Beta tensor scaling the delta update.
    /// \param fuse_qk_l2norm Enables fusing q/k L2-normalization into this op.
    /// \param q_l2_norm_eps Epsilon used for query L2-normalization when fusion is enabled.
    /// \param k_l2_norm_eps Epsilon used for key L2-normalization when fusion is enabled.
    GatedDeltaNet(स्थिरांक Output<Node>& query,
                  स्थिरांक Output<Node>& किल्ली,
                  स्थिरांक Output<Node>& value,
                  स्थिरांक Output<Node>& recurrent_state,
                  स्थिरांक Output<Node>& gate,
                  स्थिरांक Output<Node>& beta,
                  स्थिरांक तार्किक fuse_qk_l2norm = असत्य,
                  स्थिरांक दशांश q_l2_norm_eps = 1e-6F,
                  स्थिरांक दशांश k_l2_norm_eps = 1e-6F);

    /// \brief Constructs a GatedDeltaNet operation from input vector.
    ///
    /// \param args Input tensor vector in order: query, key, value, recurrent_state, gate, beta.
    /// \param fuse_qk_l2norm Enables fusing q/k L2-normalization into this op.
    /// \param q_l2_norm_eps Epsilon used for query L2-normalization when fusion is enabled.
    /// \param k_l2_norm_eps Epsilon used for key L2-normalization when fusion is enabled.
    GatedDeltaNet(स्थिरांक ov::OutputVector& args,
                  स्थिरांक तार्किक fuse_qk_l2norm = असत्य,
                  स्थिरांक दशांश q_l2_norm_eps = 1e-6F,
                  स्थिरांक दशांश k_l2_norm_eps = 1e-6F);
    शून्य validate_and_infer_types() override;
    तार्किक visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(स्थिरांक ov::OutputVector& new_args) स्थिरांक override;
    तार्किक get_fuse_qk_l2norm() स्थिरांक {
        परत m_fuse_qk_l2norm;
    }
    दशांश get_q_l2_norm_eps() स्थिरांक {
        परत m_q_l2_norm_eps;
    }
    दशांश get_k_l2_norm_eps() स्थिरांक {
        परत m_k_l2_norm_eps;
    }

खाजगी:
    तार्किक m_fuse_qk_l2norm = असत्य;
    दशांश m_q_l2_norm_eps = 1e-6F;
    दशांश m_k_l2_norm_eps = 1e-6F;
};

}  // namespace ov::op::internal
