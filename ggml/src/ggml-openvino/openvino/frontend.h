// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/frontend/frontend.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class Front_ਸਿਰਾ {
ਜਨਤਕ:
    Front_ਸਿਰਾ();

    ਸਥਿਰ std::shared_ptr<Model> convert(ਅਚਲ ਨਿਵੇਸ਼_Model::Ptr & model, bool naive = false);
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
