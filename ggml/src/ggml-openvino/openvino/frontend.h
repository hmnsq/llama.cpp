// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#निर्देशः once

#अन्तर्भाव <openvino/frontend/frontend.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class Front_अन्तः {
सार्वजनिक:
    Front_अन्तः();

    स्थिर std::shared_ptr<Model> convert(ध्रुव निवेशः_Model::Ptr & model, bool naive = false);
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
