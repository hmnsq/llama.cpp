#अन्तर्भाव "ggml-vulkan-common.h"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <mutex>
#अन्तर्भाव <shared_mutex>

स्थिर शून्य * portable_aligned_malloc(size_t align, size_t size) {
#यदि_परिभाषितम् _WIN32
    प्रत्यागम _aligned_malloc(size, align);
#अन्यथा
    शून्य * ptr = शून्यसूचकः;
    यदि (posix_memalign(&ptr, align, size) != 0) {
        प्रत्यागम शून्यसूचकः;
    }
    प्रत्यागम ptr;
#यदि_अन्तः
}

स्थिर शून्य portable_aligned_free(शून्य * ptr) {
#यदि_परिभाषितम् _WIN32
    _aligned_free(ptr);
#अन्यथा
    std::मुक्त(ptr);
#यदि_अन्तः
}

ggml_backend_बफर_type_i ggml_backend_vk_बफर_type_interface = {
    /* .get_name            = */ ggml_backend_vk_बफर_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_बफर_type_alloc_बफर,
    /* .alloc_buffer_n      = */ रिक्त,
    /* .get_alignment       = */ ggml_backend_vk_बफर_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_बफर_type_get_max_आकार,
    /* .get_alloc_size      = */ ggml_backend_vk_बफर_type_get_alloc_आकार,
    /* .get_alloc_size_n    = */ रिक्त,
    /* .is_host             = */ रिक्त,
};

स्थिर std::vector<uint32_t> ggml_vk_find_स्मृतिः_properties(ध्रुव vk::Physical_Device_स्मृतिः_Properties* mem_props, vk::स्मृतिः_Requirements* mem_req, vk::स्मृतिः_Property_Flags flags) {
    std::vector<uint32_t> indices;

    प्रति (uint32_t i = 0; i < mem_props->स्मृतिः_Type_गणनम्; ++i) {
        vk::स्मृतिः_Type स्मृतिः_type = mem_props->स्मृतिः_Types[i];
        यदि ((mem_req->स्मृतिः_Type_द्वयङ्काः & ((uint64_t)1 << i)) &&
            (flags & स्मृतिः_type.propertyFlags) == flags &&
            mem_props->स्मृतिः_Heaps[स्मृतिः_type.heap_सूचकांक].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    प्रत्यागम indices;
}

स्थिर vk_बफर ggml_vk_create_बफर(vk_device& device, size_t size, ध्रुव std::initializer_list<vk::स्मृतिः_Property_Flags> & req_flags_list,
                                       शून्य *import_ptr = शून्यसूचकः) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    यदि (size > device->max_बफर_आकार) {
        क्षिप vk::बहिर्गमनम्_सम्बन्धि_Device_स्मृतिः_Error("Requested buffer size exceeds device buffer size limit");
    }

    vk_बफर buf = std::make_shared<vk_बफर_struct>();

    यदि (size == 0) {
        buf->size = 0;
        प्रत्यागम buf;
    }

    vk::बफर_Usage_Flags usage_flags = vk::बफर_Usage_Flag_द्वयङ्काः::e_Storage_बफर | vk::बफर_Usage_Flag_द्वयङ्काः::eTransferSrc | vk::बफर_Usage_Flag_द्वयङ्काः::eTransferDst;
    vk::स्मृतिः_Allocate_Flags mem_flags {};
    यदि (device->बफर_device_स्थानाङ्कः) {
        usage_flags |= vk::बफर_Usage_Flag_द्वयङ्काः::e_Shader_Device_स्थानाङ्कः;
        mem_flags |= vk::स्मृतिः_Allocate_Flag_द्वयङ्काः::e_Device_स्थानाङ्कः;
    }

    vk::बफर_Create_Info बफर_create_info{
        vk::बफर_Create_Flags(),
        size,
        usage_flags,
        vk::Sharing_बहुलकम्::eExclusive,
        0,
        शून्यसूचकः,
    };

    vk::External_स्मृतिः_बफर_Create_Info external_स्मृतिः_bci;
    यदि (import_ptr) {
        external_स्मृतिः_bci.handleTypes = vk::External_स्मृतिः_Handle_Type_Flag_द्वयङ्काः::e_Host_आवण्टनम्_EXT;
        बफर_create_info.समुच्चयः_PNext(&external_स्मृतिः_bci);
    }

    buf->बफर = device->device.create_बफर(बफर_create_info);

    vk::स्मृतिः_Requirements mem_req = device->device.get_बफर_स्मृतिः_Requirements(buf->बफर);

    vk::Physical_Device_स्मृतिः_Properties mem_props = device->physical_device.get_स्मृतिः_Properties();

    ध्रुव vk::स्मृतिः_प्राथमिकता_Allocate_Info_EXT mem_प्राथमिकता_info { 1.0f };

    vk::स्मृतिः_Allocate_Flags_Info mem_flags_info { mem_flags };

    यदि (device->स्मृतिः_प्राथमिकता) {
        mem_flags_info.समुच्चयः_PNext(&mem_प्राथमिकता_info);
    }

    यदि (import_ptr) {
        vk::स्मृतिः_Host_Pointer_Properties_EXT host_pointer_props;
        यत्न {
            host_pointer_props = device->device.get_स्मृतिः_Host_Pointer_Properties_EXT(vk::External_स्मृतिः_Handle_Type_Flag_द्वयङ्काः::e_Host_आवण्टनम्_EXT, import_ptr);
        } गृहाण (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroy_बफर(buf->बफर);
            प्रत्यागम {};
        }
        vk::Physical_Device_स्मृतिः_Properties mem_props = device->physical_device.get_स्मृतिः_Properties();

        uint32_t स्मृतिः_type_idx;
        vk::स्मृतिः_Property_Flags property_flags = *req_flags_list.begin();
        प्रति (स्मृतिः_type_idx = 0; स्मृतिः_type_idx < 32; ++स्मृतिः_type_idx) {
            यदि (!(host_pointer_props.स्मृतिः_Type_द्वयङ्काः & (1u << स्मृतिः_type_idx))) {
                अनुवर्त;
            }
            यदि (!(mem_req.स्मृतिः_Type_द्वयङ्काः & (1u << स्मृतिः_type_idx))) {
                अनुवर्त;
            }

            vk::स्मृतिः_Type स्मृतिः_type = mem_props.स्मृतिः_Types[स्मृतिः_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            यदि ((स्मृतिः_type.propertyFlags & property_flags) == property_flags) {
                property_flags = स्मृतिः_type.propertyFlags;
                भङ्ग;
            }
        }
        यदि (स्मृतिः_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroy_बफर(buf->बफर);
            प्रत्यागम {};
        }

        buf->स्मृतिः_property_flags = mem_props.स्मृतिः_Types[स्मृतिः_type_idx].propertyFlags;
        यत्न {
            vk::Import_स्मृतिः_Host_Pointer_Info_EXT import_info;
            import_info.handleType = vk::External_स्मृतिः_Handle_Type_Flag_द्वयङ्काः::e_Host_आवण्टनम्_EXT;
            import_info.pHostPointer = import_ptr;
            import_info.समुच्चयः_PNext(&mem_flags_info);
            buf->device_स्मृतिः = device->device.allocate_स्मृतिः({ size, स्मृतिः_type_idx, &import_info });
        } गृहाण (ध्रुव vk::SystemError& e) {
        }
    } अन्यथा {
        प्रति (स्वचालित it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            ध्रुव स्वचालित & req_flags = *it;

            ध्रुव std::vector<uint32_t> स्मृतिः_type_indices = ggml_vk_find_स्मृतिः_properties(&mem_props, &mem_req, req_flags);

            यदि (स्मृतिः_type_indices.रिक्तानि()) {
                अनुवर्त;
            }

            तार्किक समाप्तम् = असत्य;

            प्रति (स्वचालित mtype_it = स्मृतिः_type_indices.begin(); mtype_it != स्मृतिः_type_indices.end(); mtype_it++) {
                यत्न {
                    buf->device_स्मृतिः = device->device.allocate_स्मृतिः({ mem_req.size, *mtype_it, &mem_flags_info });
                    buf->स्मृतिः_property_flags = mem_props.स्मृतिः_Types[*mtype_it].propertyFlags;
                    समाप्तम् = सत्य;
                    भङ्ग;
                } गृहाण (ध्रुव vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    यदि (it + 1 == req_flags_list.end() && mtype_it + 1 == स्मृतिः_type_indices.end()) {
                        device->device.destroy_बफर(buf->बफर);
                        क्षिप e;
                    }
                }
            }

            यदि (समाप्तम्) {
                भङ्ग;
            }
        }
    }

    यदि (!buf->device_स्मृतिः) {
        device->device.destroy_बफर(buf->बफर);
        क्षिप vk::बहिर्गमनम्_सम्बन्धि_Device_स्मृतिः_Error("No suitable memory type found");
    }

    buf->ptr = शून्यसूचकः;

    यदि (import_ptr) {
        buf->ptr = import_ptr;
    } अन्यथा {
        यदि (buf->स्मृतिः_property_flags & vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible) {
            buf->ptr = device->device.map_स्मृतिः(buf->device_स्मृतिः, 0, VK_सम्पूर्णम्_आकार);
        }
    }

    device->device.bind_बफर_स्मृतिः(buf->बफर, buf->device_स्मृतिः, 0);

    buf->device = device;
    buf->size = size;

    यदि (device->बफर_device_स्थानाङ्कः) {
        ध्रुव vk::बफर_Device_स्थानाङ्कः_Info स्थानाङ्कः_Info(buf->बफर);
        buf->bda_addr = device->device.get_बफर_स्थानाङ्कः(स्थानाङ्कः_Info);
    }

    device->स्मृतिः_logger->log_आवण्टनम्(buf, size);

    प्रत्यागम buf;
}

vk_बफर ggml_vk_create_बफर_check(vk_device& device, size_t size, vk::स्मृतिः_Property_Flags req_flags, vk::स्मृतिः_Property_Flags fallback_flags) {
    यत्न {
        प्रत्यागम ggml_vk_create_बफर(device, size, {req_flags, fallback_flags});
    } गृहाण (ध्रुव vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::पङ्क्त्यन्तः;
        std::cerr << "ggml_vulkan: " << e.what() << std::पङ्क्त्यन्तः;
        क्षिप e;
    }
}

vk_बफर ggml_vk_create_बफर_device(vk_device& device, size_t size) {
    vk_बफर buf;
    यत्न {
        यदि (device->prefer_host_स्मृतिः) {
            buf = ggml_vk_create_बफर(device, size, {vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCoherent,
                                                       vk::स्मृतिः_Property_Flag_द्वयङ्काः::eDeviceLocal});
        } अन्यथा यदि (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_बफर(device, size, {vk::स्मृतिः_Property_Flag_द्वयङ्काः::eDeviceLocal | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCoherent,
                                                       vk::स्मृतिः_Property_Flag_द्वयङ्काः::eDeviceLocal,
                                                       vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCoherent});
        } अन्यथा यदि (device->disable_host_visible_vidmem) {
            यदि (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_बफर(device, size, {vk::स्मृतिः_Property_Flag_द्वयङ्काः::eDeviceLocal,
                                                           vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCoherent});
            } अन्यथा {
                buf = ggml_vk_create_बफर(device, size, {vk::स्मृतिः_Property_Flag_द्वयङ्काः::eDeviceLocal});
            }
        } अन्यथा {
            // use rebar if available, otherwise fallback to device only visible memory
            यदि (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_बफर(device, size, {vk::स्मृतिः_Property_Flag_द्वयङ्काः::eDeviceLocal | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCoherent,
                                                           vk::स्मृतिः_Property_Flag_द्वयङ्काः::eDeviceLocal,
                                                           vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCoherent});
            } अन्यथा {
                buf = ggml_vk_create_बफर(device, size, {vk::स्मृतिः_Property_Flag_द्वयङ्काः::eDeviceLocal | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCoherent,
                                                           vk::स्मृतिः_Property_Flag_द्वयङ्काः::eDeviceLocal});
            }
        }
    } गृहाण (ध्रुव vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::पङ्क्त्यन्तः;
        std::cerr << "ggml_vulkan: " << e.what() << std::पङ्क्त्यन्तः;
        क्षिप e;
    }

    प्रत्यागम buf;
}

शून्य ggml_vk_destroy_बफर(vk_बफर& buf) {
    यदि (buf == शून्यसूचकः) {
        प्रत्यागम;
    }

    यदि (buf->device != शून्यसूचकः) {
        buf->device->स्मृतिः_logger->log_deallocation(buf);
    }

    buf.reset();
}

// caller must hold host_allocations_mutex
स्थिर size_t ggml_vk_host_आवण्टनम्_find(ध्रुव शून्य * ptr) {
    प्रति (size_t i = 0; i < vk_instance.host_allocations.size(); i++) {
        ध्रुव uint8_t * आधारः = (ध्रुव uint8_t *) vk_instance.host_allocations[i].आधारः;
        यदि (ptr >= आधारः && ptr < आधारः + vk_instance.host_allocations[i].size) {
            प्रत्यागम i;
        }
    }
    प्रत्यागम SIZE_MAX;
}

शून्य * ggml_vk_host_malloc(size_t size) {
    VK_LOG_स्मृतिः("ggml_vk_host_malloc(" << size << ")");

    ggml_vk_instance_init();

    यदि (vk_instance.multi_device) {
        ध्रुव size_t align      = std::max<size_t>(vk_instance.host_import_alignment, TENSOR_ALIGNMENT);
        ध्रुव size_t alloc_आकार = (size + align - 1) / align * align;

        शून्य * ptr = portable_aligned_malloc(align, alloc_आकार);
        यदि (ptr == शून्यसूचकः) {
            प्रत्यागम शून्यसूचकः;
        }
        vk_host_आवण्टनम् प्रविष्टिः;
        प्रविष्टिः.आधारः = ptr;
        प्रविष्टिः.size = alloc_आकार;

        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        vk_instance.host_allocations.push_back(std::move(प्रविष्टिः));
        प्रत्यागम ptr;
    }

    vk_device dev0 = ggml_vk_get_device(0);
    vk_बफर buf;
    यत्न {
        buf = ggml_vk_create_बफर(dev0, size,
            {vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCoherent | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCached,
             vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCoherent});
    } गृहाण (vk::SystemError&) {
        प्रत्यागम शून्यसूचकः;
    }

    यदि (!(buf->स्मृतिः_property_flags & vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n", size / 1024.0 / 1024.0);
        ggml_vk_destroy_बफर(buf);
        प्रत्यागम शून्यसूचकः;
    }

    vk_host_आवण्टनम् प्रविष्टिः;
    प्रविष्टिः.आधारः             = buf->ptr;
    प्रविष्टिः.size             = size;
    प्रविष्टिः.views[dev0->idx] = buf;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
    vk_instance.host_allocations.push_back(std::move(प्रविष्टिः));
    प्रत्यागम buf->ptr;
}

शून्य ggml_vk_host_free(शून्य * ptr) {
    यदि (ptr == शून्यसूचकः) {
        प्रत्यागम;
    }
    VK_LOG_स्मृतिः("ggml_vk_host_free(" << ptr << ")");

    vk_host_आवण्टनम् प्रविष्टिः;
    {
        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        ध्रुव size_t सूचकांक = ggml_vk_host_आवण्टनम्_find(ptr);
        यदि (सूचकांक == SIZE_MAX) {
            fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
            प्रत्यागम;
        }
        प्रविष्टिः = std::move(vk_instance.host_allocations[सूचकांक]);
        vk_instance.host_allocations.erase(vk_instance.host_allocations.begin() + सूचकांक);
    }

    प्रति (स्वचालित & view : प्रविष्टिः.views) {
        ggml_vk_destroy_बफर(view);
    }
    यदि (vk_instance.multi_device) {
        portable_aligned_free(प्रविष्टिः.आधारः);
    }
}

शून्य ggml_vk_host_get(ध्रुव vk_device& device, ध्रुव शून्य * ptr, vk_बफर& buf, size_t& buf_विस्थापनम्) {
    buf = शून्यसूचकः;
    buf_विस्थापनम् = 0;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);

    ध्रुव size_t सूचकांक = ggml_vk_host_आवण्टनम्_find(ptr);
    यदि (सूचकांक == SIZE_MAX) {
        प्रत्यागम;
    }
    vk_host_आवण्टनम् & प्रविष्टिः = vk_instance.host_allocations[सूचकांक];

    यदि (प्रविष्टिः.views[device->idx] == शून्यसूचकः) {
        ध्रुव uint32_t bit = 1u << device->idx;
        यदि (!vk_instance.multi_device || !vk_instance.device_host_बफर_import[device->idx] || (vk_instance.host_import_failed & bit)) {
            प्रत्यागम;
        }
        GGML_ASSERT(device->external_स्मृतिः_host);
        // const only propagates from the const context; the pointee is mutable
        प्रविष्टिः.views[device->idx] = ggml_vk_बफर_from_host_ptr(const_cast<vk_device&>(device), प्रविष्टिः.आधारः, प्रविष्टिः.size);
        यदि (प्रविष्टिः.views[device->idx] == शून्यसूचकः) {
            vk_instance.host_import_failed |= bit;
            GGML_LOG_WARN("ggml_vulkan: %s: host memory import failed, using staging\n", device->name.c_str());
            प्रत्यागम;
        }
    }

    buf = प्रविष्टिः.views[device->idx];
    buf_विस्थापनम् = (ध्रुव uint8_t *) ptr - (ध्रुव uint8_t *) प्रविष्टिः.आधारः;
}

शून्य ggml_vk_ensure_sync_staging_बफर(vk_device& device, size_t size) {
    यदि (device->sync_staging == शून्यसूचकः || device->sync_staging->size < size) {
        VK_LOG_स्मृतिः("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_बफर(device->sync_staging);
        device->sync_staging = ggml_vk_create_बफर_check(device, size,
            vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCoherent | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCached,
            vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCoherent);
    }
}

शून्य ggml_vk_ensure_sync_staging_बफर(ggml_backend_vk_context * ctx, size_t size) {
    यदि (ctx->sync_staging == शून्यसूचकः || ctx->sync_staging->size < size) {
        VK_LOG_स्मृतिः("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_बफर(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_बफर_check(ctx->device, size,
            vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCoherent | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCached,
            vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCoherent);
    }
}

स्थिर शून्य ggml_vk_बफर_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_बफर& dst, size_t विस्थापनम्, ध्रुव ggml_tensor * tensor, तार्किक sync_staging = असत्य) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    यदि(dst->स्मृतिः_property_flags & vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::पङ्क्त्यन्तः;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_बफर buf = शून्यसूचकः;
    size_t buf_विस्थापनम् = 0;
    ggml_vk_host_get(ctx->device, tensor->दत्त, buf, buf_विस्थापनम्);

    ध्रुव uint64_t ne0 = tensor->ne[0];
    ध्रुव uint64_t ne1 = tensor->ne[1];
    ध्रुव uint64_t ne2 = tensor->ne[2];
    ध्रुव uint64_t ne3 = tensor->ne[3];
    ध्रुव uint64_t nb0 = tensor->nb[0];
    ध्रुव uint64_t nb1 = tensor->nb[1];
    ध्रुव uint64_t nb2 = tensor->nb[2];
    ध्रुव uint64_t nb3 = tensor->nb[3];
    ध्रुव ggml_type type = tensor->type;
    ध्रुव uint64_t ts = ggml_type_आकार(type);
    ध्रुव uint64_t bs = ggml_blck_आकार(type);

    ध्रुव uint64_t dstnb0 = ts;
    ध्रुव uint64_t dstnb1 = dstnb0*(ne0/bs);
    ध्रुव uint64_t dstnb2 = dstnb1*ne1;
    ध्रुव uint64_t dstnb3 = dstnb2*ne2;

    ध्रुव uint64_t ne = ggml_nelements(tensor);

    यदि (buf != शून्यसूचकः) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::बफर_Copy> slices;

        प्रति (uint64_t i3 = 0; i3 < ne3; i3++) {
            प्रति (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                यदि (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_विस्थापनम् + i3*nb3 + i2*nb2, विस्थापनम् + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } अन्यथा {
                    प्रति (uint64_t i1 = 0; i1 < ne1; i1++) {
                        यदि (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_विस्थापनम् + i3*nb3 + i2*nb2 + i1*nb1, विस्थापनम् + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } अन्यथा {
                            ध्रुव uint64_t s_off = buf_विस्थापनम् + i3*nb3 + i2*nb2 + i1*nb1;
                            ध्रुव uint64_t d_off = विस्थापनम् + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            प्रति (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->बफर->buf.copy_बफर(buf->बफर, dst->बफर, slices);
        प्रत्यागम;
    }

    यदि (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_बफर& staging = ctx->device->sync_staging;
    ध्रुव uint64_t copy_आकार = ts*ne/bs;
    ggml_vk_ensure_sync_staging_बफर(ctx->device, copy_आकार);
    Vk_बफर_Copy buf_copy{ 0, विस्थापनम्, copy_आकार };

    ggml_vk_sync_buffers(ctx, subctx);
    vk_Cmd_Copy_बफर(subctx->s->बफर->buf, (Vk_बफर)staging->बफर, (Vk_बफर)dst->बफर, 1, &buf_copy);

    प्रति (uint64_t i3 = 0; i3 < ne3; i3++) {
        प्रति (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            यदि (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (ध्रुव uint8_t *) tensor->दत्त + buf_विस्थापनम् + i3*nb3 + i2*nb2, dstnb2, &subctx->अन्तर्_memcpys);
            } अन्यथा {
                प्रति (uint64_t i1 = 0; i1 < ne1; i1++) {
                    यदि (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (ध्रुव uint8_t *) tensor->दत्त + buf_विस्थापनम् + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->अन्तर्_memcpys);
                    } अन्यथा {
                        ध्रुव uint64_t s_off = buf_विस्थापनम् + i3*nb3 + i2*nb2 + i1*nb1;
                        ध्रुव uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        प्रति (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (ध्रुव uint8_t *) tensor->दत्त + s_off + i0*nb0, dstnb0, &subctx->अन्तर्_memcpys);
                        }
                    }
                }
            }
        }
    }
}

तार्किक ggml_vk_बफर_write_2d_async(vk_context subctx, vk_बफर& dst, size_t विस्थापनम्, ध्रुव शून्य * src, size_t spitch, size_t dpitch, size_t विस्तारः, size_t height, तार्किक sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << विस्तारः << ", " << height << ")");
    // Check if src is pinned memory
    vk_बफर buf = शून्यसूचकः;
    size_t buf_विस्थापनम् = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_विस्थापनम्);

    यदि (buf != शून्यसूचकः) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::बफर_Copy> slices(1);
        यदि (विस्तारः == spitch && विस्तारः == dpitch) {
            // Only do single write if stride is equal
            slices[0].src_विस्थापनम्_2 = buf_विस्थापनम्;
            slices[0].dst_विस्थापनम्_2 = विस्थापनम्;
            slices[0].size = विस्तारः * height;
        } अन्यथा {
            slices.resize(height);
            प्रति (size_t i = 0; i < height; i++) {
                slices[i].src_विस्थापनम्_2 = buf_विस्थापनम् + i * spitch;
                slices[i].dst_विस्थापनम्_2 = विस्थापनम् + i * dpitch;
                slices[i].size = विस्तारः;
            }
        }

        ggml_vk_sync_buffers(शून्यसूचकः, subctx);
        subctx->s->बफर->buf.copy_बफर(buf->बफर, dst->बफर, slices);
        प्रत्यागम सत्य;
    }
    VK_LOG_DEBUG("STAGING");

    यदि (!sync_staging) {
        // copy was not handled caller needs to fall back
        प्रत्यागम असत्य;
    }

    // Staging buffer required
    ध्रुव size_t staging_आकार = विस्तारः * height;
    ggml_vk_ensure_sync_staging_बफर(dst->device, staging_आकार);

    vk_बफर& staging_बफर = dst->device->sync_staging;

    std::vector<vk::बफर_Copy> slices(1);
    यदि (विस्तारः == dpitch) {
        slices[0].src_विस्थापनम्_2 = 0;
        slices[0].dst_विस्थापनम्_2 = विस्थापनम्;
        slices[0].size = staging_आकार;
    } अन्यथा {
        slices.resize(height);
        प्रति (size_t i = 0; i < height; i++) {
            slices[i].src_विस्थापनम्_2 = i * विस्तारः;
            slices[i].dst_विस्थापनम्_2 = विस्थापनम् + i * dpitch;
            slices[i].size = विस्तारः;
        }
    }

    ggml_vk_sync_buffers(शून्यसूचकः, subctx);
    subctx->s->बफर->buf.copy_बफर(staging_बफर->बफर, dst->बफर, slices);

    यदि (विस्तारः == spitch) {
        deferred_memcpy((uint8_t *)staging_बफर->ptr, src, staging_आकार, &subctx->अन्तर्_memcpys);
    } अन्यथा {
        प्रति (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *)staging_बफर->ptr + i * विस्तारः, (ध्रुव uint8_t *) src + i * spitch, विस्तारः, &subctx->अन्तर्_memcpys);
        }
    }
    प्रत्यागम सत्य;
}

तार्किक ggml_vk_बफर_write_async(vk_context subctx, vk_बफर& dst, size_t विस्थापनम्, ध्रुव शून्य * src, size_t size, तार्किक sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    प्रत्यागम ggml_vk_बफर_write_2d_async(subctx, dst, विस्थापनम्, src, size, size, size, 1, sync_staging);
}

शून्य ggml_vk_बफर_write_2d(vk_बफर& dst, size_t विस्थापनम्, ध्रुव शून्य * src, size_t spitch, size_t dpitch, size_t विस्तारः, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << विस्तारः << ", " << height << ")");
    // Buffer is already mapped
    यदि(dst->स्मृतिः_property_flags & vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible) {
        GGML_ASSERT(dst->स्मृतिः_property_flags & vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCoherent);

        यदि (विस्तारः == spitch && विस्तारः == dpitch) {
            स्मृतिं_प्रतिलिख((uint8_t *)dst->ptr + विस्थापनम्, src, विस्तारः * height);
        } अन्यथा {
            प्रति (size_t i = 0; i < height; i++) {
                स्मृतिं_प्रतिलिख((uint8_t *)dst->ptr + विस्थापनम् + i * dpitch, (ध्रुव uint8_t *) src + i * spitch, विस्तारः);
            }
        }
    } अन्यथा {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        तार्किक ret = ggml_vk_बफर_write_2d_async(subctx, dst, विस्थापनम्, src, spitch, dpitch, विस्तारः, height, सत्य);
        GGML_ASSERT(ret);
        ggml_vk_ctx_अन्तः(subctx);

        प्रति (स्वचालित& cpy : subctx->अन्तर्_memcpys) {
            स्मृतिं_प्रतिलिख(cpy.dst, cpy.src, cpy.n);
        }

        प्रति (स्वचालित& mset : subctx->memsets) {
            स्मृतिं_पूरय(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, सत्य, UINT64_MAX), "vk_buffer_write_2d waitForFences", dst->device);
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_queue_command_pools_cleanup(dst->device);
    }
}

शून्य ggml_vk_बफर_write(vk_बफर& dst, size_t विस्थापनम्, ध्रुव शून्य * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_बफर_write_2d(dst, विस्थापनम्, src, size, size, size, 1);
}

तार्किक ggml_vk_बफर_read_2d_async(vk_context subctx, vk_बफर& src, size_t विस्थापनम्, शून्य * dst, size_t spitch, size_t dpitch, size_t विस्तारः, size_t height, तार्किक sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << विस्थापनम् << ", width=" << विस्तारः << ", height=" << height << ")");
    GGML_ASSERT(विस्तारः > 0);
    GGML_ASSERT(height > 0);
    GGML_ASSERT(src != शून्यसूचकः);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_बफर buf = शून्यसूचकः;
    size_t buf_विस्थापनम् = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_विस्थापनम्);

    std::vector<vk::बफर_Copy> slices(1);
    यदि (विस्तारः == spitch && विस्तारः == dpitch) {
        // Only do single write if stride is equal
        slices[0].src_विस्थापनम्_2 = विस्थापनम्;
        slices[0].dst_विस्थापनम्_2 = buf_विस्थापनम्;
        slices[0].size = विस्तारः * height;
    } अन्यथा {
        slices.resize(height);
        प्रति (size_t i = 0; i < height; i++) {
            slices[i].src_विस्थापनम्_2 = विस्थापनम् + i * spitch;
            slices[i].dst_विस्थापनम्_2 = buf_विस्थापनम् + i * dpitch;
            slices[i].size = विस्तारः;
        }
    }

    यदि (buf != शून्यसूचकः) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(शून्यसूचकः, subctx);
        subctx->s->बफर->buf.copy_बफर(src->बफर, buf->बफर, slices);

        प्रत्यागम सत्य;
    }
    VK_LOG_DEBUG("STAGING");

    यदि (!sync_staging) {
        // copy was not handled caller needs to fall back
        प्रत्यागम असत्य;
    }

    // Fall back to staging buffer
    ध्रुव size_t staging_आकार = विस्तारः * height;
    ggml_vk_ensure_sync_staging_बफर(src->device, staging_आकार);

    vk_बफर& staging_बफर = src->device->sync_staging;

    std::vector<vk::बफर_Copy> staging_slices(1);
    यदि (विस्तारः == spitch) {
        staging_slices[0].src_विस्थापनम्_2 = विस्थापनम्;
        staging_slices[0].dst_विस्थापनम्_2 = 0;
        staging_slices[0].size = staging_आकार;
    } अन्यथा {
        staging_slices.resize(height);
        प्रति (size_t i = 0; i < height; i++) {
            staging_slices[i].src_विस्थापनम्_2 = विस्थापनम् + i * spitch;
            staging_slices[i].dst_विस्थापनम्_2 = i * विस्तारः;
            staging_slices[i].size = विस्तारः;
        }
    }

    ggml_vk_sync_buffers(शून्यसूचकः, subctx);
    subctx->s->बफर->buf.copy_बफर(src->बफर, staging_बफर->बफर, staging_slices);

    यदि (विस्तारः == dpitch) {
        deferred_memcpy(dst, staging_बफर->ptr, staging_आकार, &subctx->बहिर्गमनम्_memcpys);
    } अन्यथा {
        प्रति (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (ध्रुव uint8_t *) staging_बफर->ptr + i * विस्तारः, विस्तारः, &subctx->बहिर्गमनम्_memcpys);
        }
    }
    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_vk_बफर_read_async(vk_context subctx, vk_बफर& src, size_t विस्थापनम्, शून्य * dst, size_t size, तार्किक sync_staging = असत्य) {
    प्रत्यागम ggml_vk_बफर_read_2d_async(subctx, src, विस्थापनम्, dst, size, size, size, 1, sync_staging);
}

शून्य ggml_vk_बफर_read_2d(vk_बफर& src, size_t विस्थापनम्, शून्य * dst, size_t spitch, size_t dpitch, size_t विस्तारः, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->बफर << ", " << विस्थापनम् << ", " << विस्तारः << ", " << height << ")");

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
    ध्रुव तार्किक slow_host_read = src->device->vendor_id == VK_VENDOR_ID_AMD &&
                                !(src->स्मृतिः_property_flags & vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCached);
    यदि(src->स्मृतिः_property_flags & vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible && src->device->uma && !slow_host_read) {
        GGML_ASSERT(src->स्मृतिः_property_flags & vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCoherent);

        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        vk_context subctx = ggml_vk_create_temporary_context(src->device->compute_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        subctx->s->बफर->buf.pipelineBarrier(
            vk::Pipeline_Stage_Flag_द्वयङ्काः::eComputeShader | vk::Pipeline_Stage_Flag_द्वयङ्काः::eTransfer,
            vk::Pipeline_Stage_Flag_द्वयङ्काः::eHost,
            {},
            { { vk::Access_Flag_द्वयङ्काः::eShaderWrite | vk::Access_Flag_द्वयङ्काः::eTransferWrite,
                vk::Access_Flag_द्वयङ्काः::eHostRead } },
            {}, {});
        ggml_vk_ctx_अन्तः(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, सत्य, UINT64_MAX),
                 "vk_buffer_read_2d uma waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        यदि (विस्तारः == spitch && विस्तारः == dpitch) {
            स्मृतिं_प्रतिलिख(dst, (ध्रुव uint8_t *) src->ptr + विस्थापनम्, विस्तारः * height);
        } अन्यथा {
            प्रति (size_t i = 0; i < height; i++) {
                स्मृतिं_प्रतिलिख((uint8_t *) dst + i * dpitch, (ध्रुव uint8_t *) src->ptr + विस्थापनम् + i * spitch, विस्तारः);
            }
        }
    } अन्यथा {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        तार्किक ret = ggml_vk_बफर_read_2d_async(subctx, src, विस्थापनम्, dst, spitch, dpitch, विस्तारः, height, सत्य);
        GGML_ASSERT(ret);
        ggml_vk_ctx_अन्तः(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, सत्य, UINT64_MAX), "vk_buffer_read_2d waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        प्रति (स्वचालित& cpy : subctx->बहिर्गमनम्_memcpys) {
            स्मृतिं_प्रतिलिख(cpy.dst, cpy.src, cpy.n);
        }
    }
}

शून्य ggml_vk_बफर_read(vk_बफर& src, size_t विस्थापनम्, शून्य * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->बफर << ", " << विस्थापनम् << ", " << size << ")");
    ggml_vk_बफर_read_2d(src, विस्थापनम्, dst, size, size, size, 1);
}

शून्य ggml_vk_बफर_copy_async(vk_context& ctx, vk_बफर& dst, size_t dst_विस्थापनम्, vk_बफर& src, size_t src_विस्थापनम्, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    Vk_बफर_Copy bc{ src_विस्थापनम्, dst_विस्थापनम्, size };

    vk_Cmd_Copy_बफर(ctx->s->बफर->buf, (Vk_बफर)src->बफर, (Vk_बफर)dst->बफर, 1, &bc);
}

शून्य ggml_vk_बफर_copy(vk_बफर& dst, size_t dst_विस्थापनम्, vk_बफर& src, size_t src_विस्थापनम्, size_t size) {
    यदि (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_बफर_copy_async(subctx, dst, dst_विस्थापनम्, src, src_विस्थापनम्, size);
        ggml_vk_ctx_अन्तः(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, सत्य, UINT64_MAX), "vk_buffer_copy waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);
    } अन्यथा {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_बफर(src->device, size);

        // Copy to src staging buffer
        ggml_vk_बफर_copy(src->device->sync_staging, 0, src, src_विस्थापनम्, size);
        // Copy to dst buffer
        ggml_vk_बफर_write(dst, dst_विस्थापनम्, src->device->sync_staging->ptr, size);
    }
}

शून्य ggml_vk_बफर_memset_async(vk_context& ctx, vk_बफर& dst, size_t विस्थापनम्, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << विस्थापनम् << ", " << c << ", " << size << ")");

    यदि (dst->स्मृतिः_property_flags & vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + विस्थापनम्, c, size, &ctx->memsets);
        प्रत्यागम;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->बफर->buf.fill_बफर(dst->बफर, विस्थापनम्, size, c);
}

शून्य ggml_vk_बफर_memset(vk_बफर& dst, size_t विस्थापनम्, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << विस्थापनम् << ", " << c << ", " << size << ")");

    यदि (dst->स्मृतिः_property_flags & vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible &&
        dst->device->uma) {
        स्मृतिं_पूरय((uint8_t*)dst->ptr + विस्थापनम्, c, size);
        प्रत्यागम;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->बफर->buf.fill_बफर(dst->बफर, विस्थापनम्, size, c);
    ggml_vk_ctx_अन्तः(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, सत्य, UINT64_MAX), "vk_memset waitForFences", dst->device);
    dst->device->device.resetFences({ dst->device->fence });
    ggml_vk_queue_command_pools_cleanup(dst->device);
}

ggml_backend_बफर_i ggml_backend_vk_बफर_interface = {
    /* .free_buffer     = */ ggml_backend_vk_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_vk_बफर_get_आधारः,
    /* .init_tensor     = */ ggml_backend_vk_बफर_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_बफर_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_बफर_समुच्चयः_tensor,
    /* .get_tensor      = */ ggml_backend_vk_बफर_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_बफर_समुच्चयः_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_बफर_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_बफर_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_बफर_clear,
    /* .reset           = */ रिक्त,
};

vk_बफर ggml_vk_बफर_from_host_ptr(vk_device & device, शून्य * ptr, size_t size) {
    यदि (!device->external_स्मृतिः_host) {
        प्रत्यागम {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    यदि (uptr & (device->min_imported_host_pointer_alignment - 1)) {
        प्रत्यागम {};
    }
    यदि (size & (device->min_imported_host_pointer_alignment - 1)) {
        प्रत्यागम {};
    }

    ध्रुव vk::स्मृतिः_Property_Flags property_flags = vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostVisible | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCoherent | vk::स्मृतिः_Property_Flag_द्वयङ्काः::eHostCached;

    vk_बफर buf {};
    यत्न {
        buf = ggml_vk_create_बफर(device, size, { property_flags }, ptr);
    } गृहाण (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    प्रत्यागम buf;
}

