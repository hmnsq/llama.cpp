#समावेश "ggml-vulkan-common.h"

#समावेश <cstdint>
#समावेश <cstdlib>
#समावेश <mutex>
#समावेश <shared_mutex>

अटल व्योम * portable_aligned_malloc(माप_प्रकार align, माप_प्रकार size) {
#अगर_घोषित _WIN32
    वापस _aligned_malloc(size, align);
#अन्यथा
    व्योम * ptr = nullptr;
    अगर (posix_memalign(&ptr, align, size) != 0) {
        वापस nullptr;
    }
    वापस ptr;
#पूर्ण_अगर
}

अटल व्योम portable_aligned_free(व्योम * ptr) {
#अगर_घोषित _WIN32
    _aligned_free(ptr);
#अन्यथा
    std::मुक्त(ptr);
#पूर्ण_अगर
}

ggml_backend_बफर_type_i ggml_backend_vk_बफर_type_interface = {
    /* .get_name            = */ ggml_backend_vk_बफर_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_बफर_type_alloc_बफर,
    /* .alloc_buffer_n      = */ शून्य,
    /* .get_alignment       = */ ggml_backend_vk_बफर_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_बफर_type_get_max_आकार,
    /* .get_alloc_size      = */ ggml_backend_vk_बफर_type_get_alloc_आकार,
    /* .get_alloc_size_n    = */ शून्य,
    /* .is_host             = */ शून्य,
};

अटल std::vector<uint32_t> ggml_vk_find_memory_properties(स्थिर vk::PhysicalDeviceMemoryProperties* mem_props, vk::MemoryRequirements* mem_req, vk::MemoryPropertyFlags flags) {
    std::vector<uint32_t> indices;

    क्रम (uint32_t i = 0; i < mem_props->memory_Type_गिनती; ++i) {
        vk::MemoryType memory_type = mem_props->memoryTypes[i];
        अगर ((mem_req->memoryTypeBits & ((uint64_t)1 << i)) &&
            (flags & memory_type.propertyFlags) == flags &&
            mem_props->memoryHeaps[memory_type.heap_सूचकांक].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    वापस indices;
}

अटल vk_बफर ggml_vk_create_बफर(vk_device& device, माप_प्रकार size, स्थिर std::initializer_list<vk::MemoryPropertyFlags> & req_flags_list,
                                       व्योम *import_ptr = nullptr) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    अगर (size > device->max_बफर_आकार) {
        फेंको vk::OutOfDeviceMemoryError("Requested buffer size exceeds device buffer size limit");
    }

    vk_बफर buf = std::make_shared<vk_बफर_struct>();

    अगर (size == 0) {
        buf->size = 0;
        वापस buf;
    }

    vk::बफर_Usage_Flags usage_flags = vk::बफर_Usage_Flag_Bits::e_Storage_बफर | vk::बफर_Usage_Flag_Bits::eTransferSrc | vk::बफर_Usage_Flag_Bits::eTransferDst;
    vk::MemoryAllocateFlags mem_flags {};
    अगर (device->बफर_device_address) {
        usage_flags |= vk::बफर_Usage_Flag_Bits::eShaderDeviceAddress;
        mem_flags |= vk::MemoryAllocateFlagBits::eDeviceAddress;
    }

    vk::बफर_Create_Info बफर_create_info{
        vk::बफर_Create_Flags(),
        size,
        usage_flags,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
    };

    vk::External_Memory_बफर_Create_Info external_memory_bci;
    अगर (import_ptr) {
        external_memory_bci.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
        बफर_create_info.setPNext(&external_memory_bci);
    }

    buf->बफर = device->device.create_बफर(बफर_create_info);

    vk::MemoryRequirements mem_req = device->device.get_बफर_Memory_Requirements(buf->बफर);

    vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

    स्थिर vk::MemoryPriorityAllocateInfoEXT mem_priority_info { 1.0f };

    vk::MemoryAllocateFlagsInfo mem_flags_info { mem_flags };

    अगर (device->memory_priority) {
        mem_flags_info.setPNext(&mem_priority_info);
    }

    अगर (import_ptr) {
        vk::MemoryHostPointerPropertiesEXT host_pointer_props;
        प्रयास {
            host_pointer_props = device->device.getMemoryHostPointerPropertiesEXT(vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, import_ptr);
        } पकड़ो (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroy_बफर(buf->बफर);
            वापस {};
        }
        vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

        uint32_t memory_type_idx;
        vk::MemoryPropertyFlags property_flags = *req_flags_list.begin();
        क्रम (memory_type_idx = 0; memory_type_idx < 32; ++memory_type_idx) {
            अगर (!(host_pointer_props.memoryTypeBits & (1u << memory_type_idx))) {
                जारी;
            }
            अगर (!(mem_req.memoryTypeBits & (1u << memory_type_idx))) {
                जारी;
            }

            vk::MemoryType memory_type = mem_props.memoryTypes[memory_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            अगर ((memory_type.propertyFlags & property_flags) == property_flags) {
                property_flags = memory_type.propertyFlags;
                अवरोध;
            }
        }
        अगर (memory_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroy_बफर(buf->बफर);
            वापस {};
        }

        buf->memory_property_flags = mem_props.memoryTypes[memory_type_idx].propertyFlags;
        प्रयास {
            vk::ImportMemoryHostPointerInfoEXT import_info;
            import_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
            import_info.pHostPointer = import_ptr;
            import_info.setPNext(&mem_flags_info);
            buf->device_memory = device->device.allocateMemory({ size, memory_type_idx, &import_info });
        } पकड़ो (स्थिर vk::SystemError& e) {
        }
    } अन्यथा {
        क्रम (स्वतः it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            स्थिर स्वतः & req_flags = *it;

            स्थिर std::vector<uint32_t> memory_type_indices = ggml_vk_find_memory_properties(&mem_props, &mem_req, req_flags);

            अगर (memory_type_indices.empty()) {
                जारी;
            }

            bool done = false;

            क्रम (स्वतः mtype_it = memory_type_indices.begin(); mtype_it != memory_type_indices.end(); mtype_it++) {
                प्रयास {
                    buf->device_memory = device->device.allocateMemory({ mem_req.size, *mtype_it, &mem_flags_info });
                    buf->memory_property_flags = mem_props.memoryTypes[*mtype_it].propertyFlags;
                    done = true;
                    अवरोध;
                } पकड़ो (स्थिर vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    अगर (it + 1 == req_flags_list.end() && mtype_it + 1 == memory_type_indices.end()) {
                        device->device.destroy_बफर(buf->बफर);
                        फेंको e;
                    }
                }
            }

            अगर (done) {
                अवरोध;
            }
        }
    }

    अगर (!buf->device_memory) {
        device->device.destroy_बफर(buf->बफर);
        फेंको vk::OutOfDeviceMemoryError("No suitable memory type found");
    }

    buf->ptr = nullptr;

    अगर (import_ptr) {
        buf->ptr = import_ptr;
    } अन्यथा {
        अगर (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
            buf->ptr = device->device.mapMemory(buf->device_memory, 0, VK_WHOLE_आकार);
        }
    }

    device->device.bind_बफर_Memory(buf->बफर, buf->device_memory, 0);

    buf->device = device;
    buf->size = size;

    अगर (device->बफर_device_address) {
        स्थिर vk::बफर_Device_Address_Info addressInfo(buf->बफर);
        buf->bda_addr = device->device.get_बफर_Address(addressInfo);
    }

    device->memory_logger->log_allocation(buf, size);

    वापस buf;
}

vk_बफर ggml_vk_create_बफर_check(vk_device& device, माप_प्रकार size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags) {
    प्रयास {
        वापस ggml_vk_create_बफर(device, size, {req_flags, fallback_flags});
    } पकड़ो (स्थिर vk::SystemError& e) {
        std::अ_त्रुटि << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::प_पूर्ण;
        std::अ_त्रुटि << "ggml_vulkan: " << e.what() << std::प_पूर्ण;
        फेंको e;
    }
}

vk_बफर ggml_vk_create_बफर_device(vk_device& device, माप_प्रकार size) {
    vk_बफर buf;
    प्रयास {
        अगर (device->prefer_host_memory) {
            buf = ggml_vk_create_बफर(device, size, {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal});
        } अन्यथा अगर (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_बफर(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
        } अन्यथा अगर (device->disable_host_visible_vidmem) {
            अगर (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_बफर(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } अन्यथा {
                buf = ggml_vk_create_बफर(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        } अन्यथा {
            // use rebar if available, otherwise fallback to device only visible memory
            अगर (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_बफर(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } अन्यथा {
                buf = ggml_vk_create_बफर(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        }
    } पकड़ो (स्थिर vk::SystemError& e) {
        std::अ_त्रुटि << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::प_पूर्ण;
        std::अ_त्रुटि << "ggml_vulkan: " << e.what() << std::प_पूर्ण;
        फेंको e;
    }

    वापस buf;
}

व्योम ggml_vk_destroy_बफर(vk_बफर& buf) {
    अगर (buf == nullptr) {
        वापस;
    }

    अगर (buf->device != nullptr) {
        buf->device->memory_logger->log_deallocation(buf);
    }

    buf.reset();
}

// caller must hold host_allocations_mutex
अटल माप_प्रकार ggml_vk_host_allocation_find(स्थिर व्योम * ptr) {
    क्रम (माप_प्रकार i = 0; i < vk_instance.host_allocations.size(); i++) {
        स्थिर uint8_t * base = (स्थिर uint8_t *) vk_instance.host_allocations[i].base;
        अगर (ptr >= base && ptr < base + vk_instance.host_allocations[i].size) {
            वापस i;
        }
    }
    वापस SIZE_MAX;
}

व्योम * ggml_vk_host_malloc(माप_प्रकार size) {
    VK_LOG_MEMORY("ggml_vk_host_malloc(" << size << ")");

    ggml_vk_instance_init();

    अगर (vk_instance.multi_device) {
        स्थिर माप_प्रकार align      = std::max<माप_प्रकार>(vk_instance.host_import_alignment, TENSOR_ALIGNMENT);
        स्थिर माप_प्रकार alloc_आकार = (size + align - 1) / align * align;

        व्योम * ptr = portable_aligned_malloc(align, alloc_आकार);
        अगर (ptr == nullptr) {
            वापस nullptr;
        }
        vk_host_allocation entry;
        entry.base = ptr;
        entry.size = alloc_आकार;

        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        vk_instance.host_allocations.push_back(std::move(entry));
        वापस ptr;
    }

    vk_device dev0 = ggml_vk_get_device(0);
    vk_बफर buf;
    प्रयास {
        buf = ggml_vk_create_बफर(dev0, size,
            {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
    } पकड़ो (vk::SystemError&) {
        वापस nullptr;
    }

    अगर (!(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
        ख_लिखो(मानक_त्रुटि, "WARNING: failed to allocate %.2f MB of pinned memory\n", size / 1024.0 / 1024.0);
        ggml_vk_destroy_बफर(buf);
        वापस nullptr;
    }

    vk_host_allocation entry;
    entry.base             = buf->ptr;
    entry.size             = size;
    entry.views[dev0->idx] = buf;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
    vk_instance.host_allocations.push_back(std::move(entry));
    वापस buf->ptr;
}

व्योम ggml_vk_host_free(व्योम * ptr) {
    अगर (ptr == nullptr) {
        वापस;
    }
    VK_LOG_MEMORY("ggml_vk_host_free(" << ptr << ")");

    vk_host_allocation entry;
    {
        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        स्थिर माप_प्रकार सूचकांक = ggml_vk_host_allocation_find(ptr);
        अगर (सूचकांक == SIZE_MAX) {
            ख_लिखो(मानक_त्रुटि, "WARNING: failed to free pinned memory: memory not in map\n");
            वापस;
        }
        entry = std::move(vk_instance.host_allocations[सूचकांक]);
        vk_instance.host_allocations.erase(vk_instance.host_allocations.begin() + सूचकांक);
    }

    क्रम (स्वतः & view : entry.views) {
        ggml_vk_destroy_बफर(view);
    }
    अगर (vk_instance.multi_device) {
        portable_aligned_free(entry.base);
    }
}

व्योम ggml_vk_host_get(स्थिर vk_device& device, स्थिर व्योम * ptr, vk_बफर& buf, माप_प्रकार& buf_offset) {
    buf = nullptr;
    buf_offset = 0;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);

    स्थिर माप_प्रकार सूचकांक = ggml_vk_host_allocation_find(ptr);
    अगर (सूचकांक == SIZE_MAX) {
        वापस;
    }
    vk_host_allocation & entry = vk_instance.host_allocations[सूचकांक];

    अगर (entry.views[device->idx] == nullptr) {
        स्थिर uint32_t bit = 1u << device->idx;
        अगर (!vk_instance.multi_device || !vk_instance.device_host_बफर_import[device->idx] || (vk_instance.host_import_failed & bit)) {
            वापस;
        }
        GGML_ASSERT(device->external_memory_host);
        // const only propagates from the const context; the pointee is mutable
        entry.views[device->idx] = ggml_vk_बफर_from_host_ptr(const_cast<vk_device&>(device), entry.base, entry.size);
        अगर (entry.views[device->idx] == nullptr) {
            vk_instance.host_import_failed |= bit;
            GGML_LOG_WARN("ggml_vulkan: %s: host memory import failed, using staging\n", device->name.c_str());
            वापस;
        }
    }

    buf = entry.views[device->idx];
    buf_offset = (स्थिर uint8_t *) ptr - (स्थिर uint8_t *) entry.base;
}

व्योम ggml_vk_ensure_sync_staging_बफर(vk_device& device, माप_प्रकार size) {
    अगर (device->sync_staging == nullptr || device->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_बफर(device->sync_staging);
        device->sync_staging = ggml_vk_create_बफर_check(device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

व्योम ggml_vk_ensure_sync_staging_बफर(ggml_backend_vk_context * ctx, माप_प्रकार size) {
    अगर (ctx->sync_staging == nullptr || ctx->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_बफर(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_बफर_check(ctx->device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

अटल व्योम ggml_vk_बफर_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_बफर& dst, माप_प्रकार offset, स्थिर ggml_tensor * tensor, bool sync_staging = false) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    अगर(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        std::अ_त्रुटि << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::प_पूर्ण;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_बफर buf = nullptr;
    माप_प्रकार buf_offset = 0;
    ggml_vk_host_get(ctx->device, tensor->तथ्य, buf, buf_offset);

    स्थिर uint64_t ne0 = tensor->ne[0];
    स्थिर uint64_t ne1 = tensor->ne[1];
    स्थिर uint64_t ne2 = tensor->ne[2];
    स्थिर uint64_t ne3 = tensor->ne[3];
    स्थिर uint64_t nb0 = tensor->nb[0];
    स्थिर uint64_t nb1 = tensor->nb[1];
    स्थिर uint64_t nb2 = tensor->nb[2];
    स्थिर uint64_t nb3 = tensor->nb[3];
    स्थिर ggml_type type = tensor->type;
    स्थिर uint64_t ts = ggml_type_आकार(type);
    स्थिर uint64_t bs = ggml_blck_आकार(type);

    स्थिर uint64_t dstnb0 = ts;
    स्थिर uint64_t dstnb1 = dstnb0*(ne0/bs);
    स्थिर uint64_t dstnb2 = dstnb1*ne1;
    स्थिर uint64_t dstnb3 = dstnb2*ne2;

    स्थिर uint64_t ne = ggml_nelements(tensor);

    अगर (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::बफर_Copy> slices;

        क्रम (uint64_t i3 = 0; i3 < ne3; i3++) {
            क्रम (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                अगर (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_offset + i3*nb3 + i2*nb2, offset + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } अन्यथा {
                    क्रम (uint64_t i1 = 0; i1 < ne1; i1++) {
                        अगर (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_offset + i3*nb3 + i2*nb2 + i1*nb1, offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } अन्यथा {
                            स्थिर uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                            स्थिर uint64_t d_off = offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            क्रम (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->बफर->buf.copy_बफर(buf->बफर, dst->बफर, slices);
        वापस;
    }

    अगर (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_बफर& staging = ctx->device->sync_staging;
    स्थिर uint64_t copy_आकार = ts*ne/bs;
    ggml_vk_ensure_sync_staging_बफर(ctx->device, copy_आकार);
    Vk_बफर_Copy buf_copy{ 0, offset, copy_आकार };

    ggml_vk_sync_buffers(ctx, subctx);
    vk_Cmd_Copy_बफर(subctx->s->बफर->buf, (Vk_बफर)staging->बफर, (Vk_बफर)dst->बफर, 1, &buf_copy);

    क्रम (uint64_t i3 = 0; i3 < ne3; i3++) {
        क्रम (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            अगर (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (स्थिर uint8_t *) tensor->तथ्य + buf_offset + i3*nb3 + i2*nb2, dstnb2, &subctx->in_memcpys);
            } अन्यथा {
                क्रम (uint64_t i1 = 0; i1 < ne1; i1++) {
                    अगर (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (स्थिर uint8_t *) tensor->तथ्य + buf_offset + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->in_memcpys);
                    } अन्यथा {
                        स्थिर uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                        स्थिर uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        क्रम (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (स्थिर uint8_t *) tensor->तथ्य + s_off + i0*nb0, dstnb0, &subctx->in_memcpys);
                        }
                    }
                }
            }
        }
    }
}

bool ggml_vk_बफर_write_2d_async(vk_context subctx, vk_बफर& dst, माप_प्रकार offset, स्थिर व्योम * src, माप_प्रकार spitch, माप_प्रकार dpitch, माप_प्रकार width, माप_प्रकार height, bool sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << width << ", " << height << ")");
    // Check if src is pinned memory
    vk_बफर buf = nullptr;
    माप_प्रकार buf_offset = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_offset);

    अगर (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::बफर_Copy> slices(1);
        अगर (width == spitch && width == dpitch) {
            // Only do single write if stride is equal
            slices[0].srcOffset = buf_offset;
            slices[0].dstOffset = offset;
            slices[0].size = width * height;
        } अन्यथा {
            slices.resize(height);
            क्रम (माप_प्रकार i = 0; i < height; i++) {
                slices[i].srcOffset = buf_offset + i * spitch;
                slices[i].dstOffset = offset + i * dpitch;
                slices[i].size = width;
            }
        }

        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->बफर->buf.copy_बफर(buf->बफर, dst->बफर, slices);
        वापस true;
    }
    VK_LOG_DEBUG("STAGING");

    अगर (!sync_staging) {
        // copy was not handled caller needs to fall back
        वापस false;
    }

    // Staging buffer required
    स्थिर माप_प्रकार staging_आकार = width * height;
    ggml_vk_ensure_sync_staging_बफर(dst->device, staging_आकार);

    vk_बफर& staging_बफर = dst->device->sync_staging;

    std::vector<vk::बफर_Copy> slices(1);
    अगर (width == dpitch) {
        slices[0].srcOffset = 0;
        slices[0].dstOffset = offset;
        slices[0].size = staging_आकार;
    } अन्यथा {
        slices.resize(height);
        क्रम (माप_प्रकार i = 0; i < height; i++) {
            slices[i].srcOffset = i * width;
            slices[i].dstOffset = offset + i * dpitch;
            slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->बफर->buf.copy_बफर(staging_बफर->बफर, dst->बफर, slices);

    अगर (width == spitch) {
        deferred_memcpy((uint8_t *)staging_बफर->ptr, src, staging_आकार, &subctx->in_memcpys);
    } अन्यथा {
        क्रम (माप_प्रकार i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *)staging_बफर->ptr + i * width, (स्थिर uint8_t *) src + i * spitch, width, &subctx->in_memcpys);
        }
    }
    वापस true;
}

bool ggml_vk_बफर_write_async(vk_context subctx, vk_बफर& dst, माप_प्रकार offset, स्थिर व्योम * src, माप_प्रकार size, bool sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    वापस ggml_vk_बफर_write_2d_async(subctx, dst, offset, src, size, size, size, 1, sync_staging);
}

व्योम ggml_vk_बफर_write_2d(vk_बफर& dst, माप_प्रकार offset, स्थिर व्योम * src, माप_प्रकार spitch, माप_प्रकार dpitch, माप_प्रकार width, माप_प्रकार height) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << width << ", " << height << ")");
    // Buffer is already mapped
    अगर(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        अगर (width == spitch && width == dpitch) {
            स_नकल((uint8_t *)dst->ptr + offset, src, width * height);
        } अन्यथा {
            क्रम (माप_प्रकार i = 0; i < height; i++) {
                स_नकल((uint8_t *)dst->ptr + offset + i * dpitch, (स्थिर uint8_t *) src + i * spitch, width);
            }
        }
    } अन्यथा {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        bool ret = ggml_vk_बफर_write_2d_async(subctx, dst, offset, src, spitch, dpitch, width, height, true);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        क्रम (स्वतः& cpy : subctx->in_memcpys) {
            स_नकल(cpy.dst, cpy.src, cpy.n);
        }

        क्रम (स्वतः& mset : subctx->memsets) {
            स_रखो(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, true, UINT64_MAX), "vk_buffer_write_2d waitForFences", dst->device);
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_queue_command_pools_cleanup(dst->device);
    }
}

व्योम ggml_vk_बफर_write(vk_बफर& dst, माप_प्रकार offset, स्थिर व्योम * src, माप_प्रकार size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_बफर_write_2d(dst, offset, src, size, size, size, 1);
}

bool ggml_vk_बफर_read_2d_async(vk_context subctx, vk_बफर& src, माप_प्रकार offset, व्योम * dst, माप_प्रकार spitch, माप_प्रकार dpitch, माप_प्रकार width, माप_प्रकार height, bool sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << offset << ", width=" << width << ", height=" << height << ")");
    GGML_ASSERT(width > 0);
    GGML_ASSERT(height > 0);
    GGML_ASSERT(src != nullptr);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_बफर buf = nullptr;
    माप_प्रकार buf_offset = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_offset);

    std::vector<vk::बफर_Copy> slices(1);
    अगर (width == spitch && width == dpitch) {
        // Only do single write if stride is equal
        slices[0].srcOffset = offset;
        slices[0].dstOffset = buf_offset;
        slices[0].size = width * height;
    } अन्यथा {
        slices.resize(height);
        क्रम (माप_प्रकार i = 0; i < height; i++) {
            slices[i].srcOffset = offset + i * spitch;
            slices[i].dstOffset = buf_offset + i * dpitch;
            slices[i].size = width;
        }
    }

    अगर (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->बफर->buf.copy_बफर(src->बफर, buf->बफर, slices);

        वापस true;
    }
    VK_LOG_DEBUG("STAGING");

    अगर (!sync_staging) {
        // copy was not handled caller needs to fall back
        वापस false;
    }

    // Fall back to staging buffer
    स्थिर माप_प्रकार staging_आकार = width * height;
    ggml_vk_ensure_sync_staging_बफर(src->device, staging_आकार);

    vk_बफर& staging_बफर = src->device->sync_staging;

    std::vector<vk::बफर_Copy> staging_slices(1);
    अगर (width == spitch) {
        staging_slices[0].srcOffset = offset;
        staging_slices[0].dstOffset = 0;
        staging_slices[0].size = staging_आकार;
    } अन्यथा {
        staging_slices.resize(height);
        क्रम (माप_प्रकार i = 0; i < height; i++) {
            staging_slices[i].srcOffset = offset + i * spitch;
            staging_slices[i].dstOffset = i * width;
            staging_slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->बफर->buf.copy_बफर(src->बफर, staging_बफर->बफर, staging_slices);

    अगर (width == dpitch) {
        deferred_memcpy(dst, staging_बफर->ptr, staging_आकार, &subctx->out_memcpys);
    } अन्यथा {
        क्रम (माप_प्रकार i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (स्थिर uint8_t *) staging_बफर->ptr + i * width, width, &subctx->out_memcpys);
        }
    }
    वापस true;
}

अटल bool ggml_vk_बफर_read_async(vk_context subctx, vk_बफर& src, माप_प्रकार offset, व्योम * dst, माप_प्रकार size, bool sync_staging = false) {
    वापस ggml_vk_बफर_read_2d_async(subctx, src, offset, dst, size, size, size, 1, sync_staging);
}

व्योम ggml_vk_बफर_read_2d(vk_बफर& src, माप_प्रकार offset, व्योम * dst, माप_प्रकार spitch, माप_प्रकार dpitch, माप_प्रकार width, माप_प्रकार height) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->बफर << ", " << offset << ", " << width << ", " << height << ")");

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
    स्थिर bool slow_host_read = src->device->vendor_id == VK_VENDOR_ID_AMD &&
                                !(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCached);
    अगर(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible && src->device->uma && !slow_host_read) {
        GGML_ASSERT(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        vk_context subctx = ggml_vk_create_temporary_context(src->device->compute_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        subctx->s->बफर->buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eHost,
            {},
            { { vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eHostRead } },
            {}, {});
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX),
                 "vk_buffer_read_2d uma waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        अगर (width == spitch && width == dpitch) {
            स_नकल(dst, (स्थिर uint8_t *) src->ptr + offset, width * height);
        } अन्यथा {
            क्रम (माप_प्रकार i = 0; i < height; i++) {
                स_नकल((uint8_t *) dst + i * dpitch, (स्थिर uint8_t *) src->ptr + offset + i * spitch, width);
            }
        }
    } अन्यथा {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        bool ret = ggml_vk_बफर_read_2d_async(subctx, src, offset, dst, spitch, dpitch, width, height, true);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX), "vk_buffer_read_2d waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        क्रम (स्वतः& cpy : subctx->out_memcpys) {
            स_नकल(cpy.dst, cpy.src, cpy.n);
        }
    }
}

व्योम ggml_vk_बफर_read(vk_बफर& src, माप_प्रकार offset, व्योम * dst, माप_प्रकार size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->बफर << ", " << offset << ", " << size << ")");
    ggml_vk_बफर_read_2d(src, offset, dst, size, size, size, 1);
}

व्योम ggml_vk_बफर_copy_async(vk_context& ctx, vk_बफर& dst, माप_प्रकार dst_offset, vk_बफर& src, माप_प्रकार src_offset, माप_प्रकार size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    Vk_बफर_Copy bc{ src_offset, dst_offset, size };

    vk_Cmd_Copy_बफर(ctx->s->बफर->buf, (Vk_बफर)src->बफर, (Vk_बफर)dst->बफर, 1, &bc);
}

व्योम ggml_vk_बफर_copy(vk_बफर& dst, माप_प्रकार dst_offset, vk_बफर& src, माप_प्रकार src_offset, माप_प्रकार size) {
    अगर (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_बफर_copy_async(subctx, dst, dst_offset, src, src_offset, size);
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX), "vk_buffer_copy waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);
    } अन्यथा {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_बफर(src->device, size);

        // Copy to src staging buffer
        ggml_vk_बफर_copy(src->device->sync_staging, 0, src, src_offset, size);
        // Copy to dst buffer
        ggml_vk_बफर_write(dst, dst_offset, src->device->sync_staging->ptr, size);
    }
}

व्योम ggml_vk_बफर_memset_async(vk_context& ctx, vk_बफर& dst, माप_प्रकार offset, uint32_t c, माप_प्रकार size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << offset << ", " << c << ", " << size << ")");

    अगर (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + offset, c, size, &ctx->memsets);
        वापस;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->बफर->buf.fill_बफर(dst->बफर, offset, size, c);
}

व्योम ggml_vk_बफर_memset(vk_बफर& dst, माप_प्रकार offset, uint32_t c, माप_प्रकार size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << offset << ", " << c << ", " << size << ")");

    अगर (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        स_रखो((uint8_t*)dst->ptr + offset, c, size);
        वापस;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->बफर->buf.fill_बफर(dst->बफर, offset, size, c);
    ggml_vk_ctx_end(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, true, UINT64_MAX), "vk_memset waitForFences", dst->device);
    dst->device->device.resetFences({ dst->device->fence });
    ggml_vk_queue_command_pools_cleanup(dst->device);
}

ggml_backend_बफर_i ggml_backend_vk_बफर_interface = {
    /* .free_buffer     = */ ggml_backend_vk_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_vk_बफर_get_base,
    /* .init_tensor     = */ ggml_backend_vk_बफर_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_बफर_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_बफर_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_बफर_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_बफर_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_बफर_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_बफर_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_बफर_clear,
    /* .reset           = */ शून्य,
};

vk_बफर ggml_vk_बफर_from_host_ptr(vk_device & device, व्योम * ptr, माप_प्रकार size) {
    अगर (!device->external_memory_host) {
        वापस {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    अगर (uptr & (device->min_imported_host_pointer_alignment - 1)) {
        वापस {};
    }
    अगर (size & (device->min_imported_host_pointer_alignment - 1)) {
        वापस {};
    }

    स्थिर vk::MemoryPropertyFlags property_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached;

    vk_बफर buf {};
    प्रयास {
        buf = ggml_vk_create_बफर(device, size, { property_flags }, ptr);
    } पकड़ो (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    वापस buf;
}

