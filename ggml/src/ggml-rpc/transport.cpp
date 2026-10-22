#অন্তর্ভাৱ "transport.h"
#অন্তর্ভাৱ "log.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#অন্যথা
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif
#অন্তর্ভাৱ <cstdlib>
#অন্তর্ভাৱ <cstring>
#অন্তর্ভাৱ <mutex>
#অন্তর্ভাৱ <optional>

#ifdef GGML_RPC_RDMA
#  include <infiniband/verbs.h>
#  include <array>
#  include <cerrno>
#  include <chrono>
#  include <time.h>
#  ifndef _WIN32
#    include <poll.h>
#  endif
#  ifdef GGML_RPC_RDMA_APPLE
#    include "transport-apple.h"
#  endif
#endif // GGML_RPC_RDMA

#ifdef _WIN32
প্রকারনির্দেশ SOCKET sockfd_t;
using ssize_t = __int64;
#অন্যথা
প্রকারনির্দেশ পূর্ণাঙ্ক sockfd_t;
#endif

#ifdef GGML_RPC_RDMA
স্থির constexpr size_t RDMA_GID_SIZE = 16;            // RoCE GID / IB GID is always 16 bytes
using rdma_gid_t = std::array<uint8_t, RDMA_GID_SIZE>;
#endif // GGML_RPC_RDMA

#যদি defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
স্থির constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
স্থির constexpr পূর্ণাঙ্ক    RDMA_RX_DEPTH = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
স্থির constexpr auto   RDMA_SPIN_TIME = std::chrono::milliseconds(100);

সংরচনা rdma_conn {
    সংরচনা ibv_context * ctx = nullptr;
    সংরচনা ibv_pd * pd  = nullptr;
    সংরচনা ibv_cq * scq = nullptr;   // send completions
    সংরচনা ibv_cq * rcq = nullptr;   // recv completions
    সংরচনা ibv_qp * qp  = nullptr;
    সংরচনা ibv_comp_channel * ch = nullptr; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point last_active; // last completion or posted send

    শূন্য          * tx_buf = nullptr;
    সংরচনা ibv_mr * tx_mr  = nullptr;

    শূন্য          * rx_buf = nullptr; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    সংরচনা ibv_mr * rx_mr  = nullptr;
    পূর্ণাঙ্ক             rx_head = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(পূর্ণাঙ্ক i) ধ্রুৱ {
        ঘূৰাই static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    তার্কিক post_rx(পূর্ণাঙ্ক i) {
        সংরচনা ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.length = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        সংরচনা ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.অঙ্ক_sge = 1;
        ঘূৰাই ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        যদি (tx_mr) ibv_dereg_mr(tx_mr);
        যদি (rx_mr) ibv_dereg_mr(rx_mr);
        মুক্ত(tx_buf);
        মুক্ত(rx_buf);
        যদি (qp)  ibv_destroy_qp(qp);
        যদি (scq) ibv_destroy_cq(scq);
        যদি (rcq) ibv_destroy_cq(rcq);
        যদি (ch)  ibv_destroy_comp_channel(ch);
        যদি (pd)  ibv_dealloc_pd(pd);
        যদি (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
সংরচনা rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_SIZE] = {};
    uint8_t  ib_port = 0;
    পূর্ণাঙ্ক      gid_idx = 0;
    গণনা ibv_mtu path_mtu = IBV_MTU_1024;
};

সংরচনা rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(পরিমাণ(rdma_caps) == RPC_CONN_CAPS_SIZE, "rdma_caps must match conn_caps size");

#endif // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

সংরচনা socket_t::impl {
    impl(sockfd_t fd) : use_rdma(মিছা), fd(fd) {}
    ~impl();
    তার্কিক send_data(ধ্রুৱ শূন্য * data, size_t size);
    তার্কিক recv_data(শূন্য * data, size_t size);
    তার্কিক flush();
    শূন্য get_caps(uint8_t * local_caps);
    শূন্য update_caps(ধ্রুৱ uint8_t * remote_caps);

#ifdef GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_target_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  অন্যথা
    তার্কিক rdma_probe();
    তার্কিক rdma_send(ধ্রুৱ শূন্য * data, size_t size);
    তার্কিক rdma_recv(শূন্য * data, size_t size);
    তার্কিক tcp_peer_closed();
    তার্কিক rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ধ্রুৱ uint8_t * remote_gid);
    তার্কিক rdma_poll(সংরচনা ibv_cq * cq, সংরচনা ibv_wc * wc);
    তার্কিক rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#endif // GGML_RPC_RDMA
    তার্কিক     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#ifdef GGML_RPC_RDMA
    rdma.reset();
#endif // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, এতত্->fd);
#ifdef _WIN32
    যদি (fd != INVALID_SOCKET) closesocket(এতত্->fd);
#অন্যথা
    যদি (fd >= 0) close(এতত্->fd);
#endif
}

#ifdef GGML_RPC_RDMA

// Build a RoCE GID-shaped 16-byte target from a TCP socket's local address.
// Used to match the socket's local IP against the kernel's GID table so that
// a single memcmp handles IPv4, IPv4-mapped IPv6, and native IPv6 uniformly:
//   AF_INET                -> ::ffff:a.b.c.d  (bytes 10-11 = 0xff, last 4 = IPv4)
//   AF_INET6 (IPv4-mapped) -> ::ffff:a.b.c.d  (already in GID shape)
//   AF_INET6 (native v6)   -> the 16-byte IPv6 address as-is
// Returns std::nullopt on unsupported family or getsockname failure.
std::optional<rdma_gid_t> socket_t::impl::rdma_build_target_gid() {
    sockaddr_storage addr = {};
    socklen_t addr_len = পরিমাণ(addr);
    যদি (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        ঘূৰাই std::nullopt;
    }
    rdma_gid_t target = {};
    যদি (addr.ss_family == AF_INET) {
        ধ্রুৱ auto * a = reinterpret_cast<ধ্রুৱ sockaddr_in *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        memcpy(&target[12], &a->sin_addr, 4);
        ঘূৰাই target;
    }
    যদি (addr.ss_family == AF_INET6) {
        ধ্রুৱ auto * a = reinterpret_cast<ধ্রুৱ sockaddr_in6 *>(&addr);
        memcpy(target.data(), &a->sin6_addr, RDMA_GID_SIZE);
        ঘূৰাই target;
    }
    ঘূৰাই std::nullopt;
}

#ifndef GGML_RPC_RDMA_APPLE

তার্কিক socket_t::impl::tcp_peer_closed() {
    যদি (fd < 0) ঘূৰাই মিছা;
#ifndef _WIN32
    সংরচনা pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    পূর্ণাঙ্ক r = poll(&pfd, 1, 0);
    ঘূৰাই r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#অন্যথা
    ঘূৰাই মিছা;
#endif
}

তার্কিক socket_t::impl::rdma_probe() {
    ধ্রুৱ ৱর্ণ * dev_env = std::getenv("GGML_RDMA_DEV");
    ধ্রুৱ ৱর্ণ * gid_env = std::getenv("GGML_RDMA_GID");

    auto target_gid = rdma_build_target_gid();
    যদি (!target_gid) {
        ঘূৰাই মিছা;
    }

    ধ্রুৱ uint8_t ib_port = 1;
    পূর্ণাঙ্ক অঙ্ক_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&অঙ্ক_devs);
    যদি (!devs || অঙ্ক_devs == 0) ঘূৰাই মিছা;

    ibv_context * ibctx = nullptr;
    ধ্রুৱ ৱর্ণ * matched_dev = nullptr;
    পূর্ণাঙ্ক gid_idx = gid_env ? atoi(gid_env) : -1;
    পূর্ণাঙ্ক gid_version = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    বাবে (পূর্ণাঙ্ক d = 0; d < অঙ্ক_devs; d++) {
        ধ্রুৱ ৱর্ণ * dn = ibv_get_device_name(devs[d]);
        যদি (dev_env && strcmp(dev_env, dn) != 0) অনুৱর্ত;

        ibv_context * ctx = ibv_open_device(devs[d]);
        যদি (!ctx) অনুৱর্ত;

        ibv_port_attr pa;
        যদি (ibv_query_port(ctx, ib_port, &pa) != 0) { ibv_close_device(ctx); অনুৱর্ত; }

        পূর্ণাঙ্ক found_gid = gid_idx;
        পূর্ণাঙ্ক found_version = IBV_GID_TYPE_IB;
        যদি (found_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            পূর্ণাঙ্ক v2_idx = -1;
            পূর্ণাঙ্ক v1_idx = -1;
            বাবে (পূর্ণাঙ্ক i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_entry entry = {};
                যদি (ibv_query_gid_ex(ctx, ib_port, i, &entry, 0) != 0) অনুৱর্ত;
                যদি (memcmp(entry.gid.raw, target_gid->data(), RDMA_GID_SIZE) != 0) অনুৱর্ত;
                যদি (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } অন্যথা যদি (entry.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            যদি (v2_idx >= 0) {
                found_gid = v2_idx;
                found_version = IBV_GID_TYPE_ROCE_V2;
            } অন্যথা যদি (v1_idx >= 0) {
                found_gid = v1_idx;
                found_version = IBV_GID_TYPE_ROCE_V1;
            }
        } অন্যথা {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_entry entry = {};
            যদি (ibv_query_gid_ex(ctx, ib_port, found_gid, &entry, 0) == 0) {
                found_version = entry.gid_type;
            }
        }
        যদি (found_gid >= 0) {
            ibctx = ctx;
            gid_idx = found_gid;
            gid_version = found_version;
            matched_dev = dn;
            rdma_local.path_mtu = pa.active_mtu;
            ভঙ্গ;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    যদি (!ibctx) ঘূৰাই মিছা;

    rdma_local.ib_port = ib_port;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    যদি (!rdma->pd) ঘূৰাই মিছা;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, nullptr, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_DEPTH + 4, nullptr, rdma->ch, 0);
    যদি (!rdma->scq || !rdma->rcq) ঘূৰাই মিছা;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->scq;
    qia.recv_cq = rdma->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = 4;
    qia.cap.max_recv_wr     = RDMA_RX_DEPTH + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_data = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    যদি (!rdma->qp) ঘূৰাই মিছা;
    rdma->max_inline = qia.cap.max_inline_data;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    যদি (!rdma->tx_buf || !rdma->rx_buf) ঘূৰাই মিছা;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    যদি (!rdma->tx_mr || !rdma->rx_mr) ঘূৰাই মিছা;

    ibv_gid local_gid;
    যদি (ibv_query_gid(ibctx, ib_port, gid_idx, &local_gid) != 0) ঘূৰাই মিছা;

    rdma_local.qpn = rdma->qp->qp_অঙ্ক;
    rdma_local.psn = rdma->qp->qp_অঙ্ক & 0xffffff;
    memcpy(&rdma_local.gid, &local_gid, RDMA_GID_SIZE);

    ধ্রুৱ ৱর্ণ * ver_str = "";
    যদি (gid_version == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } অন্যথা যদি (gid_version == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    ঘূৰাই সঁচা;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
তার্কিক socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ধ্রুৱ uint8_t * remote_gid) {
    // RESET -> INIT
    {
        সংরচনা ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.port_অঙ্ক        = rdma_local.ib_port;
        a.pkey_index      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        যদি (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            ঘূৰাই মিছা;
        }
    }

    বাবে (পূর্ণাঙ্ক i = 0; i < RDMA_RX_DEPTH; i++) {
        যদি (!rdma->post_rx(i)) ঘূৰাই মিছা;
    }

    // INIT -> RTR
    {
        সংরচনা ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = rdma_local.path_mtu;
        a.dest_qp_অঙ্ক        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        memcpy(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_SIZE);
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.port_অঙ্ক   = rdma_local.ib_port;
        যদি (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            ঘূৰাই মিছা;
        }
    }

    // RTR -> RTS
    {
        সংরচনা ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        যদি (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            ঘূৰাই মিছা;
        }
    }

    rdma->last_active = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
    ঘূৰাই সঁচা;
}

// Sleep until the completion channel has an event or the TCP peer closes.
তার্কিক socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.get();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    সংরচনা pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    যদি (poll(pfds, 2, -1) < 0) {
        ঘূৰাই errno == EINTR;
    }
    যদি (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        ঘূৰাই মিছা;
    }
    যদি (pfds[0].revents & POLLIN) {
        সংরচনা ibv_cq * ev_cq  = nullptr;
        শূন্য          * ev_ctx = nullptr;
        যদি (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            ঘূৰাই মিছা;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    ঘূৰাই সঁচা;
}

তার্কিক socket_t::impl::rdma_poll(সংরচনা ibv_cq * cq, সংরচনা ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    তার্কিক armed = মিছা;
    বাবে (uint64_t s = 1; ; s++) {
        পূর্ণাঙ্ক n = ibv_poll_cq(cq, 1, wc);
        যদি (n > 0) {
            c->last_active = std::chrono::steady_clock::now();
            যদি (wc->status != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->status, ibv_wc_status_str(wc->status), wc->vendor_err);
            }
            ঘূৰাই wc->status == IBV_WC_SUCCESS;
        }
        যদি (n < 0) ঘূৰাই মিছা;
        যদি (armed) {
            // armed and still empty: sleep until the next completion
            যদি (!rdma_wait_event()) {
                ঘূৰাই মিছা;
            }
            armed = মিছা;
            অনুৱর্ত;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        যদি (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->last_active > RDMA_SPIN_TIME) {
            যদি (ibv_req_notify_cq(cq, 0) != 0) {
                ঘূৰাই মিছা;
            }
            armed = সঁচা;
            অনুৱর্ত;
        }
        যদি ((s & 0xFFFFF) == 0) {
            যদি (tcp_peer_closed()) {
                ঘূৰাই মিছা;
            }
        }
    }
}

তার্কিক socket_t::impl::rdma_send(ধ্রুৱ শূন্য * data, size_t size) {
    rdma_conn * c = rdma.get();
    ধ্রুৱ uint8_t * src = (ধ্রুৱ uint8_t *)data;
    size_t rem = size;
    যেতিয়ালৈ (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        সংরচনা ibv_sge sge = {};
        সংরচনা ibv_send_wr wr = {}, * bad = nullptr;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.অঙ্ক_sge = 1;

        যদি (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.length = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } অন্যথা {
            memcpy(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.length = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        যদি (ibv_post_send(c->qp, &wr, &bad) != 0) ঘূৰাই মিছা;
        c->last_active = std::chrono::steady_clock::now();
        সংরচনা ibv_wc wc;
        যদি (!rdma_poll(c->scq, &wc)) ঘূৰাই মিছা;

        src += chunk;
        rem -= chunk;
    }
    ঘূৰাই সঁচা;
}

তার্কিক socket_t::impl::rdma_recv(শূন্য * data, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)data;
    size_t rem = size;
    যেতিয়ালৈ (rem > 0) {
        সংরচনা ibv_wc wc;
        যদি (!rdma_poll(c->rcq, &wc)) ঘূৰাই মিছা;

        পূর্ণাঙ্ক slot = (পূর্ণাঙ্ক)wc.wr_id;
        size_t got = wc.byte_len;
        memcpy(dst, c->rx_slot(slot), got);

        যদি (!c->post_rx(slot)) ঘূৰাই মিছা;

        dst += got;
        rem -= got;
    }
    ঘূৰাই সঁচা;
}

#endif // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#endif // GGML_RPC_RDMA

তার্কিক socket_t::impl::send_data(ধ্রুৱ শূন্য * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    যদি (use_rdma) {
        ঘূৰাই rdma->send(data, size);
    }
#নোচেদ্যদি defined(GGML_RPC_RDMA)
    যদি (use_rdma) {
        ঘূৰাই rdma_send(data, size);
    }
#endif
    size_t bytes_sent = 0;
    যেতিয়ালৈ (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(fd, (ধ্রুৱ ৱর্ণ *)data + bytes_sent, size_to_send, 0);
        যদি (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           bytes_sent, size_to_send);
            ঘূৰাই মিছা;
        }
        bytes_sent += (size_t)n;
    }
    ঘূৰাই সঁচা;
}

তার্কিক socket_t::impl::recv_data(শূন্য * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    যদি (use_rdma) {
        ঘূৰাই rdma->recv(data, size);
    }
#নোচেদ্যদি defined(GGML_RPC_RDMA)
    যদি (use_rdma) {
        ঘূৰাই rdma_recv(data, size);
    }
#endif
    size_t bytes_recv = 0;
    যেতিয়ালৈ (bytes_recv < size) {
        size_t size_to_recv = std::min(size - bytes_recv, MAX_CHUNK_SIZE);
        ssize_t n = recv(fd, (ৱর্ণ *)data + bytes_recv, size_to_recv, 0);
        যদি (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           bytes_recv, size_to_recv);
            ঘূৰাই মিছা;
        }
        যদি (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            ঘূৰাই মিছা;
        }
        bytes_recv += (size_t)n;
    }
    ঘূৰাই সঁচা;
}

শূন্য socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_SIZE);
#ifdef GGML_RPC_RDMA
    যদি (std::getenv("GGML_RPC_NO_RDMA")) {
        ঘূৰাই;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    auto target_gid = rdma_build_target_gid();
    যদি (target_gid) {
        rdma = apple_rdma::probe(fd, target_gid->data(), local_caps);
    }
    যদি (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  অন্যথা
    rdma_local = {};
    যদি (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        memcpy(rc.gid, rdma_local.gid, RDMA_GID_SIZE);
        memcpy(local_caps, &rc, পরিমাণ(rc));
    } অন্যথা {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#endif // GGML_RPC_RDMA
}

শূন্য socket_t::impl::update_caps(ধ্রুৱ uint8_t * remote_caps) {
#ifdef GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    তার্কিক remote_rdma = মিছা;
    বাবে (size_t i = 0; i < RPC_CONN_CAPS_SIZE; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    যদি (!rdma || !remote_rdma) {
        যদি (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        ঘূৰাই;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    তার্কিক activated = rdma->activate(remote_caps);
#  অন্যথা
    rdma_caps rc = {};
    memcpy(&rc, remote_caps, পরিমাণ(rc));
    তার্কিক activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    যদি (activated) {
        use_rdma = সঁচা;
    } অন্যথা {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#অন্যথা
    (শূন্য)remote_caps;
#endif // GGML_RPC_RDMA
}

তার্কিক socket_t::impl::flush() {
#ifdef GGML_RPC_RDMA_APPLE
    যদি (use_rdma) {
        ঘূৰাই rdma->flush();
    }
#endif
    ঘূৰাই সঁচা;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = স্ৱভাৱ;

তার্কিক socket_t::send_data(ধ্রুৱ শূন্য * data, size_t size) {
    ঘূৰাই pimpl->send_data(data, size);
}

তার্কিক socket_t::recv_data(শূন্য * data, size_t size) {
    ঘূৰাই pimpl->recv_data(data, size);
}

তার্কিক socket_t::flush() {
    ঘূৰাই pimpl->flush();
}

শূন্য socket_t::get_caps(uint8_t * local_caps) {
    ঘূৰাই pimpl->get_caps(local_caps);
}

শূন্য socket_t::update_caps(ধ্রুৱ uint8_t * remote_caps) {
    ঘূৰাই pimpl->update_caps(remote_caps);
}

ধ্রুৱ ৱর্ণ * socket_t::transport_name() ধ্রুৱ {
    ঘূৰাই pimpl->use_rdma ? "RDMA" : "TCP";
}

স্থির তার্কিক is_valid_fd(sockfd_t sockfd) {
#ifdef _WIN32
    ঘূৰাই sockfd != INVALID_SOCKET;
#অন্যথা
    ঘূৰাই sockfd >= 0;
#endif
}

স্থির তার্কিক set_no_delay(sockfd_t sockfd) {
    পূর্ণাঙ্ক flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    পূর্ণাঙ্ক ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (ৱর্ণ *)&flag, পরিমাণ(পূর্ণাঙ্ক));
    ঘূৰাই ret == 0;
}

স্থির তার্কিক set_reuse_addr(sockfd_t sockfd) {
    পূর্ণাঙ্ক flag = 1;
    পূর্ণাঙ্ক ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (ৱর্ণ *)&flag, পরিমাণ(পূর্ণাঙ্ক));
    ঘূৰাই ret == 0;
}

socket_ptr socket_t::accept() {
    auto client_socket_fd = ::accept(pimpl->fd, রিক্ত, রিক্ত);
    যদি (!is_valid_fd(client_socket_fd)) {
        ঘূৰাই nullptr;
    }
    যদি (!set_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        ঘূৰাই nullptr;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    ঘূৰাই socket_ptr(নূতন socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(ধ্রুৱ ৱর্ণ * host, পূর্ণাঙ্ক port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    যদি (!is_valid_fd(sockfd)) {
        ঘূৰাই nullptr;
    }
    যদি (!set_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        ঘূৰাই nullptr;
    }
    যদি (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        ঘূৰাই nullptr;
    }
    সংরচনা sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    যদি (bind(sockfd, (সংরচনা sockaddr *) &serv_addr, পরিমাণ(serv_addr)) < 0) {
        ঘূৰাই nullptr;
    }
    যদি (listen(sockfd, 1) < 0) {
        ঘূৰাই nullptr;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, port);
    ঘূৰাই socket_ptr(নূতন socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(ধ্রুৱ ৱর্ণ * host, পূর্ণাঙ্ক port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    যদি (!is_valid_fd(sockfd)) {
        ঘূৰাই nullptr;
    }
    যদি (!set_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        ঘূৰাই nullptr;
    }
    সংরচনা sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    সংরচনা hostent * server = gethostbyname(host);
    যদি (server == রিক্ত) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        ঘূৰাই nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    যদি (::connect(sockfd, (সংরচনা sockaddr *)&addr, পরিমাণ(addr)) < 0) {
        ঘূৰাই nullptr;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, port);
    ঘূৰাই socket_ptr(নূতন socket_t(std::make_unique<impl>(sockfd)));
}

#ifdef _WIN32
স্থির std::mutex g_rpc_transport_mu;
স্থির তার্কিক g_rpc_transport_wsa_started = মিছা;
#endif

তার্কিক rpc_transport_init() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    যদি (g_rpc_transport_wsa_started) {
        ঘূৰাই সঁচা;
    }
    WSADATA wsaData;
    পূর্ণাঙ্ক res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    যদি (res != 0) {
        ঘূৰাই মিছা;
    }
    g_rpc_transport_wsa_started = সঁচা;
    ঘূৰাই সঁচা;
#অন্যথা
    ঘূৰাই সঁচা;
#endif
}

শূন্য rpc_transport_shutdown() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    যদি (!g_rpc_transport_wsa_started) {
        ঘূৰাই;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = মিছা;
#endif
}
