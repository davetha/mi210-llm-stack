// fa_wrapper.cpp — Bridge between raw GPU pointers and flash_attn's mha_fwd
// NO Python dependency - uses minimal ATen headers only
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <iostream>

// Forward declare the flash_attn entry point
namespace FLASH_NAMESPACE {
    at::Tensor mha_fwd(
        at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        at::Tensor &out,
        float softmax_scale,
        bool is_causal,
        int window_size_left = -1,
        int window_size_right = -1,
        float softcap = 0.0f,
        bool return_softmax = false
    );
}

extern "C" {
int fa_forward_raw(
    void* q_ptr, void* k_ptr, void* v_ptr, void* out_ptr,
    int batch_size, int seqlen_q, int seqlen_k,
    int nheads, int nheads_k, int headdim,
    int is_causal, float softmax_scale,
    void* stream_ptr
) {
    try {
        if (stream_ptr) {
            hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
            c10::cuda::CUDAStream c10_stream = c10::cuda::getStreamFromExternal(stream, 0);
            c10::cuda::setCurrentCUDAStream(c10_stream);
        }

        auto opts = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
        auto q = at::from_blob(q_ptr, {batch_size, seqlen_q, nheads, headdim}, opts);
        auto k = at::from_blob(k_ptr, {batch_size, seqlen_k, nheads_k, headdim}, opts);
        auto v = at::from_blob(v_ptr, {batch_size, seqlen_k, nheads_k, headdim}, opts);
        auto out = at::from_blob(out_ptr, {batch_size, seqlen_q, nheads, headdim}, opts);

        FLASH_NAMESPACE::mha_fwd(q, k, v, out,
            softmax_scale, is_causal != 0, -1, -1, 0.0f, false);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fa_wrapper error: " << e.what() << std::endl;
        return -1;
    }
}
}
