// CU-masking Phase 0 probe for gfx90a (MI210).
//
// Answers, with evidence rather than hipSuccess:
//   1. Does hipExtStreamCreateWithCUMask accept a mask, and does
//      hipExtStreamGetCUMask read back what we set?
//   2. Do workgroups actually execute only on the selected CUs?
//      (read HW_ID: SE_ID / SH_ID / CU_ID per workgroup)
//   3. How does a flat mask bit index map to a physical (SE, SH, CU)?
//   4. Do two disjoint-mask streams execute concurrently?
//   5. Does HIP graph capture/replay preserve the mask?
//
// Build: hipcc -O2 --offload-arch=gfx90a cu_mask_probe.hip -o cu_mask_probe

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <set>
#include <chrono>

#define HC(x)                                                                  \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      printf("FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x,                  \
             hipGetErrorString(e));                                            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// GFX9 HW_ID: CU_ID [11:8], SH_ID [12], SE_ID [15:13].
// s_getreg imm = id | (offset << 6) | ((size - 1) << 11); id=4, offset=0, size=32.
__device__ __forceinline__ uint32_t hw_id() {
  return __builtin_amdgcn_s_getreg(4 | (0u << 6) | (31u << 11));
}

// Marks which (SE, SH, CU) slots ran, and spins long enough to stay resident.
__global__ void probe(uint32_t *slots, uint64_t spin_cycles) {
  if (threadIdx.x == 0) {
    uint32_t h = hw_id();
    uint32_t cu = (h >> 8) & 0xF;
    uint32_t sh = (h >> 12) & 0x1;
    uint32_t se = (h >> 13) & 0x7;
    atomicOr(&slots[se * 32 + sh * 16 + cu], 1u);
  }
  uint64_t t0 = __builtin_amdgcn_s_memtime();
  while (__builtin_amdgcn_s_memtime() - t0 < spin_cycles) {
    __builtin_amdgcn_s_sleep(1);
  }
}

static void set_bits(std::vector<uint32_t> &m, int lo, int hi) {
  std::fill(m.begin(), m.end(), 0u);
  for (int b = lo; b < hi; ++b) m[b / 32] |= (1u << (b % 32));
}

static int popcount_mask(const std::vector<uint32_t> &m) {
  int n = 0;
  for (uint32_t w : m) n += __builtin_popcount(w);
  return n;
}

// Runs `probe` on `s` and returns the set of physical slots that executed.
static std::set<int> run_and_collect(hipStream_t s, uint32_t *d_slots,
                                     int nslots, int blocks,
                                     uint64_t spin) {
  hipMemsetAsync(d_slots, 0, nslots * sizeof(uint32_t), s);
  hipLaunchKernelGGL(probe, dim3(blocks), dim3(64), 0, s, d_slots, spin);
  hipStreamSynchronize(s);
  std::vector<uint32_t> h(nslots);
  hipMemcpy(h.data(), d_slots, nslots * sizeof(uint32_t), hipMemcpyDeviceToHost);
  std::set<int> out;
  for (int i = 0; i < nslots; ++i)
    if (h[i]) out.insert(i);
  return out;
}

static void print_slots(const std::set<int> &s) {
  printf("    %zu physical CUs:", s.size());
  int shown = 0;
  for (int v : s) {
    if (shown++ < 24) printf(" se%d/sh%d/cu%d", v / 32, (v % 32) / 16, v % 16);
  }
  if ((int)s.size() > 24) printf(" ...");
  printf("\n");
}

int main() {
  hipDeviceProp_t p;
  HC(hipGetDeviceProperties(&p, 0));
  const int NCU = p.multiProcessorCount;
  const int words = (NCU + 31) / 32;
  printf("device      : %s (%s)\n", p.name, p.gcnArchName);
  printf("CUs         : %d  -> mask words: %d\n", NCU, words);

  const int NSLOTS = 8 * 32;  // se(8) x sh(2) x cu(16)
  uint32_t *d_slots = nullptr;
  HC(hipMalloc(&d_slots, NSLOTS * sizeof(uint32_t)));

  // ---------------------------------------------------------------- test 1+2
  printf("\n== 1/2. mask set -> read back -> actual placement ==\n");
  struct Case { const char *name; int lo, hi; };
  Case cases[] = {{"bits 0-15",   0,  16},
                  {"bits 16-31", 16,  32},
                  {"bits 0-51",   0,  52},
                  {"bits 52-103",52, 104},
                  {"all bits",    0, NCU}};
  std::set<int> lower_half, upper_half;
  for (auto &c : cases) {
    std::vector<uint32_t> m(words);
    set_bits(m, c.lo, c.hi > NCU ? NCU : c.hi);
    hipStream_t s;
    hipError_t e = hipExtStreamCreateWithCUMask(&s, words, m.data());
    if (e != hipSuccess) {
      printf("  %-11s CREATE FAILED: %s\n", c.name, hipGetErrorString(e));
      continue;
    }
    std::vector<uint32_t> rb(words, 0);
    hipError_t ge = hipExtStreamGetCUMask(s, words, rb.data());
    bool same = (ge == hipSuccess) && (memcmp(m.data(), rb.data(),
                                              words * sizeof(uint32_t)) == 0);
    printf("  %-11s set=%d bits  readback=%s (%d bits)\n", c.name,
           popcount_mask(m),
           ge != hipSuccess ? "ERROR" : (same ? "IDENTICAL" : "DIFFERENT"),
           ge == hipSuccess ? popcount_mask(rb) : -1);
    auto got = run_and_collect(s, d_slots, NSLOTS, 4096, 2000000ull);
    print_slots(got);
    if (c.lo == 0 && c.hi == 52) lower_half = got;
    if (c.lo == 52) upper_half = got;
    hipStreamDestroy(s);
  }

  // ------------------------------------------------------------------ test 3
  printf("\n== 3. flat bit index -> physical (se/sh/cu) ==\n");
  for (int b = 0; b < 12; ++b) {
    std::vector<uint32_t> m(words);
    set_bits(m, b, b + 1);
    hipStream_t s;
    if (hipExtStreamCreateWithCUMask(&s, words, m.data()) != hipSuccess) {
      printf("  bit %2d: create failed\n", b);
      continue;
    }
    auto got = run_and_collect(s, d_slots, NSLOTS, 256, 200000ull);
    printf("  bit %2d ->", b);
    for (int v : got) printf(" se%d/sh%d/cu%d", v / 32, (v % 32) / 16, v % 16);
    if (got.empty()) printf(" (nothing ran)");
    printf("\n");
    hipStreamDestroy(s);
  }

  // ------------------------------------------------------------------ test 4
  printf("\n== 4. disjoint masks: do they run concurrently? ==\n");
  {
    size_t overlap = 0;
    for (int v : lower_half) if (upper_half.count(v)) overlap++;
    printf("  placement overlap between bits0-51 and bits52-103: %zu CUs\n",
           overlap);

    std::vector<uint32_t> mA(words), mB(words);
    set_bits(mA, 0, 52);
    set_bits(mB, 52, NCU);
    hipStream_t a, b;
    HC(hipExtStreamCreateWithCUMask(&a, words, mA.data()));
    HC(hipExtStreamCreateWithCUMask(&b, words, mB.data()));
    hipStream_t plain1, plain2;
    HC(hipStreamCreate(&plain1));
    HC(hipStreamCreate(&plain2));

    const uint64_t SPIN = 60000000ull;  // ~40 ms at 1.5 GHz
    const int BLK = 512;
    auto timeit = [&](hipStream_t s1, hipStream_t s2, bool serial) {
      hipDeviceSynchronize();
      auto t0 = std::chrono::steady_clock::now();
      hipLaunchKernelGGL(probe, dim3(BLK), dim3(64), 0, s1, d_slots, SPIN);
      if (serial) hipStreamSynchronize(s1);
      hipLaunchKernelGGL(probe, dim3(BLK), dim3(64), 0, s2, d_slots, SPIN);
      hipStreamSynchronize(s1);
      hipStreamSynchronize(s2);
      auto t1 = std::chrono::steady_clock::now();
      return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };
    double one = timeit(a, a, true) / 2.0;
    double ser = timeit(a, b, true);
    double dis = timeit(a, b, false);
    double plain = timeit(plain1, plain2, false);
    printf("  single kernel (masked A)        : %7.2f ms\n", one);
    printf("  serial A then B                 : %7.2f ms\n", ser);
    printf("  concurrent, DISJOINT masks      : %7.2f ms  (%.2fx vs serial)\n",
           dis, ser / dis);
    printf("  concurrent, two unmasked streams: %7.2f ms  (%.2fx vs serial)\n",
           plain, ser / plain);
    hipStreamDestroy(a);
    hipStreamDestroy(b);
    hipStreamDestroy(plain1);
    hipStreamDestroy(plain2);
  }

  // ------------------------------------------------------------------ test 5
  printf("\n== 5. does HIP graph capture/replay preserve the mask? ==\n");
  {
    std::vector<uint32_t> m(words);
    set_bits(m, 0, 16);
    hipStream_t s;
    HC(hipExtStreamCreateWithCUMask(&s, words, m.data()));

    auto direct = run_and_collect(s, d_slots, NSLOTS, 4096, 2000000ull);
    printf("  direct launch on masked stream:\n");
    print_slots(direct);

    hipGraph_t g;
    hipGraphExec_t ge;
    HC(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
    hipLaunchKernelGGL(probe, dim3(4096), dim3(64), 0, s, d_slots, 2000000ull);
    HC(hipStreamEndCapture(s, &g));
    HC(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));

    HC(hipMemset(d_slots, 0, NSLOTS * sizeof(uint32_t)));
    HC(hipGraphLaunch(ge, s));
    HC(hipStreamSynchronize(s));
    std::vector<uint32_t> h(NSLOTS);
    HC(hipMemcpy(h.data(), d_slots, NSLOTS * sizeof(uint32_t),
                 hipMemcpyDeviceToHost));
    std::set<int> replay;
    for (int i = 0; i < NSLOTS; ++i) if (h[i]) replay.insert(i);
    printf("  graph replay on the SAME masked stream:\n");
    print_slots(replay);
    printf("  -> mask %s under graph replay\n",
           replay.size() <= direct.size() + 1 ? "HELD" : "WAS DROPPED");

    hipStream_t plain;
    HC(hipStreamCreate(&plain));
    HC(hipMemset(d_slots, 0, NSLOTS * sizeof(uint32_t)));
    HC(hipGraphLaunch(ge, plain));
    HC(hipStreamSynchronize(plain));
    HC(hipMemcpy(h.data(), d_slots, NSLOTS * sizeof(uint32_t),
                 hipMemcpyDeviceToHost));
    std::set<int> onplain;
    for (int i = 0; i < NSLOTS; ++i) if (h[i]) onplain.insert(i);
    printf("  same graph replayed on an UNMASKED stream:\n");
    print_slots(onplain);

    hipGraphExecDestroy(ge);
    hipGraphDestroy(g);
    hipStreamDestroy(plain);
    hipStreamDestroy(s);
  }

  hipFree(d_slots);
  printf("\ndone\n");
  return 0;
}
