// Experimental DeepSeek-V4.1 decode kernels, compiled after the family's own (tf-dsv41-forward DS_EXP=this file).
// `_ab_` variants are ablations (wrong output; only their time means anything). The others are candidates: the same
// arithmetic in the same order as the kernel they stand in for, so their logits must equal its bit for bit.

// ---------------------------------------------------------------- ds_attend_split ablations

#define SPLIT_ARGS                                                                                                   \
  const device bfloat *Q [[buffer(0)]], const device bfloat *RING [[buffer(1)]],                                    \
      const device bfloat *COMP [[buffer(2)]], const device int *PICKS [[buffer(3)]], device float *PO [[buffer(4)]], \
      device float2 *PML [[buffer(5)]], constant AttnArgs &a [[buffer(6)]], const device int *POS [[buffer(7)]],     \
      const device float *FREQ [[buffer(8)]], uint3 tg [[threadgroup_position_in_grid]],                            \
      uint3 tgn [[threadgroups_per_grid]], uint lane [[thread_index_in_simdgroup]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]], uint3 tpos [[thread_position_in_threadgroup]]

// ds_attend_split's body with phases switched off: LOAD (keys into threadgroup memory), ROPE (the query's),
// SCORE (dots and softmax), OUT (the weighted sum and its writes).
template <bool LOAD, bool ROPE, bool SCORE, bool OUT>
inline void split_body(const device bfloat* Q, const device bfloat* RING, const device bfloat* COMP,
                       const device int* PICKS, device float* PO, device float2* PML, constant AttnArgs& a,
                       const device int* POS, const device float* FREQ, uint3 tg, uint3 tgn, uint lane, uint sg,
                       uint3 tpos, threadgroup bfloat (*ks)[HD], threadgroup float (*sc)[SPLIT]) {
  const uint t = tpos.x;
  const int h = int(tg.x) * HG + int(sg), chunk = int(tg.y), r = int(tg.z), chunks = int(tgn.y);
  const int pos = a.window_end >= 0 ? a.window_end : a.first_pos + r;
  const int nwin = min(a.window, pos + 1);
  const int nkeys = nwin + a.picks;
  const int k0 = chunk * SPLIT, k1 = min(nkeys, k0 + SPLIT);
  const device int* picks = PICKS + size_t(r) * a.picks;
  if (LOAD)
    for (int i = int(t); i < SPLIT * (HD / 8); i += HG * 32) {
      const int kk = i / (HD / 8), c8 = i % (HD / 8), j = k0 + kk;
      uint4 v = uint4(0);
      if (j < k1) v = ((const device uint4*)attn_key(RING, COMP, picks, pos, nwin, a.ring, j))[c8];
      ((threadgroup uint4*)ks[kk])[c8] = v;
    }
  float q[16];
  const device bfloat* qr = Q + (size_t(r) * 64 + h) * HD;
  for (int i = 0; i < 16; i++) q[i] = float(qr[int(lane) * 16 + i]);
  if (ROPE && lane >= uint((HD - 64) / 16)) {
    for (int p = 0; p < 8; p++) {
      const float ang = float(POS[r]) * FREQ[(int(lane) - (HD - 64) / 16) * 8 + p];
      const float cs = precise::cos(ang), sn = precise::sin(ang);
      const float re = q[2 * p], im = q[2 * p + 1];
      q[2 * p] = float(bfloat(re * cs - im * sn));
      q[2 * p + 1] = float(bfloat(re * sn + im * cs));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = -1e30f, l = 0.0f;
  if (SCORE) {
    for (int kk = 0; kk < k1 - k0; kk++) {
      float d = 0.0f;
      for (int i = 0; i < 16; i++) d += q[i] * float(ks[kk][int(lane) * 16 + i]);
      d = simd_sum(d);
      const int j = k0 + kk;
      const bool valid = j < nwin || picks[j - nwin] >= 0;
      if (lane == 0) sc[sg][kk] = valid ? d * a.scale : -INFINITY;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = int(lane); j < k1 - k0; j += 32) m = max(m, sc[sg][j]);
    m = simd_max(m);
    for (int j = int(lane); j < k1 - k0; j += 32) {
      const float p = precise::exp(sc[sg][j] - m);
      sc[sg][j] = p;
      l += p;
    }
    l = simd_sum(l);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  } else {
    m = q[0];
  }
  float o[16];
  for (int i = 0; i < 16; i++) o[i] = 0.0f;
  if (OUT)
    for (int kk = 0; kk < k1 - k0; kk++) {
      const float p = sc[sg][kk];
      for (int i = 0; i < 16; i++) o[i] += p * float(ks[kk][int(lane) * 16 + i]);
    }
  const size_t part = (size_t(r) * 64 + h) * chunks + chunk;
  for (int i = 0; i < 16; i++) PO[part * HD + int(lane) * 16 + i] = o[i] + q[i];
  if (lane == 0) PML[part] = float2(m, l);
}

#define SPLIT_VARIANT(name, L, R, S, O)                                                                     \
  [[kernel]] void name(SPLIT_ARGS) {                                                                         \
    threadgroup bfloat ks[SPLIT][HD];                                                                        \
    threadgroup float sc[HG][SPLIT];                                                                         \
    split_body<L, R, S, O>(Q, RING, COMP, PICKS, PO, PML, a, POS, FREQ, tg, tgn, lane, sg, tpos, ks, sc);    \
  }
SPLIT_VARIANT(ds_attend_split_ab_all, true, true, true, true)
SPLIT_VARIANT(ds_attend_split_ab_noload, false, true, true, true)
SPLIT_VARIANT(ds_attend_split_ab_norope, true, false, true, true)
SPLIT_VARIANT(ds_attend_split_ab_noscore, true, true, false, true)
SPLIT_VARIANT(ds_attend_split_ab_noout, true, true, true, false)
SPLIT_VARIANT(ds_attend_split_ab_none, false, false, false, false)

// ---------------------------------------------------------------- harness

// A cache-cold replay's preamble (DS_COLD): read n uint4 of src, so earlier lines leave the caches.
[[kernel]] void ds_exp_thrash(const device uint4* src [[buffer(0)]], device uint* sink [[buffer(1)]],
                              constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]],
                              uint gsz [[threads_per_grid]]) {
  uint4 acc = 0;
  for (uint i = gid; i < n; i += gsz) acc += src[i];
  if (acc.x == 0xdeadbeefu) sink[0] = acc.y;
}

// ---------------------------------------------------------------- ds_attend_split candidates (RoPE table)

// Candidate: ds_attend_split_t with the chunk's 16 dots computed before their 16 simd_sums (the same dots and sums,
// independent of each other, so they overlap instead of running one key after another).
[[kernel]] void ds_attend_split_t_ilp(SPLIT_ARGS) {
  const uint t = tpos.x;
  const int h = int(tg.x) * HG + int(sg), chunk = int(tg.y), r = int(tg.z), chunks = int(tgn.y);
  const int pos = a.window_end >= 0 ? a.window_end : a.first_pos + r;
  const int nwin = min(a.window, pos + 1);
  const int nkeys = nwin + a.picks;
  const int k0 = chunk * SPLIT, k1 = min(nkeys, k0 + SPLIT);
  const device int* picks = PICKS + size_t(r) * a.picks;
  threadgroup bfloat ks[SPLIT][HD];
  threadgroup float sc[HG][SPLIT];
  for (int i = int(t); i < SPLIT * (HD / 8); i += HG * 32) {
    const int kk = i / (HD / 8), c8 = i % (HD / 8), j = k0 + kk;
    uint4 v = uint4(0);
    if (j < k1) v = ((const device uint4*)attn_key(RING, COMP, picks, pos, nwin, a.ring, j))[c8];
    ((threadgroup uint4*)ks[kk])[c8] = v;
  }
  float q[16];
  const device bfloat* qr = Q + (size_t(r) * 64 + h) * HD;
  for (int i = 0; i < 16; i++) q[i] = float(qr[int(lane) * 16 + i]);
  if (lane >= uint((HD - 64) / 16)) {
    for (int p = 0; p < 8; p++) {
      const float2 csn = rope_cs<true>(FREQ, POS, r, (int(lane) - (HD - 64) / 16) * 8 + p);
      const float cs = csn.x, sn = csn.y;
      const float re = q[2 * p], im = q[2 * p + 1];
      q[2 * p] = float(bfloat(re * cs - im * sn));
      q[2 * p + 1] = float(bfloat(re * sn + im * cs));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int n = k1 - k0;
  float dots[SPLIT];
  for (int kk = 0; kk < SPLIT; kk++) {
    float d = 0.0f;
    for (int i = 0; i < 16; i++) d += q[i] * float(ks[kk][int(lane) * 16 + i]);
    dots[kk] = d;
  }
  for (int kk = 0; kk < SPLIT; kk++) dots[kk] = simd_sum(dots[kk]);
  if (lane == 0)
    for (int kk = 0; kk < n; kk++) {
      const int j = k0 + kk;
      const bool valid = j < nwin || picks[j - nwin] >= 0;
      sc[sg][kk] = valid ? dots[kk] * a.scale : -INFINITY;
    }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  float m = -1e30f;
  for (int j = int(lane); j < n; j += 32) m = max(m, sc[sg][j]);
  m = simd_max(m);
  float l = 0.0f;
  for (int j = int(lane); j < n; j += 32) {
    const float p = precise::exp(sc[sg][j] - m);
    sc[sg][j] = p;
    l += p;
  }
  l = simd_sum(l);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  float o[16];
  for (int i = 0; i < 16; i++) o[i] = 0.0f;
  for (int kk = 0; kk < n; kk++) {
    const float p = sc[sg][kk];
    for (int i = 0; i < 16; i++) o[i] += p * float(ks[kk][int(lane) * 16 + i]);
  }
  const size_t part = (size_t(r) * 64 + h) * chunks + chunk;
  for (int i = 0; i < 16; i++) PO[part * HD + int(lane) * 16 + i] = o[i];
  if (lane == 0) PML[part] = float2(m, l);
}

// ---------------------------------------------------------------- ds_hc_expand_mix candidate

// Candidate: ds_hc_expand_mix with the slice's mix weights (FN, static) loaded before the expand, so their DRAM
// latency overlaps it; the products and sums are the same, in the same order.
[[kernel]] void ds_hc_expand_mix_pf(const device bfloat* X [[buffer(0)]], const device bfloat* BR [[buffer(1)]],
                                    const device float* POST [[buffer(2)]], const device float* COMB [[buffer(3)]],
                                    const device float* FN [[buffer(4)]], device float* PART [[buffer(5)]],
                                    device bfloat* XO [[buffer(6)]], const device bfloat* Y [[buffer(7)]],
                                    const device bfloat* SH [[buffer(8)]], constant int2& topk_rows [[buffer(9)]],
                                    uint2 tg [[threadgroup_position_in_grid]],
                                    uint2 tpos [[thread_position_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[8][MIX + 1];
  const int j = int(tg.x), r = int(tg.y), topk = topk_rows.x;
  const uint t = tpos.x;
  const int col = 1024 * j + 4 * int(t);
  float f[MIX][4];
  for (int m = 0; m < MIX; m++)
    for (int i = 0; i < 4; i++) f[m][i] = FN[size_t(m) * F + col + i];
  const int k = col / D, d0 = col % D;
  const device bfloat* x = X + size_t(r) * F;
  const device float* cm = COMB + r * S * S;
  float v[4];
  for (int i = 0; i < 4; i++) {
    const int d = d0 + i;
    const float b = topk > 0 ? combined(Y, SH, topk, r, d) : float(BR[size_t(r) * D + d]);
    float xs[S];
    for (int jj = 0; jj < S; jj++) xs[jj] = float(x[jj * D + d]);
    float mm = 0.0f;
    for (int jj = 0; jj < S; jj++) mm += cm[jj * S + k] * xs[jj];
    const bfloat o = bfloat(POST[r * S + k] * b + mm);
    XO[size_t(r) * F + col + i] = o;
    v[i] = float(o);
  }
  float acc[MIX + 1];
  acc[MIX] = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
  for (int m = 0; m < MIX; m++) acc[m] = f[m][0] * v[0] + f[m][1] * v[1] + f[m][2] * v[2] + f[m][3] * v[3];
  for (int m = 0; m <= MIX; m++) {
    const float s = simd_sum(acc[m]);
    if (lane == 0) red[sg][m] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) <= MIX) {
    float s = 0.0f;
    for (int q = 0; q < 8; q++) s += red[q][t];
    PART[(size_t(r) * SLICES + j) * (MIX + 1) + t] = s;
  }
}
