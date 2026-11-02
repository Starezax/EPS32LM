/*
 * dlm: інференс трансформера з int8-вагами (W8A32).
 * Архітектура відповідає llm/model.py: RMSNorm, RoPE, MHA, SwiGLU, tied embeddings.
 */
#include "dlm.h"

#include <math.h>
#include <string.h>

#define MAGIC_MODEL 0x314D4C44
#define MAGIC_TOK   0x314B4F54
#define HEADER_INTS 12
#define MAX_GEN     512

/* ------------------------------------------------------------ ініціалізація */

static int read_header(const void *data, DlmConfig *c)
{
    const int32_t *h = (const int32_t *)data;
    if (h[0] != MAGIC_MODEL || h[1] != 1) return -1;
    c->dim = h[2];
    c->hidden_dim = h[3];
    c->n_layers = h[4];
    c->n_heads = h[5];
    c->vocab_size = h[6];
    c->max_seq_len = h[7];
    c->group_size = h[8];
    if (c->n_layers > DLM_MAX_LAYERS || c->vocab_size > DLM_MAX_VOCAB) return -2;
    if (c->dim % c->n_heads || c->dim % c->group_size || c->hidden_dim % c->group_size) return -3;
    return 0;
}

size_t dlm_work_size(const void *model_data, int seq_len)
{
    DlmConfig c;
    if (read_header(model_data, &c)) return 0;
    if (seq_len <= 0 || seq_len > c.max_seq_len) seq_len = c.max_seq_len;
    size_t n = (size_t)4 * c.dim + 2 * c.hidden_dim + (size_t)c.n_heads * seq_len
             + c.vocab_size + (size_t)2 * c.n_layers * seq_len * c.dim;
    return n * sizeof(float);
}

static const uint8_t *take_q(DlmQ *t, const uint8_t *p, size_t n, int gs)
{
    t->q = (const int8_t *)p;
    p += n;
    t->s = (const float *)p;
    return p + (n / gs) * sizeof(float);
}

int dlm_init(Dlm *m, const void *model_data, size_t model_size,
             void *work, size_t work_size, int seq_len)
{
    memset(m, 0, sizeof(*m));
    int err = read_header(model_data, &m->cfg);
    if (err) return err;
    DlmConfig *c = &m->cfg;
    if (seq_len <= 0 || seq_len > c->max_seq_len) seq_len = c->max_seq_len;
    m->seq_len = seq_len;
    if (work_size < dlm_work_size(model_data, seq_len)) return -4;

    const uint8_t *p = (const uint8_t *)model_data + HEADER_INTS * 4;
    size_t D = c->dim, H = c->hidden_dim, L = c->n_layers;
    int gs = c->group_size;
    m->rms_att = (const float *)p;   p += L * D * 4;
    m->rms_ffn = (const float *)p;   p += L * D * 4;
    m->rms_final = (const float *)p; p += D * 4;
    p = take_q(&m->tok_emb, p, (size_t)c->vocab_size * D, gs);
    for (size_t l = 0; l < L; l++) {
        DlmLayer *ly = &m->layers[l];
        p = take_q(&ly->wq, p, D * D, gs);
        p = take_q(&ly->wk, p, D * D, gs);
        p = take_q(&ly->wv, p, D * D, gs);
        p = take_q(&ly->wo, p, D * D, gs);
        p = take_q(&ly->w1, p, H * D, gs);
        p = take_q(&ly->w2, p, D * H, gs);
        p = take_q(&ly->w3, p, H * D, gs);
    }
    if ((size_t)(p - (const uint8_t *)model_data) != model_size) return -5;

    float *w = (float *)work;
    m->x = w;        w += D;
    m->xb = w;       w += D;
    m->xb2 = w;      w += D;
    m->q = w;        w += D;
    m->hb = w;       w += H;
    m->hb2 = w;      w += H;
    m->att = w;      w += (size_t)c->n_heads * seq_len;
    m->logits = w;   w += c->vocab_size;
    m->key_cache = w;   w += L * seq_len * D;
    m->value_cache = w;
    return 0;
}

/* ------------------------------------------------------------ математика */

static void rmsnorm(float *o, const float *x, const float *w, int n)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) o[i] = x[i] * ss * w[i];
}

static void softmax(float *x, int n)
{
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* out[d] = W[d][n] * x[n], W у int8 з масштабом на групу */
static void matmul(float *out, const float *x, const DlmQ *w, int n, int d, int gs)
{
    int ng = n / gs;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < d; i++) {
        const int8_t *wq = w->q + (size_t)i * n;
        const float *ws = w->s + (size_t)i * ng;
        float sum = 0.0f;
        for (int g = 0; g < ng; g++) {
            const int8_t *qq = wq + g * gs;
            const float *xx = x + g * gs;
            float gsum = 0.0f;
            for (int j = 0; j < gs; j++) gsum += (float)qq[j] * xx[j];
            sum += gsum * ws[g];
        }
        out[i] = sum;
    }
}

float *dlm_forward(Dlm *m, int token, int pos)
{
    const DlmConfig *c = &m->cfg;
    int D = c->dim, H = c->hidden_dim, gs = c->group_size;
    int hd = D / c->n_heads, S = m->seq_len;
    float *x = m->x;

    /* embedding: розпакувати рядок token */
    const int8_t *eq = m->tok_emb.q + (size_t)token * D;
    const float *es = m->tok_emb.s + (size_t)token * D / gs;
    for (int i = 0; i < D; i++) x[i] = eq[i] * es[i / gs];

    for (int l = 0; l < c->n_layers; l++) {
        const DlmLayer *ly = &m->layers[l];
        float *kc = m->key_cache + ((size_t)l * S + pos) * D;
        float *vc = m->value_cache + ((size_t)l * S + pos) * D;

        rmsnorm(m->xb, x, m->rms_att + (size_t)l * D, D);
        matmul(m->q, m->xb, &ly->wq, D, D, gs);
        matmul(kc, m->xb, &ly->wk, D, D, gs);
        matmul(vc, m->xb, &ly->wv, D, D, gs);

        /* RoPE: пари (2i, 2i+1) усередині кожної голови */
        for (int i = 0; i < D; i += 2) {
            int hi = i % hd;
            float freq = 1.0f / powf(10000.0f, (float)hi / (float)hd);
            float v = pos * freq, fc = cosf(v), fs = sinf(v);
            float q0 = m->q[i], q1 = m->q[i + 1];
            m->q[i] = q0 * fc - q1 * fs;
            m->q[i + 1] = q0 * fs + q1 * fc;
            float k0 = kc[i], k1 = kc[i + 1];
            kc[i] = k0 * fc - k1 * fs;
            kc[i + 1] = k0 * fs + k1 * fc;
        }

        /* увага по головах */
        float scale = 1.0f / sqrtf((float)hd);
        for (int h = 0; h < c->n_heads; h++) {
            const float *qh = m->q + h * hd;
            float *att = m->att + (size_t)h * S;
            for (int t = 0; t <= pos; t++) {
                const float *k = m->key_cache + ((size_t)l * S + t) * D + h * hd;
                float sc = 0.0f;
                for (int i = 0; i < hd; i++) sc += qh[i] * k[i];
                att[t] = sc * scale;
            }
            softmax(att, pos + 1);
            float *o = m->xb + h * hd;
            memset(o, 0, hd * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                const float *v = m->value_cache + ((size_t)l * S + t) * D + h * hd;
                float a = att[t];
                for (int i = 0; i < hd; i++) o[i] += a * v[i];
            }
        }
        matmul(m->xb2, m->xb, &ly->wo, D, D, gs);
        for (int i = 0; i < D; i++) x[i] += m->xb2[i];

        /* FFN (SwiGLU) */
        rmsnorm(m->xb, x, m->rms_ffn + (size_t)l * D, D);
        matmul(m->hb, m->xb, &ly->w1, D, H, gs);
        matmul(m->hb2, m->xb, &ly->w3, D, H, gs);
        for (int i = 0; i < H; i++) {
            float a = m->hb[i];
            m->hb[i] = a / (1.0f + expf(-a)) * m->hb2[i];
        }
        matmul(m->xb, m->hb, &ly->w2, H, D, gs);
        for (int i = 0; i < D; i++) x[i] += m->xb[i];
    }

    rmsnorm(x, x, m->rms_final, D);
    matmul(m->logits, x, &m->tok_emb, D, c->vocab_size, gs);
    return m->logits;
}

/* ------------------------------------------------------------ токенізатор */

int dlm_tok_init(DlmTokenizer *t, const void *data, size_t size)
{
    const int32_t *h = (const int32_t *)data;
    if (size < 12 || h[0] != MAGIC_TOK || h[1] > DLM_MAX_VOCAB) return -1;
    t->vocab_size = h[1];
    t->data = (const uint8_t *)data;
    uint32_t off = 12;
    for (int i = 0; i < t->vocab_size; i++) {
        if (off >= size) return -2;
        t->offset[i] = off;
        off += 1 + t->data[off];
    }
    return off == size ? 0 : -3;
}

const char *dlm_tok_piece(const DlmTokenizer *t, int id, int *len)
{
    const uint8_t *p = t->data + t->offset[id];
    *len = p[0];
    return (const char *)(p + 1);
}

int dlm_tok_find(const DlmTokenizer *t, const char *piece)
{
    int n = (int)strlen(piece);
    for (int i = 0; i < t->vocab_size; i++) {
        int len;
        const char *s = dlm_tok_piece(t, i, &len);
        if (len == n && memcmp(s, piece, n) == 0) return i;
    }
    return -1;
}

static int is_special(const DlmTokenizer *t, int id)
{
    if (id < 3) return 1;   /* unk, bos, eos */
    int len;
    const char *s = dlm_tok_piece(t, id, &len);
    return len >= 3 && s[0] == '<' && s[len - 1] == '>';
}

/* ------------------------------------------------------------ семплінг */

static uint32_t rng_u32(uint64_t *st)
{
    *st ^= *st >> 12;
    *st ^= *st << 25;
    *st ^= *st >> 27;
    return (uint32_t)((*st * 0x2545F4914F6CDD1DULL) >> 32);
}

static float rng_f32(uint64_t *st)
{
    return (rng_u32(st) >> 8) / 16777216.0f;
}

int dlm_sample(DlmSampler *s, float *logits, int n, const int *history, int n_history)
{
    if (s->rep_penalty > 1.0f) {
        for (int i = 0; i < n_history; i++) {
            float *l = &logits[history[i]];
            *l = (*l > 0) ? *l / s->rep_penalty : *l * s->rep_penalty;
        }
    }
    if (s->temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    /* top-k: k найбільших логітів (вставкою в маленький відсортований масив) */
    enum { KMAX = 128 };
    int k = (s->top_k > 0 && s->top_k < KMAX) ? s->top_k : KMAX;
    if (k > n) k = n;
    int idx[KMAX];
    float val[KMAX];
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        float v = logits[i];
        if (cnt == k && v <= val[k - 1]) continue;
        int j = (cnt < k) ? cnt++ : k - 1;
        while (j > 0 && val[j - 1] < v) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; }
        val[j] = v;
        idx[j] = i;
    }
    float sum = 0.0f;
    for (int i = 0; i < cnt; i++) {
        val[i] = expf((val[i] - val[0]) / s->temperature);
        sum += val[i];
    }
    float r = rng_f32(&s->rng) * sum, acc = 0.0f;
    for (int i = 0; i < cnt; i++) {
        acc += val[i];
        if (r < acc) return idx[i];
    }
    return idx[cnt - 1];
}

/* ------------------------------------------------------------ генерація */

int dlm_generate(Dlm *m, const DlmTokenizer *t, DlmSampler *s,
                 const int *prompt, int n_prompt, int max_new,
                 char *out, int out_size, DlmPieceFn on_piece, void *user)
{
    int hist[MAX_GEN];
    int n_hist = 0, out_len = 0, pos = 0, generated = 0;
    float *logits = 0;
    if (out_size > 0) out[0] = 0;
    if (n_prompt <= 0 || n_prompt >= m->seq_len) return 0;

    for (int i = 0; i < n_prompt; i++) logits = dlm_forward(m, prompt[i], pos++);

    while (generated < max_new && pos < m->seq_len && n_hist < MAX_GEN) {
        int tok = dlm_sample(s, logits, m->cfg.vocab_size, hist, n_hist);
        if (tok == DLM_EOS || tok == DLM_BOS) break;
        hist[n_hist++] = tok;
        generated++;
        if (!is_special(t, tok)) {
            int len;
            const char *p = dlm_tok_piece(t, tok, &len);
            if (out_len == 0) while (len > 0 && *p == ' ') { p++; len--; }   /* без пробілу на початку */
            if (out_len + len < out_size) {
                memcpy(out + out_len, p, len);
                out_len += len;
                out[out_len] = 0;
            }
            if (on_piece && len > 0) on_piece(p, len, user);
        }
        logits = dlm_forward(m, tok, pos++);
    }
    return generated;
}
