/*
 * dlm: інференс маленького трансформера (llama2.c-стиль) з int8-вагами.
 * Без динамічної пам'яті: ваги читаються напряму з переданого буфера
 * (на ESP32 це mmap розділу flash), робочу пам'ять передає викликач.
 */
#ifndef DLM_H
#define DLM_H

#include <stddef.h>
#include <stdint.h>

#define DLM_MAX_LAYERS 16
#define DLM_MAX_VOCAB  8192

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, vocab_size, max_seq_len, group_size;
} DlmConfig;

/* int8-тензор: q[n], s[n / group_size] */
typedef struct {
    const int8_t *q;
    const float *s;
} DlmQ;

typedef struct {
    DlmQ wq, wk, wv, wo, w1, w2, w3;
} DlmLayer;

typedef struct {
    DlmConfig cfg;
    int seq_len;                 /* робочий ліміт контексту (<= max_seq_len) */
    const float *rms_att, *rms_ffn, *rms_final;
    DlmQ tok_emb;                /* також вихідний шар */
    DlmLayer layers[DLM_MAX_LAYERS];
    /* робочі буфери (всередині work) */
    float *x, *xb, *xb2, *hb, *hb2, *q, *att, *logits, *key_cache, *value_cache;
} Dlm;

typedef struct {
    int vocab_size;
    const uint8_t *data;
    uint32_t offset[DLM_MAX_VOCAB];   /* зсув рядка токена в data (байт довжини + текст) */
} DlmTokenizer;

typedef struct {
    float temperature;    /* 0 = жадібно (argmax) */
    int top_k;            /* 0 = без обмеження */
    float rep_penalty;    /* 1.0 = вимкнено; >1 штрафує вже згенеровані токени */
    uint64_t rng;
} DlmSampler;

/* Розмір робочої пам'яті (байт) для заданого seq_len. 0 = помилка формату. */
size_t dlm_work_size(const void *model_data, int seq_len);

/* 0 = OK, <0 = помилка. model_data має жити весь час роботи. */
int dlm_init(Dlm *m, const void *model_data, size_t model_size,
             void *work, size_t work_size, int seq_len);

/* Один крок: токен на позиції pos -> логіти (vocab_size float). */
float *dlm_forward(Dlm *m, int token, int pos);

int dlm_tok_init(DlmTokenizer *t, const void *data, size_t size);
/* Текст токена (не нуль-термінований), довжина в *len */
const char *dlm_tok_piece(const DlmTokenizer *t, int id, int *len);
/* id токена за точним текстом (напр. "<C>"), -1 якщо немає */
int dlm_tok_find(const DlmTokenizer *t, const char *piece);

int dlm_sample(DlmSampler *s, float *logits, int n, const int *history, int n_history);

/* Колбек на кожен новий шматок тексту (для "друкування" по буквах). */
typedef void (*DlmPieceFn)(const char *piece, int len, void *user);

/*
 * Генерує текст після prompt (напр. {BOS, <C>}) до EOS/BOS або max_new токенів.
 * Результат у out (нуль-термінований UTF-8). Повертає кількість згенерованих токенів.
 */
int dlm_generate(Dlm *m, const DlmTokenizer *t, DlmSampler *s,
                 const int *prompt, int n_prompt, int max_new,
                 char *out, int out_size, DlmPieceFn on_piece, void *user);

#define DLM_BOS 1
#define DLM_EOS 2

#endif
