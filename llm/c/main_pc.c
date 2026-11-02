/*
 * Запуск dlm на ПК.
 *
 * Генерація фраз:
 *   dlm.exe model.bin tokenizer.bin -c C -n 10 -t 0.8 -k 40 -r 1.1 -s 42
 * Перевірка (для llm/check_c.py): прогнати токени і надрукувати логіти:
 *   dlm.exe model.bin tokenizer.bin --dump 1 4 812 33
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dlm.h"

static void *read_file(const char *path, size_t *size)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Не можу відкрити %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    *size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc(*size);
    if (!buf || fread(buf, 1, *size, f) != *size) { fprintf(stderr, "Помилка читання %s\n", path); exit(1); }
    fclose(f);
    return buf;
}

static void print_piece(const char *p, int len, void *user)
{
    (void)user;
    fwrite(p, 1, len, stdout);
    fflush(stdout);
}

static double now_s(void)
{
    return (double)clock() / CLOCKS_PER_SEC;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.bin tokenizer.bin [-c CAT] [-n N] [-t TEMP] [-k TOPK]"
                        " [-r REP] [-s SEED] [-l SEQ] [--dump ids...]\n", argv[0]);
        return 1;
    }
    const char *cat = 0;
    int n = 5, top_k = 40, seq_len = 0, dump_from = 0;
    float temp = 0.8f, rep = 1.1f;
    unsigned long long seed = (unsigned long long)time(0);
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--dump")) { dump_from = i + 1; break; }
        if (i + 1 >= argc) break;
        if (!strcmp(argv[i], "-c")) cat = argv[++i];
        else if (!strcmp(argv[i], "-n")) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t")) temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-k")) top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r")) rep = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-s")) seed = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "-l")) seq_len = atoi(argv[++i]);
    }

    size_t msize, tsize;
    void *mdata = read_file(argv[1], &msize);
    void *tdata = read_file(argv[2], &tsize);

    static Dlm model;
    static DlmTokenizer tok;
    size_t wsize = dlm_work_size(mdata, seq_len);
    if (!wsize) { fprintf(stderr, "Невірний формат model.bin\n"); return 1; }
    void *work = malloc(wsize);   /* на ESP32 тут буде статичний буфер у PSRAM */
    int err = dlm_init(&model, mdata, msize, work, wsize, seq_len);
    if (err) { fprintf(stderr, "dlm_init помилка %d\n", err); return 1; }
    if (dlm_tok_init(&tok, tdata, tsize)) { fprintf(stderr, "Невірний tokenizer.bin\n"); return 1; }

    if (dump_from) {
        int pos = 0;
        for (int i = dump_from; i < argc; i++, pos++) {
            float *lg = dlm_forward(&model, atoi(argv[i]), pos);
            for (int v = 0; v < model.cfg.vocab_size; v++) printf("%.6f ", lg[v]);
            printf("\n");
        }
        return 0;
    }

    const DlmConfig *c = &model.cfg;
    fprintf(stderr, "dim %d, hidden %d, layers %d, heads %d, vocab %d, seq %d | "
                    "ваги %.2f MB, робоча пам'ять %.2f MB\n",
            c->dim, c->hidden_dim, c->n_layers, c->n_heads, c->vocab_size, model.seq_len,
            msize / 1e6, wsize / 1e6);

    int prompt[2] = {DLM_BOS, 0};
    int n_prompt = 1;
    if (cat) {
        char tag[8];
        snprintf(tag, sizeof(tag), "<%s>", cat);
        prompt[1] = dlm_tok_find(&tok, tag);
        if (prompt[1] < 0) { fprintf(stderr, "Немає категорії %s\n", tag); return 1; }
        n_prompt = 2;
    }

    DlmSampler s = {temp, top_k, rep, seed * 2654435761ULL + 1};
    char out[1024];
    int total_tok = 0;
    double t0 = now_s();
    for (int i = 0; i < n; i++) {
        printf("- ");
        total_tok += dlm_generate(&model, &tok, &s, prompt, n_prompt, 120,
                                  out, sizeof(out), print_piece, 0);
        printf("\n");
    }
    double dt = now_s() - t0;
    fprintf(stderr, "%d токенів за %.2f с (%.1f ток/с)\n", total_tok, dt, total_tok / (dt > 0 ? dt : 1));
    free(work);
    free(mdata);
    free(tdata);
    return 0;
}
