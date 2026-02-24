/* Microgpt from scratch in ANSI C by @y0b1byte
 * I compiled this with `gcc -std=gnu89 -pedantic -Wall -Wextra main.c -lm -g -o gpt`
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>

#define VALUE_POOL_SIZE 100000

#define N_EMBD 16 /* embedding dim */
#define N_HEAD 4  /* number of attn heads*/
#define N_LAYER 1  /* number of layers */
#define BLOCK_SIZE 16  /* max seq len */
#define HEAD_DIM (N_EMBD / N_HEAD)

#define TRAIN_STEPS 1000
#define LR 0.01
#define BETA1 0.85
#define BETA2 0.99
#define EPS_ADAM 0.00000001
#define W_INIT_STD 0.08

#define MAX_VOCAB 100
#define TEMPERATURE 0.5 
#define INFERENCE_SAMPLES 20 

#define TWO_PI (M_PI*2)

double standard_normal() {
    /* 
     * https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform 
     * Box Mueller makes 2 standard normal vars from 2 uniform ones.
     * We cache one of them.
    */

    static int has_cached = 0;
    static double cached = 0;
    double u1,u2;
    double z;

    if (has_cached) {
        has_cached = 0;
        return cached;
    }

    u1 = (double) rand() / RAND_MAX;
    u2 = (double) rand() / RAND_MAX;
    z = sqrt(-2*log(u1)) * cos(TWO_PI*u2);
    cached = sqrt(-2*log(u1)) * sin(TWO_PI*u2);
    has_cached = 1;
    return z;
}

typedef struct Value {
    double data; /* forward pass */
    double grad ; 
    size_t num_children;
    struct Value* children[3]; /* children in the comp graph */
    double local_grads[3]; /* derivative of node w.r.t children*/
    int visited;
} Value;

typedef struct {
    Value data[VALUE_POOL_SIZE];
    size_t idx; /* Next available idx */
} ValuePool;
ValuePool comp_pool;
ValuePool w_pool;

Value *v_from_double(double d, ValuePool *pool) {
    Value *v = &pool->data[pool->idx++];
    v->data = d;
    v->grad = 0;
    v->num_children = 0;
    v->visited = 0;
    return v;
}

Value *v_add(Value *v, Value *other) {
    Value *res = v_from_double(v->data + other->data, &comp_pool);
    res->grad = 0;
    res->visited = 0;
    res->data = v->data + other->data;
    res->num_children = 2;
    res->children[0] = v;
    res->children[1] = other;
    res->local_grads[0] = 1;
    res->local_grads[1] = 1;
    return res;
}

Value *v_sum(Value **v, size_t n) {
    Value *res = v_from_double(0., &comp_pool);
    size_t i;
    for(i = 0; i < n; i++) {
        res = v_add(res, v[i]);
    }
    return res;
}

Value *v_double_add(Value *v, double other) {
    return v_add(v, v_from_double(other, &comp_pool));
}

Value *v_mul(Value *v, Value *other) {
    Value *res = &comp_pool.data[comp_pool.idx++];
    res->grad = 0;
    res->visited = 0;
    res->data = v->data * other->data;
    res->num_children = 2;
    res->children[0] = v;
    res->children[1] = other;
    res->local_grads[0] = other->data;
    res->local_grads[1] = v->data;
    return res;
}

Value *v_double_mul(Value *v, double other) {
    return v_mul(v, v_from_double(other, &comp_pool));
}

Value *v_double_pow(Value *v, double other) {
    Value *res = v_from_double(pow(v->data, other), &comp_pool);
    res->num_children = 1;
    res->children[0] = v;
    res->local_grads[0] = other * pow(v->data, other - 1);
    return res;
}
Value *v_neg(Value *v) {
    return v_double_mul(v, -1);
}
Value *v_double_sub(Value *v, double other) {
    return v_add(v, v_neg(v_from_double(other, &comp_pool)));
}
Value *v_sub(Value *v, Value *other) {
    return v_add(v, v_neg(other));
}

Value *v_div(Value *v, Value *other) {
    return v_mul(v, v_double_pow(other, -1));
}
Value *v_double_div(Value *v, double other) {
    return v_double_mul(v, pow(other, -1));
}

Value *v_log(Value *v) {
    Value *res = v_from_double(v->data, &comp_pool);
    res->data = log(v->data);
    res->num_children = 1;
    res->children[0] = v;
    res->local_grads[0] = 1 / v->data;
    res->grad = 0;
    res->visited = 0;
    return res;
}

Value *v_exp(Value *v) {
    Value *res = v_from_double(exp(v->data), &comp_pool);
    res->num_children = 1;
    res->children[0] = v;
    res->local_grads[0] = exp(v->data);
    return res;
}

Value *v_relu(Value *v) {
    Value *res = v_from_double((v->data > 0) ? v->data: 0, &comp_pool);
    res->num_children = 1;
    res->children[0] = v;
    res->local_grads[0] = (v->data > 0) ? 1 : 0;
    res->grad = 0;
    res->visited = 0;
    return res;
}

typedef struct {
    Value **data;
    size_t last_idx; /* last used index */
    size_t size; /* allocated size in items, not in bytes */
} Topo;

void build_topo(Value *node, Topo *topo) {
    /* Mom, can we have sets? We have sets at home... */
    size_t i;
    if (node->visited != 1) {
        node->visited = 1;
        for(i = 0; i < node->num_children; i++) {
            build_topo(node->children[i], topo);
        }
        /* The node is not visited yet. */
        if (topo->last_idx == topo->size) {
            /* We do not have enough space, allocate. */
            topo->data = realloc(topo->data, sizeof(Value *) * topo->size * 2);
            topo->size*=2;
        }
        topo->data[topo->last_idx++] = node;
    }
}

void v_backward(Value *loss, size_t num_params) {
    /* We will have at least num_params nodes in the computation graph. */
    size_t i,j;
    Topo topo;
    topo.size = num_params;
    topo.last_idx = 0;
    topo.data = malloc(sizeof(Value*) * num_params);
    build_topo(loss, &topo);
    loss->grad = 1;
    for(i = topo.last_idx; i-- > 0;) {
        /* reversed_topo */
        for(j = 0; j < topo.data[i]->num_children; j++) {
            topo.data[i]->children[j]->grad += topo.data[i]->local_grads[j] * topo.data[i]->grad;
        }
    }
    for(i = 0; i < topo.last_idx; i++) {
        topo.data[i] -> visited = 0;
    }
    free(topo.data);
}
typedef struct {
    Value **data;
    size_t rows;
    size_t cols;
} Matrix;

Matrix *get_weight_matrix(size_t rows, size_t cols) {
    size_t i,j;
    Matrix *mat;
    Value **data;
    mat = malloc(sizeof(Matrix));;
    data = malloc(sizeof(Value*) * rows * cols);
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            data[i * cols + j] = v_from_double(standard_normal() * W_INIT_STD, &w_pool);
        }
    }
    mat->data = data;
    mat->rows = rows;
    mat->cols = cols;
    return mat;
}

void free_mat(Matrix *mat) {
    free(mat->data);
    free(mat);
}

typedef struct {
    Matrix *attn_wq;
    Matrix *attn_wk;
    Matrix *attn_wv;
    Matrix *attn_wo;
    Matrix *mlp_fc1;
    Matrix *mlp_fc2;
} Layer;

typedef struct {
    Matrix *wte;    
    Matrix *wpe;    
    Matrix *lm_head;    
    Layer **layers;
    size_t n_params;
} Network;

Network *init_network(size_t vocab_size) {
    Network *net = malloc(sizeof(Network));
    Layer **layers = malloc(sizeof(Layer*) * N_LAYER);
    Layer *cur_layer;
    size_t layer_idx = 0;
    
    /* n_out x n_in */
    net->wte = get_weight_matrix(vocab_size, N_EMBD);
    net->wpe = get_weight_matrix(BLOCK_SIZE, N_EMBD);
    net->lm_head= get_weight_matrix(vocab_size, N_EMBD);

    for(layer_idx = 0; layer_idx < N_LAYER; layer_idx++) {
        cur_layer = malloc(sizeof(Layer));
        cur_layer->attn_wq = get_weight_matrix(N_EMBD, N_EMBD);
        cur_layer->attn_wk = get_weight_matrix(N_EMBD, N_EMBD);
        cur_layer->attn_wv = get_weight_matrix(N_EMBD, N_EMBD);
        cur_layer->attn_wo = get_weight_matrix(N_EMBD, N_EMBD);
        cur_layer->mlp_fc1 = get_weight_matrix(4 * N_EMBD, N_EMBD);
        cur_layer->mlp_fc2 = get_weight_matrix(N_EMBD, 4 * N_EMBD);
        layers[layer_idx] = cur_layer;

    }
    net->layers = layers;
    net->n_params = w_pool.idx;

    return net;
}

void free_network(Network *net) {
    size_t layer_idx;
    free_mat(net->wte);
    free_mat(net->wpe);
    free_mat(net->lm_head);
    /* We do not need to free the Value items as they are in the pool.*/
    for(layer_idx = 0; layer_idx < N_LAYER; layer_idx++) {
        free_mat(net->layers[layer_idx]->attn_wq);
        free_mat(net->layers[layer_idx]->attn_wk);
        free_mat(net->layers[layer_idx]->attn_wv);
        free_mat(net->layers[layer_idx]->attn_wo);
        free_mat(net->layers[layer_idx]->mlp_fc1);
        free_mat(net->layers[layer_idx]->mlp_fc2);
        free(net->layers[layer_idx]);
    }
    free(net->layers);
    free(net);
}

typedef struct {
    double *m; 
    double *v; 
} Adam;

Adam init_adam(size_t num_params) {
    /* First and second moment bufs */
    double *m = calloc(num_params, sizeof(double));    
    double *v = calloc(num_params, sizeof(double));    
    Adam opt;    
    opt.m = m;
    opt.v = v;
    return opt;
}

typedef struct {
    char *vocab;    
    size_t bos;    
    size_t vocab_size;
} Tokenizer;

Tokenizer init_tokenizer(char **data, size_t data_size) {
    size_t i,j,k;
    size_t vocab_size = 0;
    Tokenizer tok;
    /* We can dynamically allocate here */
    char *vocab = malloc(MAX_VOCAB);
    for(i = 0; i < data_size; i++) {
        for(j = 0; j < strlen(data[i]); j++) {
            for(k = 0; k < vocab_size; k++) {
                if (vocab[k] == data[i][j]) {
                    /* Already present, ignore */
                    break;
                }
            }
            if (k == vocab_size) {
                /* We did not break above, add. */
                vocab[vocab_size] = data[i][j];
                vocab_size++;
                if (vocab_size == MAX_VOCAB) {
                    fprintf(stderr, "Max vocab size is exhausted. Increase MAX_VOCAB\n");
                }
            }
        }
    }
    tok.vocab = vocab;
    tok.bos = vocab_size;
    tok.vocab_size = vocab_size + 1;
    return tok;
}

size_t *tokenize(char* text, Tokenizer *tok) {
    size_t i,j;
    size_t textlen = strlen(text);
    size_t *toks = malloc(sizeof(size_t)*(textlen + 2));
    toks[0] = tok->bos;
    toks[textlen + 1] = tok->bos;
    for(i = 0; i < textlen; i++) {
        for(j = 0; j < tok->vocab_size - 1; j++) {
            if (text[i] == tok->vocab[j]) {
                toks[i + 1] = j;
                break;
            }
        }
        if (j == tok->vocab_size - 1) {
            fprintf(stderr, "Unknown character %c to tokenize.\n", text[i]);
            exit(1);
        }
    }
    return toks;
}

char *detokenize(size_t *tokens, size_t len, Tokenizer *tok) {
    char *text = malloc(len + 1);
    size_t i;
    text[len] = '\0';
    for(i = 0; i < len; i++) {
        /* Do not check for the BOS */
        if(tokens[i] == tok->bos) {
            text[i] = ' ';
        } else {
            text[i] = tok->vocab[tokens[i]];
        }
    }
    return text;
}

void shuffle(char **data, unsigned int len) {
    /* The idea is swap every element with another random element from the upper array. 
    https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle 
    https://benpfaff.org/writings/clc/shuffle.html 
    */

    /* 0 is nothing to swap, 1 is nobody to swap with.*/
    char *tmp;
    if (len > 1) {
        size_t i, j;
        /* Last element is nobody to swap with */
        for(i = 0; i < len - 1; i++) {
            j = i + rand() / (RAND_MAX / (len - i) + 1);
            tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }
}

size_t sample(double* probs, size_t n) {
    /* Let's generate a number from 0 to 1
     * to go over cumulative sum of probs to quit 
     * where we are above.
     * We assume that probs are normalised!
     */
    double u = (double) rand() / (double) RAND_MAX;
    double p = 0;

    size_t i;
    for (i = 0; i < n; i++) {
        p += probs[i];
        if (u < p) {
            return i;
        }
    }
    return n - 1;
}

size_t get_data(char ***result) {
    /* We will download the data in the next version of this program. 

    For now, get it from:
        https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt
    */

    FILE *f = fopen("names.txt", "r");
    char **data;
    char buf[1024];
    unsigned int data_size = 0;

    /* Let's count first. It'll be a bit more as we filter.
       But for simplicity, let's keep it like that. 
       I assume we could fit a line into a buf.
       We could make it more robust and merge chunks that do not fit a line,
       but this is too much.
    */
    while (fgets(buf, sizeof(buf), f)) {
        data_size++;
    } 
    fclose(f);
    data = malloc(sizeof(char*) * data_size);

    data_size = 0; /* Reset for the second pass. */
    f = fopen("names.txt", "r");
    while (fgets(buf, sizeof(buf), f)) {
        size_t chunk_len = strlen(buf);
        size_t i = 0;
        size_t left_spaces = 0;
        size_t right_spaces = 0;
        size_t point_len = 0;
        while (isspace(buf[i++])) {
            left_spaces++;
        }
        i = chunk_len - 2; /* -1 will be \n */
        while (isspace(buf[i--])) {
            right_spaces++;
        }
        /* No -1 to keep \0 */
        point_len = chunk_len - right_spaces - left_spaces;
        if (point_len > 1) {
            data[data_size] = malloc(point_len);
            memcpy(data[data_size], buf + left_spaces, point_len);
            data[data_size][point_len - 1] = '\0';
            data_size++;
        }
    }
    fclose(f);
    data_size--; /* We did not insert for the last increment. */
    shuffle(data, data_size);
    *result = data;
    return data_size;
}

void v_linear(Value **x, Matrix *w, size_t n, size_t k, Value **res) {
    /* This is only to multiply vec of N elems by a matrix of KxN elems.
     *                  m11 .. m1k 
     * x1 x2 .. xn x    m21 .. m2k  
     *                      ..      = r1 r2 ... rk 
     *                      .. 
     *                  mn1 .. mnk
     */
    size_t i, j;
    for (i = 0; i < k; i++) {
        Value *elem = v_from_double(0., &comp_pool);
        for (j = 0; j < n; j++) {
            elem = v_add(elem, v_mul(w->data[i * w->cols + j], x[j])); 
        }
        res[i] = elem;
    }
}
Value **v_softmax(Value **logits, size_t n) {
    size_t i;
    double max_val = -DBL_MAX;
    Value **res = malloc(sizeof(Value*) * n);
    Value *total = v_from_double(0, &comp_pool);

    for(i = 0; i < n; i++) {
        max_val = (max_val < logits[i]->data) ? logits[i]->data : max_val;
    }
    for(i = 0; i < n; i++) {
        res[i] = v_exp(v_double_sub(logits[i], max_val));
        total = v_add(total, res[i]);    
    }
    for(i = 0; i < n; i++) {
        res[i] = v_div(res[i], total);
    }
    return res;
}

Value **v_rmsnorm(Value **x, size_t n) {
    size_t i;
    Value **res = malloc(sizeof(Value*) * n);
    Value *ms = v_from_double(0, &comp_pool);
    for(i = 0; i < n; i++) {
        ms = v_add(ms, v_mul(x[i], x[i]));
    }
    ms = v_double_div(ms, (double) n);
    ms = v_double_add(ms, 1e-5);
    ms = v_double_pow(ms, -0.5); 
    for(i = 0; i < n; i++) {
        res[i] = v_mul(x[i], ms);
    }
    return res; 
}

void gpt(size_t token_id, size_t pos_id, Network *net, Value *keys[N_LAYER][BLOCK_SIZE][N_EMBD], Value *values[N_LAYER][BLOCK_SIZE][N_EMBD], size_t vocab_size, Value *logits[MAX_VOCAB]) {
    /* t is the time dimension, in microgpt, they send keys/values and we can read lens of those.
     * Here we use t to track that. 
     */
    Value **x = malloc(sizeof(Value*) * N_EMBD);
    Value **x_residual;
    Value *q[N_EMBD], *v[N_EMBD], *k[N_EMBD];
    Value *qk_prod[HEAD_DIM];
    Value *x_attn[N_HEAD*HEAD_DIM];
    size_t i;
    size_t layer_idx = 0;
    for(i = 0; i < N_EMBD; i++) {
        x[i] = v_add(net->wte->data[token_id * net->wte->cols + i], net->wpe->data[pos_id * net->wpe->cols + i]);
    }
    x = v_rmsnorm(x, N_EMBD);
    for(layer_idx = 0; layer_idx < N_LAYER; layer_idx++) {
        Layer *layer = net->layers[layer_idx];
        size_t head_idx;

        x_residual = x;
        x = v_rmsnorm(x, N_EMBD);
        v_linear(x, layer->attn_wq, N_EMBD, N_EMBD, q);
        v_linear(x, layer->attn_wk, N_EMBD, N_EMBD, k);
        v_linear(x, layer->attn_wv, N_EMBD, N_EMBD, v);
        for(i = 0; i < N_EMBD; i++) {
            keys[layer_idx][pos_id][i] = k[i];
            values[layer_idx][pos_id][i] = v[i];
        }
        for(head_idx = 0; head_idx < N_HEAD; head_idx++) {
            size_t it, id;
            size_t hs = head_idx * HEAD_DIM;
            
            Value **attn_weights = malloc(sizeof(Value*) * (pos_id + 1));
            for(it = 0; it <= pos_id; it++) {
                for(id = 0; id < HEAD_DIM; id++) {
                    qk_prod[id] = v_mul(q[hs + id], keys[layer_idx][it][hs + id]);
                }
                attn_weights[it] = v_double_div(v_sum(qk_prod, HEAD_DIM), pow(HEAD_DIM, 0.5));
            }
            attn_weights = v_softmax(attn_weights, pos_id + 1);
            for(id = 0; id < HEAD_DIM; id++) {
                Value **head_attn_weights = malloc(sizeof(Value*) * (pos_id +1));
                for(it = 0; it <= pos_id; it++) {
                    head_attn_weights[it] = v_mul(attn_weights[it], values[layer_idx][it][hs + id]);
                }
                x_attn[hs + id] = v_sum(head_attn_weights, pos_id + 1);
            }
        }
        v_linear(x_attn, layer->attn_wo, N_EMBD, N_EMBD, x);
        for(i = 0; i < N_EMBD; i++) {
            x[i] = v_add(x[i], x_residual[i]);
        }
        x_residual = x; /* 2) MLP block */
        x = v_rmsnorm(x, N_EMBD);
        v_linear(x, layer->mlp_fc1, N_EMBD, N_EMBD, x);
        for(i = 0; i < N_EMBD; i++) {
            x[i] = v_relu(x[i]);
        }
        v_linear(x, layer->mlp_fc2, N_EMBD, N_EMBD, x);
        for(i = 0; i < N_EMBD; i++) {
            x[i] = v_add(x[i], x_residual[i]);
        }
    }
    v_linear(x, net->lm_head, N_EMBD, vocab_size, logits);
}

int main() {
    char **data = 0;
    char *cur_data;
    size_t *tokens;
    size_t step = 0;
    Value *loss;
    Tokenizer tok;
    size_t data_size = 0;
    Network *net;
    Adam opt;
    Value *keys[N_LAYER][BLOCK_SIZE][N_EMBD];
    Value *values[N_LAYER][BLOCK_SIZE][N_EMBD];
    Value *logits[MAX_VOCAB];

    srand(42);
    data_size = get_data(&data);
    printf("Dataset size: %lu \n", data_size);

    tok = init_tokenizer(data, data_size);
    printf("Vocab size: %lu \n", tok.vocab_size);
    {
        size_t i = 0;
        printf("Vocabulary: ");
        for(i = 0; i < tok.vocab_size - 1; i++) {
            printf("%c ", tok.vocab[i]);
        }
        printf("BOS");
        printf("\n");
    }
    
    net = init_network(tok.vocab_size);
    opt = init_adam(net->n_params);
    printf("num params: %lu\n", net->n_params);

    for(step = 0; step < TRAIN_STEPS; step++) {
        size_t n, n_toks;
        size_t i;
        comp_pool.idx = 0; /* Reset comp_pool for non-weight nodes */
        loss = v_from_double(0., &comp_pool);
        cur_data = data[step % data_size];
        n_toks = strlen(cur_data) + 2; /* +2 for BOS */
        n = (n_toks - 1 < BLOCK_SIZE) ? n_toks: BLOCK_SIZE;
        tokens = tokenize(cur_data, &tok);

        for(i = 0; i < n - 1; i++) {
            gpt(tokens[i], i, net, keys, values, tok.vocab_size, logits);
            memcpy(logits, v_softmax(logits, tok.vocab_size), sizeof(Value*)*MAX_VOCAB);
            loss = v_add(loss, v_neg(v_log(logits[tokens[i+1]])));
        }
        loss = v_double_div(loss, n);
        printf("step %4lu / %4d | loss %.4f | data: %s\n",step+1, TRAIN_STEPS, loss->data, cur_data);
        free(tokens);
        {
            double lr_t = LR * (1 - (double)step / TRAIN_STEPS);
            size_t i;
            double m_hat, v_hat;
            v_backward(loss, net->n_params);
            for(i = 0; i < net->n_params; i++) {
                Value *p = &w_pool.data[i];
                opt.m[i] = BETA1 * opt.m[i] + (1 - BETA1) * p->grad;
                opt.v[i] = BETA2 * opt.v[i] + (1 - BETA2) * p->grad * p->grad;
                m_hat = opt.m[i] / (1 - pow(BETA1, (step + 1)));
                v_hat = opt.v[i] / (1 - pow(BETA2, (step + 1)));
                p->data -= lr_t * m_hat / (pow(v_hat, 0.5) + EPS_ADAM);
                p->grad = 0;
            }
        }
    }

    printf("\n--- inference (new, hallucinated names) ---\n");
    {
        size_t sample_idx = 0;
        size_t token_id = 0;
        double probs[MAX_VOCAB];
        char *sampled_text;
        size_t tokens[BLOCK_SIZE];
        size_t i, j;
        Value *keys[N_LAYER][BLOCK_SIZE][N_EMBD];
        Value *values[N_LAYER][BLOCK_SIZE][N_EMBD];
        while (sample_idx < INFERENCE_SAMPLES) {
            comp_pool.idx = 0; /* Reset comp_pool for non-weight nodes */
            token_id = tok.bos;

            for(i = 0; i < BLOCK_SIZE; i++) {
                gpt(token_id, i, net, keys, values, tok.vocab_size, logits);
                for(j = 0; j < tok.vocab_size; j++) {
                    logits[j] = v_double_div(logits[j], TEMPERATURE); 
                }
                memcpy(logits, v_softmax(logits, tok.vocab_size), sizeof(Value*)*MAX_VOCAB);
                for(j = 0; j < tok.vocab_size; j++) {
                    probs[j] = logits[j]->data;   
                }
                token_id = sample(probs, tok.vocab_size);
                tokens[i] = token_id;
                if (token_id == tok.bos) {
                    break;
                }
            }
            /* Decrement if we haven't hit break. */
            sampled_text = detokenize(tokens, (i == BLOCK_SIZE) ? BLOCK_SIZE - 1: i, &tok);
            printf("Sample %3lu: %s\n", sample_idx + 1, sampled_text);
            free(sampled_text);
            sample_idx++;
        }
    }

    {
        size_t i = 0;
        free(tok.vocab);
        for(i = 0; i < data_size; i++) {
            free(data[i]);
        }
        free(data);
        free_network(net);
    }
    return 0;
}

