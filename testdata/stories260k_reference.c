#define TESTING
#include "run.c"

static void print_prefix_logits(Transformer *model, const int *tokens, int count) {
    printf("  {\"tokens\": [");
    for (int pos = 0; pos < count; pos++) {
        printf("%s%d", pos ? ", " : "", tokens[pos]);
    }
    printf("], \"logits\": [\n");
    for (int pos = 0; pos < count; pos++) {
        float *logits = forward(model, tokens[pos], pos);
        printf("    [\n      ");
        for (int id = 0; id < model->config.vocab_size; id++) {
            if (id > 0) {
                printf("%s", id % 8 == 0 ? ",\n      " : ", ");
            }
            printf("%.9g", logits[id]);
        }
        printf("\n    ]%s\n", pos + 1 < count ? "," : "");
    }
    printf("  ]}");
}

static void print_logits(char *checkpoint_path) {
    Transformer model = {0};
    build_transformer(&model, checkpoint_path);
    const int single[] = {1};
    const int multiple[] = {1, 3, 42, 3};

    printf("[\n");
    print_prefix_logits(&model, single, 1);
    printf(",\n");
    print_prefix_logits(&model, multiple, 4);
    printf("\n]\n");

    free_transformer(&model);
}

static void print_hex(const char *bytes, size_t length) {
    for (size_t i = 0; i < length; i++) {
        printf("%02x", (unsigned int)(unsigned char)bytes[i]);
    }
}

static void print_tokenizer_case(Tokenizer *tokenizer, const char *name, char *prompt) {
    size_t prompt_length = strlen(prompt);
    int *ids = malloc((prompt_length + 2) * sizeof(*ids));
    if (ids == NULL) {
        fputs("could not allocate prompt IDs\n", stderr);
        exit(EXIT_FAILURE);
    }
    int count;
    encode(tokenizer, prompt, 1, 0, ids, &count);

    printf("  {\"name\": \"%s\", \"prompt_hex\": \"", name);
    print_hex(prompt, prompt_length);
    printf("\", \"prompt_ids\": [");
    for (int i = 0; i < count; i++) {
        printf("%s%d", i ? ", " : "", ids[i]);
    }
    printf("], \"decoded_hex\": \"");
    for (int i = 1; i < count; i++) {
        char *piece = decode(tokenizer, ids[i - 1], ids[i]);
        // Each fallback ID represents one byte, including the NUL byte.
        size_t length = ids[i] >= 3 && ids[i] < 259 ? 1 : strlen(piece);
        print_hex(piece, length);
    }
    printf("\"}");

    free(ids);
}

static void print_tokenizer(char *tokenizer_path) {
    const struct {
        const char *name;
        char *prompt;
    } cases[] = {
        {"empty", ""},
        {"plain", "Once upon a time"},
        {"whitespace", "  \t\n "},
        {"unicode", "Caf\xc3\xa9 \xf0\x9f\xa6\x99\xe2\x80\x8a"},
    };
    Tokenizer tokenizer = {0};
    build_tokenizer(&tokenizer, tokenizer_path, 512);

    printf("[\n");
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (i > 0) {
            printf(",\n");
        }
        print_tokenizer_case(&tokenizer, cases[i].name, cases[i].prompt);
    }
    printf("\n]\n");

    free_tokenizer(&tokenizer);
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "logits") == 0) {
        print_logits(argv[2]);
    } else if (argc == 3 && strcmp(argv[1], "tokenizer") == 0) {
        print_tokenizer(argv[2]);
    } else {
        fprintf(stderr, "usage: %s logits CHECKPOINT | tokenizer TOKENIZER\n", argv[0]);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
