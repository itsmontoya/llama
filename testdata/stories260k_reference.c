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

int main(int argc, char **argv) {
    if (argc != 3 || strcmp(argv[1], "logits") != 0) {
        fprintf(stderr, "usage: %s logits CHECKPOINT\n", argv[0]);
        return EXIT_FAILURE;
    }

    Transformer model = {0};
    build_transformer(&model, argv[2]);
    const int single[] = {1};
    const int multiple[] = {1, 3, 42, 3};

    printf("[\n");
    print_prefix_logits(&model, single, 1);
    printf(",\n");
    print_prefix_logits(&model, multiple, 4);
    printf("\n]\n");

    free_transformer(&model);
    return EXIT_SUCCESS;
}
