// Compare little-endian FP32 artifacts on the experiment's little-endian host.
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ascending(const void *a, const void *b) {
	double left = *(const double *)a;
	double right = *(const double *)b;
	return (left > right) - (left < right);
}

static float *read_trajectory(FILE *file, size_t values, const char *path) {
	float *row = malloc(values * sizeof(*row));
	if (!row || fread(row, sizeof(*row), values, file) != values) {
		fprintf(stderr, "%s: could not read %zu values\n", path, values);
		exit(EXIT_FAILURE);
	}
	return row;
}

static void verify_repeats(FILE *file, float *baseline, size_t values, size_t repetitions, const char *path) {
	for (size_t i = 1; i < repetitions; ++i) {
		float *other = read_trajectory(file, values, path);
		if (memcmp(baseline, other, values * sizeof(*other)) != 0) {
			fprintf(stderr, "%s: phase or lane differs from first trajectory\n", path);
			exit(EXIT_FAILURE);
		}
		free(other);
	}
	if (fgetc(file) != EOF || ferror(file)) {
		fprintf(stderr, "%s: unexpected additional values or read failure\n", path);
		exit(EXIT_FAILURE);
	}
	fclose(file);
}

int main(int argc, char **argv) {
	if (argc != 5) {
		fprintf(stderr, "usage: compare_rows cpu.f32 metal.f32 rows lanes\n");
		return EXIT_FAILURE;
	}
	size_t rows = strtoul(argv[3], NULL, 10);
	size_t lanes = strtoul(argv[4], NULL, 10);
	if (rows < 1 || rows > 33 || lanes < 1 || lanes > 2) {
		fprintf(stderr, "rows must be 1..33 and lanes 1..2\n");
		return EXIT_FAILURE;
	}
	size_t count = rows * 128256;
	FILE *cpu = fopen(argv[1], "rb");
	FILE *metal = fopen(argv[2], "rb");
	if (!cpu || !metal) {
		fprintf(stderr, "open logits: %s\n", strerror(errno));
		return EXIT_FAILURE;
	}
	float *expected = read_trajectory(cpu, count, argv[1]);
	float *actual = read_trajectory(metal, count, argv[2]);
	verify_repeats(cpu, expected, count, lanes * 2, argv[1]);
	verify_repeats(metal, actual, count, lanes * 2, argv[2]);
	double *errors = malloc(count * sizeof(*errors));
	if (!errors) {
		perror("allocate errors");
		return EXIT_FAILURE;
	}
	double squared = 0, maxFraction = 0;
	size_t violations = 0;
	for (size_t i = 0; i < count; ++i) {
		if (!isfinite(expected[i]) || !isfinite(actual[i])) {
			fprintf(stderr, "nonfinite row %zu ID %zu\n", i / 128256, i % 128256);
			return EXIT_FAILURE;
		}
		double error = fabs((double)actual[i] - expected[i]);
		double fraction = error / (1e-4 + 1e-4 * fabs((double)expected[i]));
		if (fraction > 1 && violations++ == 0) {
			fprintf(stderr, "first tolerance violation at row %zu ID %zu: CPU %.9g Metal %.9g fraction %.9g\n",
				i / 128256, i % 128256, expected[i], actual[i], fraction);
		}
		maxFraction = fmax(maxFraction, fraction);
		squared += error * error;
		errors[i] = error;
	}
	qsort(errors, count, sizeof(*errors), ascending);
	printf("{\"values\":%zu,\"unique_positions\":%zu,\"p50\":%.12g,\"p95\":%.12g,"
	       "\"max\":%.12g,\"rmse\":%.12g,\"max_fraction\":%.12g,\"violations\":%zu}\n",
		count * 2 * lanes, count, errors[(count - 1) / 2], errors[(count - 1) * 95 / 100],
		errors[count - 1], sqrt(squared / count), maxFraction, violations * 2 * lanes);
	free(errors);
	free(actual);
	free(expected);
	return EXIT_SUCCESS;
}
