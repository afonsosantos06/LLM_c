#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXCHAR 1000

Img** csv_to_imgs(char* file_string, int number_of_imgs) {
	FILE *fp = fopen(file_string, "r");
	if (fp == NULL) {
		perror("Failed to open CSV file");
		exit(EXIT_FAILURE);
	}

	Img** imgs = malloc(number_of_imgs * sizeof(Img*));
	if (imgs == NULL) {
		perror("Failed to allocate imgs array");
		fclose(fp);
		exit(EXIT_FAILURE);
	}

	char *line = NULL;
	size_t cap = 0;
	ssize_t len = 0;

	// skip header line
	len = getline(&line, &cap, fp);
	if (len == -1) {
		fprintf(stderr, "CSV appears to be empty: %s\n", file_string);
		free(imgs);
		fclose(fp);
		exit(EXIT_FAILURE);
	}

	int i = 0;
	while (i < number_of_imgs && (len = getline(&line, &cap, fp)) != -1) {
		imgs[i] = malloc(sizeof(Img));
		if (imgs[i] == NULL) {
			perror("Failed to allocate Img");
			break;
		}
		imgs[i]->img_data = create_matrix(28, 28);
		if (imgs[i]->img_data == NULL) {
			perror("Failed to allocate image matrix");
			break;
		}

		int j = 0;
		char *token = strtok(line, ",");
		while (token != NULL) {
			if (j == 0) {
				imgs[i]->label = atoi(token);
				if (imgs[i]->label < 0 || imgs[i]->label > 9) {
					fprintf(stderr, "Warning: invalid label %d at image %d. Clamping to range 0..9.\n", imgs[i]->label, i);
					if (imgs[i]->label < 0) imgs[i]->label = 0;
					if (imgs[i]->label > 9) imgs[i]->label = imgs[i]->label % 10;
				}
			} else {
				int pixel_index = j - 1;
				int r = pixel_index / 28;
				int c = pixel_index % 28;
				if (r < 28 && c < 28) {
					imgs[i]->img_data->entries[r][c] = atoi(token) / 256.0;
				}
			}
			token = strtok(NULL, ",");
			j++;
		}
		// zero-fill remaining pixels if the row was short
		for (; j <= 28 * 28; j++) {
			int pixel_index = j - 1;
			int r = pixel_index / 28;
			int c = pixel_index % 28;
			if (j == 0) continue;
			if (r < 28 && c < 28) {
				imgs[i]->img_data->entries[r][c] = 0.0;
			}
		}

		i++;
	}

	free(line);
	fclose(fp);
	return imgs;
}
void print_img(Img *img){
  print_matrix(img->img_data);
  printf("Img Label: %d\n", img->label);
}

void free_img(Img *img){
  free_matrix(img->img_data);
  free(img);
  img = NULL;
}

void free_imgs(Img **imgs, int n){
  for (int i = 0; i < n; i++)
    free_img(imgs[i]);
  free(imgs);
  imgs = NULL;
}