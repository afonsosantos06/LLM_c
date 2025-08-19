#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXCHAR 1000

Img** csv_to_imgs(char* file_string, int number_of_imgs) {
	FILE *fp;
	Img** imgs = malloc(number_of_imgs * sizeof(Img*));
	char row[MAXCHAR];
	fp = fopen(file_string, "r");

	// Reads the first line 
	fgets(row, MAXCHAR, fp);
	int i = 0;
	while (feof(fp) != 1 && i < number_of_imgs) {
		imgs[i] = malloc(sizeof(Img));

		int j = 0;
		fgets(row, MAXCHAR, fp);
		char* token = strtok(row, ",");
		imgs[i]->img_data = create_matrix(28, 28);
		while (token != NULL) {
			if (j == 0) {
				imgs[i]->label = atoi(token);
			} else {
				imgs[i]->img_data->entries[(j-1) / 28][(j-1) % 28] = atoi(token) / 256.0;
			}
			token = strtok(NULL, ",");
			j++;
		}
		i++;
	}
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