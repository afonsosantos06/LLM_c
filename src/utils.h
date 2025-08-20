#ifndef UTILS
#define UTILS
#include "../matrix/matrix.h"

typedef struct img {
  Matrix *img_data;
  int label;
} Img; 

Img **csv_to_imgs(char *file_string, int number_of_imgs); // load image and convert it into matrix returning an array of img structs containing a 2D image representation
void print_img(Img *img); 
void free_img(Img *img);
void free_imgs(Img **imgs, int n);
void shuffle_imgs(Img **imgs, int n);

#endif