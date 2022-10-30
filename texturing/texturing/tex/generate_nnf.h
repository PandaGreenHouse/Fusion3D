#pragma once
#include <opencv\cv.h>
#include <opencv\highgui.h>
struct NNF_Data {
	int patch_size;
	float radius;
	IplImage* source;
	IplImage* dest;
	IplImage* nnf;
};

IplImage* generate_compeletness(IplImage* src_img, IplImage* targ_img, int patch_size, int nPass);
IplImage* generate_coherence(IplImage* src_img, IplImage* targ_img, int patch_size, int nPass);





