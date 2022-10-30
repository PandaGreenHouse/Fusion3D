#include "texturing.h"
#define ALPHA 0.5
TEX_NAMESPACE_BEGIN
/*
created at 08/15/2017
*/
float rand_float(float min, float max)
{
	return min + (max - min)*(float)rand() / RAND_MAX;
}
/*
created at 08/15/2017
*/
float rand_minus1_1()
{
	return rand_float(0.0f, 2.0f) - 1.0f;
}
/*
created at 08/15/2017
*/
void get_random_points(float x, float y, float radius, std::vector<math::Vec2f> &samples)
{
	float r = radius;
	int i = 0;
	while (r > 1.0f)
	{
		math::Vec2f sample;
		sample[0] = r*rand_minus1_1() + x;
		sample[1] = r*rand_minus1_1() + y;
		samples.push_back(sample);
		r = std::powf(ALPHA, i)*radius;
	}
}
/*
created at 08/15/2017
*/
float get_square_sample_diff(IplImage* src_img, IplImage* targ_img, int patch_width, int patch_height,
							math::Vec2f &src_sample, math::Vec2f &targ_sample)
{
	float squared_diff = 0.0f;
	int count = 0;
	for (int iy = -patch_height/2; iy < patch_height/2; iy++)
	{
		for (int ix = -patch_width/2; ix < patch_width/2; ix++)
		{
			int sx = src_sample[0] + ix;
			int sy = src_sample[1] + iy;
			int tx = targ_sample[0] + ix;
			int ty = targ_sample[1] + iy;
			float diff = CV_IMAGE_ELEM(src_img, unsigned char, sy, sx) - CV_IMAGE_ELEM(targ_img, unsigned char, ty, tx);
			squared_diff += std::powf(diff, 2);
			count++;
		}
	}
	return squared_diff/count;
}
/*
created at 08/16/2017
*/
float random_min_loc_search(IplImage* src_img, IplImage* targ_img, math::Vec2f &src_point, math::Vec2f &targ_seed, 
	int patch_width, int patch_height, float radius, math::Vec2f &minLoc)
{
	std::vector<math::Vec2f> random_points;
	get_random_points(targ_seed[0], targ_seed[1], radius, random_points);
	float min_diff = FLT_MAX;
	for (int i = 0; i < random_points.size(); i++)
	{
		float diff = get_square_sample_diff(src_img, targ_img, patch_width, patch_height, src_point, random_points[i]);
		if (diff < min_diff)
		{
			min_diff = diff;
			minLoc = random_points[i];
		}
	}
	return min_diff;
}
/*
created at 08/15/2017
*/
IplImage* init_nnf_image(IplImage* src_img, IplImage* targ_img, int patch_width, int patch_height, float radius)
{
	int width = src_img->width;
	int height = src_img->height;
	IplImage* nnf_img = cvCreateImage(cvSize(src_img->width, src_img->height), IPL_DEPTH_32F, 3);
	cvSet(nnf_img, cvScalar(FLT_MAX));
	for (int y = patch_height/2; y < height - patch_height/2; y++)
	{
		for (int x = patch_width/2; x < width - patch_width/2; x++)
		{
			math::Vec2f src_point(x, y);
			math::Vec2f targ_seed(x, y);
			math::Vec2f minLoc;
			float diff = random_min_loc_search(src_img, targ_img, src_point, targ_seed, patch_width, patch_height, radius, minLoc);
			CV_IMAGE_ELEM(nnf_img, float, y, 3 * x) = minLoc[0] - x;
			CV_IMAGE_ELEM(nnf_img, float, y, 3 * x + 1) = minLoc[1] - y;
			CV_IMAGE_ELEM(nnf_img, float, y, 3 * x + 2) = diff;
		}
	}
	return nnf_img;
}
/*
description:
update the nnf_image of source image from previous nnf_image, target image and source
created at 08/15/2017
modified at 08/16/2017
*/
void update_nnf_image(IplImage* nnf_img, IplImage* src_img, IplImage* targ_img, int patch_width, int patch_height, float radius)
{
	int width = src_img->width;
	int height = src_img->height;
	float squared_diff[2];
	for (int y = patch_height / 2; y < height - patch_height / 2; y++)
	{
		for (int x = patch_width / 2; x < width - patch_width / 2; x++)
		{
			float tx, ty;
			//propagation
			squared_diff[0] = CV_IMAGE_ELEM(nnf_img, float, y - 1, 3 * x + 2);
			squared_diff[1] = CV_IMAGE_ELEM(nnf_img, float, y, 3 * (x - 1) + 2);
			if (squared_diff[0] < squared_diff[1])
			{
				tx = x + CV_IMAGE_ELEM(nnf_img, float, y - 1, 3 * x);
				ty = y + CV_IMAGE_ELEM(nnf_img, float, y - 1, 3 * x + 1);
			}
			else
			{
				tx = x + CV_IMAGE_ELEM(nnf_img, float, y, 3 * (x-1));
				ty = y + CV_IMAGE_ELEM(nnf_img, float, y, 3 * (x-1) + 1);
			}
			//random search
			math::Vec2f src_point(x, y);
			math::Vec2f targ_seed(tx, ty);
			math::Vec2f minLoc;
			float diff = random_min_loc_search(src_img, targ_img, src_point, targ_seed, patch_width, patch_height, radius, minLoc);
			CV_IMAGE_ELEM(nnf_img, float, y, 3 * x) = minLoc[0] - x;
			CV_IMAGE_ELEM(nnf_img, float, y, 3 * x + 1) = minLoc[1] - y;
			CV_IMAGE_ELEM(nnf_img, float, y, 3 * x + 2) = diff;
		}
	}
}
/*
created at 08/15/2017
*/
IplImage* compute_nnf_image(IplImage* src_img, IplImage* targ_img, int patch_width, int patch_height, float radius, int nIterations)
{
	IplImage* nnf_img = init_nnf_image(src_img, targ_img, patch_width, patch_height, radius);
	int nLoops = nIterations;
	int width = src_img->width;
	int height = src_img->height;
	while (nLoops > 0)
	{
		update_nnf_image(nnf_img, src_img, targ_img, patch_width, patch_height, radius);
		nLoops--;
	}
	return nnf_img;
}
/*
created at 08/15/2017
updated at 08/16/2017
*/
IplImage* generate_compelete_image(IplImage* src_img, IplImage* targ_img, int patch_width, int patch_height, float radius, int nIterations)
{
	int width = src_img->width;
	int height = src_img->height;
	IplImage* src_grey_img = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	IplImage* targ_grey_img = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	cvConvertImage(src_img, src_grey_img, CV_RGB2GRAY);
	cvConvertImage(targ_img, targ_grey_img, CV_RGB2GRAY);
	IplImage* nnf_img = compute_nnf_image(src_grey_img, targ_grey_img, patch_width, patch_height, radius, nIterations);
	cvReleaseImage(&src_grey_img);
	cvReleaseImage(&targ_grey_img);
	IplImage* compeletness_img = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 4);
	cvZero(compeletness_img);
	float L = patch_width*patch_height;
	for (int y = patch_height / 2; y < height - patch_height / 2; y++)
	{
		for (int x = patch_width / 2; x < width - patch_width / 2; x++)
		{
			int sx = x - patch_width / 2;
			int sy = y - patch_height / 2;
			int tx = x + CV_IMAGE_ELEM(nnf_img, float, y, 3 * x) - patch_width/2;
			int ty = y + CV_IMAGE_ELEM(nnf_img, float, y, 3 * x+1) - patch_height/2;
			for (int iy = 0; iy < patch_height; iy++)
			{
				for (int ix = 0; ix < patch_width; ix++)
				{
					float* compele_ptr = &CV_IMAGE_ELEM(compeletness_img, float, ty+iy, 4 * (tx+ix));
					unsigned char* src_ptr = &CV_IMAGE_ELEM(src_img, unsigned char, sy+iy, 3 * (sx+ix));
					*compele_ptr += *src_ptr / L; compele_ptr++; src_ptr++;
					*compele_ptr += *src_ptr / L; compele_ptr++; src_ptr++;
					*compele_ptr += *src_ptr / L; compele_ptr++;
					*compele_ptr += 1 / L;
				}
			}
		}
	}
	return compeletness_img;
}
/*
created at 08/16/2017
*/
IplImage* generate_coherent_image(IplImage* src_img, IplImage* targ_img, int patch_width, int patch_height, float radius, int nIterations)
{
	int width = src_img->width;
	int height = src_img->height;
	IplImage* src_grey_img = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	IplImage* targ_grey_img = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	cvConvertImage(src_img, src_grey_img, CV_RGB2GRAY);
	cvConvertImage(targ_img, targ_grey_img, CV_RGB2GRAY);
	IplImage* nnf_img = compute_nnf_image(targ_grey_img, src_grey_img, patch_width, patch_height, radius, nIterations);
	cvReleaseImage(&src_grey_img);
	cvReleaseImage(&targ_grey_img);
	IplImage* coherent_img = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 4);
	cvZero(coherent_img);
	float a = 2.0f/(patch_width*patch_height);
	for (int y = patch_height / 2; y < height - patch_height / 2; y++)
	{
		for (int x = patch_width / 2; x < width - patch_width / 2; x++)
		{
			int tx = x - patch_width / 2;
			int ty = y - patch_height / 2;
			int sx = x + CV_IMAGE_ELEM(nnf_img, float, y, 3 * x) - patch_width / 2;
			int sy = y + CV_IMAGE_ELEM(nnf_img, float, y, 3 * x + 1) - patch_height / 2;
			for (int iy = 0; iy < patch_height; iy++)
			{
				for (int ix = 0; ix < patch_width; ix++)
				{
					float* coherent_ptr = &CV_IMAGE_ELEM(coherent_img, float, ty, 4 * tx);
					unsigned char* src_ptr = &CV_IMAGE_ELEM(src_img, unsigned char, sy, 3 * sx);
					*coherent_ptr += *src_ptr*a; coherent_ptr++; src_ptr++;
					*coherent_ptr += *src_ptr*a; coherent_ptr++; src_ptr++;
					*coherent_ptr += *src_ptr*a; coherent_ptr++;
					*coherent_ptr += a;
				}
			}
		}
	}
	return coherent_img;
}
TEX_NAMESPACE_END