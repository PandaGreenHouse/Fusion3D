/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <fstream>
#include <cstring>
#include <cstdint>
#include <cerrno>

#include <util/exception.h>
#include <util/file_system.h>
#include <mve/image_io.h>

#include "material_lib.h"
#include <iostream>

void
MaterialLib::save_to_files(std::string const & prefix) const {
    std::string filename = prefix + ".mtl";
    std::ofstream out(filename.c_str());
    if (!out.good())
		throw util::FileException(filename, "MaterialLib::save_to_files");// std::strerror(errno));

    std::string const name = util::fs::basename(prefix);

    for (Material const & material : *this) {
        std::string diffuse_map_postfix = "_" + material.name + "_map_Kd.png";
        out << "newmtl " << material.name << std::endl
            << "Ka 1.000000 1.000000 1.000000" << std::endl
            << "Kd 1.000000 1.000000 1.000000" << std::endl
            << "Ks 0.000000 0.000000 0.000000" << std::endl
            << "Tr 1.000000" << std::endl
            << "illum 1" << std::endl
            << "Ns 1.000000" << std::endl
            << "map_Kd " << name + diffuse_map_postfix << std::endl;
    }
    out.close();
	
    for (Material const & material : *this) {
        std::string filename = prefix + "_" + material.name + "_map_Kd.png";
		std::cout << "width:" << material.diffuse_map.get()->width() << std::endl;
		std::cout << "height:" << material.diffuse_map.get()->height() << std::endl;
		mve::image::save_png_file(material.diffuse_map, filename);
// 		save_img_file(material.diffuse_map, filename);
	}
}

// void MaterialLib::save_img_file(mve::ByteImage::ConstPtr diffuse_map, std::string const& filename) const {
// 	int width = diffuse_map->width();
// 	int height = diffuse_map->height();
// 	std::vector<uint8_t> src_data = diffuse_map->get_data();
// 	IplImage* dst_img = cvCreateImage(cvSize(width, height), 8, 3);
// 	unsigned int count = 0;
// 	unsigned char r, g, b;
// 	for (int y = 0; y < height; y++)
// 	{
// 		for (int x = 0; x < width; x++)
// 		{
// 			unsigned char* dst = &CV_IMAGE_ELEM(dst_img, unsigned char, y, 3 * x);
// 			r = src_data[count]; count++;
// 			g = src_data[count]; count++;
// 			b = src_data[count]; count++;
// 			*dst = b;
// 			dst++;
// 			*dst = g;
// 			dst++;
// 			*dst = r;
// 		}
// 	}
// 	cvSaveImage(filename.c_str(), dst_img);
// }
