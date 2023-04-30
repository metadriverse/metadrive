

#include "ArHosekSkyModel.h"
#include "PNMImage.h"
#include "Texture.h"

#include <iostream>
using namespace std;

#define ONE_PI 3.14159265359
#define TWO_PI 6.28318530718

double srgb_clamp(double v) {
	v = max(0.0, min(1.0, v / 255.0));
	return v;
}


BEGIN_PUBLISH
INLINE void generate_table()
{

	const double min_elevation = 0.0; // degree
	const double max_elevation = 40.0; // degree
	double turbidity = 3.0;
	double albedo = 0.2;

	cout << "Scattering Precomputation v0.1" << endl;
	cout << "Turbidity (double from 0 to 10): ";
	cin >> turbidity;
	cout << "Ground Albedo (double from 0 to 1): ";
	cin >> albedo;

	const double sun_azimuth = ONE_PI * 0.25;

	PNMImage img(512 * 10, 128 * 10, 3, 65535);

	for (size_t elev = 0; elev < 100; ++elev) {
		if (elev % 10 == 0)
			cout << elev << " percent" << endl;

		double factor = ((double)elev) / 99.0;
		double angle = factor * (max_elevation - min_elevation) + min_elevation;

		const double sun_elevation = angle / 360.0 * TWO_PI;
		const int num_channels = 3;

		ArHosekSkyModelState* skymodel_state = arhosek_rgb_skymodelstate_alloc_init(
			turbidity, albedo, sun_elevation);

		double skydome_result[num_channels];

		for (size_t i = 0; i < 512; ++i) {
			for (size_t k = 0; k < 128; ++k) {

				double factor_x = ((double)i) / 512.0;
				double factor_y = ((double)k) / 128.0;

				double theta = 0, gamma = 0;

				if (true) {
					// write out raw angles
					gamma = factor_x * 2.0 * ONE_PI;
					theta = factor_y * 0.5 * ONE_PI;
				}
				else {
					// Write out transformed angles
					double coord_azimuth = factor_x * TWO_PI;
					theta = factor_y * ONE_PI * 0.5;
					double neg_solar_elevation = 0.5 * ONE_PI - sun_elevation;

					double cosGamma = cos(theta) * cos(neg_solar_elevation)
						+ sin(theta) * sin(neg_solar_elevation)
						* cos(coord_azimuth - sun_azimuth);

					gamma = acos(cosGamma);
				}

				for (size_t r = 0; r < num_channels; r++) {
					skydome_result[r] = srgb_clamp(arhosek_tristim_skymodel_radiance(
							skymodel_state, theta, gamma, r));
				}

				img.set_xel(i + (elev % 10) * 512, k + (elev / 10) * 128,
					skydome_result[0], skydome_result[1], skydome_result[2]);
			}
		}

		arhosekskymodelstate_free(skymodel_state);
	}

	cout << "Writing data ..." << endl;
	// img.write("scattering_lut.png");

	Texture t;
	t.setup_2d_texture(img.get_x_size(), img.get_y_size(), Texture::T_float, Texture::F_rgb16);
	t.load(img);
	t.write("scattering_lut.txo.pz");
}

END_PUBLISH
