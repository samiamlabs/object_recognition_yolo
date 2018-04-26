#include <fstream>
// #include <iostream>
#include <sstream>
// #include <sys/stat.h>
#include <vector>

#include <boost/format.hpp>
#include <boost/range/combine.hpp>

#include <ecto/ecto.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd.hpp>

#include <opencv2/highgui/highgui.hpp>

using ecto::spore;
using ecto::tendrils;

namespace ecto_yolo {
struct ImageMux {
  static void declare_params(tendrils &params) {
    // params.declare(&TrainingImageSaver::param_visualize_, "visualize",
    //                "Visualize the output", true);
    // params.declare(&ImageMux::directory_path_, "direcotry_path",
    //                "Path to image directory.", "/tmp/yolo");
  }

  static void declare_io(const tendrils &params, tendrils &inputs,
                         tendrils &outputs) {

    inputs.declare(&ImageMux::image1_, "image1", "Rgb full image.");
    inputs.declare(&ImageMux::image2_, "image2", "Rgb full image.");

    outputs.declare(&ImageMux::image_out_, "image_out", "Rgb full image output.");

  }

  void configure(const tendrils &params, const tendrils &inputs, const tendrils &outputs) {
  }


  int process(const tendrils &inputs, const tendrils &outputs) {

    float random_float = rng_.uniform(0.0, 1.0);
    float file_percentage = 0.7;

    if (random_float < file_percentage) {
      image1_->copyTo(*image_out_);
    } else {
      image2_->copyTo(*image_out_);
    }

    return ecto::OK;
  }

  cv::RNG rng_;

  ecto::spore<cv::Mat> image1_;
  ecto::spore<cv::Mat> image2_;
  ecto::spore<cv::Mat> image_out_;
};
} // namespace ecto_yolo

ECTO_CELL(ecto_yolo, ecto_yolo::ImageMux, "ImageMux",
          "Image multiplexer.")
