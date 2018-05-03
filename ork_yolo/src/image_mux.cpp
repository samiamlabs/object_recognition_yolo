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

    inputs.declare(&ImageMux::depth1_, "depth1", "Depth full image.");

    inputs.declare(&ImageMux::K1_, "K1", "Camera intrinsic matrix.");

    outputs.declare(&ImageMux::image_out_, "image", "Rgb full image output.");
    outputs.declare(&ImageMux::depth_out_, "depth", "Depth full image output.");
    outputs.declare(&ImageMux::K_out_, "K", "Camera intrinsic matrix.");

  }

  void configure(const tendrils &params, const tendrils &inputs, const tendrils &outputs) {
  }


  int process(const tendrils &inputs, const tendrils &outputs) {

    float random_float = rng_.uniform(0.0, 1.0);
    float file_percentage = 0.7;

    if (random_float < file_percentage) {
      image1_->copyTo(*image_out_);
      depth1_->copyTo(*depth_out_);
      K1_->copyTo(*K_out_);
    } else {
      image2_->copyTo(*image_out_);
      //FIXME: generate depth somehow instaid of using empty matrix
      cv::Mat depth(image2_->rows, image2_->cols, CV_16U, cv::Scalar(0));
      depth.copyTo(*depth_out_);

      float focal_length_x = 525.0;
      float focal_length_y = 525.0;

      float principal_point_x = image2_->cols/2;
      float principal_point_y = image2_->rows/2;

      cv::Mat K = (
        cv::Mat_<float>(3, 3) << focal_length_x, 0.0, principal_point_x,
                                0.0, focal_length_y, principal_point_y,
                                0.0, 0.0, 1.0);
      K.copyTo(*K_out_);
    }

    return ecto::OK;
  }

  cv::RNG rng_;

  ecto::spore<cv::Mat> image1_;
  ecto::spore<cv::Mat> image2_;

  ecto::spore<cv::Mat> depth1_;

  ecto::spore<cv::Mat> K1_;

  ecto::spore<cv::Mat> image_out_;
  ecto::spore<cv::Mat> depth_out_;

  ecto::spore<cv::Mat> K_out_;
};
} // namespace ecto_yolo

ECTO_CELL(ecto_yolo, ecto_yolo::ImageMux, "ImageMux",
          "Image multiplexer.")
