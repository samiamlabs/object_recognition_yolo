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

typedef struct stat Stat;

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

    inputs.declare(&ImageMux::image_, "image", "Rgb full image.");
    outputs.declare(&ImageMux::image_out_, "image_out", "Rgb full image output.");

  }

  void configure(const tendrils &params, const tendrils &inputs, const tendrils &outputs) {
    std::ifstream infile("/home/sam/fast_ws/src/ork_yolo/data/imagenet/train.txt");
    std::string line;
    while (std::getline(infile, line))
    {
      image_paths_.push_back(line);
    }

  }


  int process(const tendrils &inputs, const tendrils &outputs) {

    float random_float = rng_.uniform(0.0, 1.0);

    float file_percentage = 0.7;

    if (random_float < file_percentage) {
      int random_file_index = rng_.uniform(0, image_paths_.size() -1);
      cv::Mat image_from_file = cv::imread(image_paths_[random_file_index].c_str(), 1);

      if(! image_from_file.data || image_from_file.cols < 100){
        std::cout << "File import failed, path: " << image_paths_[random_file_index] << '\n';
        image_->copyTo(*image_out_);
        return ecto::OK;
      }
      image_from_file.copyTo(*image_out_);
    } else {
      image_->copyTo(*image_out_);
    }

    return ecto::OK;
  }

  cv::RNG rng_;

  std::vector<std::string> image_paths_;

  ecto::spore<cv::Mat> image_;
  ecto::spore<cv::Mat> image_out_;
};
} // namespace ecto_yolo

ECTO_CELL(ecto_yolo, ecto_yolo::ImageMux, "ImageMux",
          "Image multiplexer.")
