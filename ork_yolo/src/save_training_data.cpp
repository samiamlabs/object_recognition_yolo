#include <fstream>
#include <iostream>
#include <sys/stat.h>
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
struct TrainingImageSaver {
  static void declare_params(tendrils &params) {
    // params.declare(&TrainingImageSaver::param_visualize_, "visualize",
    //                "Visualize the output", true);
    params.declare(&TrainingImageSaver::directory_path_, "direcotry_path",
                   "Path to image directory.", "/tmp/yolo");
  }

  static void declare_io(const tendrils &params, tendrils &inputs,
                         tendrils &outputs) {

    inputs.declare(&TrainingImageSaver::color_images_, "color_images",
                   "Rgb full frame images with object.");

    inputs.declare(&TrainingImageSaver::box_labels_, "box_labels",
                   "Box labels.");

    inputs.declare(&TrainingImageSaver::object_id_index_, "object_id_index",
                  "Index of object to represent yolo class.");
  }

  void configure(const tendrils &params, const tendrils &inputs, const tendrils &outputs) {
    file_counter_ = -1;
  }

  inline bool file_exists(const std::string &name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
  }

  std::string make_box_label_string(cv::Rect box_label, cv::Mat image) {
    int category_number = *object_id_index_;
    double image_width = image.size().width;
    double image_height = image.size().height;
    double object_center_x = (box_label.x + box_label.width/2)/image_width;
    double object_center_y = (box_label.y + box_label.height/2)/image_height;
    double object_width = box_label.width/image_width;
    double object_height = box_label.height/image_height;

    char template_str[] = "%d %f %f %f %f";
    std::string box_label_string = boost::str(boost::format(template_str)
                                   % category_number
                                   % object_center_x
                                   % object_center_y
                                   % object_width
                                   % object_height);

    return box_label_string;
  }

  int process(const tendrils &inputs, const tendrils &outputs) {

    // Save the rendered image

    std::string directory_path = "/tmp/yolo/";
    std::string filename = "pos-";
    std::string image_extension = ".jpg";
    std::string image_path;
    std::string box_label_extension = ".txt";
    std::string box_label_path;

    Stat st;
    if (stat(directory_path.c_str(), &st) != 0)
    {
      mode_t mode = 0777;
      mkdir(directory_path.c_str(), mode);
    }

    if (file_counter_ == -1) {
      do {
        file_counter_++;
        image_path = directory_path + filename + std::to_string(file_counter_) + image_extension;
        // std::cout << "Checking path: " << image_path << '\n';
      } while (file_exists(image_path));
    }

    image_path = directory_path + filename + std::to_string(file_counter_) + image_extension;
    box_label_path = directory_path + filename + std::to_string(file_counter_) + box_label_extension;

    std::cout << "Empty path found: " << image_path << '\n';

    std::cout << "Number of box labels: " << box_labels_->size() << '\n';
    std::cout << "Number of images: " << color_images_->size() << '\n';

    // std::string box_label_string = "1 0 1 0 20 0";

    for (auto tup : boost::combine(*box_labels_, *color_images_)) {
      cv::Rect box_label;
      cv::Mat color_image;
      boost::tie(box_label, color_image) = tup;
      // cv::rectangle(color_image, box_label, cv::Scalar(255), 1, 8, 0);

      std::cout << "Saving file: " << image_path << '\n';
      cv::imwrite(boost::str(boost::format(image_path)), color_image);

      std::ofstream out(box_label_path);
      out << make_box_label_string(box_label, color_image);
      out.close();

      file_counter_++;
      image_path = directory_path + filename + std::to_string(file_counter_) + image_extension;
      box_label_path = directory_path + filename + std::to_string(file_counter_) + box_label_extension;

      // if (true) {
      //   cv::namedWindow("Saving");
      //   cv::imshow("Saving", color_image);
      //   cv::waitKey(1000);
      // }
    }

    return ecto::OK;
  }

  int file_counter_;

  ecto::spore<std::string> directory_path_;
  ecto::spore<std::vector<cv::Mat> > color_images_;
  ecto::spore<std::vector<cv::Rect> > box_labels_;
  ecto::spore<int> object_id_index_;
};
} // namespace ecto_yolo

ECTO_CELL(ecto_yolo, ecto_yolo::TrainingImageSaver, "TrainingImageSaver",
          "Save the labeled YOLO images.")
