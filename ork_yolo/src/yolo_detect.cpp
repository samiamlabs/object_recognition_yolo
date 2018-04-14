#include <fstream>
#include <iostream>
#include <sstream>

#include <boost/format.hpp>
#include <boost/range/combine.hpp>

#include <ecto/ecto.hpp>

#include <object_recognition_core/common/pose_result.h>
#include <object_recognition_core/db/ModelReader.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

extern "C" {
#include "darknet.h"
}

using object_recognition_core::common::PoseResult;

using ecto::tendrils;

namespace ecto_yolo {

struct Detector {

  // virtual void parameter_callback(
  //     const object_recognition_core::db::Documents &db_documents) {}

  // virtual void parameterCallbackJsonDb(const std::string &json_db) {}

  static void declare_params(ecto::tendrils &params) {
    // object_recognition_core::db::bases::declare_params_impl(params, "mesh");

    params.declare(&Detector::network_config_, "network_config", "Path to network config file.",
                   "/home/sam/OrkData/cfg/tiny-yolo-voc.cfg");

    params.declare(&Detector::data_config_, "data_config", "Path to data config.",
                   "/home/sam/OrkData/cfg/tiny-yolo-voc.names");

    params.declare(&Detector::weights_, "weights", "Path to weigths.",
                   "/home/sam/OrkData/weights/tiny-yolo-voc.weights");

    params.declare(&Detector::detection_threshold_, "detection_threshold",
                   "Threshold for detection.", 0.5);

    params.declare(&Detector::nms_, "nms",
                   "Non-maximum supression.", 0.45);
  }

  static void declare_io(const tendrils &params, tendrils &inputs, tendrils &outputs) {

    inputs.declare(&Detector::image_, "image", "An rgb full frame image.");

    outputs.declare(&Detector::pose_results_, "pose_results",
                    "The results of object recognition");

    outputs.declare(&Detector::bounding_boxes_, "bounding_boxes",
                    "Bounding boxes of detected objects.");

    outputs.declare(&Detector::detected_objects_, "detected_objects",
                    "Names of of detected objects.");

    outputs.declare(&Detector::probabilities_, "probabilities",
                    "Probabilities of detected objects.");
  }

  void configure(const tendrils &params, const tendrils &inputs, const tendrils &outputs) {

    read_data_config();
    setupNetwork();
    allocate_variables();

  }

  void allocate_variables() {

    layer l = net_->layers[net_->n - 1];

    boxes_ = (box *)calloc(l.w * l.h * l.n, sizeof(box));

    float **probabilities = (float **)calloc(l.w * l.h * l.n, sizeof(float *));
    for (int i = 0; i < l.w * l.h * l.n; ++i)
      probabilities[i] = (float *)calloc(l.classes + 1, sizeof(float));
  }

  void read_data_config() {
    char *data_config_ptr = new char[data_config_->length() + 1];
    strcpy(data_config_ptr, data_config_->c_str());

    list *options = read_data_cfg(data_config_ptr);
    char *name_list = option_find_str(options, (char*)"names", (char*)"data/ork.names");

    class_names_ptr_ = get_labels(name_list);

    std::ifstream infile(name_list);
    std::string line;
    while (std::getline(infile, line)) {
      class_names_.push_back(line);
    }
  }

  void setupNetwork() {
    network_config_ptr_ = new char[network_config_->length() + 1];
    strcpy(network_config_ptr_, network_config_->c_str());

    weights_ptr_ = new char[weights_->length() + 1];
    strcpy(weights_ptr_, weights_->c_str());

    net_ = parse_network_cfg(network_config_ptr_);

    load_weights(net_, weights_ptr_);
    set_batch_network(net_, 1);
  }

  void clearOutputs() {
    bounding_boxes_->clear();
    detected_objects_->clear();
    probabilities_->clear();
  }

  int process(const tendrils &inputs, const tendrils &outputs) {
    std::cout << "Process start...\n";
    clearOutputs();

    updateImage();
    detect();
    visualize();

    std::cout << "Process end...\n";
    return ecto::OK;
  }

  void ipl_into_image(IplImage *src, image &im) {
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    int i, j, k;

    for (i = 0; i < h; ++i) {
      for (k = 0; k < c; ++k) {
        for (j = 0; j < w; ++j) {
          im.data[k * w * h + i * w + j] = data[i * step + j * c + k] / 255.;
        }
      }
    }
  }

  image ipl_to_image(IplImage *src) {
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image out = make_image(w, h, c);
    ipl_into_image(src, out);
    return out;
  }

  void updateImage() {
    IplImage *ipl_img = new IplImage(*image_);
    // TODO: use ipl_into_image when size is the same as last image?
    yolo_image_ = ipl_to_image(ipl_img);
    rgbgr_image(yolo_image_);
    yolo_image_letterbox_ = letterbox_image(yolo_image_, net_->w, net_->h);
    // letterbox_image_into(yolo_image_, net_->w, net_->h, yolo_image_letterbox_);
  }

  void addObjectName(std::string name) {
    std::cout << boost::format("Object name: %s\n") % name;
    detected_objects_->push_back(name);
  }

  void addProbability(float probability) {
    std::cout << boost::format("Probability: %f\n") % probability;
    probabilities_->push_back(probability);
  }

  void addBoundingBox(float relative_x, float relative_y, float relative_width, float relative_height) {

    int x = relative_x * image_->cols;
    int y = relative_y * image_->rows;
    int width = relative_width * image_->cols;
    int height = relative_height * image_->rows;

    std::cout << boost::format(
      "Adding bounding box, x: %f, y: %f, height: %f, width: %f\n")
      % x % y % height % width;

    cv::Rect bounding_box(x, y, width, height);
    bounding_boxes_->push_back(bounding_box);
  }

  void detect() {
    // Non-maximum suppression
    // image **alphabet_ptr_ = load_alphabet();
    // char **names = get_labels(*names_);

    layer l = net_->layers[net_->n - 1];
    float *X = yolo_image_letterbox_.data;

    double time_start = what_time_is_it_now();
    float *prediction = network_predict(net_, X);
    std::cout << boost::format("Predicted in %f seconds.\n") %
                     (what_time_is_it_now() - time_start);

    int nboxes = 0;

    detection *detections = get_network_boxes(
        net_, yolo_image_.w, yolo_image_.h, *detection_threshold_,
        hierarchy_threshold_, 0, 1, &nboxes);

    if (*nms_ < 0)
      do_nms_sort(detections, nboxes, l.classes, *nms_);

    for(int box_index = 0; box_index < nboxes; ++box_index) {

      int detected_class = -1;
      for(int object_class = 0; object_class < l.classes; ++object_class) {
        if (detections[box_index].prob[object_class] > *detection_threshold_) {
          detected_class = object_class;
        }
      }

      if(detected_class >= 0) {
        box bounding_box = detections[box_index].bbox;

        float left  = (bounding_box.x-bounding_box.w/2.);
        float right = (bounding_box.x+bounding_box.w/2.);
        float top   = (bounding_box.y-bounding_box.h/2.);
        float bot   = (bounding_box.y+bounding_box.h/2.);

        if(left < 0) left = 0;
        if(right > yolo_image_.w-1) right = yolo_image_.w-1;
        if(top < 0) top = 0;
        if(bot > yolo_image_.h-1) bot = yolo_image_.h-1;

        float width = right - left;
        float height = bot - top;

        addObjectName(class_names_[detected_class]);
        addProbability(detections[box_index].prob[detected_class]);

        addBoundingBox(
          left, top,
          width, height
        );
      }
    }
    free_detections(detections, nboxes);
  }

  void visualize() {
    std::string text;
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 1;
    int thickness = 2;
    int baseline = 0;
    baseline += thickness;

    cv::Mat image;
    image_->copyTo(image);

    for (auto tup : boost::combine(*detected_objects_, *bounding_boxes_)) {
      std::string class_name;
      cv::Rect bounding_box;
      boost::tie(class_name, bounding_box) = tup;

      cv::Size text_size = cv::getTextSize(class_name, font_face, font_scale,
                                           thickness, &baseline);

      cv::Point text_org(bounding_box.x, bounding_box.y - text_size.height / 2);
      cv::Point pt1(bounding_box.x - 1, bounding_box.y - 2 * text_size.height);
      cv::Point pt2(bounding_box.x + bounding_box.width, bounding_box.y);
      cv::rectangle(image, pt1, pt2, cv::Scalar(255, 0, 0), -1, 8, 0);

      cv::putText(image, class_name, text_org, font_face, font_scale,
                  cv::Scalar(255, 255, 255), thickness, 8);
      cv::rectangle(image, bounding_box, cv::Scalar(255, 0, 0), 2, 8, 0);
    }

    cv::namedWindow("Detections");
    cv::imshow("Detections", image);
    cv::waitKey(1);
  }

private:
  network *net_;

  char *network_config_ptr_;
  char *weights_ptr_;
  char **class_names_ptr_;

  image yolo_image_;
  image yolo_image_letterbox_;

  box *boxes_;

  std::vector<std::string> class_names_;

  float hierarchy_threshold_ = 0.5;
  ecto::spore<cv::Mat> image_;
  ecto::spore<std::vector<PoseResult>> pose_results_;

  ecto::spore<std::vector<cv::Rect>> bounding_boxes_;
  ecto::spore<std::vector<std::string>> detected_objects_;
  ecto::spore<std::vector<float>> probabilities_;

  ecto::spore<std::string> weights_;
  ecto::spore<std::string> network_config_;
  ecto::spore<std::string> data_config_;
  ecto::spore<float> nms_;

  ecto::spore<float> detection_threshold_;
};
} // namespace ecto_yolo

ECTO_CELL(ecto_yolo, ecto_yolo::Detector, "Detector",
          "Given image, identify objects.")
