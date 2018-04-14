#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <map>

#include <boost/format.hpp>

#include <ecto/ecto.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd.hpp>

namespace cv {
using namespace cv::rgbd;
}

#include <object_recognition_core/common/json.hpp>
#include <object_recognition_core/db/db.h>
#include <object_recognition_core/db/document.h>
#include <object_recognition_core/db/model_utils.h>

#include <object_recognition_renderer/renderer3d.h>
#include <object_recognition_renderer/utils.h>

#include <opencv2/highgui/highgui.hpp>

using ecto::spore;
using ecto::tendrils;

namespace ecto_yolo {
struct Trainer {
  static void declare_params(tendrils &params) {
    params.declare(&Trainer::param_visualize_, "visualize",
                   "Visualize the output", true);
    params.declare(&Trainer::param_wait_key_, "wait_key",
                   "Wait for key when visualizing", true);
    params.declare(&Trainer::param_n_points_, "renderer_n_points",
                   "Renderer parameter: the number of points on the sphere.",
                   10);
    params.declare(&Trainer::param_blur_, "blur",
                   "Blur the rendered object.", 2);

    params.declare(&Trainer::param_radius_min_, "renderer_radius_min",
                   "Renderer parameter: the minimum scale sampling.", 0.6);
    params.declare(&Trainer::param_radius_max_, "renderer_radius_max",
                   "Renderer parameter: the maximum scale sampling.", 2.5);
    params.declare(&Trainer::param_radius_step_, "renderer_radius_step",
                   "Renderer parameter: the step scale sampling.", 0.4);
    params.declare(&Trainer::param_width_, "renderer_width",
                   "Renderer parameter: the image width.", 640);
    params.declare(&Trainer::param_height_, "renderer_height",
                   "Renderer parameter: the image height.", 480);
    params.declare(&Trainer::param_focal_length_x_, "renderer_focal_length_x",
                   "Renderer parameter: the focal length x.", 525.0);
    params.declare(&Trainer::param_focal_length_y_, "renderer_focal_length_y",
                   "Renderer parameter: the focal length y.", 525.0);
    params.declare(&Trainer::param_near_, "renderer_near",
                   "Renderer parameter: near distance.", 0.1);
    params.declare(&Trainer::param_far_, "renderer_far",
                   "Renderer parameter: far distance.", 1000.0);
    params.declare(&Trainer::param_camera_min_y_, "camera_min_y",
                   "Camera parameter: minimum y.", 0.0);
    params.declare(&Trainer::param_camera_max_y_, "camera_max_y",
                   "Camera parameter: maximum y.", 1.2);
    params.declare(&Trainer::param_camera_roll_, "camera_roll",
                   "Camera parameter: roll.", 0.2);
  }

  static void declare_io(const tendrils &params, tendrils &inputs,
                         tendrils &outputs) {
    inputs.declare(&Trainer::json_db_, "json_db", "The DB parameters", "{}")
        .required(true);
    inputs .declare(&Trainer::object_id_, "object_id", "The object id, to associate this model with.")
        .required(true);

    inputs.declare(&Trainer::color_, "image", "An rgb full frame image.");

    outputs.declare(&Trainer::renderer_n_points_, "renderer_n_points",
                    "Renderer parameter: the number of points on the sphere.");
    outputs.declare(&Trainer::renderer_radius_min_, "renderer_radius_min",
                    "Renderer parameter: the minimum scale sampling.");
    outputs.declare(&Trainer::renderer_radius_max_, "renderer_radius_max",
                    "Renderer parameter: the maximum scale sampling.");
    outputs.declare(&Trainer::renderer_radius_step_, "renderer_radius_step",
                    "Renderer parameter: the step scale sampling.");
    outputs.declare(&Trainer::renderer_width_, "renderer_width",
                    "Renderer parameter: the image width.");
    outputs.declare(&Trainer::renderer_height_, "renderer_height",
                    "Renderer parameter: the image height.");
    outputs.declare(&Trainer::renderer_focal_length_x_,
                    "renderer_focal_length_x",
                    "Renderer parameter: the focal length x.");
    outputs.declare(&Trainer::renderer_focal_length_y_,
                    "renderer_focal_length_y",
                    "Renderer parameter: the focal length y.");
    outputs.declare(&Trainer::renderer_near_, "renderer_near",
                    "Renderer parameter: near distance.");
    outputs.declare(&Trainer::renderer_far_, "renderer_far",
                    "Renderer parameter: far distance.");

    outputs.declare(&Trainer::color_images_, "color_images",
                    "Rgb full frame images with object.");
    outputs.declare(&Trainer::debug_image_, "debug_image",
                    "Dubug image of rendered objects.");
    outputs.declare(&Trainer::box_labels_, "box_labels",
                    "Box labels for training YOLO.");
  }

  void configure(const tendrils &params, const tendrils &inputs,
                 const tendrils &outputs) {
  }

  cv::Vec3d get_random_camera_translation(float min_y_norm, float max_y_norm) {
    cv::Vec3d point;

    double theta;
    double phi;
    double x, y, z;

    do {
      theta = 2 * M_PI * rng_.uniform(0.0, 1.0);
      phi = M_PI * rng_.uniform(0.0, 1.0);
      x = std::sin(phi) * std::cos(theta);
      y = std::sin(phi) * std::sin(theta);
      z = std::cos(phi);
    } while (y > min_y_norm || y < -max_y_norm);

    float radius = rng_.uniform(*renderer_radius_min_, *renderer_radius_max_);

    x *= radius;
    y *= radius;
    z *= radius;

    point = cv::Vec3d(x, y, z);

    return point;
  }

  cv::Vec3d get_up_vector(cv::Vec3d translation, float rotation) {
    float radius = cv::norm(translation);

    cv::Vec3d up;

    float up_x = rotation;
    float up_z = 0;
    float up_y = 1;
    normalize_vector(up_x, up_y, up_z);

    up = cv::Vec3d(up_x, up_y, up_z);
    // std::cout << "Up vector" << up << '\n';

    return up;
  }

  std::string load_mesh(){

    std::cout << "JSON db string: "<< *json_db_ << '\n';

    // Get the document for the object_id_ from the DB
    object_recognition_core::db::ObjectDbPtr db =
        object_recognition_core::db::ObjectDbParameters(*json_db_).generateDb();

    object_recognition_core::db::Documents documents =
        object_recognition_core::db::ModelDocuments(
            db,
            std::vector<object_recognition_core::db::ObjectId>(1, *object_id_),
            "mesh");

    if (documents.empty()) {
      std::cerr << "Skipping object id \"" << *object_id_
                << "\" : no mesh in the DB" << std::endl;
      // return ecto::OK;
    }

    // Get the list of _attachments and figure out the original one
    object_recognition_core::db::Document document = documents[0];

    std::vector<std::string> attachments_names = document.attachment_names();
    std::string mesh_path;
    std::vector<std::string> possible_names(2);

    /* Create the temporary directory */
    char template_str[] = "/tmp/tmpdir.XXXXXX";
    char *tmp_dirname = mkdtemp(template_str);

    if (tmp_dirname == NULL) {
      perror("tempdir: error: Could not create tmp directory");
      exit(EXIT_FAILURE);
    }

    /* Change directory */
    if (chdir (tmp_dirname) == -1)
    {
       perror ("tempdir: error: ");
       exit (EXIT_FAILURE);
    }

    possible_names[0] = "original.obj";
    possible_names[1] = "mesh.stl";

    BOOST_FOREACH (const std::string &attachment_name, attachments_names) {


      std::string file_path = std::string(tmp_dirname) + '/' + attachment_name;

      std::cout << file_path << '\n';

      // Load the mesh and save it to the temporary file
      std::ofstream temp_file;
      temp_file.open(file_path.c_str());
      document.get_attachment_stream(attachment_name, temp_file);
      temp_file.close();

      for (size_t i = 0; i < possible_names.size() && mesh_path.empty(); ++i) {
        std::vector<std::string>::iterator it = find(
                                  possible_names.begin(),
                                  possible_names.end(),
                                  attachment_name);

        if (it != possible_names.end()){
          std::cout << "Setting file name: " + *it + '\n';
          mesh_path = std::string(tmp_dirname) + '/' + *it;
        }
      }
    }

    return mesh_path;
  }

  Renderer3d get_object_renderer() {

    // Define the display
    // assign the parameters of the renderer
    *renderer_n_points_ = *param_n_points_;
    *renderer_radius_min_ = *param_radius_min_;
    *renderer_radius_max_ = *param_radius_max_;
    *renderer_width_ = *param_width_;
    *renderer_height_ = *param_height_;
    *renderer_near_ = *param_near_;
    *renderer_far_ = *param_far_;
    *renderer_focal_length_x_ = *param_focal_length_x_;
    *renderer_focal_length_y_ = *param_focal_length_y_;


    // Renderer3d renderer = Renderer3d(mesh_path);
    Renderer3d renderer = Renderer3d("/dummy_path");

    return renderer;
  }

  int process(const tendrils &inputs, const tendrils &outputs) {
    box_labels_->clear();
    color_images_->clear();

    Renderer3d renderer2 = get_object_renderer();

    std::string mesh_path = load_mesh();
    if (mesh_path.empty()) {
      std::remove(mesh_path.c_str());
      std::cerr << "The mesh path is empty for the object id \"" << *object_id_
                << std::endl;
      // return ecto::OK;
    }

    renderer.set_parameters(
        *renderer_width_, *renderer_height_, *renderer_focal_length_x_,
        *renderer_focal_length_y_, *renderer_near_, *renderer_far_);
    renderer.set_mesh_path(mesh_path);


    cv::Mat image, depth, mask;
    cv::Mat debug_image;
    color_->copyTo(debug_image);

    for (size_t i = 0; i < *param_n_points_; ++i) {
      std::stringstream status;
      status << "Rendering images " << (i + 1) << '/' << *param_n_points_ << '\n';

      std::cout << status.str();

      float min_y_norm = *param_camera_min_y_;
      float max_y_norm = *param_camera_max_y_;
      float rotation = rng_.uniform(0.0, *param_camera_roll_);

      cv::Vec3d translation = get_random_camera_translation(min_y_norm, max_y_norm);
      cv::Vec3d up = get_up_vector(translation, rotation);

      cv::Rect rect;

      float brightness_variance = 0.4;
      float brightness = std::max(0.0, std::min(1.0, 0.5 + rng_.gaussian(brightness_variance)));
      float color_variance = 0.1;
      float red = std::max(0.0, std::min(1.0, brightness + rng_.gaussian(color_variance)));
      float green = std::max(0.0, std::min(1.0, brightness + rng_.gaussian(color_variance)));
      float blue = std::max(0.0, std::min(1.0, brightness + rng_.gaussian(color_variance)));
      renderer.set_lighting_color(red, green, blue);

      float x = rng_.gaussian(10);
      float y = rng_.gaussian(10);
      float z = 15.0 + rng_.gaussian(10);
      renderer.set_lighting_position(x, y, z);

      int height = color_->rows; //480;
      int width = color_->cols; //640;

      renderer.lookAt(translation(0), translation(1), translation(2), up(0), up(1), up(2));
      renderer.renderDepthOnly(depth, mask, rect);
      // rect = cv::Rect(0, 0, width, height);
      renderer.renderImageOnly(image, rect);


      // cv::namedWindow("Rendering");
      // cv::imshow("Rendering", image);
      // cv::waitKey(100000);

      int x_offset = rng_.uniform(0, (width-rect.width));
      int y_offset = rng_.uniform(0, (height-rect.height));

      rect.x = x_offset;
      rect.y = y_offset;

      cv::Mat training_image;
      color_->copyTo(training_image);
      image.copyTo(training_image(rect), mask);

      cv::Mat blurred_object_image;

      int blur = rng_.uniform(0, *param_blur_);
      int kernel_size = 1 + blur*2;
      cv::GaussianBlur(training_image(rect), blurred_object_image, cv::Size(kernel_size, kernel_size), 0, 0);

      float blur_rect = rng_.uniform(0.0, 1.0);
      if (blur_rect < 0.5){
        blurred_object_image.copyTo(training_image(rect));
        blurred_object_image.copyTo(debug_image(rect));
      } else {
        blurred_object_image.copyTo(training_image(rect), mask);
        blurred_object_image.copyTo(debug_image(rect), mask);
      }

      cv::rectangle(debug_image, rect, cv::Scalar(255), 1, 8, 0);

      box_labels_->push_back(rect);
      color_images_->push_back(training_image);
      // Delete the status
      for (size_t j = 0; j < status.str().size(); ++j)
        std::cout << '\b';
    }

    if (*param_visualize_) {
      cv::namedWindow("Rendering");
      cv::imshow("Rendering", debug_image);
      if (*param_wait_key_) {
        std::cout << "Wait key true\n";
        cv::waitKey(100000);
      } else {
        // cv::waitKey(0);
      }
    }

    //FIXME: Figure out how to prevent memory leak
    renderer.~Renderer3d();

    return ecto::OK;
  }

  Renderer3d renderer = Renderer3d("/dummy_path");

  cv::RNG rng_;

  std::map<std::string, std::string> mesh_paths_;

  /** The DB parameters as a JSON string */
  ecto::spore<std::string> json_db_;
  /** The id of the object to generate a trainer for */
  ecto::spore<std::string> object_id_;

  ecto::spore<cv::Mat> color_;

  ecto::spore<int> param_n_points_;
  ecto::spore<int> param_blur_;

  ecto::spore<bool> param_visualize_;
  ecto::spore<bool> param_wait_key_;
  ecto::spore<double> param_radius_min_;
  ecto::spore<double> param_radius_max_;
  ecto::spore<double> param_radius_step_;
  ecto::spore<int> param_width_;
  ecto::spore<int> param_height_;
  ecto::spore<double> param_near_;
  ecto::spore<double> param_far_;
  ecto::spore<double> param_focal_length_x_;
  ecto::spore<double> param_focal_length_y_;
  ecto::spore<double> param_camera_min_y_;
  ecto::spore<double> param_camera_max_y_;
  ecto::spore<double> param_camera_roll_;

  ecto::spore<int> renderer_n_points_;
  ecto::spore<double> renderer_radius_min_;
  ecto::spore<double> renderer_radius_max_;
  ecto::spore<double> renderer_radius_step_;
  ecto::spore<int> renderer_width_;
  ecto::spore<int> renderer_height_;
  ecto::spore<double> renderer_near_;
  ecto::spore<double> renderer_far_;
  ecto::spore<double> renderer_focal_length_x_;
  ecto::spore<double> renderer_focal_length_y_;

  ecto::spore<cv::Mat> debug_image_;
  ecto::spore<std::vector<cv::Mat> > color_images_;
  ecto::spore<std::vector<cv::Rect> > box_labels_;
};
} // namespace ecto_yolo

ECTO_CELL(ecto_yolo, ecto_yolo::Trainer, "Trainer",
          "Train the YOLO object detection algorithm.")
