#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <map>

#include <Eigen/Dense>

#include <boost/format.hpp>

#include <ecto/ecto.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>

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
struct ObservationRenderer {
  static void declare_params(tendrils &params) {
    params.declare(&ObservationRenderer::param_visualize_, "visualize",
                   "Visualize the output", true);
    params.declare(&ObservationRenderer::param_wait_key_, "wait_key",
                   "Wait for key when visualizing", true);
    params.declare(&ObservationRenderer::param_n_points_, "renderer_n_points",
                   "Renderer parameter: the number of points on the sphere.",
                   10);
    params.declare(&ObservationRenderer::param_blur_, "blur",
                   "Blur the rendered object.", 2);

    params.declare(&ObservationRenderer::param_radius_min_, "renderer_radius_min",
                   "Renderer parameter: the minimum scale sampling.", 0.6);
    params.declare(&ObservationRenderer::param_radius_max_, "renderer_radius_max",
                   "Renderer parameter: the maximum scale sampling.", 2.5);
    params.declare(&ObservationRenderer::param_radius_step_, "renderer_radius_step",
                   "Renderer parameter: the step scale sampling.", 0.4);
    params.declare(&ObservationRenderer::param_near_, "renderer_near",
                   "Renderer parameter: near distance.", 0.1);
    params.declare(&ObservationRenderer::param_far_, "renderer_far",
                   "Renderer parameter: far distance.", 1000.0);
    params.declare(&ObservationRenderer::param_camera_min_y_, "camera_min_y",
                   "Camera parameter: minimum y.", 0.0);
    params.declare(&ObservationRenderer::param_camera_max_y_, "camera_max_y",
                   "Camera parameter: maximum y.", 1.2);
    params.declare(&ObservationRenderer::param_camera_roll_, "camera_roll",
                   "Camera parameter: roll.", 0.2);
  }

  static void declare_io(const tendrils &params, tendrils &inputs,
                         tendrils &outputs) {
    inputs.declare(&ObservationRenderer::json_db_, "json_db", "The DB parameters", "{}") .required(true);
    inputs.declare(&ObservationRenderer::object_id_, "object_id", "The object id, to associate this model with.") .required(true);

    inputs.declare(&ObservationRenderer::K_in_, "K", "Intrinsic camera matrix.").required(true);
    inputs.declare(&ObservationRenderer::background_, "image", "An rgb full frame image.").required(true);
    inputs.declare(&ObservationRenderer::background_depth_, "depth", "An depth full frame image.").required(true);

    outputs.declare(&ObservationRenderer::debug_image_, "debug_image", "Dubug image of rendered objects.");

    outputs.declare(&ObservationRenderer::image_, "image", "An rgb image.");
    outputs.declare(&ObservationRenderer::depth_, "depth", "A depth image");
    outputs.declare(&ObservationRenderer::mask_, "mask", "A mask for the object.");
    outputs.declare(&ObservationRenderer::box_label_, "box_label", "Box labels.");

    outputs.declare(&ObservationRenderer::K_, "K", "The camera intrinsic matrix.");
    outputs.declare(&ObservationRenderer::R_, "R", "The orientation.");
    outputs.declare(&ObservationRenderer::T_, "T", "The translation.");


  }

  void configure(const tendrils &params, const tendrils &inputs,
                 const tendrils &outputs) {
  }

  bool in_frustum() {
    float modelview_matrix[16];
    renderer_.get_modelview_matrix(modelview_matrix);
    Eigen::Map<Eigen::Matrix4f> MV(modelview_matrix);

    float projection_matrix[16];
    renderer_.get_projection_matrix(projection_matrix);
    Eigen::Map<Eigen::Matrix4f> P(projection_matrix);

    Eigen::Matrix4f MVP = P*MV;

    Eigen::Transform<float, 3, Eigen::Affine> transform(MVP);

    SceneBoundingBox bounding_box = renderer_.get_bounding_box();

    cv::Mat K, R, T;

    R_->convertTo(R, CV_64F);
    T_->convertTo(T, CV_64F);
    K_->convertTo(K, CV_64F);

    Eigen::Vector3f projected_point;

    std::vector<Eigen::Vector3f> corner_points = get_rotated_bounding_box();
    std::vector<cv::Point3f> object_points;

    for (auto &corner_point : corner_points) {
      cv::Point3f object_point(corner_point(0), corner_point(1), corner_point(2));
      object_points.push_back(object_point);
    }


    std::vector<cv::Point2f> image_points;
    projectPoints(object_points, R, T, K, cv::Mat(4, 1, CV_64FC1, cv::Scalar(0)), image_points);

    for (auto &image_point : image_points) {
      // std::cout << "Image point: " << image_point << std::endl;
      if (image_point.x < 1 || image_point.x > background_->cols) {
        // std::cout << "Outside width \n";
        return false;
      } else if (image_point.y < 1 || image_point.y > background_->rows) {
        // std::cout << "Outside height\n";
        return false;
      }
    }


    return true;
  }

  void draw_3d_bounding_box() {

    std::vector<Eigen::Vector3f> corner_points = get_rotated_bounding_box();

    // bottom
    draw_projected_line(corner_points[0], corner_points[1]);
    draw_projected_line(corner_points[0], corner_points[4]);
    draw_projected_line(corner_points[1], corner_points[5]);
    draw_projected_line(corner_points[4], corner_points[5]);

    // sides
    draw_projected_line(corner_points[0], corner_points[2]);
    draw_projected_line(corner_points[1], corner_points[3]);
    draw_projected_line(corner_points[4], corner_points[6]);
    draw_projected_line(corner_points[5], corner_points[7]);

    // top
    draw_projected_line(corner_points[2], corner_points[3]);
    draw_projected_line(corner_points[2], corner_points[6]);
    draw_projected_line(corner_points[3], corner_points[7]);
    draw_projected_line(corner_points[6], corner_points[7]);

  }

  std::vector<Eigen::Vector3f> get_rotated_bounding_box() {
    float modelview_matrix[16];
    renderer_.get_modelview_matrix(modelview_matrix);
    Eigen::Map<Eigen::Matrix4f> MV(modelview_matrix);

    float projection_matrix[16];
    renderer_.get_projection_matrix(projection_matrix);
    Eigen::Map<Eigen::Matrix4f> P(projection_matrix);

    Eigen::Matrix4f MVP = P*MV;

    Eigen::Transform<float, 3, Eigen::Affine> transform(MVP);

    SceneBoundingBox bounding_box = renderer_.get_bounding_box();

    std::vector<Eigen::Vector3f> corner_points;

    // std::cout << "BB: max_x: " << bounding_box.max_x << " BB: min_x: " << bounding_box.min_x << std::endl;
    // std::cout << "BB: max_y: " << bounding_box.max_y << " BB: min_y: " << bounding_box.min_y << std::endl;
    // std::cout << "BB: max_z: " << bounding_box.max_z << " BB: min_z: " << bounding_box.min_z << std::endl;

    corner_points.push_back(Eigen::Vector3f(bounding_box.min_x, bounding_box.min_y, bounding_box.min_z));
    corner_points.push_back(Eigen::Vector3f(bounding_box.min_x, bounding_box.min_y, bounding_box.max_z));
    corner_points.push_back(Eigen::Vector3f(bounding_box.min_x, bounding_box.max_y, bounding_box.min_z));
    corner_points.push_back(Eigen::Vector3f(bounding_box.min_x, bounding_box.max_y, bounding_box.max_z));
    corner_points.push_back(Eigen::Vector3f(bounding_box.max_x, bounding_box.min_y, bounding_box.min_z));
    corner_points.push_back(Eigen::Vector3f(bounding_box.max_x, bounding_box.min_y, bounding_box.max_z));
    corner_points.push_back(Eigen::Vector3f(bounding_box.max_x, bounding_box.max_y, bounding_box.min_z));
    corner_points.push_back(Eigen::Vector3f(bounding_box.max_x, bounding_box.max_y, bounding_box.max_z));

    Eigen::Vector3f unit_x(Eigen::Vector3f::UnitX());
    Eigen::Vector3f unit_z(Eigen::Vector3f::UnitZ());
    Eigen::Transform<float, 3, Eigen::Affine> scene_rotation(Eigen::Affine3f::Identity());
    scene_rotation.rotate(Eigen::AngleAxisf(M_PI, unit_x));
    scene_rotation.rotate(Eigen::AngleAxisf(-M_PI/2, unit_z));

    for (auto &corner_point : corner_points) {
      corner_point = scene_rotation*corner_point;
    }

    return corner_points;
  }

  void draw_projected_line(Eigen::Vector3f start_point, Eigen::Vector3f end_point) {
    using namespace cv;

    Scalar color(0, 0, 255);
    Mat K, R, T;

    K_->convertTo(K, CV_64F);
    R_->convertTo(R, CV_64F);
    T_->convertTo(T, CV_64F);

    std::vector<Point3f> object_points(2);
    object_points[0] = Point3f(start_point(0), start_point(1), start_point(2));
    object_points[1] = Point3f(end_point(0), end_point(1), end_point(2));
    std::vector<Point2f> image_points;
    projectPoints(Mat(object_points), R, T, K, Mat(4, 1, CV_64FC1, Scalar(0)), image_points);

    line(*debug_image_, image_points[0], image_points[1], color);
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

    float radius = rng_.uniform(*param_radius_min_, *param_radius_max_);

    x *= radius;
    y *= radius;
    z *= radius;

    point = cv::Vec3d(x, y, z);

    return point;
  }

  cv::Vec3d get_up_vector(cv::Vec3d translation, float rotation) {
    float radius = cv::norm(translation);

    cv::Vec3d up;

    float up_x = 0; //rotation;
    float up_z = 0;
    float up_y = 1;

    normalize_vector(up_x, up_y, up_z);

    up = cv::Vec3d(up_x, up_y, up_z);

    // std::cout << "up = "<< std::endl << " "  << up << std::endl << std::endl;

    return up;
  }

  std::string load_mesh(){

    std::string mesh_path;

    if(mesh_directories_.find(*object_id_) != mesh_directories_.end()) {

      // std::cout << boost::format("Path found: %s\n") % mesh_directories_[*object_id_];

      /* Change directory */
      if (chdir (mesh_directories_[*object_id_].c_str()) == -1)
      {
         perror ("tempdir: error: ");
         exit (EXIT_FAILURE);
      }

      // TODO: support other possible meshnames
      mesh_path = mesh_directories_[*object_id_] + "/original.obj";
      return mesh_path;
    }

    // std::cout << "JSON db string: "<< *json_db_ << '\n';

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
    }

    // Get the list of _attachments and figure out the original one
    object_recognition_core::db::Document document = documents[0];

    std::vector<std::string> attachments_names = document.attachment_names();
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
      // std::cout << file_path << '\n';

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
          // std::cout << "Setting file name: " + *it + '\n';
          mesh_path = std::string(tmp_dirname) + '/' + *it;
        }
      }
    }

    mesh_directories_[*object_id_] = std::string(tmp_dirname);
    return mesh_path;
  }

  void setup_renderer(std::string mesh_path) {
    int width = background_->cols;
    int height = background_->rows;

    float focal_length_x_ = K_in_->at<float>(0, 0);
    float focal_length_y_ = K_in_->at<float>(1, 1);

    // std::cout << boost::format("focal length x: %f, y: %f \n") % focal_length_x_ % focal_length_y_;
    // std::cout << boost::format("cx: %f, cy: %f \n") % K_in_->at<float>(2, 0) % K_in_->at<float>(2, 1);
    std::cout << "K in:\n" << *K_in_ << std::endl;

    renderer_.set_parameters(
      width, height,
      focal_length_x_, focal_length_y_,
      *param_near_, *param_far_
    );

    renderer_.set_mesh_path(mesh_path);
  }

  int render_observation() {
    cv::Mat image, depth, mask;
    cv::Mat debug_image;
    background_->copyTo(debug_image);

    std::cout << "Rendering image\n";

    randomize_view();

    //std::cout << boost::format("Bg image width: %d, height: %d\n") % background_->cols % background_->rows;

    cv::Rect rect;
    renderer_.renderDepthOnly(depth, mask, rect);
    renderer_.renderImageOnly(image, rect);

    int height = background_->rows;
    int width = background_->cols;

    if( width < rect.width || height < rect.height) {
      // break ecto
      return -1;
    }

    // FIXME: Bug in renderDepthOnly ?
    if( rect.width < 0|| rect.height < 0) {
      std::cout << "Rendered object size is negative\n";
      std::cout << boost::format("Rect x: %s, y: %f, w: %f, h: %f\n") % rect.x % rect.y % rect.width % rect.height;
      image.copyTo(*debug_image_);
      return -1;
    }

    set_output_orientation();

    int x_offset = rng_.uniform(0, (width-rect.width));
    int y_offset = rng_.uniform(0, (height-rect.height));

    cv::Mat training_image;
    background_->copyTo(training_image);
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

    training_image.copyTo(*image_);

    //FIXME: Don't use temporary matrices
    cv::Mat full_mask(background_->size(), CV_8U, cv::Scalar(0));
    mask.copyTo(full_mask(rect));
    *mask_ = full_mask;

    cv::Mat full_depth;
    background_depth_->copyTo(full_depth);
    depth.copyTo(full_depth(rect), mask);
    full_depth.copyTo(*depth_);

    debug_image.copyTo(*debug_image_);

    // temp
    draw_3d_bounding_box();

    *box_label_ = rect;

    return 0;
  }

  void randomize_view() {
    float min_y_norm = *param_camera_min_y_;
    float max_y_norm = *param_camera_max_y_;

    randomize_lighting();

    int height = background_->rows;
    int width = background_->cols;

    int outside_image_count = -1;
    do
    {
      ++outside_image_count;
      float rotation = rng_.uniform(0.0, *param_camera_roll_);
      cv::Vec3d eye = get_random_camera_translation(min_y_norm, max_y_norm);
      cv::Vec3d up = get_up_vector(eye, rotation);

      float center_displacement = cv::norm(eye)*1.0;

      float center_x = rng_.uniform(-center_displacement, center_displacement);
      float center_y = rng_.uniform(-center_displacement, center_displacement);
      float center_z = rng_.uniform(-center_displacement, center_displacement);

      renderer_.lookAt(eye(0), eye(1), eye(2),
                       center_x, center_y, center_z,
                       up(0), up(1), up(2));

      set_output_orientation();

    } while(!in_frustum());

    std::cout << "Object outside image count: " << outside_image_count << std::endl;

  }

  void randomize_lighting() {
    float brightness_variance = 0.4;
    float brightness = std::max(0.0, std::min(1.0, 0.5 + rng_.gaussian(brightness_variance)));
    float color_variance = 0.05;
    float red = std::max(0.0, std::min(1.0, brightness + rng_.gaussian(color_variance)));
    float green = std::max(0.0, std::min(1.0, brightness + rng_.gaussian(color_variance)));
    float blue = std::max(0.0, std::min(1.0, brightness + rng_.gaussian(color_variance)));
    renderer_.set_lighting_color(red, green, blue);

    float x = rng_.gaussian(10);
    float y = rng_.gaussian(10);
    float z = 15.0 + rng_.gaussian(10);
    renderer_.set_lighting_position(x, y, z);
  }

  void set_output_orientation() {
    float modelview_matrix[16];
    renderer_.get_modelview_matrix(modelview_matrix);

    Eigen::Map<Eigen::Matrix4f> MV(modelview_matrix);
    Eigen::Transform<float, 3, Eigen::Affine> transform(MV);

    Eigen::Vector3f unit_z(Eigen::Vector3f::UnitZ());
    Eigen::Vector3f unit_x(Eigen::Vector3f::UnitX());
    Eigen::Vector3f unit_y(Eigen::Vector3f::UnitY());

    transform.prerotate(Eigen::AngleAxisf(M_PI, unit_z));
    transform.rotate(Eigen::AngleAxisf(3*M_PI/2, unit_z));

    // Translation
    Eigen::MatrixXf T_eigen(transform.translation());
    T_eigen = -T_eigen; // OpenGL standard to OpenCV standard
    cv::Mat T;
    cv::eigen2cv(T_eigen, T);
    T.copyTo(*T_);

    // Orientation
    Eigen::MatrixXf R_eigen(transform.rotation());
    cv::Mat R;
    cv::eigen2cv(R_eigen, R);
    R.copyTo(*R_);

    K_in_->copyTo(*K_);
  }

  void visualize() {
    if (*param_visualize_) {
      cv::namedWindow("Rendering");
      cv::imshow("Rendering", *debug_image_);
      if (*param_wait_key_) {
        std::cout << "Wait key true\n";
        cv::waitKey(100000);
      } else {
        // cv::waitKey(0);
      }
    }
  }

  int process(const tendrils &inputs, const tendrils &outputs) {
    // FIXME: Figure out how to prevent memory leak
    std::string mesh_path = load_mesh();
    setup_renderer(mesh_path);

    // FIXME: Figure out how to make texture rendering work when height > width
    // if (background_->rows > background_->cols) {
    //   return ecto::BREAK;
    // }

    if (mesh_path.empty()) {
      std::remove(mesh_path.c_str());
      std::cerr << "The mesh path is empty for the object id \"" << *object_id_
                << std::endl;

      return ecto::BREAK;
    }

    if(render_observation() == -1) {
      return ecto::BREAK;
    }

    // visualize();

    return ecto::OK;
  }

  Renderer3d renderer_ = Renderer3d("/dummy_path");

  cv::RNG rng_;

  std::map<std::string, std::string> mesh_directories_;

  /** The DB parameters as a JSON string */
  ecto::spore<std::string> json_db_;
  /** The id of the object to generate a trainer for */
  ecto::spore<std::string> object_id_;

  ecto::spore<cv::Mat> background_;
  ecto::spore<cv::Mat> background_depth_;

  ecto::spore<cv::Mat> K_in_;

  ecto::spore<int> param_n_points_;
  ecto::spore<int> param_blur_;

  ecto::spore<bool> param_visualize_;
  ecto::spore<bool> param_wait_key_;
  ecto::spore<double> param_radius_min_;
  ecto::spore<double> param_radius_max_;
  ecto::spore<double> param_radius_step_;
  ecto::spore<double> param_near_;
  ecto::spore<double> param_far_;
  ecto::spore<double> param_camera_min_y_;
  ecto::spore<double> param_camera_max_y_;
  ecto::spore<double> param_camera_roll_;

  ecto::spore<cv::Mat> debug_image_;

  ecto::spore<cv::Mat> image_;
  ecto::spore<cv::Mat> depth_;
  ecto::spore<cv::Mat> mask_;
  ecto::spore<cv::Rect> box_label_;

  ecto::spore<cv::Mat> R_;
  ecto::spore<cv::Mat> T_;
  ecto::spore<cv::Mat> K_;

};
} // namespace ecto_yolo

ECTO_CELL(ecto_yolo, ecto_yolo::ObservationRenderer, "ObservationRenderer",
          "Render observations for training.")
