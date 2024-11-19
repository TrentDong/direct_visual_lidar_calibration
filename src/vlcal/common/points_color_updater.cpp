#include <vlcal/common/points_color_updater.hpp>
#include <vlcal/common/estimate_fov.hpp>

#include <glk/primitives/icosahedron.hpp>
#include <guik/viewer/light_viewer.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>

namespace vlcal {

PointsColorUpdater::PointsColorUpdater(const camera::GenericCameraBase::ConstPtr& proj, const cv::Mat& image)
: proj(proj),
  min_nz(std::cos(estimate_camera_fov(proj, {image.cols, image.rows}) + 0.5 * M_PI / 180.0)),
  image(image) {
  glk::Icosahedron icosahedron;
  for (int i = 0; i < 6; i++) {
    icosahedron.subdivide();
  }
  icosahedron.spherize();

  points = std::make_shared<FrameCPU>(icosahedron.vertices);
  intensity_colors.resize(points->size(), Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f));

  cloud_buffer = std::make_shared<glk::PointCloudBuffer>(points->points, points->size());
}

PointsColorUpdater::PointsColorUpdater(const camera::GenericCameraBase::ConstPtr& proj, const cv::Mat& image, const FrameCPU::ConstPtr& points)
: proj(proj),
  min_nz(std::cos(estimate_camera_fov(proj, {image.cols, image.rows}) + 0.5 * M_PI / 180.0)),
  image(image),
  points(points) {
  cloud_buffer = std::make_shared<glk::PointCloudBuffer>(points->points, points->size());
  intensity_colors.resize(points->size());
  for (int i = 0; i < points->size(); i++) {
    intensity_colors[i] = glk::colormapf(glk::COLORMAP::TURBO, points->intensities[i]);
  }
}

void PointsColorUpdater::update(const Eigen::Isometry3d& T_camera_liar, const double blend_weight) {
  std::shared_ptr<std::vector<Eigen::Vector4f>> colors(new std::vector<Eigen::Vector4f>(points->size(), Eigen::Vector4f::Zero()));

  if (!cachedColoredCloud) {
    cachedColoredCloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    cachedColoredCloud->resize(points->size());
  
  for (int i = 0; i < points->size(); i++) {
    const Eigen::Vector4d pt_camera = T_camera_liar * points->points[i];

    if (pt_camera.head<3>().normalized().z() < min_nz) {
      // Out of FoV
      continue;
    }

    const Eigen::Vector2i pt_2d = proj->project(pt_camera.head<3>()).cast<int>();
    if ((pt_2d.array() < Eigen::Array2i::Zero()).any() || (pt_2d.array() >= Eigen::Array2i(image.cols, image.rows)).any()) {
      // Out of Image
      continue;
    }

    // const unsigned char pix = image.at<std::uint8_t>(pt_2d.y(), pt_2d.x());
    // const Eigen::Vector4f color(pix / 255.0f, pix / 255.0f, pix / 255.0f, 1.0f);
    const cv::Vec3b& pix = image.at<cv::Vec3b>(pt_2d.y(), pt_2d.x());
    const Eigen::Vector4f color(pix[2] / 255.0f, pix[1] / 255.0f, pix[0] / 255.0f, 1.0f);  // OpenCV 默认是 BGR

    // colors->at(i) = color * blend_weight + intensity_colors[i] * (1.0 - blend_weight);
    colors->at(i) = color;

    // 填充到新的点云
    auto& pcl_point = cachedColoredCloud->points[i];
      pcl_point.x = points->points[i].x();
      pcl_point.y = points->points[i].y();
      pcl_point.z = points->points[i].z();
      pcl_point.r = pix[2];
      pcl_point.g = pix[1];
      pcl_point.b = pix[0];
  }

    guik::LightViewer::instance()->invoke([cloud_buffer = cloud_buffer, colors = colors] { cloud_buffer->add_color(*colors); });
  }
}

void PointsColorUpdater::save_colored_pointcloud(const std::string& filename) const {
  if (is_saved) {
    std::cout << "Colored point cloud already saved, skipping." << std::endl;
    return;
  }

  if (!cachedColoredCloud || cachedColoredCloud->empty()) {
    throw std::runtime_error("Colored point cloud has not been generated yet!");
  }

  pcl::io::savePLYFileASCII(filename, *cachedColoredCloud);
  is_saved = true;  // 标记点云已保存
  std::cout << "Colored point cloud saved to " << filename << std::endl;
}

}  // namespace vlcal
