import os
import json
from argparse import ArgumentParser
import sys
from PIL import Image
import numpy as np
import math
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
import csv
from PIL.ExifTags import TAGS, GPSTAGS
from coords import *

# 添加utils模块到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from read_write_model import read_model, write_model, Image as ColmapImage, Camera, Point3D, rotmat2qvec, qvec2rotmat

def read_exif_gps(image_path):
    """
    从图像中读取GPS信息
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        tuple: (纬度, 经度, 高度) 或 None
    """
    try:
        image = Image.open(image_path)
        exif = image._getexif()
        if not exif:
            return None

        # 查找GPS信息的标签ID
        gps_info = None
        for tag_id in exif:
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'GPSInfo':
                gps_info = exif[tag_id]
                break

        if not gps_info:
            return None

        # 提取GPS数据
        gps_data = {}
        for key in gps_info.keys():
            decode = GPSTAGS.get(key, key)
            gps_data[decode] = gps_info[key]

        # 提取经纬度和高度
        if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
            def convert_to_degrees(value):
                d, m, s = value
                degrees = float(d)
                minutes = float(m)
                seconds = float(s[0]) / float(s[1]) if isinstance(s, tuple) else float(s)
                return degrees + (minutes / 60.0) + (seconds / 3600.0)

            lat = convert_to_degrees(gps_data['GPSLatitude'])
            lon = convert_to_degrees(gps_data['GPSLongitude'])

            # 处理南纬和西经
            if gps_data.get('GPSLatitudeRef', 'N') == 'S':
                lat = -lat
            if gps_data.get('GPSLongitudeRef', 'E') == 'W':
                lon = -lon

            # 读取高度
            alt = 0
            if 'GPSAltitude' in gps_data:
                alt_value = gps_data['GPSAltitude']
                if isinstance(alt_value, tuple):
                    alt = float(alt_value[0]) / float(alt_value[1])
                else:
                    alt = float(alt_value)

                # 处理高度参考
                if gps_data.get('GPSAltitudeRef', b'0') == b'1':
                    alt = -alt

            return lat, lon, alt

    except Exception as e:
        print(f"警告: 无法从图像 {image_path} 读取GPS数据: {e}")
        return None

    return None

def gps_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """
    将GPS坐标转换为ENU坐标
    
    Args:
        lat, lon, alt: 目标点的GPS坐标
        ref_lat, ref_lon, ref_alt: 参考点的GPS坐标
        
    Returns:
        numpy.array: ENU坐标 [东, 北, 上]
    """
    # WGS84椭球体参数
    a = 6378137.0  # 长半轴 (m)
    f = 1/298.257223563  # 扁率
    b = a * (1 - f)  # 短半轴
    e2 = 2*f - f*f  # 第一偏心率平方
    
    def lla_to_ecef(lat, lon, alt):
        """将经纬度高度转换为ECEF坐标"""
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        # 计算卯酉圈曲率半径
        N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
        
        # 计算ECEF坐标
        x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
        y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
        z = (N * (1 - e2) + alt) * math.sin(lat_rad)
        
        return x, y, z
    
    # 转换目标点和参考点到ECEF坐标
    x, y, z = lla_to_ecef(lat, lon, alt)
    x0, y0, z0 = lla_to_ecef(ref_lat, ref_lon, ref_alt)
    
    # 计算相对位置
    dx, dy, dz = x - x0, y - y0, z - z0
    
    # 计算ENU旋转矩阵
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)
    
    # 构建从ECEF到ENU的旋转矩阵
    R = np.array([
        [-np.sin(ref_lon_rad), np.cos(ref_lon_rad), 0],
        [-np.sin(ref_lat_rad)*np.cos(ref_lon_rad), -np.sin(ref_lat_rad)*np.sin(ref_lon_rad), np.cos(ref_lat_rad)],
        [np.cos(ref_lat_rad)*np.cos(ref_lon_rad), np.cos(ref_lat_rad)*np.sin(ref_lon_rad), np.sin(ref_lat_rad)]
    ])
    
    # 转换到ENU坐标系
    enu = R @ np.array([dx, dy, dz])
    return enu

def compute_alignment(colmap_pts, gps_pts):
    """
    计算COLMAP坐标系到GPS坐标系的对齐变换
    
    Args:
        colmap_pts: COLMAP坐标系中的点
        gps_pts: GPS坐标系中的对应点
        
    Returns:
        tuple: (旋转矩阵, 缩放因子, COLMAP中心点, GPS中心点)
    """
    colmap_pts = np.asarray(colmap_pts)
    gps_pts = np.asarray(gps_pts)

    # 确保输入数据格式正确
    if colmap_pts.shape != gps_pts.shape or colmap_pts.ndim != 2 or colmap_pts.shape[1] != 3:
        raise ValueError("Input point arrays must both be of shape (N, 3)")

    # 确保有足够的点进行对齐
    if colmap_pts.shape[0] < 3:
        raise ValueError("At least 3 points are required for alignment")

    # 计算两个坐标系中的点对之间的距离
    colmap_distances = []
    gps_distances = []
    for i in range(len(colmap_pts)):
        for j in range(i+1, len(colmap_pts)):
            colmap_distances.append(np.linalg.norm(colmap_pts[i] - colmap_pts[j]))
            gps_distances.append(np.linalg.norm(gps_pts[i] - gps_pts[j]))
    
    # 使用距离比例计算缩放因子
    scale = np.mean(np.array(gps_distances) / np.array(colmap_distances))

    # 计算中心点
    colmap_mean = colmap_pts.mean(axis=0)
    gps_mean = gps_pts.mean(axis=0)

    # 去中心化
    X = colmap_pts - colmap_mean
    Y = gps_pts - gps_mean

    # 计算旋转矩阵
    S = X.T @ Y
    U, _, Vt = np.linalg.svd(S)
    R_mat = Vt.T @ U.T
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T

    return R_mat, scale, colmap_mean, gps_mean

def transform_pose(Rwc, twc, R_align, scale):
    """
    转换相机位姿
    
    Args:
        Rwc: 相机到世界坐标系的旋转矩阵
        twc: 相机到世界坐标系的平移向量
        R_align: 对齐旋转矩阵
        scale: 缩放因子
        
    Returns:
        tuple: (新旋转矩阵, 新平移向量, 新四元数)
    """
    # 1. 计算原始相机中心
    C_orig = -Rwc.T @ twc
    
    # 2. 对相机中心应用旋转和缩放
    C_new = R_align @ (C_orig * scale)
    
    # 3. 计算新的相机旋转
    R_new = Rwc @ R_align.T
    
    # 4. 计算新的平移向量
    t_new = -R_new @ C_new

    # 将旋转矩阵转换为四元数
    qvec_new = rotmat2qvec(R_new)
    
    return R_new, t_new, qvec_new

def align_camera_pose(Rwc, twc, R_align, scale, shift):
    C = -Rwc.T @ twc
    C_new = scale * (R_align @ C) + shift

    # 新旋转（世界坐标变换 ⇒ 右乘 R^T）
    R_new = Rwc @ R_align.T
    qvec_new = rotmat2qvec(R_new)
    t_new = -R_new @ C_new

    return R_new, t_new, qvec_new    

def save_points_to_ply(points3D, output_path):
    """
    将点云保存为PLY格式
    
    Args:
        points3D: COLMAP点云字典
        output_path: 输出PLY文件路径
    """
    with open(output_path, 'w') as f:
        # 写入PLY头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points3D)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 写入点云数据
        for point_id, point in points3D.items():
            x, y, z = point.xyz
            r, g, b = point.rgb
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
    
    print(f"保存点云到: {output_path}")

def create_metadata_xml(lon, lat, alt, output_path):
    """
    创建metadata.xml文件
    
    Args:
        lon: 经度
        lat: 纬度
        alt: 高度
        output_path: 输出文件路径
    """
    # 创建XML根元素
    root = ET.Element("ModelMetadata")
    root.set("version", "1")
    
    # 添加SRS元素（坐标参考系统）
    srs = ET.SubElement(root, "SRS")
    srs.text = f"ENU:{lat},{lon}"
    
    # 添加SRSOrigin元素（坐标原点）
    srs_origin = ET.SubElement(root, "SRSOrigin")
    srs_origin.text = f"0,0,{alt}"
    
    # 添加ColorSource元素
    color_source = ET.SubElement(root, "ColorSource")
    color_source.text = "Visible"
    
    # 创建漂亮的XML输出
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="\t")
    
    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    
    print(f"已创建metadata.xml文件: {output_path}")


def umeyama_align(X, Y, with_scale=True):
    # X, Y: Nx3 arrays; find s,R,t so that Y ~ s*R*X + t
    assert X.shape == Y.shape
    n, m = X.shape
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc = X - muX
    Yc = Y - muY

    Sxx = (Xc.T @ Xc) / n
    cov = (Yc.T @ Xc) / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    if with_scale:
        varX = np.sum(Xc ** 2) / n
        s = np.trace(np.diag(D) @ S) / varX
    else:
        s = 1.0

    t = muY - s * R @ muX
    return s, R, t


def umeyama_align_ransac(X, Y, with_scale=True, max_iters=1000, inlier_threshold=1.0, min_inliers=None, random_seed=None):
    """
    使用RANSAC对umeyama对齐进行鲁棒估计，处理外点。

    Args:
        X, Y: 对应的 Nx3 点阵列，求解 Y ~ s*R*X + t
        with_scale: 是否估计缩放
        max_iters: RANSAC最大迭代次数
        inlier_threshold: 判断内点的距离阈值（米）
        min_inliers: 最少内点数量（默认为 max(3, 50% * N)）
        random_seed: 随机种子，便于复现

    Returns:
        s, R, t, inlier_mask
        - s: 缩放因子
        - R: 3x3 旋转矩阵
        - t: 平移向量 (length-3)
        - inlier_mask: 布尔数组，长度为 N，标记内点
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    assert X.shape == Y.shape
    n, m = X.shape

    # 如果点少于3个，无法可靠估计，返回默认恒等变换并打印警告
    if n < 3:
        print("警告: 输入点数少于3，无法进行RANSAC估计，返回 s=1, R=I, t=0")
        s = 1.0
        R = np.eye(3)
        t = np.zeros(3)
        return s, R, t, np.ones(n, dtype=bool)

    if min_inliers is None:
        min_inliers = max(3, int(0.5 * n))

    rng = np.random.RandomState(random_seed)
    best_inliers = None
    best_model = None
    best_count = 0

    for _ in range(max_iters):
        # 从样本中随机选择最小样本数（若干 个 3D 点）
        try:
            idx = rng.choice(n, min_inliers, replace=False)
        except ValueError:
            continue

        Xs = X[idx]
        Ys = Y[idx]

        # 避免退化样本（共面/共线等）: 检查中心化后的秩
        if np.linalg.matrix_rank(Xs - Xs.mean(axis=0)) < 3:
            continue

        # 估计模型
        try:
            s_try, R_try, t_try = umeyama_align(Xs, Ys, with_scale=with_scale)
        except Exception:
            continue

        # 将所有 X 变换到估计的模型下并计算与 Y 的距离
        X_trans = (s_try * (R_try @ X.T)).T + t_try
        dists = np.linalg.norm(X_trans - Y, axis=1)
        # 计算并打印dists的统计信息
        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        print(f"colmap与 gps 对齐后，距离误差统计信息: 均值={mean_dist:.3f} 米, 标准差={std_dist:.3f} 米")
        inliers = dists <= inlier_threshold
        count = int(inliers.sum())

        # 更新最优模型
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_model = (s_try, R_try, t_try)

            # 如果达到所有点都是内点，提前退出
            if best_count >= n:
                break

    # 如果没有找到足够的内点，则回退到对全部点的估计
    if best_inliers is None or best_count < min_inliers:
        s, R, t = umeyama_align(X, Y, with_scale=with_scale)
        return s, R, t, np.ones(n, dtype=bool)

    # 使用所有内点重新拟合以获得更好估计
    s_ref, R_ref, t_ref = umeyama_align(X[best_inliers], Y[best_inliers], with_scale=with_scale)
    return s_ref, R_ref, t_ref, best_inliers


def visualize_alignment(colmap_pts, gps_pts, labels=None, show_lines=True, figsize=(10, 8), point_size=40, save_path=None, title=None):
    """
    在可旋转/缩放的3D窗口中可视化对齐后的COLMAP相机中心和GPS(ENU)点，
    并按一一对应关系将它们连线。

    Args:
        colmap_pts: (N,3) array-like of COLMAP points (camera centers)
        gps_pts: (N,3) array-like of GPS points (ENU coordinates)
        labels: optional list of N labels (strings) to annotate points
        show_lines: whether to draw lines connecting corresponding points
        figsize: figure size tuple
        point_size: marker size for scatter
        save_path: optional path to save the figure (PNG)
        title: optional figure title

    Returns:
        (fig, ax): matplotlib figure and 3D axes
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    colmap_pts = np.asarray(colmap_pts)
    gps_pts = np.asarray(gps_pts)

    if colmap_pts.shape != gps_pts.shape or colmap_pts.ndim != 2 or colmap_pts.shape[1] != 3:
        raise ValueError("`colmap_pts` and `gps_pts` must both be shape (N,3)")

    n = colmap_pts.shape[0]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # scatter
    ax.scatter(colmap_pts[:, 0], colmap_pts[:, 1], colmap_pts[:, 2], c='tab:blue', s=point_size, label='colmap')
    ax.scatter(gps_pts[:, 0], gps_pts[:, 1], gps_pts[:, 2], c='tab:red', s=point_size, label='gps')

    # lines: draw as a single Line3DCollection for performance
    if show_lines and n > 0:
        try:
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            lines = [[colmap_pts[i], gps_pts[i]] for i in range(n)]
            lc = Line3DCollection(lines, colors='gray', linewidths=0.7, alpha=0.8)
            ax.add_collection3d(lc)
        except Exception:
            # fallback to per-line plotting if Line3DCollection is unavailable
            for i in range(n):
                xs = [colmap_pts[i, 0], gps_pts[i, 0]]
                ys = [colmap_pts[i, 1], gps_pts[i, 1]]
                zs = [colmap_pts[i, 2], gps_pts[i, 2]]
                ax.plot(xs, ys, zs, c='gray', linewidth=0.8, alpha=0.8)

    # labels
    if labels is not None:
        for i, lab in enumerate(labels):
            try:
                ax.text(colmap_pts[i, 0], colmap_pts[i, 1], colmap_pts[i, 2], f"C:{lab}", color='blue')
                ax.text(gps_pts[i, 0], gps_pts[i, 1], gps_pts[i, 2], f"G:{lab}", color='red')
            except Exception:
                pass

    # set equal aspect ratio
    all_pts = np.vstack((colmap_pts, gps_pts))
    x_limits = (all_pts[:, 0].min(), all_pts[:, 0].max())
    y_limits = (all_pts[:, 1].min(), all_pts[:, 1].max())
    z_limits = (all_pts[:, 2].min(), all_pts[:, 2].max())

    x_center = 0.5 * (x_limits[0] + x_limits[1])
    y_center = 0.5 * (y_limits[0] + y_limits[1])
    z_center = 0.5 * (z_limits[0] + z_limits[1])

    max_range = max(x_limits[1] - x_limits[0], y_limits[1] - y_limits[0], z_limits[1] - z_limits[0])
    if max_range == 0:
        max_range = 1.0

    half = max_range / 2.0
    ax.set_xlim(x_center - half, x_center + half)
    ax.set_ylim(y_center - half, y_center + half)
    ax.set_zlim(z_center - half, z_center + half)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        try:
            fig.savefig(save_path, dpi=200)
            print(f"Saved alignment visualization to: {save_path}")
        except Exception as e:
            print(f"Warning: cannot save figure to {save_path}: {e}")

    plt.show()
    return fig, ax


def visualize_alignment_open3d(
    colmap_pts,
    gps_pts,
    labels=None,
    point_size=6,
    line_color=(0.7, 0.7, 0.7),
    show_camera_frames=True,
    camera_frame_size=0.35,
    show_matplotlib_fallback=True,
    camera_rotations=None
):
    """
    使用 Open3D 可视化对齐的 COLMAP 相机中心与 GPS 点，并用线段连接一一对应点。
    增强内容：
      - 优先在 3D 窗口中显示点云和连线
      - 可选择为每个相机绘制小的坐标轴（相机方向可视化）
      - 在每个点附近显示 3D 文本标签（例如相机名或 GPS 名称），类似 matplotlib 的文本注释

    如果 Open3D 未安装或当前环境不支持图形窗口，会抛出异常，由调用者回退到 matplotlib。

    Args:
        colmap_pts, gps_pts: (N,3) numpy arrays
        labels: optional list of N labels (strings) to annotate points; if provided, used for both camera and gps with prefixes
        point_size: unused for draw_geometries; kept for API parity
        line_color: RGB tuple for line color
        show_camera_frames: whether to draw small coordinate frames at camera centers
        camera_frame_size: size of coordinate frames
        camera_rotations: (N,3,3) numpy array of camera rotation matrices

    Returns:
        True on success
    """
    try:
        import open3d as o3d
    except Exception:
        raise

    colmap_pts = np.asarray(colmap_pts)
    gps_pts = np.asarray(gps_pts)
    if colmap_pts.shape != gps_pts.shape or colmap_pts.ndim != 2 or colmap_pts.shape[1] != 3:
        raise ValueError("`colmap_pts` and `gps_pts` must both be shape (N,3)")

    n = colmap_pts.shape[0]

    # point clouds (kept for lightweight rendering). Use brighter colors for visibility.
    pc_cam = o3d.geometry.PointCloud()
    pc_cam.points = o3d.utility.Vector3dVector(colmap_pts)
    pc_cam.colors = o3d.utility.Vector3dVector(np.tile([0.0, 0.6, 1.0], (n, 1)))

    pc_gps = o3d.geometry.PointCloud()
    pc_gps.points = o3d.utility.Vector3dVector(gps_pts)
    pc_gps.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.4, 0.4], (n, 1)))

    # lines connecting corresponding points
    pts = np.vstack([colmap_pts, gps_pts])
    lines = [[i, i + n] for i in range(n)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(line_color, (len(lines), 1)))

    # prepare optional camera frames and label points
    geom_list = [pc_cam, pc_gps, line_set]

    label_supported = True

    if show_camera_frames:
        for i in range(n):
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_frame_size)
            frame.translate(colmap_pts[i])
            # 应用相机的旋转（如果提供了旋转信息）
            if camera_rotations is not None and i < len(camera_rotations):
                # Open3D 的 rotate 方法需要一个 3x3 的旋转矩阵
                frame.rotate(camera_rotations[i], center=colmap_pts[i])
            geom_list.append(frame)

    # create visualizer to add 3D labels (add_3d_label may not be available in all builds)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Alignment (Open3D)', width=1280, height=960)

    for g in geom_list:
        vis.add_geometry(g)

    # Improve render options: dark background, larger points
    try:
        ro = vis.get_render_option()
        ro.background_color = np.asarray([0.08, 0.08, 0.08])
        ro.point_size = max(6, int(point_size))
    except Exception:
        pass

    # Add 3D text labels near points if provided. Offset labels slightly in Z to avoid overlap.
    if labels is not None:
        label_added = False
        for i, lab in enumerate(labels):
            try:
                cam_pt = colmap_pts[i] + np.array([0.0, 0.0, camera_frame_size * 0.8])
                vis.add_3d_label(cam_pt, f"C:{lab}")
                gps_pt = gps_pts[i] + np.array([0.0, 0.0, camera_frame_size * 0.8])
                vis.add_3d_label(gps_pt, f"G:{lab}")
                label_added = True
            except Exception as e:
                # some Open3D builds may not support add_3d_label; mark unsupported and continue
                label_supported = False
                print(f"警告: 无法添加3D标签 '{lab}': {e}")
                break
        
        # 如果没有成功添加任何标签，尝试使用matplotlib作为备选方案
        # if not label_added and show_matplotlib_fallback:
            # try:
            #     print("可视化：Open3D 未能添加 3D 标签，改为显示 matplotlib 静态带标签快照")
            #     visualize_alignment(colmap_pts, gps_pts, labels=labels, show_lines=True, point_size=60, title='Aligned cameras vs GPS (snapshot)')
            # except Exception as e:
            #     print(f"警告: matplotlib 备选可视化失败: {e}")
    vis.run()
    vis.destroy_window()

    return True


def perform_gps_alignment(
    gps_images_dir,
    model_dir,
    aligned_output_dir,
    gps_image_filter=None,
    gps_metadata=None,
    show_viz=False,
):
    """
    执行GPS对齐功能
    
    Args:
        gps_images_dir: 带有GPS信息的原始图像目录
        model_dir: COLMAP稀疏模型目录（通常为 sparse/0）
        aligned_output_dir: 输出对齐结果与metadata的目录
        gps_image_filter: 可选，仅使用指定文件名集合
        gps_metadata: 可选，直接提供的图像->(lat,lon,alt)映射
    """
    print("\n开始执行GPS对齐...")
    print(f"GPS图像目录: {gps_images_dir}")
    print(f"COLMAP模型目录: {model_dir}")
    print(f"输出目录: {aligned_output_dir}")
    if gps_image_filter:
        gps_image_filter = set(gps_image_filter)
        print(f"仅使用用户指定的 {len(gps_image_filter)} 张图像进行GPS对齐")
    
    # 检查必要的文件是否存在
    if not os.path.exists(gps_images_dir) and gps_metadata is None:
        print(f"错误: 图像目录不存在, 且没有提供GPS元数据: {gps_images_dir}")
        return False
        
    if not os.path.exists(model_dir):
        print(f"错误: COLMAP模型目录不存在: {model_dir}")
        return False
    
    # 创建对齐输出目录
    os.makedirs(aligned_output_dir, exist_ok=True)
    
    try:
        # 读取图像 GPS 信息
        gps_coords = []
        specified_found = set()
        if gps_metadata:
            candidate_names = (
                gps_image_filter if gps_image_filter else gps_metadata.keys()
            )
            for name in sorted(candidate_names):
                if name not in gps_metadata:
                    continue
                lat, lon, alt = gps_metadata[name]
                if None in (lat, lon, alt):
                    print(f"警告: JSON中的 {name} 数据不完整，跳过该图像")
                    continue
                gps_coords.append((name, lat, lon, alt))
                specified_found.add(name)
        if len(gps_coords) == 0:
            print("警告: gpsmetadata不存在，或未从其中找到任何图像数据，将尝试从图像中提取GPS信息")
            image_names = sorted(os.listdir(gps_images_dir))
            for name in image_names:
                if gps_image_filter and name not in gps_image_filter:
                    continue
                path = os.path.join(gps_images_dir, name)
                if os.path.isfile(path):
                    gps_info = read_exif_gps(path)
                    if gps_info:
                        lat, lon, alt = gps_info
                        gps_coords.append((name, lat, lon, alt))
                        specified_found.add(name)
        
        if len(gps_coords) == 0:
            print("错误: 未找到任何带有GPS信息的图像，跳过GPS对齐")
            return False
        if gps_image_filter:
            missing = gps_image_filter - specified_found
            if missing:
                print(f"警告: 以下图像在目录中不存在或缺少GPS信息：{', '.join(sorted(missing))}")
        
        print(f"找到{len(gps_coords)}张带有GPS信息的图像")
        
        # 设置参考点（第一个图像的 GPS）
        ref_lat, ref_lon, ref_alt = gps_coords[0][1:]
        print(f"使用第一个图像作为参考点: 纬度={ref_lat}, 经度={ref_lon}, 高度={ref_alt}")

        # 转换 GPS 到 ENU
        gps_enu = {}
        for name, lat, lon, alt in gps_coords:
            enu = gps_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)
            if '.' in name:
                name = '.'.join(name.split('.')[:-1])            
            gps_enu[name] = enu

        # 读取 COLMAP 模型
        cameras, images, points3D = read_model(model_dir, ext='.bin')

        # 提取相机中心和对应的 ENU 坐标
        colmap_pts = []
        gps_pts = []
        for image_id, image in images.items():
            # 去除文件名后缀
            name = image.name
            if '.' in name:
                name = '.'.join(name.split('.')[:-1])
            Rcw = image.qvec2rotmat()
            tcw = image.tvec
            C = -Rcw.T @ tcw  # 相机中心
            if name in gps_enu:
                colmap_pts.append(C)
                gps_pts.append(gps_enu[name])

        colmap_pts = np.array(colmap_pts)
        gps_pts = np.array(gps_pts)

        if len(colmap_pts) < 3:
            print("警告: 找到的GPS对应点少于3个，无法进行对齐")
            return False

        # 计算对齐变换
        # R_align, scale, center_colmap, center_gps = compute_alignment(colmap_pts, gps_pts)
        # print("R_align:\n", R_align)
        # print("scale:", scale)
        # print("center_colmap:", center_colmap)
        # print("center_gps:", center_gps)
        print("找到 {} 个GPS对应点".format(len(gps_pts)))
        scale, R_align, shift, _ = umeyama_align_ransac(colmap_pts, gps_pts, max_iters=1000, inlier_threshold=2.0)
        print("scale:", scale)
        print("R:\n", R_align)
        print("t:", shift)        

        # origin_lat, origin_lon, origin_alt = enu_to_wgs84(center_gps[0], center_gps[1], center_gps[2], ref_lat, ref_lon, ref_alt)
        origin_lat = ref_lat
        origin_lon = ref_lon
        origin_alt = ref_alt

        # 创建metadata.xml文件
        metadata_path = os.path.join(aligned_output_dir, "metadata.xml")
        create_metadata_xml(origin_lon, origin_lat, origin_alt, metadata_path)

        # 转换相机位姿
        for image_id, image in images.items():
            Rcw = qvec2rotmat(image.qvec)
            tcw = image.tvec
            # Rcw_new0, tcw_new0, qvec_new0 = transform_pose(Rcw, tcw, R_align, scale)
            Rcw_new, tcw_new, qvec_new = align_camera_pose(Rcw, tcw, R_align, scale, shift)

            images[image_id] = ColmapImage(
                id=image.id,
                qvec=qvec_new,
                tvec=tcw_new,
                camera_id=image.camera_id,
                name=image.name,
                xys=image.xys,
                point3D_ids=image.point3D_ids,
            )
        

        
        # 转换3D点
        for point_id, point in points3D.items():
            X = point.xyz
            # X_new0 = R_align @ (X * scale)
            X_new = scale * (R_align @ X) + shift

            points3D[point_id] = Point3D(
                id=point.id,
                xyz=X_new,
                rgb=point.rgb,
                error=point.error,
                image_ids=point.image_ids,
                point2D_idxs=point.point2D_idxs,
            )

        # 写入新的模型
        write_model(cameras, images, points3D, aligned_output_dir, ext='.bin')
        
        # 保存点云为PLY格式
        ply_path = os.path.join(aligned_output_dir, 'aligned_model.ply')
        save_points_to_ply(points3D, ply_path)
        
        # 生成metadata.xml文件
        # if len(gps_coords) > 0:
        #     lat, lon, alt = gps_coords[0][1:]
        #     # 计算原点的GPS坐标
        #     lat_scale = 1.0 / (111132.92 + -559.82 * np.cos(2 * np.radians(lat)) + 1.175 * np.cos(4 * np.radians(lat)))
        #     lon_scale = 1.0 / (111412.84 * np.cos(np.radians(lat)) + -93.5 * np.cos(3 * np.radians(lat)))
            
        #     # 获取第一个相机的位置
        #     first_camera_center = -qvec2rotmat(images[list(images.keys())[0]].qvec).T @ images[list(images.keys())[0]].tvec
            
        #     # 计算原点(0,0,0)的GPS坐标
        #     origin_lat = lat - first_camera_center[1] * lat_scale
        #     origin_lon = lon - first_camera_center[0] * lon_scale
        #     origin_alt = alt - first_camera_center[2]
            
        #     # 创建metadata.xml文件
        #     metadata_path = os.path.join(aligned_output_dir, "metadata.xml")
        #     create_metadata_xml(origin_lon, origin_lat, origin_alt, metadata_path)
        
        # 可选：弹出可视化窗口，展示对齐后的相机中心与GPS点并按一一对应连线
        if show_viz:
            cam_centers = []
            gps_centers = []
            cam_rotations = []
            labels = []
            for image_id, image in images.items():
                name = image.name
                if '.' in name:
                    name = '.'.join(name.split('.')[:-1])
                if name in gps_enu:
                    Rcw = qvec2rotmat(image.qvec)
                    C = -Rcw.T @ image.tvec
                    cam_centers.append(C)
                    cam_rotations.append(Rcw)
                    gps_centers.append(gps_enu[name])
                    labels.append(name)

            if len(cam_centers) > 0:
                try:
                    # 优先使用 Open3D（更流畅），若不可用则回退到 matplotlib
                    try:
                        import open3d  # type: ignore
                        use_o3d = True
                    except Exception:
                        use_o3d = False

                    if use_o3d:
                        visualize_alignment_open3d(np.array(cam_centers), np.array(gps_centers), labels=labels, camera_rotations=np.array(cam_rotations))
                    else:
                        visualize_alignment(np.array(cam_centers), np.array(gps_centers), labels=labels, title='Aligned cameras vs GPS')
                except Exception as e:
                    print(f"警告: 可视化失败: {e}")
            else:
                print("警告: 未找到匹配点用于可视化")

        print(f"GPS对齐完成！对齐后的模型保存在: {aligned_output_dir}")
        return True
        
    except Exception as e:
        print(f"GPS对齐过程中出现错误: {str(e)}")
        return False

def load_gps_image_filter(gps_image_arg):
    """
    解析用户指定的GPS图像列表
    gps_image_arg: txt文件路径或逗号分隔字符串
    """
    if not gps_image_arg:
        return None
    candidates = []
    if os.path.isfile(gps_image_arg):
        with open(gps_image_arg, 'r', encoding='utf-8') as f:
            candidates = [line.strip() for line in f if line.strip()]
    else:
        candidates = [item.strip() for item in gps_image_arg.split(',') if item.strip()]
    if not candidates:
        return None
    return set(candidates)


def load_gps_metadata_from_json(json_path):
    """
    加载外部提供的GPS信息
    期望格式：{ "image_name.jpg": {"GPSLatitude": "...", "GPSLongitude": "...", "AbsoluteAltitude": "..."} }
    """
    if not json_path:
        return None
    if not os.path.isfile(json_path):
        print(f"警告: GPS JSON文件不存在: {json_path}")
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"警告: 无法解析GPS JSON文件: {e}")
        return None

    def to_float(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            if value.startswith("+"):
                value = value[1:]
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def extract_entry(entry):
        if not isinstance(entry, dict):
            return None
        lat = (
            entry.get("GPSLatitude")
            or entry.get("latitude")
            or entry.get("Latitude")
            or entry.get("lat")
        )
        lon = (
            entry.get("GPSLongitude")
            or entry.get("longitude")
            or entry.get("Longitude")
            or entry.get("lon")
        )
        alt = (
            entry.get("AbsoluteAltitude")
            or entry.get("absolute_altitude")
            or entry.get("Altitude")
            or entry.get("alt")
            or entry.get("GPSAltitude")
            or entry.get("RelativeAltitude")
        )
        return to_float(lat), to_float(lon), to_float(alt)

    metadata = {}
    if isinstance(data, dict):
        iterator = data.items()
    elif isinstance(data, list):
        iterator = []
        for item in data:
            if isinstance(item, dict):
                name = (
                    item.get("name")
                    or item.get("file")
                    or item.get("filename")
                    or item.get("image")
                )
                if name:
                    iterator.append((name, item))
    else:
        print("警告: GPS JSON格式未知，跳过该文件")
        return None

    for name, entry in iterator:
        lat, lon, alt = extract_entry(entry)
        if None in (lat, lon, alt):
            continue
        metadata[name] = (lat, lon, alt)

    if not metadata:
        print("警告: GPS JSON中未找到有效的经纬度信息")
        return None
    print(f"已从JSON加载 {len(metadata)} 条GPS记录")
    return metadata


def load_gps_metadata_from_csv(csv_path):
    """
    加载外部提供的GPS信息（CSV格式）
    支持两种格式：
    1. 包含列名的CSV文件，列包括图像文件名、纬度、经度和高度
    2. 不包含列名的CSV文件，默认第一列为图像名，第二列为纬度，第三列为经度，第四列为高度
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        dict: {image_name: (lat, lon, alt)}
    """
    if not csv_path:
        return None
    if not os.path.isfile(csv_path):
        print(f"警告: GPS CSV文件不存在: {csv_path}")
        return None
    
    try:
        metadata = {}
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            # 读取前几行来判断是否有列标题
            sample = csvfile.read(1024)
            csvfile.seek(0)
            
            # 尝试检测是否有列标题
            try:
                dialect = csv.Sniffer().sniff(sample)
                has_header = csv.Sniffer().has_header(sample)
                
                if has_header:
                    # 有列标题的情况
                    reader = csv.DictReader(csvfile, dialect=dialect)
                    fieldnames = reader.fieldnames
                    
                    if not fieldnames:
                        print("警告: CSV文件没有列标题")
                        return None
                        
                    # 寻找文件名列
                    filename_keys = ['filename', 'file', 'name', 'image', 'image_name']
                    filename_key = None
                    for key in filename_keys:
                        if key in fieldnames:
                            filename_key = key
                            break
                            
                    if not filename_key:
                        print(f"警告: CSV文件中未找到文件名列，可用列: {fieldnames}")
                        return None
                    
                    # 寻找经纬度高度列
                    lat_keys = ['lat', 'latitude', 'Lat', 'Latitude']
                    lon_keys = ['lon', 'lng', 'longitude', 'Lon', 'Lng', 'Longitude']
                    alt_keys = ['alt', 'altitude', 'Alt', 'Altitude', 'height', 'Height']
                    
                    lat_key, lon_key, alt_key = None, None, None
                    for key in lat_keys:
                        if key in fieldnames:
                            lat_key = key
                            break
                    for key in lon_keys:
                        if key in fieldnames:
                            lon_key = key
                            break
                    for key in alt_keys:
                        if key in fieldnames:
                            alt_key = key
                            break
                    
                    if not lat_key or not lon_key:
                        print(f"警告: CSV文件中未找到必需的经纬度列，可用列: {fieldnames}")
                        return None
                    
                    # 读取数据
                    for row in reader:
                        try:
                            filename = row[filename_key]
                            lat = float(row[lat_key])
                            lon = float(row[lon_key])
                            alt = float(row[alt_key]) if alt_key and row[alt_key] else 0.0
                            
                            metadata[filename] = (lat, lon, alt)
                        except (ValueError, KeyError) as e:
                            print(f"警告: CSV文件中某行数据格式错误，跳过该行: {e}")
                            continue
                else:
                    # 没有列标题的情况，使用默认列顺序
                    reader = csv.reader(csvfile, dialect=dialect)
                    
                    # 跳过可能的标题行（即使检测为无标题，也可能存在标题）
                    first_row = next(reader, None)
                    if first_row is None:
                        print("警告: CSV文件为空")
                        return None
                    
                    # 检查第一行是否可能是标题（包含非数字内容）
                    try:
                        # 尝试将第一行的第2-4列转为浮点数
                        float(first_row[1])  # 纬度
                        float(first_row[2])  # 经度
                        # 如果成功，说明第一行是数据行
                        filename = first_row[0]
                        lon = float(first_row[1])
                        lat = float(first_row[2])
                        alt = float(first_row[3]) if len(first_row) > 3 else 0.0
                        metadata[filename] = (lat, lon, alt)
                    except (ValueError, IndexError):
                        # 如果失败，说明第一行是标题，跳过它
                        pass
                    
                    # 读取剩余数据行
                    for row in reader:
                        try:
                            if len(row) >= 3:  # 至少需要文件名、纬度、经度
                                filename = row[0]
                                lon = float(row[1])
                                lat = float(row[2])
                                alt = float(row[3]) if len(row) > 3 else 0.0
                                
                                metadata[filename] = (lat, lon, alt)
                        except (ValueError, IndexError) as e:
                            print(f"警告: CSV文件中某行数据格式错误，跳过该行: {e}")
                            continue
            except csv.Error:
                # 如果嗅探失败，假设是没有标题的CSV文件
                csvfile.seek(0)
                reader = csv.reader(csvfile)
                
                for row in reader:
                    try:
                        if len(row) >= 3:  # 至少需要文件名、纬度、经度
                            filename = row[0]
                            lat = float(row[1])
                            lon = float(row[2])
                            alt = float(row[3]) if len(row) > 3 else 0.0
                            
                            metadata[filename] = (lat, lon, alt)
                    except (ValueError, IndexError) as e:
                        print(f"警告: CSV文件中某行数据格式错误，跳过该行: {e}")
                        continue
        
        print(f"已从CSV加载 {len(metadata)} 条GPS记录")
        return metadata
        
    except Exception as e:
        print(f"警告: 无法解析GPS CSV文件: {e}")
        return None


def resolve_aligned_dir(workspace, aligned_dir):
    if not aligned_dir:
        aligned_dir = "sparse_align"
    if os.path.isabs(aligned_dir):
        return aligned_dir
    return os.path.join(workspace, aligned_dir)


def main():
    parser = ArgumentParser("COLMAP GPS对齐工具")
    parser.add_argument(
        "--workspace",
        "-w",
        required=True,
        type=str,
        help="包含 input / images / sparse 的根目录",
    )
    parser.add_argument(
        "--gps_source_path",
        type=str,
        help="可选：单独指定带GPS信息图像的目录，默认使用 <workspace>/input",
    )
    parser.add_argument(
        "--gps_image_list",
        type=str,
        help="可选：限定参与对齐的图像名（txt文件每行一个或逗号分隔字符串）",
    )
    parser.add_argument(
        "--gps_json",
        type=str,
        help="可选：提供包含GPS经纬度的JSON文件（键为图像文件名）",
    )
    parser.add_argument(
        "--gps_csv",
        type=str,
        help="可选：提供包含GPS经纬度的CSV文件（键为图像文件名）",
    )    
    parser.add_argument(
        "--aligned_dir",
        type=str,
        default="sparse_align",
        help="存放对齐结果的目录（默认 workspace/sparse_align）",
    )
    parser.add_argument(
        "--show_viz",
        action="store_true",
        help="可选：在对齐完成后弹出可视化窗口，显示相机中心与GPS点的对应关系",
    )
    args = parser.parse_args()

    workspace = os.path.abspath(args.workspace)
    if not os.path.isdir(workspace):
        print(f"错误: workspace目录不存在: {workspace}")
        sys.exit(1)

    gps_images_dir = (
        os.path.abspath(args.gps_source_path)
        if args.gps_source_path
        else os.path.join(workspace, "input")
    )
    model_dir = os.path.join(workspace, "sparse", "0")
    aligned_output_dir = resolve_aligned_dir(workspace, args.aligned_dir)

    gps_image_filter = load_gps_image_filter(args.gps_image_list)
    gps_metadata = load_gps_metadata_from_json(args.gps_json)
    
    # 如果提供了CSV文件，则从中加载GPS数据
    if args.gps_csv and not gps_metadata:
        gps_metadata = load_gps_metadata_from_csv(args.gps_csv)

    success = perform_gps_alignment(
        gps_images_dir=gps_images_dir,
        model_dir=model_dir,
        aligned_output_dir=aligned_output_dir,
        gps_image_filter=gps_image_filter,
        gps_metadata=gps_metadata,
        show_viz=args.show_viz,
    )

    if success:
        print(f"\n对齐完成。结果与metadata位于: {aligned_output_dir}")
    else:
        print("\n对齐失败，请根据上方日志检查问题。")
        sys.exit(1)


if __name__ == "__main__":
    main()
