#include "voxel_map2d.h"

VoxelKey2D VoxelMapBuilder2D::index(const Vec2f &point)
{
    auto idx = (point / m_config.resolution).array().floor();
    return VoxelKey2D(static_cast<int64_t>(idx(0)), static_cast<int64_t>(idx(1)));
}

bool VoxelMapBuilder2D::addPoint(const VoxelKey2D &key, const Vec2f &point)
{
    VoxelGrid2D *value;
    auto iter = voxel_map.find(key);
    if (iter == voxel_map.end())
    {
        value = &(voxel_map.insert(std::make_pair(key, VoxelGrid2D())).first->second);
        m_cache.push_front(key);//将新体素键插入到链表头部
        value->position_iter = m_cache.begin();//保存迭代器位置，便于后续快速删除
        //LRU（最近最少使用）策略：
        if (m_cache.size() > m_config.max_voxel_number)
        {
            voxel_map.erase(m_cache.back());//从哈希表中删除该体素
            m_cache.pop_back();//从链表中删除该键
        }
    }
    else
    {
        // m_cache.splice(m_cache.begin(),   // 目标位置：链表开头
        //        m_cache,            // 源链表（同一个链表）
        //        value->position_iter); // 要移动的元素迭代器
        value = &(iter->second);
        m_cache.splice(m_cache.begin(), m_cache, value->position_iter);
    }

    if (value->count >= m_config.max_update_thresh)
        return false;
    value->mean += (point - value->mean) / (value->count + 1.0f);
    value->ppt += point * point.transpose();
    value->count += 1.0f;
    return true;
}
void VoxelMapBuilder2D::addClouds(const Vec3f &pose, const std::vector<Vec2f> &cloud)
{
    Eigen::Affine2f transform = Eigen::Translation2f(pose.head<2>()) * Eigen::Rotation2Df(pose(2));
    std::unordered_set<VoxelKey2D, VoxelKeyHash2D> visited;
    visited.reserve(cloud.size());
    for (auto &point : cloud)
    {
        Vec2f p = transform * point;// 将点从局部坐标系转换到全局坐标系
        VoxelKey2D idx = index(p);
        if (addPoint(idx, p))
            visited.insert(idx);
    }
    for (auto &idx : visited)
        updatePlane(idx);
}
bool VoxelMapBuilder2D::optimize(ScanPack &package)
{

    Vec3f pose = package.pose;// 初始位姿 [x, y, theta]
    Mat3f Hess;// 3×3海森矩阵（信息矩阵）
    Vec3f lb;// 雅可比矩阵×残差（负梯度方向）
    Vec3f delta;// 位姿更新量
    //高斯-牛顿迭代主循环
    for (unsigned int i = 0; i < m_config.max_iteration; i++)
    {
        Eigen::Affine2f transform = Eigen::Translation2f(pose.head<2>()) * Eigen::Rotation2Df(pose(2));
        Hess.setZero();//
        lb.setZero();
        unsigned int valid_count = 0;//有效匹配点计数
        for (unsigned int i = 0; i < package.points.size(); i++)
        {
            Vec2f point = package.points[i];
            Vec2f point_world = transform * point;
            VoxelKey2D pid = index(point_world);
            auto iter = voxel_map.find(pid);
            if (iter == voxel_map.end())
                continue;

            if (iter->second.is_plane)
            {
                Eigen::Matrix<float, 2, 3> dp;//雅可比矩阵计算
                dp.block<2, 2>(0, 0).setIdentity();
                dp.col(2) = Vec2f(-sin(pose(2)) * point.x() - cos(pose(2)) * point.y(), cos(pose(2)) * point.x() - sin(pose(2)) * point.y());
                float loss = iter->second.norm.transpose() * (point_world - iter->second.mean);//残差计算，点到线段的距离
                Vec3f jacc = dp.transpose() * iter->second.norm;//完整雅可比计算
                Hess += jacc * 1000.0 * jacc.transpose();//构建高斯-牛顿方程
                lb -= 1000.0 * jacc * loss;
                valid_count++;
            }
        }
        if (valid_count == 0)
        {
            std::cout << "no valid point" << std::endl;
            return false;
        }

        delta = Hess.inverse() * lb;
        pose += delta;
        //收敛判断
        if (abs(delta.x()) < 0.001 && abs(delta.y()) < 0.001 && abs(delta.z()) < 0.002)
            break;
    }
    pose(2) = normalize_angle(pose(2));//角度归一化-180~180度
    package.pose = pose;
    return true;
}
void VoxelMapBuilder2D::update(ScanPack &package)
{   
    //首次更新，直接添加点云，无需优化
    if (!m_is_initialized)
    {
        m_is_initialized = true;
        addClouds(package.pose, package.points);
        return;
    }
    optimize(package);

    addClouds(package.pose, package.points);
}
void VoxelMapBuilder2D::updatePlane(VoxelGrid2D *grid)
{   
    //点数太少，统计不可靠，直接返回，不进行平面检测
    if (grid->count < m_config.update_thresh)
        return;
    //点数太多，可能已稳定或过度更新，直接返回，不进行平面检测
    if (grid->count > m_config.max_update_thresh)
        return;
    //计算协方差矩阵    
    Mat2f cov = grid->ppt / grid->count - grid->mean * grid->mean.transpose();
    //对2×2的协方差矩阵进行特征值分解
    Eigen::SelfAdjointEigenSolver<Mat2f> es(cov);
    //提取最小特征值
    float eval = es.eigenvalues()(0);
    if (eval > m_config.plane_threshold)
    {
        grid->is_plane = false;
        return;
    }
    grid->is_plane = true;
    grid->norm = es.eigenvectors().col(0);//对应最小特征值的特征向量
}
void VoxelMapBuilder2D::updatePlane(const VoxelKey2D &key)
{
    auto iter = voxel_map.find(key);
    if (iter == voxel_map.end())
        return;
    updatePlane(&(iter->second));
}