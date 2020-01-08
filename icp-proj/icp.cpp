#include <iostream>
#include <cassert>
#include <queue>
#include <pthread.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

#include <pcl/point_cloud.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <opencv2/core/core.hpp>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <condition_variable>
#include <pcl/features/normal_3d_omp.h>

#include <Eigen/Dense>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

using namespace std;

struct multiThreadInput {
    pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr icp = NULL;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr current_cloud = NULL;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr last_cloud = NULL;
    cv::Mat trans = cv::Mat::eye(4,4,CV_32F);
    bool finish = false;
};

g2o::SE3Quat toSE3Quat(const cv::Mat &cvT)
{
    Eigen::Matrix<double,3,3> R;
    R << cvT.at<float>(0,0), cvT.at<float>(0,1), cvT.at<float>(0,2),
         cvT.at<float>(1,0), cvT.at<float>(1,1), cvT.at<float>(1,2),
         cvT.at<float>(2,0), cvT.at<float>(2,1), cvT.at<float>(2,2);

    Eigen::Matrix<double,3,1> t(cvT.at<float>(0,3), cvT.at<float>(1,3), cvT.at<float>(2,3));

    return g2o::SE3Quat(R,t);
}

cv::Mat toCvMat(const Eigen::Matrix<float, 4, 4> &m)
{
    cv::Mat cvMat(4,4,CV_32F);
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::Mat toCvMat(const Eigen::Matrix3d &m)
{
    cv::Mat cvMat(3,3,CV_32F);
    for(int i=0;i<3;i++)
        for(int j=0; j<3; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::Mat toCvMat( const std::vector<float>& v )
{
    Eigen::Quaterniond q;
    q.x()  = v[0];
    q.y()  = v[1];
    q.z()  = v[2];
    q.w()  = v[3];
    Eigen::Matrix<double,3,3>eigMat(q);
    cv::Mat M = toCvMat(eigMat);
    return M;
}

cv::Mat toCvMat( Eigen::Matrix<double,7,1>& v)
{
    cv::Mat T = cv::Mat::eye(4,4,CV_32F);
    for (int j = 0; j < 3; j++) {
        T.at<float>(j,3) = v[j];
    }
    Eigen::Quaterniond q;
    q.x()  = v[3];
    q.y()  = v[4];
    q.z()  = v[5];
    q.w()  = v[6];
    Eigen::Matrix<double,3,3>eigMat(q);
    cv::Mat R = toCvMat(eigMat);
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            T.at<float>(j,k) = R.at<float>(j,k);
        }
    }
    return T;
}

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr generatePointCloud(cv::Mat& frame, long unsigned int id, cv::Mat& color, cv::Mat& depth, double fx, double fy, double cx, double cy)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGBA>);
    for ( int m=int(depth.rows*0.1); m<int(depth.rows*0.9); m+=3 )
    {
        for ( int n=int(depth.cols*0.1); n<int(depth.cols*0.9); n+=3 )
        {
            float d = depth.ptr<float>(m)[n];
            // cout << "depth:" << d << endl;
            if (d < 0.1 || d>3) // default d>10
                continue;
            pcl::PointXYZRGBA p;
            p.z = d;
            p.x = ( n - cx) * p.z / fx;
            p.y = ( m - cy) * p.z / fy;
            
            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];
                
            tmp->points.push_back(p);
        }
    }
    
    Eigen::Isometry3d T = toSE3Quat( frame );
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;
    
    cout<<"generate point cloud for frame "<< id <<", size="<<cloud->points.size()<<endl;
    return cloud;
}


void addNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	       pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals
) {
    pcl::PointCloud<pcl::Normal>::Ptr normals ( new pcl::PointCloud<pcl::Normal> );

    pcl::search::KdTree<pcl::PointXYZ>::Ptr searchTree (new pcl::search::KdTree<pcl::PointXYZ>);
    searchTree->setInputCloud ( cloud );

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEstimator(16);
    normalEstimator.setInputCloud ( cloud );
    normalEstimator.setSearchMethod ( searchTree );
    normalEstimator.setKSearch ( 100 );
    // normalEstimator.setRadiusSearch (0.05);
    normalEstimator.compute ( *normals );
    
    pcl::concatenateFields( *cloud, *normals, *cloud_with_normals );
}

// void *icp_multi_thread(void *input_param) {
//     struct multiThreadInput * input = (struct multiThreadInput *)input_param;
//     pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBA, pcl::PointXYZRGBA> icp;
//     icp.setMaxCorrespondenceDistance(0.05);
// 	icp.setTransformationEpsilon(1e-9);
// 	icp.setEuclideanFitnessEpsilon(1);
// 	icp.setMaximumIterations (30);
//     pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudxyz1(new pcl::PointCloud<pcl::PointXYZRGBA> ());
//     pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudxyz2(new pcl::PointCloud<pcl::PointXYZRGBA> ());
//     pcl::copyPointCloud(*input->current_cloud, *cloudxyz1);
//     pcl::copyPointCloud(*input->last_cloud, *cloudxyz2);
//     icp.setInputSource(cloudxyz1);
//     // cout << input->current_cloud->points.size() << endl;
//     icp.setInputTarget(cloudxyz2);
//     // cout << input->last_cloud->points.size() << endl;
//     icp.align(*cloudxyz1);
//     if (icp.hasConverged()) { 
//         input->trans = toCvMat(icp.getFinalTransformation());
//     }
//     input->finish = true;
// }

void *icp_multi_thread(void *input_param) {
    struct multiThreadInput * input = (struct multiThreadInput *)input_param;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudxyzrgb1(new pcl::PointCloud<pcl::PointXYZRGBA> ());
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudxyzrgb2(new pcl::PointCloud<pcl::PointXYZRGBA> ());
    pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBA, pcl::PointXYZRGBA>::Ptr icp(new pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA>());
    icp->setMaxCorrespondenceDistance(0.05);
	icp->setTransformationEpsilon(1e-9);
	icp->setEuclideanFitnessEpsilon(1);
	icp->setMaximumIterations (30);
    pcl::copyPointCloud(*input->current_cloud, *cloudxyzrgb1);
    icp->setInputSource(cloudxyzrgb1);
    // cout << input->current_cloud->points.size() << endl;
    pcl::copyPointCloud(*input->last_cloud, *cloudxyzrgb2);
    icp->setInputTarget(cloudxyzrgb2);
    // cout << input->last_cloud->points.size() << endl;
    // sleep(10);
    icp->align(*(cloudxyzrgb1));
    if (icp->hasConverged()) { 
        input->trans = toCvMat(icp->getFinalTransformation());
    }
    input->finish = true;
}

int main (int argc, char** argv)
{
    string rootdir = argv[1];
    ifstream f_associated, f_originpose, f_calibration;
    f_associated.open( rootdir + "associated.txt", ios_base::in );
    f_originpose.open( rootdir + "originpose.txt",  ios_base::in );
    f_calibration.open( rootdir + "calibration.txt",  ios_base::in );

    // read input data filenames
    std::vector<std::string> vstrTimeStamp;
    std::vector<std::string> vstrImageFilenamesD;
    std::vector<std::string> vstrImageFilenamesRGB;
    while(!f_associated.eof()) {
        string s;
        getline(f_associated,s);
        if(!s.empty()) {
            stringstream ss;
            ss << s;
            string t;
            string sRGB, sD;
            ss >> t;
            vstrTimeStamp.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
    f_associated.close();

    // read transform message
    std::vector<std::vector<double>> vvTransform;
    int index = 0;
    while(!f_originpose.eof()) {
        string s;
        getline(f_originpose, s);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            string t;
            double val;
            vector<double> trans;
            ss >> t;
            assert(t == vstrTimeStamp[index] && "timestamp of associated and originpose different");
            index++;
            for (int i = 0; i < 7; i++) {
                ss >> val;
                trans.push_back(val);
            }
            vvTransform.push_back(trans);
        }
    }
    f_originpose.close();

    // read calibration
    double fx, fy, cx, cy;
    f_calibration >> fx >> fy >> cx >> cy;
    f_calibration.close();

    // start icp
    pcl::VoxelGrid<pcl::PointXYZRGBA> voxel;
    voxel.setLeafSize( 0.01, 0.01, 0.01 );
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
    vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> last_cloud_queue;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr global_cloud(new pcl::PointCloud<pcl::PointXYZRGBA> ());
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudxyzrgb(new pcl::PointCloud<pcl::PointXYZRGBA> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudxyz(new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGBA>);
    cv::Mat accumulate_transform = cv::Mat::eye(4,4,CV_32F);
    pcl::visualization::CloudViewer viewer("viewer");
    // icp algorithm and parameters setting, we can also set outlier percent ......
    pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr icp(new pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>());
    icp->setMaxCorrespondenceDistance(0.05);
	icp->setTransformationEpsilon(1e-9);
	icp->setEuclideanFitnessEpsilon(1);
	icp->setMaximumIterations (30);

    for (int i = 0; i < vstrTimeStamp.size(); i+=1) {

        cv::Mat T = cv::Mat::zeros(4,4,CV_32F);
        std::vector<float> Quat(4);
        for (int j = 0; j < 3; j++) {
            T.at<float>(j,3) = vvTransform[i][j];
        }
        for (int j = 3; j < 7; j++) {
            Quat[j-3] = vvTransform[i][j];
        }
        cv::Mat R = toCvMat( Quat );
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                T.at<float>(j,k) = R.at<float>(j,k);
            }
        }
        T.at<float>(3,3) = 1;

        cv::Mat rgb_img, depth_img;
        try {
            rgb_img = cv::imread(rootdir + vstrImageFilenamesRGB[i], cv::IMREAD_COLOR);
            depth_img = cv::imread(rootdir + vstrImageFilenamesD[i], cv::IMREAD_ANYDEPTH );
        } catch (exception e) {
            cout << "depth or rgb image loss";
        }
        depth_img.convertTo(depth_img, CV_32F, 1.0f/1000.0);
        cloudxyzrgb = generatePointCloud( T, i, rgb_img, depth_img, fx, fy, cx, cy );
        // 滤波下采样
        voxel.setInputCloud( cloudxyzrgb );
        voxel.filter( *tmp );
        cloudxyzrgb->swap( *tmp );
        // 转到世界坐标系
        Eigen::Isometry3d Tworld = toSE3Quat( accumulate_transform );
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr _cloudxyzrgb(new pcl::PointCloud<pcl::PointXYZRGBA> ());
        pcl::transformPointCloud( *cloudxyzrgb, *_cloudxyzrgb, Tworld.matrix());
        cloudxyzrgb = _cloudxyzrgb;
        // 好像只有cloudxyz可以添加normal，所以需要做这样的转换，目的是给cloudxyzrgb current_cloud添加normal
        pcl::copyPointCloud(*cloudxyzrgb, *current_cloud);
        pcl::copyPointCloud(*cloudxyzrgb, *cloudxyz);
        addNormal(cloudxyz, current_cloud);

        if (!current_cloud->empty() && !last_cloud_queue.empty()) {
            // 并行计算前五帧的icp然后取均值
            pthread_t tids[5];
            int num = last_cloud_queue.size();
            struct multiThreadInput input_param[num];
            for (int j = 0; j < num; j++) {
                pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>::Ptr _icp(new pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>());
                *_icp = *icp;
                input_param[j].icp = _icp;
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr _current_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
                *_current_cloud = *current_cloud;
                input_param[j].current_cloud = _current_cloud;
                input_param[j].last_cloud = last_cloud_queue[j];
            }
            // cout << "num:" << num << endl;
            
            Eigen::Matrix<double,7,1> mean_trans;
            for (int j = 0; j < 7; j++) {
                mean_trans[j] = 0;
                if (j == 6) mean_trans[j] = 1;
            }
            for(int j = 0; j < num; ++j) {
                int ret = pthread_create(&tids[j], NULL, icp_multi_thread, (void *)(&input_param[j]));
                if (ret != 0) {
                    cout << "pthread_create error: error_code=" << ret << endl;
                }
            }
            while (1) {
                int flag = 0;
                for (int j = 0; j < num; j++) {
                    if (!input_param[j].finish) flag = 1; // 只要有一个子线程没结束，父线程就循环等待
                }
                if (flag == 0) break;
                usleep(300);
                cout << "num:" << num << endl;
            }
            //等各个线程退出后，进程才结束
            for (int j = 0; j < num; j++) {
                for (int k = 0; k < 7; k++) {
                    mean_trans[k] += toSE3Quat(input_param[j].trans).toVector()[k];
                }
            }
            // cout << mean_trans << endl;
            Tworld = toSE3Quat( toCvMat(mean_trans)/num );
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr _current_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
            pcl::transformPointCloud( *current_cloud, *_current_cloud, Tworld.matrix());
            current_cloud = _current_cloud;
            accumulate_transform = accumulate_transform*(toCvMat(mean_trans));
        } else if (i!=0) {
            cout << "some cloud was empty";
            exit(0);
        }
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr last_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
        pcl::copyPointCloud(*current_cloud, *last_cloud);
        last_cloud_queue.push_back(last_cloud);
        if (last_cloud_queue.size() > 5) {
            last_cloud_queue.erase(last_cloud_queue.begin());
        }

        pcl::copyPointCloud(*current_cloud, *cloudxyzrgb);
        *global_cloud += *cloudxyzrgb;
        voxel.setInputCloud( global_cloud );
        voxel.filter( *tmp );
        global_cloud->swap( *tmp );
        viewer.showCloud(global_cloud);
    }

    std::string pcd_path = std::string(rootdir + std::string("map.pcd"));
    pcl::io::savePCDFileBinary(pcd_path, *global_cloud);
    return 0;
}

