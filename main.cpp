#include <iostream>
#include <immintrin.h>
#include <smmintrin.h>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace cv;
using namespace std;

inline __m128 bilinearInterpolationFloatSSE(const float *const mat, const __m128 &x, const __m128 &y, const int &width)
{
    const __m128 w = _mm_set_ps1((float)width); //_mm_set_ps1创建一个存有四个相同值的SIMD向量
    const __m128 ones = _mm_set_ps1(1.f);

    const __m128 int_x = _mm_floor_ps(x); // 向下取整
    const __m128 int_y = _mm_floor_ps(y);

    const __m128 dx = _mm_sub_ps(x, int_x); // x - int_x
    const __m128 dy = _mm_sub_ps(y, int_y);
    const __m128 dxdy = _mm_mul_ps(dx, dy); // dx * dy

    alignas(16) float index[4]; // 开辟一块16对齐的内存用于存放4个float变量
    _mm_store_ps(index, _mm_add_ps(_mm_mul_ps(int_y, w), int_x));

    __m128 val1 = _mm_loadu_ps(mat + (int)index[0]);
    __m128 val2 = _mm_loadu_ps(mat + (int)index[0] + width);

    val1 = _mm_movelh_ps(val1, _mm_loadu_ps(mat + (int)index[1]));
    val2 = _mm_movelh_ps(val2, _mm_loadu_ps(mat + (int)index[1] + width));

    __m128 val3 = _mm_loadu_ps(mat + (int)index[2]);
    __m128 val4 = _mm_loadu_ps(mat + (int)index[2] + width);

    val3 = _mm_movelh_ps(val3, _mm_loadu_ps(mat + (int)index[3]));
    val4 = _mm_movelh_ps(val4, _mm_loadu_ps(mat + (int)index[3] + width));

    // reorder
    const __m128 sourceAt0 = _mm_shuffle_ps(val1, val3, _MM_SHUFFLE(2, 0, 2, 0));
    const __m128 sourceAt1 = _mm_shuffle_ps(val1, val3, _MM_SHUFFLE(3, 1, 3, 1));
    const __m128 sourceAtW0 = _mm_shuffle_ps(val2, val4, _MM_SHUFFLE(2, 0, 2, 0));
    const __m128 sourceAtW1 = _mm_shuffle_ps(val2, val4, _MM_SHUFFLE(3, 1, 3, 1));

    // (1.f - dx - dy + dxdy) * source[0] // [(1.f + dxdy) - (dx - dy)] * source[0]
    val1 = _mm_mul_ps(_mm_sub_ps(_mm_add_ps(ones, dxdy), _mm_add_ps(dx, dy)), sourceAt0);

    // (dx - dxdy) * source[1]
    val2 = _mm_mul_ps(_mm_sub_ps(dx, dxdy), sourceAt1);

    // (dy - dxdy) * source[width]
    val3 = _mm_mul_ps(_mm_sub_ps(dy, dxdy), sourceAtW0);

    // dxdy * source[1 + width]
    val4 = _mm_mul_ps(dxdy, sourceAtW1);

    return _mm_add_ps(_mm_add_ps(_mm_add_ps(val1, val2), val3), val4);
}

float bilinearInterpolation(const float* const mat, const float &x, const float &y, const int &width)
{
  // 这部分代码的变量名称我都按上文的示意图一一对应了，饭都喂到嘴边了，还看不懂就不礼貌了兄弟.jpg
  const int x1 = static_cast<const int>(x);// 代表x的整数部分
  const int y1 = static_cast<const int>(y);// 代表y的整数部分
  // 对于一副内存起点在mat，length行width列的图像，mat + x1 + y1*width就代表指向第y1行第x1列的像素的指针
  const float* const V1_ptr = mat + x1 + y1*width; // 指向V1的指针

#if 1// 这么写好理解些
  const float x2 = x1 + 1;
  const float y2 = y1 + 1;

  const float V1 = V1_ptr[0]; // 根据指针提取指针所指向的值，并赋值给V1，下同
  const float V2 = V1_ptr[1]; // 指针+1，也就是V1_ptr[0]右边的像素
  const float V3 = V1_ptr[width]; // width是行数，实际是0+width，V1_ptr[0+width]也就是V1_ptr[0]像素下一行的像素
  const float V4 = V1_ptr[1 + width];

  const float Va = (x2-x) * V1 + (y2-y) * V2;// 一次插值 // 由于x2-x1=1，故此处将其省略，下同
  const float Vb = (x2-x) * V3 + (y2-y) * V4;
  const float V  = (y-y1) * Va + (y2-y) * Vb;// 二次插值
 
  return  V;
#else // 这么写会快些，实际上是将公式合并同类项，简化了计算流程，与上面的写法功能上是等价的
  const float dx = x - x1; // 代表x的小数部分
  const float dy = y - y1; // 代表y的小数部分
  const float dxdy = dx*dy;

  return (dxdy * V1_ptr[1 + width] + (dx - dxdy) * V1_ptr[1] 
       + (dy - dxdy) * V1_ptr[width] + (1.f - dx - dy + dxdy) * V1_ptr[0]);
#endif
}


int main()
{
    Mat src = imread("../anya.jpg", IMREAD_GRAYSCALE); // 读取输入图片
    if (src.empty())
    {
        cout << "无法读取输入图片！" << std::endl;
        return -1;
    }

    vector<Mat> vImgs1, vImgs2;
    Mat src2, src3;

    Mat rows_pixel(1, 640, CV_8UC1); //1行640列,也就是1行长度为640个单位的像素
    Mat cols_pixel(481, 1, CV_8UC1); //481行1列,也就是1列长度为481个单位的像素
 
    //垂直方向扩维(下边加一行)
    vImgs1.push_back(src);
    vImgs1.push_back(rows_pixel);
    vconcat(vImgs1, src2); // 垂直方向拼接

    //水平方向扩维(右边加一列)
    vImgs2.push_back(src2);
    vImgs2.push_back(cols_pixel);
    hconcat(vImgs2, src3); // 水平方向拼接

    //cout << src3.rows << " " << src3.cols << endl;

    alignas(16) Mat src3_32;
    src3.convertTo(src3_32, CV_32FC1,1 / 255.0);//  扩维后的图转换为32位浮点型灰度图

    auto startTime1 = chrono::high_resolution_clock::now();
    alignas(16) Mat src4(480, 640, CV_32FC1);
    for (int y = 0; y < 480; y++)
    {
        for (int x = 0; x < 640; x += 4)
        {
            // cout << "x=" << x << " " <<"y=" << y << endl;
            __m128 srcXVec = _mm_set_ps(float(x + 3.5), float(x + 2.5), float(x + 1.5), float(x + 0.5));
            __m128 srcYVec = _mm_set_ps1(float(y + 0.5));

            __m128 result = bilinearInterpolationFloatSSE(src3_32.ptr<float>(), srcXVec, srcYVec, 641);
           
            _mm_store_ps(src4.ptr<float>(y,x), result);
        }
    }
    auto startTime2 = chrono::high_resolution_clock::now();

    alignas(16) Mat src5(480, 640, CV_32FC1);
    for (int y = 0; y < 480; y++)
    {
        for (int x = 0; x < 640; x ++)
        {
            float result = bilinearInterpolation(src3_32.ptr<float>(), float(x + 0.5), float(y + 0.5), 641);

            memcpy(src5.ptr<float>(y,x), &result, 4);
        }
    }
    auto endTime = chrono::high_resolution_clock::now();

    auto time1 = chrono::duration_cast<chrono::microseconds>(startTime2 - startTime1).count();
    auto time2 = chrono::duration_cast<chrono::microseconds>(endTime - startTime1).count();

    // imshow("src", src);
    src4.convertTo(src4, CV_8UC1, 255);//,1 / 255.0);
    imshow("result_sse", src4);
    // src5.convertTo(src5, CV_8UC1, 255);//,1 / 255.0);
    // imshow("result", src5);
    waitKey(0);

    cout << "SSE Version cost:" << time1 << endl 
         << "Normal Version cost:" << time2 << endl
         << "Speed up rate:" << (time2-time1)/100 << endl;  

    return 0;
}
