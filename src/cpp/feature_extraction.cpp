#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Helper Functions (Internal) ---

// Helper for mean
double calculate_mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}

// Helper for standard deviation
double calculate_std(const std::vector<double>& v, double mean) {
    if (v.empty()) return 0.0;
    double sq_sum = 0.0;
    for (double d : v) {
        sq_sum += (d - mean) * (d - mean);
    }
    return std::sqrt(sq_sum / v.size());
}

// Struct to hold skewness and kurtosis
struct SkewKurt {
    double skew;
    double kurtosis;
};

// Calculates Skewness and Kurtosis for a dataset
SkewKurt calculate_skew_kurtosis(const cv::Mat& data) {
    if (data.empty()) return {0.0, 0.0};
    
    cv::Mat mean_mat, std_mat;
    cv::meanStdDev(data, mean_mat, std_mat);
    double mean = mean_mat.at<double>(0);
    double std_dev = std_mat.at<double>(0);

    if (std_dev < 1e-9) return {0.0, 0.0};

    double n = (double)data.total();
    double sum_cubed = 0.0;
    double sum_quad = 0.0;

    // Iterate efficiently
    if (data.isContinuous()) {
        const double* ptr = data.ptr<double>(0); 
        for (size_t i = 0; i < data.total(); ++i) {
            double diff = (ptr[i] - mean) / std_dev;
            double diff_sq = diff * diff;
            sum_cubed += diff_sq * diff;
            sum_quad += diff_sq * diff_sq;
        }
    } else {
        for (int i = 0; i < data.rows; ++i) {
            for (int j = 0; j < data.cols; ++j) {
                double val = data.at<double>(i, j);
                double diff = (val - mean) / std_dev;
                double diff_sq = diff * diff;
                sum_cubed += diff_sq * diff;
                sum_quad += diff_sq * diff_sq;
            }
        }
    }

    double skew = sum_cubed / n;
    double kurtosis = (sum_quad / n) - 3.0; // Fisher kurtosis
    
    return {skew, kurtosis};
}

// Global cache for Gabor kernels to avoid re-creation overhead
static std::vector<cv::Mat> GABOR_KERNELS_REAL;
static std::vector<cv::Mat> GABOR_KERNELS_IMAG;
static bool GABOR_INIT = false;

void init_gabor_kernels() {
    if (GABOR_INIT) return;

    double sigmas[] = {1.0, 3.0};
    double freqs[] = {0.05, 0.25};
    
    // Order loops to match Python: theta, sigma, frequency
    for (int t = 0; t < 4; ++t) {
        double theta = CV_PI * t / 4.0;
        for (double sigma : sigmas) {
            for (double freq : freqs) {
                double lambd = 1.0 / freq;
                // Real part (psi=0)
                cv::Mat k_real = cv::getGaborKernel(cv::Size(5,5), sigma, theta, lambd, 1.0, 0, CV_64F);
                // Imaginary part (psi=pi/2)
                cv::Mat k_imag = cv::getGaborKernel(cv::Size(5,5), sigma, theta, lambd, 1.0, CV_PI/2, CV_64F);
                
                GABOR_KERNELS_REAL.push_back(k_real);
                GABOR_KERNELS_IMAG.push_back(k_imag);
            }
        }
    }
    GABOR_INIT = true;
}

extern "C" {

    /**
     * @brief Applies CLAHE contrast enhancement.
     */
    void contrast_enhancement_c(
        unsigned char* src_data, int rows, int cols, 
        double clip_limit, int grid_x, int grid_y, 
        unsigned char* dst_data) 
    {
        cv::Mat src(rows, cols, CV_8UC3, src_data);
        cv::Mat dst(rows, cols, CV_8UC3, dst_data);

        cv::Mat hsv;
        cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

        std::vector<cv::Mat> channels;
        cv::split(hsv, channels);

        auto clahe = cv::createCLAHE(clip_limit, cv::Size(grid_x, grid_y));
        clahe->apply(channels[2], channels[2]);

        cv::merge(channels, hsv);
        cv::cvtColor(hsv, dst, cv::COLOR_HSV2BGR);
    }

    /**
     * @brief Applies contrast stretching based on percentiles.
     */
    void contrast_stretch_c(
        unsigned char* src_data, int rows, int cols, int channels,
        unsigned char* dst_data)
    {
        int type = (channels == 3) ? CV_8UC3 : CV_8UC1;
        cv::Mat src(rows, cols, type, src_data);
        cv::Mat dst(rows, cols, type, dst_data);
        
        cv::Mat src_f;
        src.convertTo(src_f, CV_32F);

        std::vector<cv::Mat> split_channels;
        cv::split(src_f, split_channels);
        std::vector<cv::Mat> dst_channels;

        // Process each channel independently
        for (auto& ch : split_channels) {
            // Find percentiles using sorting (std::nth_element for efficiency)
            cv::Mat flat = ch.reshape(1, 1).clone();
            std::vector<float> data(flat.begin<float>(), flat.end<float>());
            
            size_t n = data.size();
            size_t idx_low = (size_t)(0.015 * n);
            size_t idx_high = (size_t)(0.985 * n);

            std::nth_element(data.begin(), data.begin() + idx_low, data.end());
            float min_val = data[idx_low];

            std::nth_element(data.begin(), data.begin() + idx_high, data.end());
            float max_val = data[idx_high];

            if (std::abs(max_val - min_val) < 1e-6) {
                max_val = min_val + 1.0f;
            }

            // Stretch: (img - min) / (max - min) * 255
            cv::Mat stretched = (ch - min_val) / (max_val - min_val) * 255.0f;
            
            // Clip
            cv::threshold(stretched, stretched, 255, 255, cv::THRESH_TRUNC);
            cv::threshold(stretched, stretched, 0, 0, cv::THRESH_TOZERO);
            
            dst_channels.push_back(stretched);
        }

        cv::Mat merged;
        cv::merge(dst_channels, merged);
        merged.convertTo(dst, CV_8U);
    }

    /**
     * @brief Calculates normalized color histogram.
     */
    void extract_color_features_c(
        unsigned char* img_data, int rows, int cols, 
        int bins, int range_min, int range_max,
        double* out_features)
    {
        cv::Mat img(rows, cols, CV_8UC1, img_data);
        
        int histSize[] = {bins};
        float hranges[] = { (float)range_min, (float)range_max };
        const float* ranges[] = { hranges };
        int channels[] = {0};

        cv::Mat hist;
        cv::calcHist(&img, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

        // Normalize density=True
        double sum = cv::sum(hist)[0];
        if (sum > 0) hist /= sum;

        for (int i = 0; i < bins; ++i) {
            out_features[i] = (double)hist.at<float>(i);
        }
    }

    /**
     * @brief Calculates LBP features.
     */
    void extract_lbp_features_c(
        unsigned char* gray_data, int rows, int cols, int n_points,
        double* out_features)
    {
        cv::Mat gray(rows, cols, CV_8UC1, gray_data);
        cv::Mat lbp_img = cv::Mat::zeros(rows, cols, CV_32S);

        // Compute LBP codes
        // Skipping 1-pixel border
        for (int i = 1; i < rows - 1; ++i) {
            const unsigned char* prev = gray.ptr<unsigned char>(i - 1);
            const unsigned char* curr = gray.ptr<unsigned char>(i);
            const unsigned char* next = gray.ptr<unsigned char>(i + 1);
            int* dst = lbp_img.ptr<int>(i);

            for (int j = 1; j < cols - 1; ++j) {
                unsigned char center = curr[j];
                int code = 0;
                code |= (prev[j-1] >= center) << 7;
                code |= (prev[j]   >= center) << 6;
                code |= (prev[j+1] >= center) << 5;
                code |= (curr[j+1] >= center) << 4;
                code |= (next[j+1] >= center) << 3;
                code |= (next[j]   >= center) << 2;
                code |= (next[j-1] >= center) << 1;
                code |= (curr[j-1] >= center) << 0;
                dst[j] = code;
            }
        }

        // Compute Histogram of LBP codes
        cv::Mat lbp_float;
        lbp_img.convertTo(lbp_float, CV_32F);

        int n_bins = n_points + 2; 
        int histSize[] = {n_bins};
        float range[] = { 0.0f, (float)n_bins };
        const float* ranges[] = { range };
        int channels[] = {0};

        cv::Mat hist;
        cv::calcHist(&lbp_float, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

        double sum = cv::sum(hist)[0];
        if (sum > 0) hist /= sum;

        for (int i = 0; i < n_bins; ++i) {
            out_features[i] = (double)hist.at<float>(i);
        }
    }

    /**
     * @brief Calculates GLCM features for 4 angles.
     */
    void extract_glcm_features_c(
        unsigned char* img_data, int rows, int cols,
        double* out_features)
    {
        cv::Mat img(rows, cols, CV_8UC1, img_data);
        
        // Angles: 0, 45, 90, 135
        int drs[] = {0, -1, -1, -1};
        int dcs[] = {1, 1, 0, -1};

        int feat_idx = 0;

        for (int k = 0; k < 4; ++k) {
            int dr = drs[k];
            int dc = dcs[k];

            double glcm[256][256] = {0};
            double total_sum = 0.0;

            int r_start = std::max(0, -dr);
            int r_end = std::min(rows, rows - dr);
            int c_start = std::max(0, -dc);
            int c_end = std::min(cols, cols - dc);

            for (int r = r_start; r < r_end; ++r) {
                const unsigned char* ptr_src = img.ptr<unsigned char>(r);
                const unsigned char* ptr_dst = img.ptr<unsigned char>(r + dr);
                for (int c = c_start; c < c_end; ++c) {
                    glcm[ptr_src[c]][ptr_dst[c + dc]]++;
                    total_sum++;
                }
            }

            // Symmetric Normalization Factor (sum(M + M.T) = 2 * sum(M))
            double norm = (total_sum > 0) ? 1.0 / (2.0 * total_sum) : 0.0;

            double contrast = 0, dissimilarity = 0, homogeneity = 0, asm_val = 0;
            double mean_i = 0, mean_j = 0;

            // Pass 1: Basic stats + Means
            for(int i=0; i<256; ++i) {
                for(int j=0; j<256; ++j) {
                    double p = (glcm[i][j] + glcm[j][i]) * norm;
                    if (p > 0) {
                        double diff = (double)(i - j);
                        contrast += p * diff * diff;
                        dissimilarity += p * std::abs(diff);
                        homogeneity += p / (1.0 + diff * diff);
                        asm_val += p * p;
                        mean_i += i * p;
                        mean_j += j * p;
                    }
                }
            }

            double energy = std::sqrt(asm_val);
            
            // Pass 2: Correlation (needs means)
            double var_i = 0, var_j = 0, cov = 0;
            for(int i=0; i<256; ++i) {
                for(int j=0; j<256; ++j) {
                    double p = (glcm[i][j] + glcm[j][i]) * norm;
                    if (p > 0) {
                        var_i += p * (i - mean_i) * (i - mean_i);
                        var_j += p * (j - mean_j) * (j - mean_j);
                        cov   += p * (i - mean_i) * (j - mean_j);
                    }
                }
            }
            
            double std_i = std::sqrt(var_i);
            double std_j = std::sqrt(var_j);
            double correlation = (std_i * std_j > 1e-10) ? cov / (std_i * std_j) : 0.0;

            out_features[feat_idx++] = contrast;
            out_features[feat_idx++] = dissimilarity;
            out_features[feat_idx++] = homogeneity;
            out_features[feat_idx++] = energy;
            out_features[feat_idx++] = asm_val;
            out_features[feat_idx++] = correlation;
        }
    }

    /**
     * @brief Calculates Gabor filter features.
     */
    void extract_gabor_features_c(
        unsigned char* img_data, int rows, int cols,
        double* out_features)
    {
        if (!GABOR_INIT) init_gabor_kernels();

        cv::Mat img(rows, cols, CV_8UC1, img_data);
        cv::Mat img_f;
        img.convertTo(img_f, CV_64F); 

        int feat_idx = 0;
        size_t n_kernels = GABOR_KERNELS_REAL.size();

        for (size_t i = 0; i < n_kernels; ++i) {
            cv::Mat r_real, r_imag;
            cv::filter2D(img_f, r_real, -1, GABOR_KERNELS_REAL[i]);
            cv::filter2D(img_f, r_imag, -1, GABOR_KERNELS_IMAG[i]);

            cv::Mat r_sq = r_real.mul(r_real) + r_imag.mul(r_imag);
            
            // 1. Local Energy
            double local_energy = cv::sum(r_sq)[0];

            // 2. Mean Amplitude
            cv::Mat amp;
            cv::sqrt(r_sq, amp);
            double mean_amplitude = cv::mean(amp)[0];

            // 3. Phase Stats
            cv::Mat phase;
            cv::phase(r_real, r_imag, phase); // returns radians
            
            // Reshape for stats
            cv::Mat phase_flat = phase.reshape(1, phase.total());
            cv::Mat phase_64; 
            phase_flat.convertTo(phase_64, CV_64F);

            cv::Scalar mean_std = cv::mean(phase_64);
            cv::Mat m, s;
            cv::meanStdDev(phase_64, m, s);
            SkewKurt sk = calculate_skew_kurtosis(phase_64);

            out_features[feat_idx++] = local_energy;
            out_features[feat_idx++] = mean_amplitude;
            out_features[feat_idx++] = m.at<double>(0);     // Mean
            out_features[feat_idx++] = s.at<double>(0);     // Std
            out_features[feat_idx++] = sk.skew;
            out_features[feat_idx++] = sk.kurtosis;
        }
    }

    /**
     * @brief Calculates principal plane fitting features (3D).
     */
    void extract_principal_plane_features_c(
        double* depth_data, int rows, int cols, double eps,
        double* out_features)
    {
        // Copy depth to vector for processing
        int n_pixels = rows * cols;
        Eigen::VectorXd z(n_pixels);
        Eigen::MatrixXd A(n_pixels, 9);
        
        // Also compute rugosity
        double As = 0.0;
        double Ap = (double)((cols - 1) * (rows - 1));

        int idx = 0;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                double val_z = depth_data[r * cols + c];
                z(idx) = val_z;

                double x = (double)c;
                double y = (double)r;

                // Design matrix for polynomial
                A(idx, 0) = 1.0;
                A(idx, 1) = x;
                A(idx, 2) = y;
                A(idx, 3) = x*x;
                A(idx, 4) = x*y;
                A(idx, 5) = y*y;
                A(idx, 6) = x*x*y;
                A(idx, 7) = x*y*y;
                A(idx, 8) = y*y*y;

                // Rugosity Area Calculation
                if (r < rows - 1 && c < cols - 1) {
                    double z1 = depth_data[r * cols + c];
                    double z2 = depth_data[(r + 1) * cols + c];
                    double z3 = depth_data[r * cols + (c + 1)];
                    double z4 = depth_data[(r + 1) * cols + (c + 1)];

                    double area1 = 0.5 * std::sqrt(std::pow(z3 - z1, 2) + std::pow(z2 - z1, 2) + 1.0);
                    double area2 = 0.5 * std::sqrt(std::pow(z4 - z2, 2) + std::pow(z4 - z3, 2) + 1.0);
                    As += area1 + area2;
                }
                idx++;
            }
        }

        // 1. Z Statistics
        // Use OpenCV for easy stats on the buffer
        cv::Mat z_mat(rows, cols, CV_64F, depth_data);
        cv::Scalar z_mean, z_std;
        cv::meanStdDev(z_mat, z_mean, z_std);
        SkewKurt z_sk = calculate_skew_kurtosis(z_mat);

        // 2. Least Squares Solve
        Eigen::VectorXd coeffs = A.colPivHouseholderQr().solve(z);

        // 3. Distance Statistics
        Eigen::VectorXd z_fitted = A * coeffs;
        Eigen::VectorXd diff = (z - z_fitted).cwiseAbs();
        double dist_mean = diff.mean();
        double dist_std = std::sqrt((diff.array() - dist_mean).square().sum() / (diff.size()));

        // 4. Angle
        double n_x = coeffs(1);
        double n_y = coeffs(2);
        double n_z = -1.0; 
        double norm_mag = std::sqrt(n_x*n_x + n_y*n_y + n_z*n_z);
        double theta = std::acos(n_z / (norm_mag + eps)) * 180.0 / M_PI;

        double rugosity = (Ap > 0) ? As / Ap : 0.0;

        // Fill Output
        int out_idx = 0;
        out_features[out_idx++] = z_std[0];
        out_features[out_idx++] = z_sk.skew;
        out_features[out_idx++] = z_sk.kurtosis;
        for(int i=0; i<9; ++i) out_features[out_idx++] = coeffs(i);
        out_features[out_idx++] = theta;
        out_features[out_idx++] = dist_mean;
        out_features[out_idx++] = dist_std;
        out_features[out_idx++] = rugosity;
    }

    /**
     * @brief Calculates Curvature and Surface Normal features.
     */
    void extract_curvatures_c(
        double* depth_data, int rows, int cols, double eps,
        double* out_features)
    {
        cv::Mat z(rows, cols, CV_64F, depth_data);

        // Calculate Derivatives using central difference (replicate border)
        // Kernel: [-0.5, 0, 0.5]
        cv::Mat kx = (cv::Mat_<double>(1,3) << -0.5, 0, 0.5);
        cv::Mat ky = (cv::Mat_<double>(3,1) << -0.5, 0, 0.5);

        cv::Mat dx, dy, dxdx, dydy, dxdy, dydx;
        cv::filter2D(z, dx, -1, kx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(z, dy, -1, ky, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(dx, dxdx, -1, kx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(dy, dydy, -1, ky, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(dx, dxdy, -1, ky, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(dy, dydx, -1, kx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

        // Feature Accumulators
        std::vector<double> G_vec, M_vec, k1_vec, k2_vec, S_vec, C_vec, alpha_vec, beta_vec;
        int n_pixels = rows * cols;
        G_vec.reserve(n_pixels); M_vec.reserve(n_pixels); 
        k1_vec.reserve(n_pixels); k2_vec.reserve(n_pixels);
        S_vec.reserve(n_pixels); C_vec.reserve(n_pixels);
        alpha_vec.reserve(n_pixels); beta_vec.reserve(n_pixels);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double _dx = dx.at<double>(i, j);
                double _dy = dy.at<double>(i, j);
                double _dxdx = dxdx.at<double>(i, j);
                double _dydy = dydy.at<double>(i, j);
                double _dxdy = dxdy.at<double>(i, j);
                double _dydx = dydx.at<double>(i, j);

                double denom = 1.0 + _dx*_dx + _dy*_dy;
                double G = (_dxdx * _dydy - _dxdy * _dydx) / (denom * denom);
                double M = (_dydy + _dxdx) / (2.0 * std::pow(denom, 1.5));

                double discriminant = std::sqrt(std::max(M*M - G, 0.0));
                double k1 = M + discriminant;
                double k2 = M - discriminant;

                double S = (2.0 / M_PI) * std::atan2(k2 + k1, k2 - k1);
                double C = std::sqrt((k1*k1 + k2*k2) / 2.0);

                // Normals
                double nx = -_dx;
                double ny = -_dy;
                double nz = 1.0;
                double norm = std::sqrt(nx*nx + ny*ny + nz*nz) + eps;
                nx /= norm; ny /= norm; nz /= norm;

                double alpha = std::atan2(ny, nx);
                double beta = std::atan2(nz, std::sqrt(nx*nx + ny*ny));

                G_vec.push_back(G);
                M_vec.push_back(M);
                k1_vec.push_back(k1);
                k2_vec.push_back(k2);
                S_vec.push_back(S);
                C_vec.push_back(C);
                alpha_vec.push_back(alpha);
                beta_vec.push_back(beta);
            }
        }

        // Compute Mean and Std for all and fill output
        int out_idx = 0;
        auto push_stats = [&](const std::vector<double>& v) {
            double mean = calculate_mean(v);
            double std = calculate_std(v, mean);
            out_features[out_idx++] = mean;
            out_features[out_idx++] = std;
        };

        push_stats(G_vec);
        push_stats(M_vec);
        push_stats(k1_vec);
        push_stats(k2_vec);
        push_stats(S_vec);
        push_stats(C_vec);
        push_stats(alpha_vec);
        push_stats(beta_vec);
    }
}