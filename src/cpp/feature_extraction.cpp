#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <eigen3/Eigen/Dense>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Helper Functions (Internal) ---

// Struct to hold skewness and kurtosis
struct SkewKurt {
    double skew;
    double kurtosis;
};

// Calculates Mean and Standard Deviation
std::pair<double, double> calculate_mean_std(const double* data, size_t size) {
    if (size == 0) return {0.0, 0.0};
    
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += data[i];
    }
    double mean = sum / size;

    double sq_sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double diff = data[i] - mean;
        sq_sum += diff * diff;
    }
    double std_dev = std::sqrt(sq_sum / size);

    return {mean, std_dev};
}

// Overload for vector
std::pair<double, double> calculate_mean_std(const std::vector<double>& data) {
    return calculate_mean_std(data.data(), data.size());
}

// Calculates Skewness and Kurtosis
SkewKurt calculate_skew_kurtosis(const double* data, size_t size, double mean, double std_dev) {
    if (size == 0 || std_dev < 1e-9) return {0.0, 0.0};

    double n = (double)size;
    double sum_cubed = 0.0;
    double sum_quad = 0.0;

    for (size_t i = 0; i < size; ++i) {
        double diff = (data[i] - mean) / std_dev;
        double diff_sq = diff * diff;
        sum_cubed += diff_sq * diff;
        sum_quad += diff_sq * diff_sq;
    }

    double skew = sum_cubed / n;
    double kurtosis = (sum_quad / n) - 3.0; // Fisher kurtosis
    
    return {skew, kurtosis};
}

// Helper to access pixels with Border Replicate (Clamp)
inline int clamp_idx(int idx, int max_size) {
    if (idx < 0) return 0;
    if (idx >= max_size) return max_size - 1;
    return idx;
}

/**
 * @brief Templated HOG implementation to support both byte images and float depth maps.
 */
template <typename T>
void compute_hog_impl(
    const T* img_data, int rows, int cols, int n_bins,
    double* out_features)
{
    // 1. Clear output buffer
    // Total features = n_bins + 4 (Mean, Std, Skew, Kurt)
    int stats_offset = n_bins;
    int total_len = n_bins + 4;
    for(int i = 0; i < total_len; ++i) out_features[i] = 0.0;

    std::vector<double> mags;
    mags.reserve(rows * cols);

    double bin_width = 180.0 / n_bins;
    double total_mag_sum = 0.0;

    // 2. Compute Gradients and Accumulate Histogram
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            // Calculate indices with clamping
            int c_left  = clamp_idx(c - 1, cols);
            int c_right = clamp_idx(c + 1, cols);
            int r_up    = clamp_idx(r - 1, rows);
            int r_down  = clamp_idx(r + 1, rows);

            // Central Difference (Auto-casts T to double)
            double dx = (double)img_data[r * cols + c_right] - (double)img_data[r * cols + c_left];
            double dy = (double)img_data[r_down * cols + c]  - (double)img_data[r_up * cols + c];

            double mag = std::sqrt(dx * dx + dy * dy);
            double ang = std::atan2(dy, dx) * 180.0 / M_PI; // Result in [-180, 180]

            // Convert to Unsigned Orientation [0, 180)
            if (ang < 0) ang += 180.0;
            if (ang >= 180.0) ang -= 180.0;

            // Determine Bin
            int bin = (int)(ang / bin_width);
            if (bin >= n_bins) bin = n_bins - 1;

            // Weighted Voting
            out_features[bin] += mag;
            
            // Store magnitude for stats
            mags.push_back(mag);
            total_mag_sum += mag;
        }
    }

    // 3. Normalize Histogram (L1 Norm)
    if (total_mag_sum > 1e-9) {
        for(int i = 0; i < n_bins; ++i) {
            out_features[i] /= total_mag_sum;
        }
    }

    // 4. Calculate Magnitude Statistics
    std::pair<double, double> ms = calculate_mean_std(mags);
    SkewKurt sk = calculate_skew_kurtosis(mags.data(), mags.size(), ms.first, ms.second);

    out_features[stats_offset + 0] = ms.first;     // Mean
    out_features[stats_offset + 1] = ms.second;    // Std Dev
    out_features[stats_offset + 2] = sk.skew;      // Skewness
    out_features[stats_offset + 3] = sk.kurtosis;  // Kurtosis
}

extern "C" {
    /**
     * @brief Calculates normalized color histogram.
     */
    void extract_color_features_c(
        unsigned char* img_data, int rows, int cols, 
        int bins, int range_min, int range_max,
        double* out_features)
    {
        for (int i = 0; i < bins; ++i) out_features[i] = 0.0;

        int num_pixels = rows * cols;
        float range_width = (float)(range_max - range_min);
        if (range_width <= 0) range_width = 1.0f;

        for (int i = 0; i < num_pixels; ++i) {
            int val = img_data[i];
            if (val >= range_min && val < range_max) {
                int bin_idx = (int)((val - range_min) * bins / range_width);
                if (bin_idx >= bins) bin_idx = bins - 1; 
                out_features[bin_idx]++;
            }
        }

        double sum = 0.0;
        for (int i = 0; i < bins; ++i) sum += out_features[i];
        if (sum > 0) {
            for (int i = 0; i < bins; ++i) out_features[i] /= sum;
        }
    }

    /**
     * @brief Calculates LBP features.
     */
    void extract_lbp_features_c(
        unsigned char* gray_data, int rows, int cols, int n_points,
        double* out_features)
    {
        int n_bins = n_points + 2;
        for(int i=0; i<n_bins; ++i) out_features[i] = 0.0;

        for (int i = 1; i < rows - 1; ++i) {
            for (int j = 1; j < cols - 1; ++j) {
                unsigned char center = gray_data[i * cols + j];
                int code = 0;
                
                code |= (gray_data[(i-1) * cols + (j-1)] >= center) << 7;
                code |= (gray_data[(i-1) * cols + (j)]   >= center) << 6;
                code |= (gray_data[(i-1) * cols + (j+1)] >= center) << 5;
                code |= (gray_data[(i)   * cols + (j+1)] >= center) << 4;
                code |= (gray_data[(i+1) * cols + (j+1)] >= center) << 3;
                code |= (gray_data[(i+1) * cols + (j)]   >= center) << 2;
                code |= (gray_data[(i+1) * cols + (j-1)] >= center) << 1;
                code |= (gray_data[(i)   * cols + (j-1)] >= center) << 0;

                float val = (float)code;
                int bin_idx = (int)(val * n_bins / 256.0f);
                if (bin_idx >= n_bins) bin_idx = n_bins - 1;
                out_features[bin_idx]++;
            }
        }

        double sum = 0.0;
        for (int i = 0; i < n_bins; ++i) sum += out_features[i];
        if (sum > 0) {
            for (int i = 0; i < n_bins; ++i) out_features[i] /= sum;
        }
    }

    /**
     * @brief Calculates GLCM features for 4 angles.
     */
    void extract_glcm_features_c(
        unsigned char* img_data, int rows, int cols,
        double* out_features)
    {
        int drs[] = {0, -1, -1, -1};
        int dcs[] = {1, 1, 0, -1};
        int feat_idx = 0;

        std::vector<double> glcm(256 * 256);

        for (int k = 0; k < 4; ++k) {
            int dr = drs[k];
            int dc = dcs[k];

            std::fill(glcm.begin(), glcm.end(), 0.0);
            double total_sum = 0.0;

            int r_start = std::max(0, -dr);
            int r_end = std::min(rows, rows - dr);
            int c_start = std::max(0, -dc);
            int c_end = std::min(cols, cols - dc);

            for (int r = r_start; r < r_end; ++r) {
                for (int c = c_start; c < c_end; ++c) {
                    unsigned char src_val = img_data[r * cols + c];
                    unsigned char dst_val = img_data[(r + dr) * cols + (c + dc)];
                    glcm[src_val * 256 + dst_val]++;
                    total_sum++;
                }
            }

            double norm = (total_sum > 0) ? 1.0 / (2.0 * total_sum) : 0.0;
            double contrast = 0, dissimilarity = 0, homogeneity = 0, asm_val = 0;
            double mean_i = 0, mean_j = 0;

            for(int i=0; i<256; ++i) {
                for(int j=0; j<256; ++j) {
                    double val_ij = glcm[i * 256 + j];
                    double val_ji = glcm[j * 256 + i];
                    double p = (val_ij + val_ji) * norm;

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
            double var_i = 0, var_j = 0, cov = 0;
            for(int i=0; i<256; ++i) {
                for(int j=0; j<256; ++j) {
                    double val_ij = glcm[i * 256 + j];
                    double val_ji = glcm[j * 256 + i];
                    double p = (val_ij + val_ji) * norm;

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
     * @brief Standard HOG for 8-bit Images.
     */
    void extract_hog_features_c(
        unsigned char* img_data, int rows, int cols, int n_bins,
        double* out_features)
    {
        compute_hog_impl<unsigned char>(img_data, rows, cols, n_bins, out_features);
    }

    /**
     * @brief "Symmetry" HOG for Double Precision Depth Maps.
     */
    void extract_hog_features_depth_c(
        double* depth_data, int rows, int cols, int n_bins,
        double* out_features)
    {
        compute_hog_impl<double>(depth_data, rows, cols, n_bins, out_features);
    }

    /**
     * @brief Calculates principal plane fitting features (3D).
     */
    void extract_principal_plane_features_c(
        double* depth_data, int rows, int cols, double eps,
        double* out_features)
    {
        int n_pixels = rows * cols;
        Eigen::VectorXd z(n_pixels);
        Eigen::MatrixXd A(n_pixels, 9);

        double As = 0.0;
        double Ap = (double)((cols - 1) * (rows - 1));

        int idx = 0;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                double val_z = depth_data[r * cols + c];
                z(idx) = val_z;

                double x = (double)c;
                double y = (double)r;

                A(idx, 0) = 1.0; A(idx, 1) = x; A(idx, 2) = y;
                A(idx, 3) = x*x; A(idx, 4) = x*y; A(idx, 5) = y*y;
                A(idx, 6) = x*x*y; A(idx, 7) = x*y*y; A(idx, 8) = y*y*y;

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

        std::pair<double, double> z_ms = calculate_mean_std(depth_data, n_pixels);
        SkewKurt z_sk = calculate_skew_kurtosis(depth_data, n_pixels, z_ms.first, z_ms.second);

        Eigen::VectorXd coeffs = A.colPivHouseholderQr().solve(z);
        Eigen::VectorXd z_fitted = A * coeffs;
        Eigen::VectorXd diff = (z - z_fitted).cwiseAbs();
        double dist_mean = diff.mean();
        double dist_std = std::sqrt((diff.array() - dist_mean).square().sum() / (diff.size()));

        double n_x = coeffs(1);
        double n_y = coeffs(2);
        double n_z = -1.0; 
        double norm_mag = std::sqrt(n_x*n_x + n_y*n_y + n_z*n_z);
        double theta = std::acos(n_z / (norm_mag + eps)) * 180.0 / M_PI;

        double rugosity = (Ap > 0) ? As / Ap : 0.0;

        int out_idx = 0;
        out_features[out_idx++] = z_ms.second; // std
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
        std::vector<double> dx(rows * cols);
        std::vector<double> dy(rows * cols);
        std::vector<double> dxdx(rows * cols);
        std::vector<double> dydy(rows * cols);
        std::vector<double> dxdy(rows * cols);
        std::vector<double> dydx(rows * cols);

        auto compute_deriv_x = [&](const double* src, std::vector<double>& dst) {
            for(int r=0; r<rows; ++r) {
                for(int c=0; c<cols; ++c) {
                    double v_left = src[r*cols + clamp_idx(c-1, cols)];
                    double v_right = src[r*cols + clamp_idx(c+1, cols)];
                    dst[r*cols+c] = 0.5 * (v_right - v_left);
                }
            }
        };

        auto compute_deriv_y = [&](const double* src, std::vector<double>& dst) {
            for(int r=0; r<rows; ++r) {
                for(int c=0; c<cols; ++c) {
                    double v_up = src[clamp_idx(r-1, rows)*cols + c];
                    double v_down = src[clamp_idx(r+1, rows)*cols + c];
                    dst[r*cols+c] = 0.5 * (v_down - v_up);
                }
            }
        };

        compute_deriv_x(depth_data, dx);
        compute_deriv_y(depth_data, dy);
        compute_deriv_x(dx.data(), dxdx);
        compute_deriv_y(dy.data(), dydy);
        compute_deriv_y(dx.data(), dxdy); 
        compute_deriv_x(dy.data(), dydx); 

        std::vector<double> stats_buffers[8];
        
        for(int i = 0; i < rows * cols; ++i) {
            double _dx = dx[i];
            double _dy = dy[i];
            double _dxdx = dxdx[i];
            double _dydy = dydy[i];
            double _dxdy = dxdy[i];
            double _dydx = dydx[i];

            double denom = 1.0 + _dx*_dx + _dy*_dy;
            double denom_sq = denom * denom;
            double G = (_dxdx * _dydy - _dxdy * _dydx) / denom_sq;
            double M = (_dydy + _dxdx) / (2.0 * std::pow(denom, 1.5));

            double discriminant = std::sqrt(std::max(M*M - G, 0.0));
            double k1 = M + discriminant;
            double k2 = M - discriminant;

            double S = (2.0 / M_PI) * std::atan2(k2 + k1, k2 - k1);
            double C = std::sqrt((k1*k1 + k2*k2) / 2.0);

            double nx = -_dx;
            double ny = -_dy;
            double nz = 1.0;
            double norm = std::sqrt(nx*nx + ny*ny + nz*nz) + eps;
            nx /= norm; ny /= norm; nz /= norm;

            double alpha = std::atan2(ny, nx);
            double beta = std::atan2(nz, std::sqrt(nx*nx + ny*ny));

            stats_buffers[0].push_back(G);
            stats_buffers[1].push_back(M);
            stats_buffers[2].push_back(k1);
            stats_buffers[3].push_back(k2);
            stats_buffers[4].push_back(S);
            stats_buffers[5].push_back(C);
            stats_buffers[6].push_back(alpha);
            stats_buffers[7].push_back(beta);
        }

        int out_idx = 0;
        for(int i=0; i<8; ++i) {
            auto ms = calculate_mean_std(stats_buffers[i]);
            out_features[out_idx++] = ms.first;
            out_features[out_idx++] = ms.second;
        }
    }
}