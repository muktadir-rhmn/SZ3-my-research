#ifndef _SZ_INTERPOLATION_DECOMPOSITION_HPP
#define _SZ_INTERPOLATION_DECOMPOSITION_HPP

#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>


#include "Decomposition.hpp"
#include "SZ3/def.hpp"
#include "SZ3/quantizer/Quantizer.hpp"
#include "SZ3/utils/Config.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "SZ3/utils/Interpolators.hpp"
#include "SZ3/utils/Iterator.hpp"
#include "SZ3/utils/MemoryUtil.hpp"
#include "SZ3/utils/Timer.hpp"

namespace SZ3 {
class EquationSolver {
   public:
    static std::vector<double> solve(std::vector<std::vector<double>> &A, std::vector<double> &B) {
        return solve_using_Cramers_rule(A, B);
    }

   private:
    static std::vector<double> solve_using_Cramers_rule(std::vector<std::vector<double>> &A, std::vector<double> &B) {
        size_t nUnknowns = B.size();
        double detA = determinant(A);
        if (detA == 0) {
            throw std::runtime_error("No solution of the equations since the determinant is 0.");
        }
        std::vector<double> solution(nUnknowns);
        for (int i = 0; i < nUnknowns; ++i) {
            std::vector<std::vector<double>> tempA = A;
            for (int j = 0; j < nUnknowns; ++j) {
                tempA[j][i] = B[j];
            }
            solution[i] = determinant(tempA) / detA;
        }
        return solution;
    }

    static double determinant(std::vector<std::vector<double>> matrix) {
        double det = 1.0;
        for (int i = 0; i < matrix.size(); i++) {
            int pivot = i;
            for (int j = i + 1; j < matrix.size(); j++) {
                if (std::abs(matrix[j][i]) > std::abs(matrix[pivot][i])) {
                    pivot = j;
                }
            }
            if (pivot != i) {
                std::swap(matrix[i], matrix[pivot]);
                det *= -1;
            }
            if (matrix[i][i] == 0) {
                return 0;
            }
            det *= matrix[i][i];
            for (int j = i + 1; j < matrix.size(); j++) {
                double factor = matrix[j][i] / matrix[i][i];
                for (int k = i + 1; k < matrix.size(); k++) {
                    matrix[j][k] -= factor * matrix[i][k];
                }
            }
        }
        return det;
    }
};

template <class T, uint N, class Quantizer>
class InterpolationDecomposition : public concepts::DecompositionInterface<T, int, N> {
   public:
    InterpolationDecomposition(const Config &conf, Quantizer quantizer) : quantizer(quantizer) {
        static_assert(std::is_base_of<concepts::QuantizerInterface<T, int>, Quantizer>::value,
                      "must implement the quatizer interface");
    }

    T *decompress(const Config &conf, std::vector<int> &quant_inds, T *dec_data) override {
        init();
        linear_interpolator.print_weight();
        this->quant_inds = quant_inds.data();
        //            lossless.postdecompress_data(buffer);
        double eb = quantizer.get_eb();

        *dec_data = quantizer.recover(0, this->quant_inds[quant_index++]);

        for (uint level = interpolation_level; level > 0 && level <= interpolation_level; level--) {
            if (level >= 3) {
                quantizer.set_eb(eb * eb_ratio);
            } else {
                quantizer.set_eb(eb);
            }
            size_t stride = 1U << (level - 1);
            auto inter_block_range = std::make_shared<multi_dimensional_range<T, N>>(
                dec_data, std::begin(global_dimensions), std::end(global_dimensions), stride * blocksize, 0);
            auto inter_begin = inter_block_range->begin();
            auto inter_end = inter_block_range->end();
            for (auto block = inter_begin; block != inter_end; ++block) {
                auto end_idx = block.get_global_index();
                for (int i = 0; i < N; i++) {
                    end_idx[i] += stride * blocksize;
                    if (end_idx[i] > global_dimensions[i] - 1) {
                        end_idx[i] = global_dimensions[i] - 1;
                    }
                }
                block_traversal(Interpolation, dec_data, block.get_global_index(), end_idx, PB_recover,
                                interpolators[interpolator_id], direction_sequence_id, stride);
            }
        }

        quantizer.postdecompress_data();
        //            timer.stop("Interpolation Decompress");

        return dec_data;
    }

    // compress given the error bound
    std::vector<int> compress(const Config &conf, T *data) override {
        std::copy_n(conf.dims.begin(), N, global_dimensions.begin());
        blocksize = 32;
        interpolator_id = conf.interpAlgo;
        direction_sequence_id = conf.interpDirection;

        log_compression();
        init();

        //for analysis purpose:
//        std::ofstream order_file_inst("/home/foo/datasets/acs_wht_order_file.txt");
//        order_file = &order_file_inst;

        auto quant_inds_vec = compress_using_sz3s_original_order(data);
//        auto quant_inds_vec = compress_using_my_custom_order(data);

        // print some analysis data
//        auto avg_prediction_error = prediction_error_sum / num_data_points_compressed;
//        auto avg_diff = diff_sum / num_data_points_compressed;
//        auto percent_predicted = ( 100.0 * predicted_data_points) / num_data_points_compressed;
//        auto mse = prediction_error_squared_sum / num_data_points_compressed;
//        auto variance = mse - avg_prediction_error * avg_prediction_error;
//        std::cout << "percent predicted = " << percent_predicted << std::endl;
//        std::cout << "average prediction error = " <<  avg_prediction_error << std::endl;
//        std::cout << "mean squared error = " << mse << std::endl;
//        std::cout << "variance = " << variance << std::endl;
//        std::cout << "max prediction error = " << max_prediction_error << std::endl;
//        std::cout << "average difference between original and decompressed = " <<  avg_diff << std::endl;
//
//        std::map<int, int> freq_map; // ordered map
//        for (auto i: quant_inds_vec) {
//            freq_map[i] += 1;
//        }
//        std::cout << "Unique indices = " << freq_map.size() <<std::endl;
//
//        std::map<int, int> freq_to_index;
//        for (auto & i : freq_map) {
//            auto index = i.first;
//            auto freq = i.second;
//            freq_to_index[freq] = index;
//        }
//
//        auto it = freq_to_index.rbegin();
//        for(auto i = 0; i < 10; i++) {
//            std::cout << it-> first << ", " << it-> second <<std::endl;
//            it++;
//        }
//
//        order_file->close();
        return quant_inds_vec;
    }

    void save(uchar *&c) override {
        write(global_dimensions.data(), N, c);
        write(blocksize, c);
        write(interpolator_id, c);
        write(direction_sequence_id, c);

        linear_interpolator.save(c);

        quantizer.save(c);
    }

    void load(const uchar *&c, size_t &remaining_length) override {
        read(global_dimensions.data(), N, c, remaining_length);
        read(blocksize, c, remaining_length);
        read(interpolator_id, c, remaining_length);
        read(direction_sequence_id, c, remaining_length);

        linear_interpolator.load(c, remaining_length);

        quantizer.load(c, remaining_length);
    }

    std::pair<int, int> get_out_range() override { return quantizer.get_out_range(); }

   private:
    enum PredictorBehavior { PB_predict_overwrite, PB_predict, PB_recover };
    enum TraversalPurpose {Learning, Interpolation};
    class LinearWeightLearningInterpolator {
       private:
        // initializing to original SZ3 weights: (a + b) / 2;
        double w1 = 1.0/2.0;
        double w2 = 1.0/2.0;

        long num_datapoints = 0;
        double a_1 = 0;
        double a_2 = 0;
        double a_3 = 0;
        double b_1 = 0;
        double b_2 = 0;
        double b_3 = 0;

       public:
        T interp(T d_i_minus_1, T d_i_plus_1) {
            return w1 * d_i_minus_1 + w2 * d_i_plus_1;
        }

        void learn(T d_i_minus_1, T d_i, T d_i_plus_1){
            num_datapoints++;

            a_1 += d_i_minus_1 * d_i_minus_1;
            a_2 += d_i_plus_1 * d_i_minus_1;
            a_3 += d_i * d_i_minus_1;

            b_1 += d_i_minus_1 * d_i_plus_1;
            b_2 += d_i_plus_1 * d_i_plus_1;
            b_3 += d_i * d_i_plus_1;
        }

        void finalize_learning() {
            std::cout << "Computing weights from " << num_datapoints << " data points" <<std::endl;
            double d1 = a_1 * b_2 - a_2 * b_1;
            double d2 = a_1 * b_2 - a_2 * b_1;
            if (d1 != 0 && d2 != 0) {
                w1 = (b_2 * a_3 - a_2 * b_3) / d1;
                w2 = (a_1 * b_3 - a_3 * b_1) / d2;
            }

        }

        void print_weight(){
//            std::cout << "Linear Weights(" << w1 << "," << w2 << ")" << std::endl;
        }

        void save(uchar *&c) {
            write(w1, c);
            write(w2, c);
        }
        void load(const uchar *&c, size_t &remaining_length) {
            read(w1, c, remaining_length);
            read(w2, c, remaining_length);
        }
    };

    class LinearWeightWithConstantLearningInterpolator {
       private:
        // initializing to original SZ3 weights: (a + b) / 2;
        double w1 = 1.0/2.0;
        double w2 = 1.0/2.0;
        double w3 = 0;

        long num_datapoints = 0;
        double a_1 = 0;
        double a_2 = 0;
        double a_3 = 0;
        double a_4 = 0;
        double b_1 = 0;
        double b_2 = 0;
        double b_3 = 0;
        double b_4 = 0;
        double c_3 = 0;
        double c_4 = 0;

       public:
        T interp(T d_i_minus_1, T d_i_plus_1) {
            return w1 * d_i_minus_1 + w2 * d_i_plus_1 + w3;
        }

        void learn(T d_i_minus_1, T d_i, T d_i_plus_1){
            num_datapoints++;

            a_1 += d_i_minus_1 * d_i_minus_1;
            a_2 += d_i_plus_1 * d_i_minus_1;
            a_3 += d_i_minus_1;
            a_4 += d_i * d_i_minus_1;

            b_1 += d_i_minus_1 * d_i_plus_1;
            b_2 += d_i_plus_1 * d_i_plus_1;
            b_3 += d_i_plus_1;
            b_4 += d_i * d_i_plus_1;

            c_3 += 1;
            c_4 += d_i;
        }

        void finalize_learning() {
            std::cout << "Computing weights from " << num_datapoints << " data points" <<std::endl;
            std::vector<std::vector<double>> A = {
                {a_1, a_2, a_3},
                {b_1, b_2, b_3},
                {a_3, b_3, c_3}
            };
            std::vector<double> B = {
                a_4,
                b_4,
                c_4
            };
            std::vector<double> weights = EquationSolver::solve(A, B);
            w1 = weights[0];
            w2 = weights[1];
            w3 = weights[2];
        }

        void print_weight(){
            std::cout << "Linear Weights(" << w1 << "," << w2<< "," << w3 << ")" << std::endl;
        }

        void save(uchar *&c) {
            write(w1, c);
            write(w2, c);
            write(w3, c);
        }
        void load(const uchar *&c, size_t &remaining_length) {
            read(w1, c, remaining_length);
            read(w2, c, remaining_length);
            read(w3, c, remaining_length);
        }
    };

    // terrible performance
    class NonLinearWeightWithConstantLearningInterpolator {
       private:
        // initializing to original SZ3 weights: (a + b) / 2;
        double w1 = 1.0/2.0;
        double w2 = 1.0/2.0;
        double w3 = 0;

        long num_datapoints = 0;
        double a_1 = 0;
        double a_2 = 0;
        double a_3 = 0;
        double a_4 = 0;
        double b_1 = 0;
        double b_2 = 0;
        double b_3 = 0;
        double b_4 = 0;
        double c_3 = 0;
        double c_4 = 0;

       public:
        T interp(T d_i_minus_1, T d_i_plus_1) {
            return w1 * f1(d_i_minus_1) + w2 * f2(d_i_plus_1) + w3;
        }

        void learn(T d_i_minus_1, T d_i, T d_i_plus_1){
            num_datapoints++;

            a_1 += f1(d_i_minus_1) * f1(d_i_minus_1);
            a_2 += f2(d_i_plus_1) * f1(d_i_minus_1);
            a_3 += f1(d_i_minus_1);
            a_4 += d_i * f1(d_i_minus_1);

            b_1 += f1(d_i_minus_1) * f2(d_i_plus_1);
            b_2 += f2(d_i_plus_1) * f2(d_i_plus_1);
            b_3 += f2(d_i_plus_1);
            b_4 += d_i * f2(d_i_plus_1);

            c_3 += 1;
            c_4 += d_i;
        }

        T f1(T d_i_minus_1) {
            return d_i_minus_1 * std::sin(d_i_minus_1);
        }

        T f2(T d_i_plus_1) {
            return d_i_plus_1 * std::sin(d_i_plus_1) ;
        }

        void finalize_learning() {
            std::cout << "Computing weights from " << num_datapoints << " data points" <<std::endl;
            std::vector<std::vector<double>> A = {
                {a_1, a_2, a_3},
                {b_1, b_2, b_3},
                {a_3, b_3, c_3}
            };
            std::vector<double> B = {
                a_4,
                b_4,
                c_4
            };
            std::vector<double> weights = EquationSolver::solve(A, B);
            w1 = weights[0];
            w2 = weights[1];
            w3 = weights[2];
        }

        void print_weight(){
            std::cout << "Non-Linear Weights(" << w1 << "," << w2<< "," << w3 << ")" << std::endl;
        }

        void save(uchar *&c) {
            write(w1, c);
            write(w2, c);
            write(w3, c);
        }
        void load(const uchar *&c, size_t &remaining_length) {
            read(w1, c, remaining_length);
            read(w2, c, remaining_length);
            read(w3, c, remaining_length);
        }
    };
    class LinearWeightByBlocksLearningInterpolator {
       private:
        std::vector<LinearWeightLearningInterpolator> interpolators;
        size_t training_data_index = 0;
        size_t interpolation_data_index = 0;
        size_t block_size;
       public:
        void set_block_size(size_t block_size_){
            block_size = block_size_;
        }
        void learn(T d_i_minus_1, T d_i, T d_i_plus_1) {
            if (training_data_index == 0) {
                interpolators.push_back(LinearWeightLearningInterpolator());
            }
            interpolators[interpolators.size() - 1].learn(d_i_minus_1, d_i, d_i_plus_1);
            training_data_index++;
            if (training_data_index == block_size) {
                training_data_index = 0;
            }
        }

        void finalize_learning() {
            for (int i = 0; i < interpolators.size(); i++) {
                interpolators[i].finalize_learning();
            }
        }

        T interp(T d_i_minus_1, T d_i_plus_1) {
            size_t interpolator_i = interpolation_data_index / block_size;
            interpolation_data_index++;
            return interpolators[interpolator_i].interp(d_i_minus_1, d_i_plus_1);
        }

        void print_weight(){
            for (int i = 0; i < interpolators.size(); i++) {
                interpolators[i].print_weight();
            }
        }

        void save(uchar *&c) {
            size_t num_interpolators = interpolators.size();
            write(num_interpolators, c);

            for (int i = 0; i < interpolators.size(); i++) {
                interpolators[i].save(c);
            }
        }
        void load(const uchar *&c, size_t &remaining_length) {
            size_t num_interpolators;
            read(num_interpolators, c, remaining_length);

            for(int i = 0; i < num_interpolators; i++) {
                interpolators.push_back(LinearWeightLearningInterpolator());
                interpolators[i].load(c, remaining_length);
            }
        }
    };

    class CubicWeightLearningInterpolator {
       private:
        // initializing to original SZ3 weights: (-a + 9 * b + 9 * c - d) / 16;
        double w1 = -1.0 / 16.0;
        double w2 = 9.0 / 16.0;
        double w3 = 9.0 / 16.0;
        double w4 = -1.0 / 16.0;
       public:
        T interp(T a, T b, T c, T d) {
            return w1 * a +
                   w2 * b +
                   w3 * c +
                   w4 * d;
        }

        void print_weights() {
            std::cout << "Cubic Weights(" << w1 << "," << w2 << "," << w3<< "," << w4 << ")" << std::endl;
        }

        //todo: learning (similar to LinearWeightLearningInterpolator)
        //todo: write save() for storing the weights
        //todo: write load() for loading the weights
    };

    void log_compression() {
//        if (interpolators[interpolator_id] == "linear") {
//            std::cout <<"running_compression_algo=linear_spline_interpolation" << std::endl;
//        } else if (interpolators[interpolator_id] == "cubic") {
//            std::cout <<"running_compression_algo=cubic_spline_interpolation" << std::endl;
//        }
//        std::cout << "dimension=";
//        for (int i = 0; i < global_dimensions.size(); i++) {
//            std::cout << global_dimensions[i];
//            if (i < global_dimensions.size() - 1) std::cout << "x";
//        }
//        std::cout<<std::endl;
    }

    void init() {
        quant_index = 0;
        assert(blocksize % 2 == 0 && "Interpolation block size should be even numbers");
        num_elements = 1;
        interpolation_level = -1;
        for (int i = 0; i < N; i++) {
            if (interpolation_level < ceil(log2(global_dimensions[i]))) {
                interpolation_level = static_cast<uint>(ceil(log2(global_dimensions[i])));
            }
            num_elements *= global_dimensions[i];
        }

        dimension_offsets[N - 1] = 1;
        for (int i = N - 2; i >= 0; i--) {
            dimension_offsets[i] = dimension_offsets[i + 1] * global_dimensions[i + 1];
        }

        dimension_sequences = std::vector<std::array<int, N>>();
        auto sequence = std::array<int, N>();
        for (int i = 0; i < N; i++) {
            sequence[i] = i;
        }
        do {
            dimension_sequences.push_back(sequence);
        } while (std::next_permutation(sequence.begin(), sequence.end()));
    }


    std::vector<int> compress_using_my_custom_order(T *data){
        size_t BLOCK_SIZE = num_elements;
//        linear_interpolator.set_block_size(BLOCK_SIZE);
        learn_weights_custom_order(data, BLOCK_SIZE);
        return interpolate_custom_order(data, BLOCK_SIZE);
    }

    void learn_weights_custom_order(T * data, size_t block_size) {
//        std::cout << "Learning weights using custom order" << std::endl;
        traverse_customer_order(Learning, data, block_size);
        linear_interpolator.finalize_learning();
    }

    std::vector<int> interpolate_custom_order(T *data, size_t block_size){
        std::cout << "Interpolating using custom order" <<std::endl;
        linear_interpolator.print_weight();
        cubic_interpolator.print_weights();
        std::vector<int> quant_inds_vec(num_elements);
        quant_inds = quant_inds_vec.data();

        //todo: change it

        quant_inds[quant_index++] = quantizer.quantize_and_overwrite(*(data+1), *(data+1));
        traverse_customer_order(Interpolation, data, block_size);
        return quant_inds_vec;
    }

    void traverse_customer_order(TraversalPurpose purpose, T * data, size_t block_size) {
//        if (N != 1) throw std::runtime_error("multidimensional data is not supported.");
        for(int i = 0 ; i < num_elements; i++) {
            if (i % block_size == 0) {
                if (purpose == Learning) {
                    linear_interpolator.learn(*(data + i - 2), *(data + i), *(data + i - 1));
                } else if (purpose == Interpolation) {
                    quant_inds[quant_index++] = quantizer.quantize_and_overwrite(*(data+i), 0);
                }
            } else if (i % block_size == 1) {
                if (purpose == Learning) {
                    linear_interpolator.learn(*(data + i - 2), *(data + i), *(data + i - 1));
                } else if (purpose == Interpolation) {
                    quantize(0, *(data + i), linear_interpolator.interp(*(data + i - 2), *(data + i - 1)));
                }
            } else {
                if (purpose == Learning) {
                    linear_interpolator.learn(*(data + i - 2), *(data + i), *(data + i - 1));
                } else if (purpose == Interpolation) {
                    quantize(0, *(data + i), linear_interpolator.interp(*(data + i - 2), *(data + i - 1)));
                }
            }

        }
    }

    std::vector<int> compress_using_sz3s_original_order(T *data){
        /// just comment out to use the original SZ3
        ///     except, weights are stored
        learn_weights(data);
        return interpolate(data);
    }

    void learn_weights(T * data) {
        std::cout << "Learning weights" << std::endl;
        traverse(Learning, data);
        linear_interpolator.finalize_learning();
    }

    std::vector<int> interpolate(T * data) {
//        std::cout << "Interpolating" <<std::endl;
        linear_interpolator.print_weight();
        cubic_interpolator.print_weights();
        std::vector<int> quant_inds_vec(num_elements);
        quant_inds = quant_inds_vec.data();

        quant_inds[quant_index++] = quantizer.quantize_and_overwrite(*data, 0);
        traverse(Interpolation, data);
        return quant_inds_vec;
    }

    void traverse(TraversalPurpose purpose, T * data) {
        double eb = quantizer.get_eb();

        for (uint level = interpolation_level; level > 0 && level <= interpolation_level; level--) {
            if (level >= 3) {
                quantizer.set_eb(eb * eb_ratio); //todo: why?
            } else {
                quantizer.set_eb(eb);
            }
//            std::cout << "abs error bound: " << quantizer.get_eb() << std::endl;
            size_t stride = 1U << (level - 1);

            auto inter_block_range = std::make_shared<multi_dimensional_range<T, N>>(
                data, std::begin(global_dimensions), std::end(global_dimensions), blocksize * stride, 0);

            auto inter_begin = inter_block_range->begin();
            auto inter_end = inter_block_range->end();

            for (auto block = inter_begin; block != inter_end; ++block) {
                auto end_idx = block.get_global_index();
                for (int i = 0; i < N; i++) {
                    end_idx[i] += blocksize * stride;
                    if (end_idx[i] > global_dimensions[i] - 1) {
                        end_idx[i] = global_dimensions[i] - 1;
                    }
                }

                block_traversal(purpose, data, block.get_global_index(), end_idx, PB_predict_overwrite,
                                interpolators[interpolator_id], direction_sequence_id, stride);
            }
        }

        quantizer.postcompress_data();

    }

    inline void quantize(size_t idx, T &d, T pred) {
//        num_data_points_compressed++;
//        T original_value = d;
//        T prediction_error = std::abs(original_value - pred);
//        if (prediction_error > max_prediction_error) max_prediction_error = prediction_error;
//        prediction_error_sum += prediction_error;
//        prediction_error_squared_sum += prediction_error * prediction_error;

        quant_inds[quant_index++] = quantizer.quantize_and_overwrite(d, pred);
//        d = original_value; //todo: remove it. Used for experimental purposes to compress using the original data.

//        if (quantization_index != 0) {
//            predicted_data_points++;
//        }

//        T decompressed_value = d;
//        diff_sum += std::abs(original_value - decompressed_value);
    }

    inline void recover(size_t idx, T &d, T pred) { d = quantizer.recover(pred, quant_inds[quant_index++]); }

    double block_traversal_1d(TraversalPurpose purpose, T *data, size_t begin, size_t end, size_t stride, const std::string &interp_func,
                                  const PredictorBehavior pb) {
        size_t n = (end - begin) / stride + 1;
        if (n <= 1) {
            return 0;
        }
        double predict_error = 0;

        size_t stride3x = 3 * stride;
        size_t stride5x = 5 * stride;
        if (interp_func == "linear" || n < 5) {
            if (pb == PB_predict_overwrite) {
                if (purpose == Interpolation){
                    for (size_t i = 1; i + 1 < n; i += 2) {
                        T *d = data + begin + i * stride;
                        quantize(d - data, *d, linear_interpolator.interp(*(d - stride), *(d + stride)));
                    }
                } else if (purpose == Learning) {
                    for (size_t i = 1; i + 1 < n; i += 2) {
                        T *d = data + begin + i * stride;
                        linear_interpolator.learn(*(d - stride), *d, *(d + stride));
                    }
                }

                if (n % 2 == 0) {
                    T *d = data + begin + (n - 1) * stride;
                    if (n < 4) {
                        if (purpose == Interpolation){
                            //todo: what should I do with this predictions?
                            //todo: change the corresponding recovering accordingly
                            quantize(d - data, *d, *(d - stride));
                        } else if (purpose == Learning) {
                            //todo: what should I do here?
                        }
                    } else {
                        if (purpose == Interpolation){
                            //todo: what should I with interp_linear1
                            //todo: change the corresponding recovering accordingly
                            quantize(d - data, *d, interp_linear1(*(d - stride3x), *(d - stride)));
//                            quantize(d - data, *d, linear_interpolator.interp(*(d - stride3x), *(d - stride)));
                        } else if (purpose == Learning) {
                            //todo: what should I do here?
//                            linear_interpolator.learn(*(d - stride3x), *d, *(d - stride));
                        }
                    }
                }
            } else {
                for (size_t i = 1; i + 1 < n; i += 2) {
                    T *d = data + begin + i * stride;
                    recover(d - data, *d, linear_interpolator.interp(*(d - stride), *(d + stride)));
                }
                if (n % 2 == 0) {
                    T *d = data + begin + (n - 1) * stride;
                    if (n < 4) {
                        recover(d - data, *d, *(d - stride));
                    } else {
                        recover(d - data, *d, interp_linear1(*(d - stride3x), *(d - stride)));
                    }
                }
            }
        } else {
            if (pb == PB_predict_overwrite) {
                T *d;
                size_t i;

                if (purpose == Interpolation){
                    for (i = 3; i + 3 < n; i += 2) {
                        d = data + begin + i * stride;
                        quantize(d - data, *d,
                                 cubic_interpolator.interp(*(d - stride3x), *(d - stride), *(d + stride),
                                                           *(d + stride3x)));
                    }
                } else if (purpose == Learning) {
                    //todo: what should I do here?

                }

                d = data + begin + stride;
                if (purpose == Interpolation) {
                    //todo: what should I with interp_quad_1?
                    //todo: change the corresponding recovering accordingly
                    quantize(d - data, *d, interp_quad_1(*(d - stride), *(d + stride), *(d + stride3x)));
                } else if (purpose == Learning) {
                    //todo: what should I do here?
                }

                d = data + begin + i * stride;
                if (purpose == Interpolation) {
                    //todo: what should I with interp_quad_2?
                    //todo: change the corresponding recovering accordingly
                    quantize(d - data, *d, interp_quad_2(*(d - stride3x), *(d - stride), *(d + stride)));
                } else if (purpose == Learning) {
                    //todo: what should I do here?
                }
                if (n % 2 == 0) {
                    d = data + begin + (n - 1) * stride;
                    if (purpose == Interpolation){
                        //todo: what should I with interp_quad_3?
                        //todo: change the corresponding recovering accordingly
                        quantize(d - data, *d, interp_quad_3(*(d - stride5x), *(d - stride3x), *(d - stride)));
                    } else if (purpose == Learning) {
                        //todo: what should I do here?
                    }
                }

            } else {
                T *d;

                size_t i;
                for (i = 3; i + 3 < n; i += 2) {
                    d = data + begin + i * stride;
                    recover(d - data, *d,
                            cubic_interpolator.interp(*(d - stride3x), *(d - stride), *(d + stride), *(d + stride3x)));
                }
                d = data + begin + stride;

                recover(d - data, *d, interp_quad_1(*(d - stride), *(d + stride), *(d + stride3x)));

                d = data + begin + i * stride;
                recover(d - data, *d, interp_quad_2(*(d - stride3x), *(d - stride), *(d + stride)));

                if (n % 2 == 0) {
                    d = data + begin + (n - 1) * stride;
                    recover(d - data, *d, interp_quad_3(*(d - stride5x), *(d - stride3x), *(d - stride)));
                }
            }
        }

        return predict_error;
    }

    template <uint NN = N>
    typename std::enable_if<NN == 1, double>::type block_traversal(TraversalPurpose purpose,
                                                                       T *data, std::array<size_t, N> begin,
                                                                       std::array<size_t, N> end,
                                                                       const PredictorBehavior pb,
                                                                       const std::string &interp_func,
                                                                       const int direction, size_t stride = 1) {
        return block_traversal_1d(purpose, data, begin[0], end[0], stride, interp_func, pb);
    }

    template <uint NN = N>
    typename std::enable_if<NN == 2, double>::type block_traversal(TraversalPurpose purpose,
                                                                       T *data, std::array<size_t, N> begin,
                                                                       std::array<size_t, N> end,
                                                                       const PredictorBehavior pb,
                                                                       const std::string &interp_func,
                                                                       const int direction, size_t stride = 1) {
        double predict_error = 0;
        size_t stride2x = stride * 2;
        const std::array<int, N> dims = dimension_sequences[direction];
        for (size_t j = (begin[dims[1]] ? begin[dims[1]] + stride2x : 0); j <= end[dims[1]]; j += stride2x) {
            size_t begin_offset = begin[dims[0]] * dimension_offsets[dims[0]] + j * dimension_offsets[dims[1]];
            predict_error +=
                block_traversal_1d(purpose, data, begin_offset,
                                   begin_offset + (end[dims[0]] - begin[dims[0]]) * dimension_offsets[dims[0]],
                                   stride * dimension_offsets[dims[0]], interp_func, pb);
        }
        for (size_t i = (begin[dims[0]] ? begin[dims[0]] + stride : 0); i <= end[dims[0]]; i += stride) {
            size_t begin_offset = i * dimension_offsets[dims[0]] + begin[dims[1]] * dimension_offsets[dims[1]];
            predict_error +=
                block_traversal_1d(purpose, data, begin_offset,
                                   begin_offset + (end[dims[1]] - begin[dims[1]]) * dimension_offsets[dims[1]],
                                   stride * dimension_offsets[dims[1]], interp_func, pb);
        }
        return predict_error;
    }

    template <uint NN = N>
    typename std::enable_if<NN == 3, double>::type block_traversal(TraversalPurpose purpose,
                                                                       T *data, std::array<size_t, N> begin,
                                                                       std::array<size_t, N> end,
                                                                       const PredictorBehavior pb,
                                                                       const std::string &interp_func,
                                                                       const int direction, size_t stride = 1) {
        double predict_error = 0;
        size_t stride2x = stride * 2;
        const std::array<int, N> dims = dimension_sequences[direction];
        for (size_t j = (begin[dims[1]] ? begin[dims[1]] + stride2x : 0); j <= end[dims[1]]; j += stride2x) {
            for (size_t k = (begin[dims[2]] ? begin[dims[2]] + stride2x : 0); k <= end[dims[2]]; k += stride2x) {
                size_t begin_offset = begin[dims[0]] * dimension_offsets[dims[0]] + j * dimension_offsets[dims[1]] +
                                      k * dimension_offsets[dims[2]];
                predict_error +=
                    block_traversal_1d(purpose, data, begin_offset,
                                       begin_offset + (end[dims[0]] - begin[dims[0]]) * dimension_offsets[dims[0]],
                                       stride * dimension_offsets[dims[0]], interp_func, pb);
            }
        }
        for (size_t i = (begin[dims[0]] ? begin[dims[0]] + stride : 0); i <= end[dims[0]]; i += stride) {
            for (size_t k = (begin[dims[2]] ? begin[dims[2]] + stride2x : 0); k <= end[dims[2]]; k += stride2x) {
                size_t begin_offset = i * dimension_offsets[dims[0]] + begin[dims[1]] * dimension_offsets[dims[1]] +
                                      k * dimension_offsets[dims[2]];
                predict_error +=
                    block_traversal_1d(purpose, data, begin_offset,
                                       begin_offset + (end[dims[1]] - begin[dims[1]]) * dimension_offsets[dims[1]],
                                       stride * dimension_offsets[dims[1]], interp_func, pb);
            }
        }
        for (size_t i = (begin[dims[0]] ? begin[dims[0]] + stride : 0); i <= end[dims[0]]; i += stride) {
            for (size_t j = (begin[dims[1]] ? begin[dims[1]] + stride : 0); j <= end[dims[1]]; j += stride) {
                size_t begin_offset = i * dimension_offsets[dims[0]] + j * dimension_offsets[dims[1]] +
                                      begin[dims[2]] * dimension_offsets[dims[2]];
                predict_error +=
                    block_traversal_1d(purpose, data, begin_offset,
                                       begin_offset + (end[dims[2]] - begin[dims[2]]) * dimension_offsets[dims[2]],
                                       stride * dimension_offsets[dims[2]], interp_func, pb);
            }
        }
        return predict_error;
    }

    template <uint NN = N>
    typename std::enable_if<NN == 4, double>::type block_traversal(TraversalPurpose purpose,
                                                                       T *data, std::array<size_t, N> begin,
                                                                       std::array<size_t, N> end,
                                                                       const PredictorBehavior pb,
                                                                       const std::string &interp_func,
                                                                       const int direction, size_t stride = 1) {
        double predict_error = 0;
        size_t stride2x = stride * 2;
        max_error = 0;
        const std::array<int, N> dims = dimension_sequences[direction];
        for (size_t j = (begin[dims[1]] ? begin[dims[1]] + stride2x : 0); j <= end[dims[1]]; j += stride2x) {
            for (size_t k = (begin[dims[2]] ? begin[dims[2]] + stride2x : 0); k <= end[dims[2]]; k += stride2x) {
                for (size_t t = (begin[dims[3]] ? begin[dims[3]] + stride2x : 0); t <= end[dims[3]]; t += stride2x) {
                    size_t begin_offset = begin[dims[0]] * dimension_offsets[dims[0]] + j * dimension_offsets[dims[1]] +
                                          k * dimension_offsets[dims[2]] + t * dimension_offsets[dims[3]];
                    predict_error +=
                        block_traversal_1d(purpose, data, begin_offset,
                                           begin_offset + (end[dims[0]] - begin[dims[0]]) * dimension_offsets[dims[0]],
                                           stride * dimension_offsets[dims[0]], interp_func, pb);
                }
            }
        }
        max_error = 0;
        for (size_t i = (begin[dims[0]] ? begin[dims[0]] + stride : 0); i <= end[dims[0]]; i += stride) {
            for (size_t k = (begin[dims[2]] ? begin[dims[2]] + stride2x : 0); k <= end[dims[2]]; k += stride2x) {
                for (size_t t = (begin[dims[3]] ? begin[dims[3]] + stride2x : 0); t <= end[dims[3]]; t += stride2x) {
                    size_t begin_offset = i * dimension_offsets[dims[0]] + begin[dims[1]] * dimension_offsets[dims[1]] +
                                          k * dimension_offsets[dims[2]] + t * dimension_offsets[dims[3]];
                    predict_error +=
                        block_traversal_1d(purpose, data, begin_offset,
                                           begin_offset + (end[dims[1]] - begin[dims[1]]) * dimension_offsets[dims[1]],
                                           stride * dimension_offsets[dims[1]], interp_func, pb);
                }
            }
        }
        max_error = 0;
        for (size_t i = (begin[dims[0]] ? begin[dims[0]] + stride : 0); i <= end[dims[0]]; i += stride) {
            for (size_t j = (begin[dims[1]] ? begin[dims[1]] + stride : 0); j <= end[dims[1]]; j += stride) {
                for (size_t t = (begin[dims[3]] ? begin[dims[3]] + stride2x : 0); t <= end[dims[3]]; t += stride2x) {
                    size_t begin_offset = i * dimension_offsets[dims[0]] + j * dimension_offsets[dims[1]] +
                                          begin[dims[2]] * dimension_offsets[dims[2]] + t * dimension_offsets[dims[3]];
                    predict_error +=
                        block_traversal_1d(purpose, data, begin_offset,
                                           begin_offset + (end[dims[2]] - begin[dims[2]]) * dimension_offsets[dims[2]],
                                           stride * dimension_offsets[dims[2]], interp_func, pb);
                }
            }
        }

        max_error = 0;
        for (size_t i = (begin[dims[0]] ? begin[dims[0]] + stride : 0); i <= end[dims[0]]; i += stride) {
            for (size_t j = (begin[dims[1]] ? begin[dims[1]] + stride : 0); j <= end[dims[1]]; j += stride) {
                for (size_t k = (begin[dims[2]] ? begin[dims[2]] + stride : 0); k <= end[dims[2]]; k += stride) {
                    size_t begin_offset = i * dimension_offsets[dims[0]] + j * dimension_offsets[dims[1]] +
                                          k * dimension_offsets[dims[2]] + begin[dims[3]] * dimension_offsets[dims[3]];
                    predict_error +=
                        block_traversal_1d(purpose, data, begin_offset,
                                           begin_offset + (end[dims[3]] - begin[dims[3]]) * dimension_offsets[dims[3]],
                                           stride * dimension_offsets[dims[3]], interp_func, pb);
                }
            }
        }
        return predict_error;
    }

    LinearWeightLearningInterpolator linear_interpolator;
//    NonLinearWeightWithConstantLearningInterpolator linear_interpolator;
//    LinearWeightByBlocksLearningInterpolator linear_interpolator;
//    LinearWeightWithConstantLearningInterpolator linear_interpolator;

    // for analysis purpose
//    std::ofstream* order_file;
//    double prediction_error_sum = 0;
//    double prediction_error_squared_sum = 0;
//    double max_prediction_error = -9999999999999;
//    double diff_sum = 0; // sum of difference between original data and decompressed data
//    size_t num_data_points_compressed = 0;
//    size_t predicted_data_points = 0;

    CubicWeightLearningInterpolator cubic_interpolator;
    int interpolation_level = -1;
    uint blocksize;
    int interpolator_id;
    double eb_ratio = 0.5;
    std::vector<std::string> interpolators = {"linear", "cubic"};
    int *quant_inds;
    size_t quant_index = 0;
    double max_error;
    Quantizer quantizer;
    size_t num_elements;
    std::array<size_t, N> global_dimensions;
    std::array<size_t, N> dimension_offsets;
    std::vector<std::array<int, N>> dimension_sequences;
    int direction_sequence_id;
};

template <class T, uint N, class Quantizer>
InterpolationDecomposition<T, N, Quantizer> make_decomposition_interpolation(const Config &conf, Quantizer quantizer) {
    return InterpolationDecomposition<T, N, Quantizer>(conf, quantizer);
}

}  // namespace SZ3

#endif
