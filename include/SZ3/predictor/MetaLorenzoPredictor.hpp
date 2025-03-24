#ifndef _meta_lorenzo_hpp
#define _meta_lorenzo_hpp

#include<vector>
#include<iostream>

#include "SZ3/utils/MetaDef.hpp"
#include "SZ3/utils/MemoryUtil.hpp"

namespace SZMETA {

using namespace SZ3;

template <class T>
class GeneralWeightLearningPredictor {
    int num_weights;
    std::vector<double> weights;
    
    size_t num_data_points = 0;
    std::vector<std::vector<double>> as;
    std::vector<double> bs;

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


public:
    template <typename... Types>
    GeneralWeightLearningPredictor(Types... initial_weights) {
        weights = std::vector<double>{static_cast<double>(std::forward<Types>(initial_weights))...};    
        this->num_weights = weights.size();

        as.resize(num_weights);
        for(int i = 0; i < num_weights; i++) {
            as[i].resize(num_weights, 0.0);
        }

        bs.resize(num_weights, 0.0);
    }

    // GeneralWeightLearningPredictor(int num_weights) {
    //     this->num_weights = num_weights;

    //     as.resize(num_weights);
    //     for(int i = 0; i < num_weights; i++) {
    //         as[i].resize(num_weights, 0.0);
    //     }

    //     bs.resize(num_weights, 0.0);
    //     weights.resize(num_weights, 0.0);
    // }

    void learn(T original, std::vector<T> values) {
        if (values.size() != num_weights) throw std::runtime_error("Number of data points not equal to number of weights");
        num_data_points++;
        
        for(int i = 0; i < num_weights; i++) {
            for(int j = 0; j < num_weights; j++) {
                as[i][j] += values[i] * values[j];
            }
        }

        for(int i = 0; i < num_weights; i++) {
            bs[i] += original * values[i];
        }
    }

    template <typename... Types>
    void learn(T original, Types... values){
        auto values_parsed = std::vector<T>{static_cast<T>(std::forward<Types>(values))...};
        learn(original, values_parsed);
    }
    
    void finalize_learning() {
        std::cout << "Computing weights from " << num_data_points << " data points" << std::endl;

        for(int i = 0; i < num_weights; i++) {
            for(int j = 0; j < num_weights; j++) {
                std::cout << as[i][j] << ", ";
            }
            std::cout <<std::endl;
        }

        for(int i = 0; i < num_weights; i++) {
            std::cout << bs[i] <<", ";
        }
        std::cout << std::endl;

        try {
            weights = EquationSolver::solve(as, bs);    
        } catch (const std::runtime_error& error) {
            std::cout << "determinant is 0. So, using initial weights" << std::endl;
        }

    }

    T predict(std::vector<T> values) {
        if (values.size() != num_weights) throw std::runtime_error("Number of data points not equal to number of weights");
        T prediction = 0;
        for(int i = 0; i < num_weights; i++) {
            prediction += weights[i] * values[i];
        }
        return prediction;
    }

    template <typename... Types>
    T predict(Types... values) {
        auto values_parsed = std::vector<T>{static_cast<T>(std::forward<Types>(values))...};
        return predict(values_parsed);
    }

    void print_weights(){
        std::cout << "Weights: ";
        for(int i = 0; i < num_weights; i++) {
            std::cout << weights[i] << ", ";
        }
        std::cout << std::endl;
    }

    void save(unsigned char *&c) {
        write(num_weights, c);
        for(int i = 0; i < num_weights; i++) {
            write(weights[i], c);
        }
    }
    void load(const unsigned char *&c, size_t &remaining_length) {
        read(num_weights, c, remaining_length);
        for(int i = 0 ; i < num_weights; i++) {
            double w = 0.0;
            read(w, c, remaining_length);
            weights.push_back(w);
        }
        
    }
};

template <class T>
class WeightLearningLorenzo3D {
    GeneralWeightLearningPredictor<T> predictor;
    int t = 0;

    std::vector<T> find_points_for_prediction(const T *data_pos, size_t dim0_offset, size_t dim1_offset) {
        std::vector<T> points = {
            data_pos[-1], data_pos[-dim1_offset], data_pos[-dim0_offset],
            data_pos[-dim1_offset - 1], data_pos[-dim0_offset - 1], data_pos[-dim0_offset - dim1_offset],
            data_pos[-dim0_offset - dim1_offset - 1]
        };
        
        return points;
    }
public:
    WeightLearningLorenzo3D(): predictor(1, 1, 1, -1, -1, -1, 1){

    }

    void learn(T original, const T *data_pos, size_t dim0_offset, size_t dim1_offset) {
        predictor.learn(original, find_points_for_prediction(data_pos, dim0_offset, dim1_offset));
    }

    T predict(const T *data_pos, size_t dim0_offset, size_t dim1_offset) {
        return predictor.predict(find_points_for_prediction(data_pos, dim0_offset, dim1_offset));
        // return 0.5 * data_pos[-1] + 0.5 * data_pos[-dim0_offset - dim1_offset - 1]; // better ratio than Lorenzo for QMCPACK
        // return 0.5 * data_pos[-1] + 0.5 * data_pos[-2]; // better ratio than Lorenzo for QMCPACK
        // return (-data_pos[-1] + 9 * data_pos[-dim1_offset] + 9 * data_pos[-dim0_offset] - data_pos[-dim1_offset - 1]) / 16.0;
    }

};

template <typename T>
inline T lorenzo_predict_1d(const T *data_pos, size_t dim0_offset) {
    return data_pos[-1];
}

template <typename T>
inline T lorenzo_predict_1d_2layer(const T *data_pos, size_t dim0_offset) {
    return 2 * data_pos[-1] - data_pos[-2];
}

template <typename T>
inline T lorenzo_predict_2d(const T *data_pos, size_t dim0_offset, size_t dim1_offset) {
    return data_pos[-1] + data_pos[-dim0_offset] - data_pos[-1 - dim0_offset];
}

template <typename T>
inline T lorenzo_predict_2d_2layer(const T *data_pos, size_t dim0_offset, size_t dim1_offset) {
    return 2 * data_pos[-dim0_offset] - data_pos[-2 * dim0_offset] + 2 * data_pos[-1] - 4 * data_pos[-1 - dim0_offset] +
           2 * data_pos[-1 - 2 * dim0_offset] - data_pos[-2] + 2 * data_pos[-2 - dim0_offset] -
           data_pos[-2 - 2 * dim0_offset];
}

template <typename T>
inline T lorenzo_predict_3d(const T *data_pos, size_t dim0_offset, size_t dim1_offset) {
    return data_pos[-1] + data_pos[-dim1_offset] + data_pos[-dim0_offset] - data_pos[-dim1_offset - 1] -
           data_pos[-dim0_offset - 1] - data_pos[-dim0_offset - dim1_offset] + data_pos[-dim0_offset - dim1_offset - 1];
}

template <typename T>
inline T lorenzo_predict_3d_2layer(const T *data_pos, size_t dim0_offset, size_t dim1_offset) {
    return 2 * data_pos[-1] - data_pos[-2] + 2 * data_pos[-dim1_offset] - 4 * data_pos[-dim1_offset - 1] +
           2 * data_pos[-dim1_offset - 2] - data_pos[-2 * dim1_offset] + 2 * data_pos[-2 * dim1_offset - 1] -
           data_pos[-2 * dim1_offset - 2] + 2 * data_pos[-dim0_offset] - 4 * data_pos[-dim0_offset - 1] +
           2 * data_pos[-dim0_offset - 2] - 4 * data_pos[-dim0_offset - dim1_offset] +
           8 * data_pos[-dim0_offset - dim1_offset - 1] - 4 * data_pos[-dim0_offset - dim1_offset - 2] +
           2 * data_pos[-dim0_offset - 2 * dim1_offset] - 4 * data_pos[-dim0_offset - 2 * dim1_offset - 1] +
           2 * data_pos[-dim0_offset - 2 * dim1_offset - 2] - data_pos[-2 * dim0_offset] +
           2 * data_pos[-2 * dim0_offset - 1] - data_pos[-2 * dim0_offset - 2] +
           2 * data_pos[-2 * dim0_offset - dim1_offset] - 4 * data_pos[-2 * dim0_offset - dim1_offset - 1] +
           2 * data_pos[-2 * dim0_offset - dim1_offset - 2] - data_pos[-2 * dim0_offset - 2 * dim1_offset] +
           2 * data_pos[-2 * dim0_offset - 2 * dim1_offset - 1] - data_pos[-2 * dim0_offset - 2 * dim1_offset - 2];
}

template <typename T, class Quantizer>
inline void lorenzo_predict_quantize_3d(const meanInfo<T> &mean_info, const T *data_pos, T *buffer, T precision,
                                        T recip_precision, int capacity, int intv_radius, int size_x, int size_y,
                                        int size_z, size_t buffer_dim0_offset, size_t buffer_dim1_offset,
                                        size_t dim0_offset, size_t dim1_offset, int *&type_pos,
                                        int *unpred_count_buffer, T *unpred_buffer, size_t offset, int padding_layer,
                                        bool use_2layer, Quantizer &quantizer, int pred_dim) {
    const T *cur_data_pos = data_pos;
    T *buffer_pos = buffer + padding_layer * (buffer_dim0_offset + buffer_dim1_offset + 1);
    int radius = (quantizer.get_out_range().second - quantizer.get_out_range().first) / 2;
    for (int i = 0; i < size_x; i++) {
        for (int j = 0; j < size_y; j++) {
            for (int k = 0; k < size_z; k++) {
                T *cur_buffer_pos = buffer_pos + k;
                T cur_data = cur_data_pos[k];
                T pred;
                if (mean_info.use_mean && fabs(cur_data - mean_info.mean) <= precision) {
                    type_pos[k] = radius;
                    *cur_buffer_pos = mean_info.mean;
                } else {
                    if (use_2layer) {
                        if (pred_dim == 3) {
                            pred = lorenzo_predict_3d_2layer(cur_buffer_pos, buffer_dim0_offset, buffer_dim1_offset);
                        } else if (pred_dim == 2) {
                            pred = lorenzo_predict_2d_2layer(cur_buffer_pos, buffer_dim0_offset, buffer_dim1_offset);
                        } else {
                            pred = lorenzo_predict_1d_2layer(cur_buffer_pos, buffer_dim0_offset);
                        }
                    } else {
                        if (pred_dim == 3) {
                            pred = lorenzo_predict_3d(cur_buffer_pos, buffer_dim0_offset, buffer_dim1_offset);
                        } else if (pred_dim == 2) {
                            pred = lorenzo_predict_2d(cur_buffer_pos, buffer_dim0_offset, buffer_dim1_offset);
                        } else {
                            pred = lorenzo_predict_1d(cur_buffer_pos, buffer_dim0_offset);
                        }
                    }
                    //                    *cur_buffer_pos = cur_data;
                    //                    type_pos[k] = quantizer.quantize_and_overwrite(*cur_buffer_pos, pred);
                    // Me: for training, *cur_buffer_pos = cur_data
                    type_pos[k] = quantizer.quantize_and_overwrite(cur_data, pred, *cur_buffer_pos); //Me: for training comment this out
                    if (mean_info.use_mean && type_pos[k] >= radius) {
                        type_pos[k] += 1;
                    }
                }
            }
            type_pos += size_z;
            buffer_pos += buffer_dim1_offset;
            cur_data_pos += dim1_offset;
        }
        buffer_pos += buffer_dim0_offset - size_y * buffer_dim1_offset;
        cur_data_pos += dim0_offset - size_y * dim1_offset;
    }
}

template <typename T, class Quantizer>
inline void lorenzo_predict_recover_3d(const meanInfo<T> &mean_info, T *buffer, T precision, int intv_radius,
                                       int size_x, int size_y, int size_z, size_t buffer_dim0_offset,
                                       size_t buffer_dim1_offset, size_t dim0_offset, size_t dim1_offset,
                                       const int *&type_pos, int *unpred_count_buffer, const T *unpred_data_buffer,
                                       const int offset, T *dec_data_pos, const int layer, bool use_2layer,
                                       Quantizer &quantizer, int pred_dim) {
    T *cur_data_pos = dec_data_pos;
    T *buffer_pos = buffer + layer * (buffer_dim0_offset + buffer_dim1_offset + 1);
    int radius = (quantizer.get_out_range().second - quantizer.get_out_range().first) / 2;
    for (int i = 0; i < size_x; i++) {
        for (int j = 0; j < size_y; j++) {
            for (int k = 0; k < size_z; k++) {
                int index = j * size_z + k;
                int type_val = type_pos[index];
                T *cur_buffer_pos = buffer_pos + k;
                if (type_val == 0) {
                    cur_data_pos[k] = *cur_buffer_pos = quantizer.recover_unpred();
                } else if (mean_info.use_mean && type_val == radius) {
                    cur_data_pos[k] = *cur_buffer_pos = mean_info.mean;
                } else {
                    T pred;
                    //                        pred = predict(cur_buffer_pos, buffer_dim1_offset, buffer_dim0_offset);
                    if (use_2layer) {
                        if (pred_dim == 3) {
                            pred = lorenzo_predict_3d_2layer(cur_buffer_pos, buffer_dim0_offset, buffer_dim1_offset);
                        } else if (pred_dim == 2) {
                            pred = lorenzo_predict_2d_2layer(cur_buffer_pos, buffer_dim0_offset, buffer_dim1_offset);
                        } else {
                            pred = lorenzo_predict_1d_2layer(cur_buffer_pos, buffer_dim0_offset);
                        }
                    } else {
                        if (pred_dim == 3) {
                            pred = lorenzo_predict_3d(cur_buffer_pos, buffer_dim0_offset, buffer_dim1_offset);
                        } else if (pred_dim == 2) {
                            pred = lorenzo_predict_2d(cur_buffer_pos, buffer_dim0_offset, buffer_dim1_offset);
                        } else {
                            pred = lorenzo_predict_1d(cur_buffer_pos, buffer_dim0_offset);
                        }
                    }
                    if (mean_info.use_mean && type_val > radius) {
                        type_val -= 1;
                    }
                    cur_data_pos[k] = *cur_buffer_pos = quantizer.recover_pred(pred, type_val);
                }
            }
            buffer_pos += buffer_dim1_offset;
            cur_data_pos += dim1_offset;
        }
        type_pos += size_y * size_z;
        buffer_pos += buffer_dim0_offset - size_y * buffer_dim1_offset;
        cur_data_pos += dim0_offset - size_y * dim1_offset;
    }
}

}  // namespace SZMETA
#endif
