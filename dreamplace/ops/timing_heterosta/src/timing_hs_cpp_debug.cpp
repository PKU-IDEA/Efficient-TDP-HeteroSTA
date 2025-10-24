#include "timing_hs_cpp.h"
#include "timing_hs_io_cpp.h"
#include <cstring>
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>

// Debug output control macros for HeteroSTA
#ifndef HETEROSTA_DEBUG_LEVEL
#define HETEROSTA_DEBUG_LEVEL 3  // 0: no debug, 1: basic, 2: detailed, 3: verbose
#endif

// Debug helper functions for HeteroSTA
namespace heterosta_debug {

    // Get current timestamp string
    inline std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
        return ss.str();
    }

    // Get iteration counter (static to persist across calls)
    inline int get_iteration_counter() {
        static int iteration = 0;
        return ++iteration;
    }

    // Create filename with iteration and timestamp
    inline std::string create_debug_filename(const std::string& base_name, int iteration) {
        std::stringstream ss;
        ss << base_name << "_round_" << iteration << "_" << get_timestamp();
        return ss.str();
    }

    // Log pin2pin weight change details for HeteroSTA
    inline void log_weight_change_heterosta(std::ofstream& log_file, int iteration,
                                           long pin1, long pin2, float old_weight, float new_weight,
                                           float delta, float path_slack, float wns, bool is_new_key) {
        if (log_file.is_open()) {
            log_file << "[HeteroSTA-第" << iteration << "轮] ";
            if (is_new_key) {
                log_file << "新Key";
            } else {
                log_file << "更新Key";
            }
            log_file << "(" << pin1 << "," << pin2 << "): ";
            log_file << std::fixed << std::setprecision(6);
            log_file << old_weight << " -> " << new_weight
                    << " (delta: " << (delta >= 0 ? "+" : "") << delta
                    << ", path_slack/wns: " << (path_slack/wns) << ")" << std::endl;
        }
    }

    // Write HeteroSTA statistics summary
    inline void write_heterosta_stats_summary(const std::string& filename, int iteration,
                                             int num_unique_pairs, int num_all_pairs,
                                             int dict_size, float wns, float tns,
                                             float wns_hold, float tns_hold, int num_paths,
                                             const std::chrono::steady_clock::time_point& start_time,
                                             const std::chrono::steady_clock::time_point& end_time) {
        std::ofstream stats_file(filename);
        if (stats_file.is_open()) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

            stats_file << "=== HeteroSTA PIN2PIN 第" << iteration << "轮优化统计 ===\n";
            stats_file << "时间戳: " << get_timestamp() << "\n";
            stats_file << "执行时间: " << (duration * 0.001) << " 秒\n\n";

            stats_file << "== 时序信息 ==\n";
            stats_file << "Setup WNS: " << std::fixed << std::setprecision(6) << wns << "\n";
            stats_file << "Setup TNS: " << tns << "\n";
            stats_file << "Hold WNS: " << wns_hold << "\n";
            stats_file << "Hold TNS: " << tns_hold << "\n";
            stats_file << "提取路径数: " << num_paths << "\n\n";

            stats_file << "== Pin2Pin权重统计 ==\n";
            stats_file << "新增唯一键值对: " << num_unique_pairs << "\n";
            stats_file << "处理的总键值对: " << num_all_pairs << "\n";
            stats_file << "字典当前大小: " << dict_size << "\n";
            stats_file << "权重更新率: " << std::fixed << std::setprecision(2)
                      << (dict_size > 0 ? (double)num_all_pairs / dict_size * 100 : 0) << "%\n";

            stats_file.close();
        }
    }

    // Append to global HeteroSTA summary log
    inline void append_heterosta_global_summary(int iteration, int num_unique_pairs, int num_all_pairs,
                                               int dict_size, float wns, float tns, float wns_hold, float tns_hold,
                                               int num_paths, double duration_seconds) {
        std::ofstream global_log("heterosta_pin2pin_global_summary.log", std::ios::app);
        if (global_log.is_open()) {
            // Write header if this is the first iteration
            if (iteration == 1) {
                global_log << "=== HeteroSTA PIN2PIN 全局优化汇总日志 ===\n";
                global_log << "开始时间: " << get_timestamp() << "\n\n";
                global_log << "轮次\t时间戳\t\t执行时间(s)\t路径数\tSetup WNS\tSetup TNS\tHold WNS\tHold TNS\t新增pairs\t处理pairs\t字典大小\n";
                global_log << "────────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n";
            }

            global_log << iteration << "\t" << get_timestamp() << "\t"
                      << std::fixed << std::setprecision(3) << duration_seconds << "\t\t"
                      << num_paths << "\t"
                      << std::setprecision(6) << wns << "\t" << tns << "\t"
                      << wns_hold << "\t" << tns_hold << "\t"
                      << num_unique_pairs << "\t\t" << num_all_pairs << "\t\t" << dict_size << "\n";
            global_log.close();
        }
    }
}


DREAMPLACE_BEGIN_NAMESPACE

#ifdef CUDA_FOUND
// Forward declaration of CUDA launcher
template <typename T>
void updateNetWeightCudaLauncher(
    STAHoldings* sta,
    int num_nets,
    int num_pins,
    const int* flat_netpin,
    const int* netpin_start,
    T* net_criticality,
    T* net_criticality_deltas,
    T* net_weights,
    T* net_weight_deltas,
    const int* degree_map,
    T momentum_decay_factor,
    T max_net_weight,
    int ignore_net_degree);
#endif


/// 
/// @brief Perform timing analysis using HeteroSTA engine.
///// @param sta the HeteroSTA holdings object containing timing database.
///// @param x the horizontal coordinates of cell locations.
///// @param y the vertical coordinates of cell locations.
///// @param num_pins the number of pins in the design.
///// @param wire_resistance_per_micron unit-length resistance value.
///// @param wire_capacitance_per_micron unit-length capacitance value.
///// @param scale_factor the scaling factor to be applied to the design.
///// @param lef_unit the unit distance microns defined in the LEF file.
///// @param def_unit the unit distance microns defined in the DEF file.
///// @param slacks_rf output array for pin slacks (rise/fall).
///// @param ignore_net_degree the degree threshold for ignoring high-degree nets.
///// @param use_cuda whether to use CUDA for computation.
///
template <typename T>
int timingHeterostaCppLauncher(
		STAHoldings& sta,
		const T* x,const T* y,
		size_t num_pins,
		T wire_resistance_per_micron,
		T wire_capacitance_per_micron,
		double scale_factor, int lef_unit, int def_unit,
		float (*slacks_rf)[2],
		int ignore_net_degree, bool use_cuda){
	dreamplacePrint(kINFO, "HeteroSTA launcher started\n");

	// Convert coordinates and apply scaling for HeteroSTA units
	double unit_to_micron = scale_factor * def_unit;
	double res_unit = 1e3;  // Rust canonical units
	double cap_unit = 1e-15;
	double rf = static_cast<double>(wire_resistance_per_micron) / res_unit;
	double cf = static_cast<double>(wire_capacitance_per_micron) / cap_unit;		
	double unit_cap_xy = cf / unit_to_micron;
	double unit_res_xy = rf / unit_to_micron;

	auto beg = std::chrono::steady_clock::now();

	dreamplacePrint(kINFO,"extract rc from placement...\n");

	auto device = use_cuda ? torch::kCUDA : torch::kCPU;

	auto via_res_tensor = torch::tensor(0.0f, torch::TensorOptions().dtype(torch::kFloat32).device(device));
	auto flute_accuracy_tensor = torch::tensor(8, torch::TensorOptions().dtype(torch::kInt32).device(device));
	auto pdr_alpha_tensor = torch::tensor(0.3f, torch::TensorOptions().dtype(torch::kFloat32).device(device));
	auto use_flute_or_pdr_tensor = torch::tensor(0, torch::TensorOptions().dtype(torch::kUInt8).device(device));

	float via_res = via_res_tensor.item<float>();
	uint32_t flute_accuracy= static_cast<uint32_t>(flute_accuracy_tensor.item<int32_t>());
	float pdr_alpha = pdr_alpha_tensor.item<float>();
	uint8_t use_flute_or_pdr = use_flute_or_pdr_tensor.item<uint8_t>();	

	// If heterosta_extract_rc_from_placement panics, the process will terminate
	heterosta_extract_rc_from_placement(&sta, 
			reinterpret_cast<const float*>(x), 
			reinterpret_cast<const float*>(y),
			static_cast<float>(unit_cap_xy), 
			static_cast<float>(unit_cap_xy),
			static_cast<float>(unit_res_xy), 
			static_cast<float>(unit_res_xy),
			via_res,flute_accuracy,
			pdr_alpha,use_flute_or_pdr,	
			use_cuda);
	dreamplacePrint(kINFO,"finish rc extraction...\n");
	heterosta_update_delay(&sta, use_cuda);

	// Update arrival times and required arrival times for timing analysis
	heterosta_update_arrivals(&sta, use_cuda);

	dreamplacePrint(kINFO,"finish state updates...\n");

	static_assert(sizeof(bool) == 1);
	static std::vector<uint8_t> timingpin_is_endpoint;
	timingpin_is_endpoint.resize(num_pins);
	heterosta_get_is_endpoint(&sta, (bool *) timingpin_is_endpoint.data());
	//heterosta_launch_debug_shell(&sta);

	// Report pin slacks
	if(!use_cuda && slacks_rf!=nullptr)
	{
		heterosta_report_slacks_at_max(&sta, slacks_rf, use_cuda);
		dreamplacePrint(kDEBUG, "HeteroSTA slack report completed\n");
		dreamplacePrint(kDEBUG, "Sample slacks (rise/fall) for first 10 endpoints:\n");
		int endpoint_count = 0;
		for (size_t i = 0; i < num_pins && endpoint_count < 10; ++i) {
			if (timingpin_is_endpoint[i]) {
				dreamplacePrint(kDEBUG, "  Endpoint Pin %s (%zu): Rise=%.3f, Fall=%.3f\n",
						TimingHeterostaIO::getPinName(i), i, slacks_rf[i][0], slacks_rf[i][1]);
				endpoint_count++;
			}
		}
	}
	auto end = std::chrono::steady_clock::now();
	auto usc = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);
	dreamplacePrint(kINFO, "finish state updates (%f s)\n", usc.count() * 0.001);
	return 0;
}

// Implementation of the forward method
void TimingHeterostaCpp::forward(
		STAHoldings& sta, torch::Tensor pos,
		size_t num_pins,
		double wire_resistance_per_micron,
		double wire_capacitance_per_micron,
		double scale_factor, int lef_unit, int def_unit,
		torch::Tensor slacks_rf,
		int ignore_net_degree, bool use_cuda) {
	// Check tensor properties for input validation
	CHECK_EVEN(pos);
	CHECK_CONTIGUOUS(pos);
	// Device consistency check
	bool pos_is_cuda = pos.is_cuda();

	if (use_cuda != pos_is_cuda) {
		dreamplacePrint(kWARN, "Device mismatch detected but proceeding (pos_pin should handle device placement)\n");
	}
	// Check slack tensor if provided
	float (*slacks_rf_ptr)[2] = nullptr;
	if (slacks_rf.numel() > 0) {
		CHECK_CONTIGUOUS(slacks_rf);
		float* slacks_rf_data = slacks_rf.data_ptr<float>();
		slacks_rf_ptr = reinterpret_cast<float (*)[2]>(slacks_rf_data);
	}

	DREAMPLACE_DISPATCH_FLOATING_TYPES(
			pos, "timingHeterostaCppLauncher",
			[&] {
			timingHeterostaCppLauncher<scalar_t>(
					sta,
					DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
					DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() / 2,
					num_pins,
					wire_resistance_per_micron,
					wire_capacitance_per_micron,
					scale_factor, lef_unit, def_unit,
					slacks_rf_ptr,
					ignore_net_degree,
					use_cuda
					);
			});
}


///
/// @brief Update net weights based on timing criticality using HeteroSTA (template launcher).
/// @param sta the HeteroSTA holdings object containing timing database.
/// @param n the maximum number of critical paths to analyze.
/// @param num_nets the number of nets in the design.
/// @param num_pins the number of pins in the design.
/// @param flat_netpin the flattened netpin array.
/// @param netpin_start the starting indices for each net.
/// @param pin_to_node_map the pin to node mapping array.
/// @param net_criticality the criticality values of nets (array).
/// @param net_criticality_deltas the criticality delta values of nets (array).
/// @param net_weights the weights of nets (array).
/// @param net_weight_deltas the increment of net weights.
/// @param degree_map the degree map of nets.
/// @param max_net_weight the maximum net weight in timing optimization.
/// @param pin2pin_net_weight 
/// @param net_weighting_scheme the net-weighting scheme.
/// @param momentum_decay_factor the decay factor in momentum iteration.
/// @param ignore_net_degree the net degree threshold for ignoring high-degree nets.
/// @param num_threads number of threads for parallel computing.
/// @param pin2pin_max_weight 
/// @param pin2pin_min_weight
/// @param pin2pin_accumulate_weight 
///
template <typename T>
void updateNetWeightCppLauncher(
		STAHoldings& sta,
		int num_nets, 
		int num_pins,
		const int* flat_netpin, 
		const int* netpin_start,
		const int* pin_to_node_map,
		const int* pin_to_net_map,
		T* net_criticality, 
		T* net_criticality_deltas,
		T* net_weights, 
		T* net_weight_deltas,
		const int* degree_map, 
		T momentum_decay_factor, 
		pybind11::dict& pin2pin_net_weight,
		int net_weighting_scheme,
		T max_net_weight,
		int ignore_net_degree, 
		int num_threads,
		int pin2pin_max_weight, int pin2pin_min_weight, double pin2pin_accumulate_weight,
		bool use_cuda
		){
	
	if(net_weighting_scheme == 1) //LILITH
	{
	dreamplacePrint(kINFO, "apply lilith net-weighting scheme...\n");
    dreamplacePrint(kINFO, "lilith mode momentum decay factor: %f\n", momentum_decay_factor);

	// Calculate run-time of net-weighting update.
    auto beg = std::chrono::steady_clock::now();
	// Get WNS/TNS from HeteroSTA
	float wns, tns;
	bool success = heterosta_report_wns_tns(&sta, &wns, &tns, true,false);

	// Get pin slacks from HeteroSTA
	static std::vector<float> slack_data;
	if (slack_data.size() < num_pins * 2) {
		slack_data.resize(num_pins * 2);
	}
	float (*slack_array)[2] = reinterpret_cast<float(*)[2]>(slack_data.data());
	heterosta_report_slacks_at_max(&sta, slack_array, false);

	// Apply net weighting using momentum-based criticality update
	if(!(wns < 0)) wns = 0;

#pragma omp parallel for num_threads(num_threads)
	for(int net_i = 0; net_i < num_nets; ++net_i) {

		float net_slack = std::numeric_limits<float>::max();
		int np_s = netpin_start[net_i], np_e = netpin_start[net_i + 1];

		// Calculate net slack as minimum of all pin slacks in the net
		for(int np_i = np_s; np_i < np_e; ++np_i) {
			int pin_i = flat_netpin[np_i];
			if (pin_i < num_pins) {
				// Take the worst (minimum) of rise/fall slack for this pin
				float pin_worst_slack = std::min(slack_array[pin_i][0], 
						slack_array[pin_i][1]);
				net_slack = std::min(net_slack, pin_worst_slack);
			}
		}


		if(wns < 0) {
			// Calculate normalized criticality
			float nc = (net_slack < 0) ? std::max(0.f, net_slack / wns) : 0;
			// Apply momentum-based criticality update 
			net_criticality[net_i] = std::pow(1 + net_criticality[net_i], momentum_decay_factor) * 
				std::pow(1 + nc, 1 - momentum_decay_factor) - 1;
		}

		if(degree_map[net_i]>ignore_net_degree) continue;
		net_weights[net_i] *= (1 + net_criticality[net_i]);

		if(net_weights[net_i] > max_net_weight)        net_weights[net_i] = max_net_weight;

		auto end = std::chrono::steady_clock::now();
		dreamplacePrint(kINFO, "finish net-weighting (%f s)\n",
		std::chrono::duration_cast<std::chrono::milliseconds>(
			end - beg).count() * 0.001); 
	}

	}
	else if(net_weighting_scheme==2) //PIN2PIN
	{
        // Get iteration counter and setup debug output
        int current_iteration = heterosta_debug::get_iteration_counter();

		// Apply net-weighting scheme.
        dreamplacePrint(kINFO, "=== HeteroSTA 第%d轮 PIN2PIN net-weighting 开始 ===\n", current_iteration);
        // Calculate run-time of net-weighting update.
        auto begT = std::chrono::steady_clock::now();

        // Setup debug output files if enabled
        std::ofstream weight_log, path_log, summary_log;

#if HETEROSTA_DEBUG_LEVEL >= 1
        std::string weight_filename = heterosta_debug::create_debug_filename("heterosta_pin2pin_weights", current_iteration) + ".log";
        std::string path_filename = heterosta_debug::create_debug_filename("heterosta_pin2pin_paths", current_iteration) + ".txt";
        std::string summary_filename = heterosta_debug::create_debug_filename("heterosta_pin2pin_summary", current_iteration) + ".log";

        weight_log.open(weight_filename);
        path_log.open(path_filename);
        summary_log.open(summary_filename);

        if (weight_log.is_open()) {
            weight_log << "=== HeteroSTA PIN2PIN 第" << current_iteration << "轮权重变化日志 ===\n";
            weight_log << "时间戳: " << heterosta_debug::get_timestamp() << "\n\n";
        }

        if (summary_log.is_open()) {
            summary_log << "=== HeteroSTA PIN2PIN 第" << current_iteration << "轮优化汇总 ===\n";
            summary_log << "开始时间: " << heterosta_debug::get_timestamp() << "\n\n";
        }
#endif

		dreamplacePrint(kINFO, "extracting paths...\n");
		size_t num_paths = 100000000;  // a big enough number to extract all violated paths
		size_t nworst = 1;
		float slack_threshold = 0.0;
		bool split_endpoint_rf = true;
        bool result_use_cuda = false;
		PBAPathCollectionCppInterface* path_interface_max = heterosta_report_paths(&sta, num_paths, nworst, slack_threshold, split_endpoint_rf, true, use_cuda,result_use_cuda);
		PBAPathCollectionCppInterface* path_interface_min = heterosta_report_paths(&sta, num_paths, nworst, slack_threshold, split_endpoint_rf, false, use_cuda,result_use_cuda);
		dreamplacePrint(kINFO, "paths extraction done, found %d paths...\n", path_interface_max->num_paths);

		int num_unique_pairs = 0;
        int num_all_pairs = 0;
		int num_paths_max = path_interface_max->num_paths;
		int num_paths_min = path_interface_min->num_paths;
		dreamplacePrint(kINFO,"第%d轮: 路径数量:%d\n", current_iteration, num_paths_max);
		dreamplacePrint(kINFO,"num_paths_min:%d\n",num_paths_min);
		dreamplacePrint(kINFO,"nworst:%zu\n",nworst);

		const uintptr_t* path_st_max = path_interface_max->path_st;
		const uintptr_t* pin_rfs_max = path_interface_max->pin_rfs;
		const uintptr_t* path_st_min = path_interface_min->path_st;
		const uintptr_t* pin_rfs_min = path_interface_min->pin_rfs;
		long path_st_indice = 0;
		long path_st_indice_nx = 0;

		float wns, tns,wns_hold, tns_hold;
		bool success = heterosta_report_wns_tns(&sta, &wns, &tns, true, use_cuda);
		dreamplacePrint(kINFO,"node_id !=last_node\n");
		heterosta_report_wns_tns(&sta, &wns_hold, &tns_hold, false, use_cuda);
		dreamplacePrint(kINFO,"第%d轮: setup wns:%f,tns:%f\n", current_iteration, wns,tns);
		dreamplacePrint(kINFO,"第%d轮: hold wns:%f,tns:%f\n", current_iteration, wns_hold,tns_hold);

		// Create path dump filename with timestamp
		std::string file_path = heterosta_debug::create_debug_filename("heterosta_paths_dump_setup", current_iteration) + ".txt";
		heterosta_dump_paths_setup_to_file(&sta, num_paths_max, nworst, true, file_path.c_str(), use_cuda);
		file_path = heterosta_debug::create_debug_filename("heterosta_paths_dump_hold", current_iteration) + ".txt";
		heterosta_dump_paths_to_file(&sta, num_paths_min, nworst, true,false, file_path.c_str(), use_cuda);


#if HETEROSTA_DEBUG_LEVEL >= 2
        if (summary_log.is_open()) {
            summary_log << "== 路径提取结果 ==\n";
            summary_log << "提取路径数: " << num_paths_max << "\n";
            summary_log << "Setup WNS: " << std::fixed << std::setprecision(6) << wns << "\n";
            summary_log << "Setup TNS: " << tns << "\n";
            summary_log << "Hold WNS: " << wns_hold << "\n";
            summary_log << "Hold TNS: " << tns_hold << "\n";
            summary_log << "路径dump文件: " << file_path << "\n\n";
        }
#endif

		// #pragma omp parallel for num_threads(52)
        for (long path_idx = 0; path_idx < num_paths_max; path_idx++) {
            bool first = true;
            long last_id = -1;
			long last_node = -1;
			path_st_indice = path_st_max[path_idx];
			path_st_indice_nx = path_st_max[path_idx+1];
			float slack = path_interface_max->slacks[path_idx];

#if HETEROSTA_DEBUG_LEVEL >= 2
            // Log path header
            if (path_log.is_open()) {
                path_log << "\n=== HeteroSTA 第" << current_iteration << "轮优化 - 路径" << (path_idx + 1)
                        << "/" << num_paths_max << " ===\n";
                path_log << "路径slack: " << std::fixed << std::setprecision(6) << slack
                        << ", WNS: " << wns << ", 比值: " << (slack/wns) << "\n";
                path_log << "路径点数: " << (path_st_indice_nx - path_st_indice) << "\n";
                path_log << "------------------------\n";
            }
#endif

			for(long indice = path_st_indice; indice < path_st_indice_nx; indice++)
			{
				long pin_id = pin_rfs_max[indice]/2;
				int node_id = pin_to_node_map[pin_id];

#if HETEROSTA_DEBUG_LEVEL >= 3
                dreamplacePrint(kDEBUG, "第%d轮: 路径%ld点%ld: pin_id=%ld, node_id=%d\n",
                               current_iteration, path_idx, indice - path_st_indice, pin_id, node_id);
#endif

				if (first) {
					int net_id = pin_to_net_map[pin_id];
					if (heterosta_is_clock_network_net(&sta, net_id))
					{
						//skip the clock tree iccad_clk -> lcb:a -> lcb:o
						continue;
					}
					else
					{
						last_id = pin_id;
						last_node = node_id;
						first = false;
					}
                }
				else{
					if(node_id !=last_node)
					//if(true)
					{

						#pragma omp critical
						{
							pybind11::tuple key = pybind11::make_tuple(last_id, pin_id);
                            float old_weight = 0.0f;
                            float new_weight = 0.0f;
                            float delta = pin2pin_accumulate_weight * slack / wns;
                            bool is_new_key = false;

#if HETEROSTA_DEBUG_LEVEL >= 2
                            dreamplacePrint(kDEBUG, "第%d轮: 处理key:(%ld,%ld)\n", current_iteration, last_id, pin_id);
#endif

							if (pin2pin_net_weight.contains(key)) {
								{
								old_weight = pin2pin_net_weight[key].cast<float>();
								num_all_pairs += 1;
								new_weight = old_weight + delta;
								if (new_weight > pin2pin_max_weight){
									new_weight = pin2pin_max_weight;
								}
								pin2pin_net_weight[key] = new_weight;
                                is_new_key = false;
								}
							}
							else{
								num_unique_pairs += 1;
								old_weight = 0.0f;
                                new_weight = pin2pin_min_weight;
								pin2pin_net_weight[key] = new_weight;
                                is_new_key = true;
							}

#if HETEROSTA_DEBUG_LEVEL >= 1
                            // Log weight change details
                            heterosta_debug::log_weight_change_heterosta(weight_log, current_iteration,
                                                                        last_id, pin_id, old_weight, new_weight,
                                                                        delta, slack, wns, is_new_key);
#endif
						}

					}

					last_id = pin_id;
					last_node = node_id;

				}

			}

        }

		// #pragma omp parallel for num_threads(52)
        for (long path_idx = 0; path_idx < num_paths_min; path_idx++) {
            bool first = true;
            long last_id = -1;
			long last_node = -1;
			path_st_indice = path_st_min[path_idx];
			path_st_indice_nx = path_st_min[path_idx+1];
			float slack = path_interface_min->slacks[path_idx];

#if HETEROSTA_DEBUG_LEVEL >= 2
            // Log path header
            if (path_log.is_open()) {
                path_log << "\n=== HeteroSTA 第" << current_iteration << "轮优化 - 路径" << (path_idx + 1)
                        << "/" << num_paths_min << " ===\n";
                path_log << "路径slack: " << std::fixed << std::setprecision(6) << slack
                        << ", WNS: " << wns << ", 比值: " << (slack/wns) << "\n";
                path_log << "路径点数: " << (path_st_indice_nx - path_st_indice) << "\n";
                path_log << "------------------------\n";
            }
#endif

			for(long indice = path_st_indice; indice < path_st_indice_nx; indice++)
			{
				long pin_id = pin_rfs_min[indice]/2;
				int node_id = pin_to_node_map[pin_id];

#if HETEROSTA_DEBUG_LEVEL >= 3
                dreamplacePrint(kDEBUG, "第%d轮: 路径%ld点%ld: pin_id=%ld, node_id=%d\n",
                               current_iteration, path_idx, indice - path_st_indice, pin_id, node_id);
#endif

				if (first) {
					int net_id = pin_to_net_map[pin_id];
					if (heterosta_is_clock_network_net(&sta, net_id))
					{
						//skip the clock tree iccad_clk -> lcb:a -> lcb:o
						continue;
					}
					else
					{
						last_id = pin_id;
						last_node = node_id;
						first = false;
					}
                }
				else{
					if(node_id !=last_node)
					//if(true)
					{

						#pragma omp critical
						{
							pybind11::tuple key = pybind11::make_tuple(last_id, pin_id);
                            float old_weight = 0.0f;
                            float new_weight = 0.0f;
                            float delta = pin2pin_accumulate_weight * slack / wns;
                            bool is_new_key = false;

#if HETEROSTA_DEBUG_LEVEL >= 2
                            dreamplacePrint(kDEBUG, "第%d轮: 处理key:(%ld,%ld)\n", current_iteration, last_id, pin_id);
#endif

							if (pin2pin_net_weight.contains(key)) {
								{
								old_weight = pin2pin_net_weight[key].cast<float>();
								num_all_pairs += 1;
								new_weight = old_weight + delta;
								if (new_weight > pin2pin_max_weight){
									new_weight = pin2pin_max_weight;
								}
								pin2pin_net_weight[key] = new_weight;
                                is_new_key = false;
								}
							}
							else{
								num_unique_pairs += 1;
								old_weight = 0.0f;
                                new_weight = pin2pin_min_weight;
								pin2pin_net_weight[key] = new_weight;
                                is_new_key = true;
							}

#if HETEROSTA_DEBUG_LEVEL >= 1
                            // Log weight change details
                            heterosta_debug::log_weight_change_heterosta(weight_log, current_iteration,
                                                                        last_id, pin_id, old_weight, new_weight,
                                                                        delta, slack, wns, is_new_key);
#endif
						}

					}

					last_id = pin_id;
					last_node = node_id;

				}

			}

        }

    

		heterosta_free_pba_path_collection(path_interface_max);
		heterosta_free_pba_path_collection(path_interface_min);

		auto endT = std::chrono::steady_clock::now();
        double duration_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(endT - begT).count() * 0.001;

        // Enhanced statistics output
        dreamplacePrint(kINFO, "=== HeteroSTA 第%d轮 PIN2PIN 统计信息 ===\n", current_iteration);
        dreamplacePrint(kINFO, "新增唯一pairs: %d\n", num_unique_pairs);
        dreamplacePrint(kINFO, "处理总pairs: %d\n", num_all_pairs);
        dreamplacePrint(kINFO, "字典当前大小: %zu\n", pin2pin_net_weight.size());
        dreamplacePrint(kINFO, "执行时间: %f 秒\n", duration_seconds);

#if HETEROSTA_DEBUG_LEVEL >= 1
        // Write detailed statistics summary
        std::string stats_filename = heterosta_debug::create_debug_filename("heterosta_pin2pin_stats", current_iteration) + ".txt";
        heterosta_debug::write_heterosta_stats_summary(stats_filename, current_iteration,
                                                      num_unique_pairs, num_all_pairs,
                                                      pin2pin_net_weight.size(), wns, tns,
                                                      wns_hold, tns_hold, num_paths_max+num_paths_min, begT, endT);

        // Append to global summary log
        heterosta_debug::append_heterosta_global_summary(current_iteration, num_unique_pairs, num_all_pairs,
                                                        pin2pin_net_weight.size(), wns, tns, wns_hold, tns_hold,
                                                         num_paths_max+num_paths_min, duration_seconds);

        // Weight distribution analysis
        if (summary_log.is_open()) {
            summary_log << "== 权重分布分析 ==\n";
            std::map<int, int> weight_distribution;
            for (auto item : pin2pin_net_weight) {
                float value = item.second.cast<float>();
                int weight_range = static_cast<int>(value / 10) * 10; // Group by 10s
                weight_distribution[weight_range]++;
            }

            for (const auto& [range, count] : weight_distribution) {
                summary_log << "权重范围 [" << range << "-" << (range + 9) << "]: " << count << " 个键值对\n";
            }
            summary_log << "\n";
        }

        // Close debug files
        if (weight_log.is_open()) {
            weight_log << "\n=== HeteroSTA 第" << current_iteration << "轮权重变化完成 ===\n";
            weight_log.close();
        }
        if (path_log.is_open()) {
            path_log.close();
        }
        if (summary_log.is_open()) {
            summary_log << "结束时间: " << heterosta_debug::get_timestamp() << "\n";
            summary_log.close();
        }
#endif

#if HETEROSTA_DEBUG_LEVEL >= 2
        // Print sample of weight changes for console output
        dreamplacePrint(kINFO, "=== HeteroSTA 权重样本输出 (前10个) ===\n");
        int sample_count = 0;
        for (auto item : pin2pin_net_weight) {
            if (sample_count >= 10) break;
            pybind11::tuple key = item.first.cast<pybind11::tuple>();
            float value = item.second.cast<float>();
            long pin1 = key[0].cast<long>();
            long pin2 = key[1].cast<long>();
            dreamplacePrint(kINFO, "Key: (%ld, %ld) -> Weight: %.6f\n", pin1, pin2, value);
            sample_count++;
        }
#endif

        dreamplacePrint(kINFO, "=== HeteroSTA 第%d轮 PIN2PIN net-weighting 完成 ===\n", current_iteration);

	}
	else{
		dreamplacePrint(kERROR, "Unsupported net-weighting scheme\n");
	}

}

// Implementation of the update_net_weights method
void TimingHeterostaCpp::update_net_weights(
		STAHoldings& sta, int n,
		int num_nets,
		int num_pins,
		torch::Tensor flat_netpin, torch::Tensor netpin_start,
		torch::Tensor pin2node_map, torch::Tensor pin2net_map,
		torch::Tensor net_criticality, torch::Tensor net_criticality_deltas,
		torch::Tensor net_weights, torch::Tensor net_weight_deltas,
		torch::Tensor degree_map,
		double max_net_weight,
		pybind11::dict& pin2pin_net_weight,
		int net_weighting_scheme,
		double momentum_decay_factor,
		int ignore_net_degree, 
		int pin2pin_max_weight, int pin2pin_min_weight, double pin2pin_accumulate_weight,
		bool use_cuda) {

	// Check tensor properties
	CHECK_CONTIGUOUS(flat_netpin);
	CHECK_CONTIGUOUS(netpin_start);
	CHECK_CONTIGUOUS(pin2node_map);
	CHECK_CONTIGUOUS(pin2net_map);
	CHECK_CONTIGUOUS(net_criticality);
	CHECK_CONTIGUOUS(net_weights);
	CHECK_CONTIGUOUS(net_weight_deltas);
	CHECK_CONTIGUOUS(degree_map);


	if (use_cuda && net_weighting_scheme ==1) {
#ifdef CUDA_FOUND
        DREAMPLACE_DISPATCH_FLOATING_TYPES(
            net_criticality, "updateNetWeightCudaLauncher",
            [&] {
                updateNetWeightCudaLauncher<scalar_t>(
                    &sta,
                    num_nets,
                    num_pins,
                    DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
                    DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
                    DREAMPLACE_TENSOR_DATA_PTR(net_criticality, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(net_criticality_deltas, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(net_weight_deltas, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(degree_map, int),
                    static_cast<scalar_t>(momentum_decay_factor),
                    static_cast<scalar_t>(max_net_weight),
                    ignore_net_degree);
            });
#else
        dreamplacePrint(kERROR, "CUDA not available but use_cuda=true\n");
#endif
    } else {
	DREAMPLACE_DISPATCH_FLOATING_TYPES(
			net_criticality, "updateNetWeightCppLauncher",
			[&] {
			updateNetWeightCppLauncher<scalar_t>(
					sta,
					num_nets,
					num_pins,
					DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
					DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
					DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int),
					DREAMPLACE_TENSOR_DATA_PTR(pin2net_map, int),
					DREAMPLACE_TENSOR_DATA_PTR(net_criticality, scalar_t),
					DREAMPLACE_TENSOR_DATA_PTR(net_criticality_deltas, scalar_t),
					DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t),
					DREAMPLACE_TENSOR_DATA_PTR(net_weight_deltas, scalar_t),
					DREAMPLACE_TENSOR_DATA_PTR(degree_map, int),
					static_cast<scalar_t>(momentum_decay_factor),
					pin2pin_net_weight,
					net_weighting_scheme,
					static_cast<scalar_t>(max_net_weight),
					ignore_net_degree,
					at::get_num_threads(),
					pin2pin_max_weight, 
					pin2pin_min_weight, 
					pin2pin_accumulate_weight,
					use_cuda);
			});
	}

}

DREAMPLACE_END_NAMESPACE

