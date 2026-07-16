#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sys/select.h>
#include <unistd.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fmt/format.h>
#include <yaml-cpp/yaml.h>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "onnxruntime_cxx_api.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"

#include "unitree/idl/go2/LowCmd_.hpp"
#include "unitree/idl/go2/LowState_.hpp"
#include "unitree/robot/b2/motion_switcher/motion_switcher_client.hpp"
#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include "unitree/robot/go2/sport/sport_client.hpp"

using namespace std::chrono_literals;
using Eigen::Quaternionf;
using Eigen::Vector3f;
using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelPublisherPtr;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;
using unitree::robot::b2::MotionSwitcherClient;

namespace {

std::atomic<bool> g_stop_requested(false);

void signal_handler(int) {
    g_stop_requested.store(true, std::memory_order_relaxed);
}

float yaml_float(const YAML::Node &node, const std::string &key, float default_value) {
    return node[key] ? node[key].as<float>() : default_value;
}

int yaml_int(const YAML::Node &node, const std::string &key, int default_value) {
    return node[key] ? node[key].as<int>() : default_value;
}

std::string yaml_string(
    const YAML::Node &node, const std::string &key, const std::string &default_value) {
    return node[key] ? node[key].as<std::string>() : default_value;
}

bool yaml_bool(const YAML::Node &node, const std::string &key, bool default_value) {
    return node[key] ? node[key].as<bool>() : default_value;
}

std::filesystem::path absolute_from(
    const std::filesystem::path &base, const std::filesystem::path &path) {
    return path.is_absolute() ? path : base / path;
}

// MuJoCo Playground Go2 actor observation uses local gravity:
// data.site_xmat[imu].T @ [0, 0, -1].
Vector3f projected_gravity_from_quat(const Quaternionf &q) {
    const Eigen::Matrix3f rotation = q.normalized().toRotationMatrix();
    return rotation.transpose() * Vector3f(0.0f, 0.0f, -1.0f);
}

uint32_t crc32_core(uint32_t *ptr, uint32_t len) {
    uint32_t crc = 0xFFFFFFFF;
    constexpr uint32_t polynomial = 0x04c11db7;
    for (uint32_t i = 0; i < len; ++i) {
        uint32_t xbit = 1u << 31;
        uint32_t data = ptr[i];
        for (uint32_t bits = 0; bits < 32; ++bits) {
            crc = (crc & 0x80000000) ? (crc << 1) ^ polynomial : (crc << 1);
            if (data & xbit) {
                crc ^= polynomial;
            }
            xbit >>= 1;
        }
    }
    return crc;
}

}  // namespace

struct DeployConfig {
    int control_freq = 50;
    float control_dt = 0.02f;

    std::string lowcmd_topic = "rt/lowcmd";
    std::string lowstate_topic = "rt/lowstate";
    std::string velocity_command_topic = "/velocity_command";

    std::filesystem::path config_dir;
    std::filesystem::path policy_path;
    std::string policy_input_name = "obs";
    std::string policy_output_name = "actions";

    float kp = 35.0f;
    float kd = 0.5f;
    float action_scale = 0.5f;
    float action_clip = 1.0f;
    float obs_clip = 100.0f;
    float command_clip_x = 1.5f;
    float command_clip_y = 1.0f;
    float command_clip_yaw = 1.5f;
    float command_timeout_s = 0.25f;

    float stand_time_s = 2.0f;
    float stand_tau_s = 1.2f;
    float stand_kp_start = 20.0f;
    float stand_kp_end = 50.0f;
    float stand_kd = 3.5f;
    float hold_default_s = 1.0f;
    float sit_time_s = 2.0f;
    bool enable_unitree_services = false;

    static constexpr int kNumActions = 12;
    static constexpr int kSingleObsDim = 45;
    static constexpr int kHistoryLen = 10;
    static constexpr int kObsDim = kSingleObsDim * kHistoryLen;

    // MuJoCo Playground policy action order. This is the qpos[7:] / actuator
    // order from go2_mjx_fullcollisions.xml, not the sensor or Unitree order.
    std::array<std::string, kNumActions> policy_joint_names = {
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    };

    // Unitree Go2 motor order used by LowState/LowCmd.
    std::array<std::string, kNumActions> motor_joint_names = {
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    };

    std::array<int, kNumActions> sim_to_motor{};
    std::array<float, kNumActions> default_angles{};
    std::array<float, kNumActions> stand_angles{};
    std::array<float, kNumActions> sit_angles{};

    void load_joint_arrays(const YAML::Node &yaml) {
        constexpr std::array<int, kNumActions> expected_policy_to_motor = {
            3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8,
        };

        const YAML::Node default_node = yaml["default_angles"];
        const YAML::Node stand_node = yaml["stand_angles"];
        const YAML::Node sit_node = yaml["sit_angles"];
        if (!default_node || !stand_node || !sit_node) {
            throw std::runtime_error(
                "config.yaml must define default_angles, stand_angles, and sit_angles");
        }

        for (int i = 0; i < kNumActions; ++i) {
            const std::string &name = policy_joint_names[i];
            default_angles[i] = default_node[name].as<float>();
            stand_angles[i] = stand_node[name].as<float>();
            sit_angles[i] = sit_node[name].as<float>();

            auto it = std::find(motor_joint_names.begin(), motor_joint_names.end(), name);
            if (it == motor_joint_names.end()) {
                throw std::runtime_error("Joint not found in motor order: " + name);
            }
            sim_to_motor[i] = static_cast<int>(std::distance(motor_joint_names.begin(), it));
        }

        if (sim_to_motor != expected_policy_to_motor) {
            throw std::runtime_error(
                "Unexpected policy-to-motor joint order. Expected FL,FR,RL,RR policy "
                "order to map to Unitree FR,FL,RR,RL motor indices: "
                "3 4 5 0 1 2 9 10 11 6 7 8");
        }
    }

    static DeployConfig from_yaml(const std::filesystem::path &config_path) {
        DeployConfig cfg;
        cfg.config_dir = std::filesystem::absolute(config_path).parent_path();
        const YAML::Node yaml = YAML::LoadFile(config_path.string());

        cfg.control_freq = yaml_int(yaml, "control_freq", cfg.control_freq);
        cfg.control_dt = 1.0f / static_cast<float>(cfg.control_freq);

        const YAML::Node topics = yaml["topics"];
        cfg.lowcmd_topic = yaml_string(topics, "lowcmd", cfg.lowcmd_topic);
        cfg.lowstate_topic = yaml_string(topics, "lowstate", cfg.lowstate_topic);
        cfg.velocity_command_topic =
            yaml_string(topics, "velocity_command", cfg.velocity_command_topic);

        const YAML::Node policy = yaml["policy"];
        cfg.policy_path = absolute_from(
            cfg.config_dir, yaml_string(policy, "path", "model/policy_deploy.onnx"));
        cfg.policy_input_name = yaml_string(policy, "input_name", cfg.policy_input_name);
        cfg.policy_output_name = yaml_string(policy, "output_name", cfg.policy_output_name);

        const YAML::Node control = yaml["control"];
        cfg.kp = yaml_float(control, "kp", cfg.kp);
        cfg.kd = yaml_float(control, "kd", cfg.kd);
        cfg.action_scale = yaml_float(control, "action_scale", cfg.action_scale);
        cfg.action_clip = yaml_float(control, "action_clip", cfg.action_clip);
        cfg.obs_clip = yaml_float(control, "obs_clip", cfg.obs_clip);
        cfg.command_timeout_s = yaml_float(control, "command_timeout_s", cfg.command_timeout_s);

        const YAML::Node command_clip = control["command_clip"];
        cfg.command_clip_x = yaml_float(command_clip, "x", cfg.command_clip_x);
        cfg.command_clip_y = yaml_float(command_clip, "y", cfg.command_clip_y);
        cfg.command_clip_yaw = yaml_float(command_clip, "yaw", cfg.command_clip_yaw);

        const YAML::Node startup = yaml["startup"];
        cfg.stand_time_s = yaml_float(startup, "stand_time_s", cfg.stand_time_s);
        cfg.stand_tau_s = yaml_float(startup, "stand_tau_s", cfg.stand_tau_s);
        cfg.stand_kp_start = yaml_float(startup, "stand_kp_start", cfg.stand_kp_start);
        cfg.stand_kp_end = yaml_float(startup, "stand_kp_end", cfg.stand_kp_end);
        cfg.stand_kd = yaml_float(startup, "stand_kd", cfg.stand_kd);
        cfg.hold_default_s = yaml_float(startup, "hold_default_s", cfg.hold_default_s);
        cfg.sit_time_s = yaml_float(startup, "sit_time_s", cfg.sit_time_s);

        const YAML::Node services = yaml["unitree_services"];
        cfg.enable_unitree_services = yaml_bool(services, "enable", cfg.enable_unitree_services);

        cfg.load_joint_arrays(yaml);
        return cfg;
    }
};

class Go2JoystickDeployNode final : public rclcpp::Node {
  public:
    explicit Go2JoystickDeployNode(DeployConfig cfg)
        : Node("go2_joystick_deploy_node"), cfg_(std::move(cfg)),
          ort_env_(ORT_LOGGING_LEVEL_WARNING, "go2_joystick_deploy") {
        command_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
            cfg_.velocity_command_topic, 10,
            std::bind(&Go2JoystickDeployNode::velocity_command_callback, this, std::placeholders::_1));
        frequency_pub_ = create_publisher<std_msgs::msg::Float32>(
            "/go2_joystick_deploy/control_frequency", 10);

        action_.fill(0.0f);
        target_q_.fill(0.0f);
        obs_history_.fill(0.0f);
        command_.setZero();

        for (int i = 0; i < DeployConfig::kNumActions; ++i) {
            target_q_[i] = cfg_.default_angles[i];
        }
    }

    void initialize() {
        lowcmd_pub_ = std::make_unique<ChannelPublisher<unitree_go::msg::dds_::LowCmd_>>(
            cfg_.lowcmd_topic);
        lowcmd_pub_->InitChannel();

        lowstate_sub_ = std::make_unique<ChannelSubscriber<unitree_go::msg::dds_::LowState_>>(
            cfg_.lowstate_topic);
        lowstate_sub_->InitChannel(
            std::bind(&Go2JoystickDeployNode::low_state_callback, this, std::placeholders::_1), 10);

        initialize_low_cmd();
        load_policy();
        wait_for_low_state();

        if (cfg_.enable_unitree_services) {
            enter_unitree_release_mode();
        }

        stand_up_motion();

        wait_for_policy_activation();
        initialized_ = true;
    }

    void run() {
        rclcpp::Rate rate(cfg_.control_freq);
        auto last_loop = std::chrono::steady_clock::now();

        while (rclcpp::ok() && !g_stop_requested.load(std::memory_order_relaxed)) {
            const auto now = std::chrono::steady_clock::now();
            const float loop_dt = std::chrono::duration<float>(now - last_loop).count();
            last_loop = now;
            if (loop_dt > 0.0f) {
                publish_frequency(1.0f / loop_dt);
            }

            step_policy();
            rate.sleep();
        }

        wait_for_shutdown_confirmation();
        shutdown_motion();
    }

  private:
    void velocity_command_callback(const geometry_msgs::msg::PointStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(command_mutex_);
        command_(0) = std::clamp(
            static_cast<float>(msg->point.x), -cfg_.command_clip_x, cfg_.command_clip_x);
        command_(1) = std::clamp(
            static_cast<float>(msg->point.y), -cfg_.command_clip_y, cfg_.command_clip_y);
        command_(2) = std::clamp(
            static_cast<float>(msg->point.z), -cfg_.command_clip_yaw, cfg_.command_clip_yaw);
        last_command_time_ = std::chrono::steady_clock::now();
        received_command_ = true;
    }

    void low_state_callback(const void *msg) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        low_state_ = *static_cast<const unitree_go::msg::dds_::LowState_ *>(msg);
    }

    void initialize_low_cmd() {
        low_cmd_.head()[0] = 0xFE;
        low_cmd_.head()[1] = 0xEF;
        low_cmd_.level_flag() = 0xFF;
        low_cmd_.gpio() = 0;
        for (auto &motor : low_cmd_.motor_cmd()) {
            motor.mode() = 0x0A;
            motor.q() = 2.146e9f;
            motor.dq() = 16000.0f;
            motor.kp() = 0.0f;
            motor.kd() = 0.0f;
            motor.tau() = 0.0f;
        }
    }

    void load_policy() {
        if (!std::filesystem::exists(cfg_.policy_path)) {
            throw std::runtime_error("ONNX policy not found: " + cfg_.policy_path.string());
        }

        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        policy_session_ = std::make_unique<Ort::Session>(
            ort_env_, cfg_.policy_path.string().c_str(), options);
        RCLCPP_INFO(get_logger(), "Loaded ONNX policy: %s", cfg_.policy_path.string().c_str());
    }

    void wait_for_low_state() {
        RCLCPP_INFO(get_logger(), "Waiting for %s...", cfg_.lowstate_topic.c_str());
        const auto start = std::chrono::steady_clock::now();
        while (rclcpp::ok()) {
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                if (low_state_.tick() != 0) {
                    RCLCPP_INFO(get_logger(), "LowState received.");
                    return;
                }
            }
            if (std::chrono::steady_clock::now() - start > 5s) {
                throw std::runtime_error("Timed out waiting for LowState.");
            }
            std::this_thread::sleep_for(20ms);
        }
        throw std::runtime_error("ROS shutdown while waiting for LowState.");
    }

    void enter_unitree_release_mode() {
        RCLCPP_INFO(get_logger(), "Disabling sport mode and entering release mode...");
        auto sport_client = std::make_unique<unitree::robot::go2::SportClient>();
        auto motion_switcher = std::make_unique<MotionSwitcherClient>();

        sport_client->SetTimeout(5.0f);
        sport_client->Init();
        sport_client->StandDown();
        std::this_thread::sleep_for(2s);

        motion_switcher->SetTimeout(5.0f);
        motion_switcher->Init();
        if (motion_switcher->ReleaseMode() != 0) {
            throw std::runtime_error("MotionSwitcher ReleaseMode failed.");
        }
        RCLCPP_INFO(get_logger(), "MotionSwitcher ReleaseMode succeeded.");
        std::this_thread::sleep_for(2s);
    }

    void wait_for_policy_activation() {
        RCLCPP_INFO(get_logger(), "Robot is standing in policy default pose.");
        std::cout << "\nPress ENTER to activate the locomotion policy..." << std::flush;

        rclcpp::Rate rate(cfg_.control_freq);
        while (rclcpp::ok() && !g_stop_requested.load(std::memory_order_relaxed) &&
               !stdin_has_line()) {
            send_position_command(cfg_.default_angles, cfg_.kp, cfg_.kd);
            rate.sleep();
        }
        if (!rclcpp::ok() || g_stop_requested.load(std::memory_order_relaxed)) {
            throw std::runtime_error("Policy activation cancelled by shutdown request.");
        }

        std::string line;
        if (!std::getline(std::cin, line)) {
            throw std::runtime_error("Policy activation cancelled: stdin closed before ENTER.");
        }

        RCLCPP_INFO(get_logger(), "ENTER received. Activating locomotion policy.");
    }

    unitree_go::msg::dds_::LowState_ state_snapshot() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return low_state_;
    }

    std::array<float, DeployConfig::kNumActions> current_joint_positions() const {
        const auto state = state_snapshot();
        std::array<float, DeployConfig::kNumActions> q{};
        for (int i = 0; i < DeployConfig::kNumActions; ++i) {
            q[i] = state.motor_state()[cfg_.sim_to_motor[i]].q();
        }
        return q;
    }

    Vector3f command_snapshot() {
        std::lock_guard<std::mutex> lock(command_mutex_);
        if (!received_command_ ||
            std::chrono::steady_clock::now() - last_command_time_ >
                std::chrono::duration<float>(cfg_.command_timeout_s)) {
            return Vector3f::Zero();
        }
        return command_;
    }

    std::array<float, DeployConfig::kSingleObsDim> make_actor_frame(
        const unitree_go::msg::dds_::LowState_ &state) {
        std::array<float, DeployConfig::kSingleObsDim> frame{};

        frame[0] = state.imu_state().gyroscope()[0];
        frame[1] = state.imu_state().gyroscope()[1];
        frame[2] = state.imu_state().gyroscope()[2];

        const Quaternionf quat(
            state.imu_state().quaternion()[0], state.imu_state().quaternion()[1],
            state.imu_state().quaternion()[2], state.imu_state().quaternion()[3]);
        const Vector3f gravity = projected_gravity_from_quat(quat);
        frame[3] = gravity.x();
        frame[4] = gravity.y();
        frame[5] = gravity.z();

        const Vector3f command = command_snapshot();
        frame[6] = command.x();
        frame[7] = command.y();
        frame[8] = command.z();

        for (int i = 0; i < DeployConfig::kNumActions; ++i) {
            const int motor = cfg_.sim_to_motor[i];
            frame[9 + i] = state.motor_state()[motor].q() - cfg_.default_angles[i];
            frame[21 + i] = state.motor_state()[motor].dq();
            frame[33 + i] = action_[i];
        }

        return frame;
    }

    void update_observation_history(const std::array<float, DeployConfig::kSingleObsDim> &frame) {
        if (!obs_history_initialized_) {
            for (int h = 0; h < DeployConfig::kHistoryLen; ++h) {
                std::copy(frame.begin(), frame.end(),
                          obs_history_.begin() + h * DeployConfig::kSingleObsDim);
            }
            obs_history_initialized_ = true;
        } else {
            std::move(
                obs_history_.begin() + DeployConfig::kSingleObsDim, obs_history_.end(),
                obs_history_.begin());
            std::copy(
                frame.begin(), frame.end(),
                obs_history_.begin() + (DeployConfig::kHistoryLen - 1) * DeployConfig::kSingleObsDim);
        }

        for (float &v : obs_history_) {
            v = std::clamp(v, -cfg_.obs_clip, cfg_.obs_clip);
        }
    }

    std::array<float, DeployConfig::kNumActions> run_policy() {
        std::array<float, DeployConfig::kNumActions> output{};
        std::array<int64_t, 2> input_shape = {1, DeployConfig::kObsDim};

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, obs_history_.data(), obs_history_.size(), input_shape.data(),
            input_shape.size());

        const char *input_names[] = {cfg_.policy_input_name.c_str()};
        const char *output_names[] = {cfg_.policy_output_name.c_str()};

        auto output_tensors = policy_session_->Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        const float *ptr = output_tensors[0].GetTensorData<float>();
        for (int i = 0; i < DeployConfig::kNumActions; ++i) {
            float value = ptr[i];
            if (!std::isfinite(value)) {
                value = 0.0f;
            }
            output[i] = std::clamp(value, -cfg_.action_clip, cfg_.action_clip);
        }
        return output;
    }

    void step_policy() {
        if (!initialized_) {
            return;
        }

        const auto state = state_snapshot();
        update_observation_history(make_actor_frame(state));
        action_ = run_policy();

        for (int i = 0; i < DeployConfig::kNumActions; ++i) {
            target_q_[i] = cfg_.default_angles[i] + action_[i] * cfg_.action_scale;
        }
        send_position_command(target_q_, cfg_.kp, cfg_.kd);
    }

    void send_position_command(
        const std::array<float, DeployConfig::kNumActions> &q, float kp, float kd) {
        for (int i = 0; i < DeployConfig::kNumActions; ++i) {
            const int motor = cfg_.sim_to_motor[i];
            low_cmd_.motor_cmd()[motor].mode() = 0x0A;
            low_cmd_.motor_cmd()[motor].q() = q[i];
            low_cmd_.motor_cmd()[motor].dq() = 0.0f;
            low_cmd_.motor_cmd()[motor].kp() = kp;
            low_cmd_.motor_cmd()[motor].kd() = kd;
            low_cmd_.motor_cmd()[motor].tau() = 0.0f;
        }
        low_cmd_.crc() = crc32_core(
            reinterpret_cast<uint32_t *>(&low_cmd_),
            (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
        lowcmd_pub_->Write(low_cmd_);
    }

    void interpolate_pose(
        const std::array<float, DeployConfig::kNumActions> &start,
        const std::array<float, DeployConfig::kNumActions> &end, float duration_s,
        float kp, float kd) {
        const int steps = std::max(1, static_cast<int>(duration_s * cfg_.control_freq));
        for (int step = 0; step <= steps && rclcpp::ok(); ++step) {
            const float alpha = static_cast<float>(step) / static_cast<float>(steps);
            std::array<float, DeployConfig::kNumActions> q{};
            for (int i = 0; i < DeployConfig::kNumActions; ++i) {
                q[i] = start[i] * (1.0f - alpha) + end[i] * alpha;
            }
            send_position_command(q, kp, kd);
            std::this_thread::sleep_for(std::chrono::duration<float>(cfg_.control_dt));
        }
    }

    void interpolate_pose_tanh(
        const std::array<float, DeployConfig::kNumActions> &start,
        const std::array<float, DeployConfig::kNumActions> &end, float duration_s,
        float tau_s, float kp_start, float kp_end, float kd) {
        const int steps = std::max(1, static_cast<int>(duration_s * cfg_.control_freq));
        const float tau = std::max(tau_s, 1.0e-3f);
        for (int step = 0; step <= steps && rclcpp::ok(); ++step) {
            const float elapsed_s = static_cast<float>(step) * cfg_.control_dt;
            const float phase = std::tanh(elapsed_s / tau);
            std::array<float, DeployConfig::kNumActions> q{};
            for (int i = 0; i < DeployConfig::kNumActions; ++i) {
                q[i] = phase * end[i] + (1.0f - phase) * start[i];
            }
            const float kp = phase * kp_end + (1.0f - phase) * kp_start;
            send_position_command(q, kp, kd);
            std::this_thread::sleep_for(std::chrono::duration<float>(cfg_.control_dt));
        }
    }

    void stand_up_motion() {
        RCLCPP_INFO(get_logger(), "Standing up directly into policy default pose.");
        interpolate_pose_tanh(
            current_joint_positions(), cfg_.default_angles, cfg_.stand_time_s,
            cfg_.stand_tau_s, cfg_.stand_kp_start, cfg_.stand_kp_end, cfg_.stand_kd);
    }

    void wait_for_shutdown_confirmation() {
        if (!lowcmd_pub_) {
            return;
        }

        action_.fill(0.0f);
        target_q_ = cfg_.default_angles;

        RCLCPP_WARN(
            get_logger(),
            "Policy paused. Moving to default pose before sit-down shutdown.");
        interpolate_pose(current_joint_positions(), target_q_, cfg_.hold_default_s, cfg_.kp, cfg_.kd);

        std::cout << "\nPolicy paused. Press ENTER to sit down and close..." << std::flush;

        rclcpp::Rate rate(cfg_.control_freq);
        while (!stdin_has_line()) {
            send_position_command(target_q_, cfg_.kp, cfg_.kd);
            rate.sleep();
        }

        std::string line;
        if (!std::getline(std::cin, line)) {
            RCLCPP_WARN(get_logger(), "stdin closed while waiting; continuing sit-down shutdown.");
        } else {
            RCLCPP_INFO(get_logger(), "ENTER received. Closing policy and sitting down.");
        }
    }

    bool stdin_has_line() const {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(STDIN_FILENO, &read_fds);

        timeval timeout{};
        timeout.tv_sec = 0;
        timeout.tv_usec = 0;

        return select(STDIN_FILENO + 1, &read_fds, nullptr, nullptr, &timeout) > 0 &&
               FD_ISSET(STDIN_FILENO, &read_fds);
    }

    void shutdown_motion() {
        if (!lowcmd_pub_) {
            return;
        }
        RCLCPP_INFO(get_logger(), "Sitting down...");
        interpolate_pose_tanh(
            current_joint_positions(), cfg_.sit_angles, cfg_.sit_time_s,
            cfg_.stand_tau_s, cfg_.stand_kp_end, cfg_.stand_kp_end, cfg_.stand_kd);

        for (auto &motor : low_cmd_.motor_cmd()) {
            motor.q() = 0.0f;
            motor.dq() = 0.0f;
            motor.kp() = 0.0f;
            motor.kd() = 8.0f;
            motor.tau() = 0.0f;
        }
        low_cmd_.crc() = crc32_core(
            reinterpret_cast<uint32_t *>(&low_cmd_),
            (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
        lowcmd_pub_->Write(low_cmd_);
    }

    void publish_frequency(float frequency) {
        if (++frequency_counter_ < cfg_.control_freq) {
            return;
        }
        frequency_counter_ = 0;
        std_msgs::msg::Float32 msg;
        msg.data = frequency;
        frequency_pub_->publish(msg);
    }

    DeployConfig cfg_;

    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr command_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr frequency_pub_;

    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_pub_;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_sub_;
    unitree_go::msg::dds_::LowCmd_ low_cmd_{};
    mutable std::mutex state_mutex_;
    unitree_go::msg::dds_::LowState_ low_state_{};

    std::mutex command_mutex_;
    Vector3f command_{0.0f, 0.0f, 0.0f};
    std::chrono::steady_clock::time_point last_command_time_{};
    bool received_command_ = false;

    Ort::Env ort_env_;
    std::unique_ptr<Ort::Session> policy_session_;
    std::array<float, DeployConfig::kObsDim> obs_history_{};
    std::array<float, DeployConfig::kNumActions> action_{};
    std::array<float, DeployConfig::kNumActions> target_q_{};
    bool obs_history_initialized_ = false;
    bool initialized_ = false;
    int frequency_counter_ = 0;
};

int main(int argc, char **argv) {
    std::string net_interface = "lo";
    std::filesystem::path config_path = "config.yaml";
    std::filesystem::path policy_override;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--net" || arg == "-n") && i + 1 < argc) {
            net_interface = argv[++i];
        } else if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            config_path = argv[++i];
        } else if ((arg == "--policy" || arg == "-p") && i + 1 < argc) {
            policy_override = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: locomotion_node [--net IFACE] [--config PATH] [--policy PATH]\n";
            return 0;
        }
    }

    std::thread spin_thread;
    try {
        const int net_index = net_interface == "lo" ? 1 : 0;
        ChannelFactory::Instance()->Init(net_index, net_interface);

        rclcpp::init(argc, argv);
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        DeployConfig cfg = DeployConfig::from_yaml(config_path);
        if (!policy_override.empty()) {
            cfg.policy_path = absolute_from(std::filesystem::current_path(), policy_override);
        }

        auto node = std::make_shared<Go2JoystickDeployNode>(cfg);
        spin_thread = std::thread([node]() {
            rclcpp::executors::MultiThreadedExecutor executor;
            executor.add_node(node);
            executor.spin();
        });

        node->initialize();
        node->run();

        if (rclcpp::ok()) {
            rclcpp::shutdown();
        }
        if (spin_thread.joinable()) {
            spin_thread.join();
        }
    } catch (const std::exception &e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        if (rclcpp::ok()) {
            rclcpp::shutdown();
        }
        if (spin_thread.joinable()) {
            spin_thread.join();
        }
        return 1;
    }

    return 0;
}
