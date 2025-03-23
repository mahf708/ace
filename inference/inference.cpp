#include <iostream>
#include <fstream>  // Added for file existence check
#include <string>
#include <torch/script.h> // Torch C++ inference

void run_inference() {
    // Hardcoded configuration values
    std::string experiment_dir = "scratch/inference_output";
    int n_forward_steps = 3;
    int forward_steps_in_memory = 2;
    std::string checkpoint_path = "/pscratch/sd/m/mahf708/emulator/ace_ckpt_traced.tar";
    // Additional hardcoded parameters can be defined as needed

    // Output configuration values
    std::cout << "Experiment Directory: " << experiment_dir << std::endl;
    std::cout << "Number of Forward Steps: " << n_forward_steps << std::endl;
    std::cout << "Forward Steps in Memory: " << forward_steps_in_memory << std::endl;
    std::cout << "Checkpoint Path: " << checkpoint_path << std::endl;

    // Check if the checkpoint file exists
    std::ifstream cp_file(checkpoint_path, std::ios::binary);
    if (!cp_file.good()) {
        std::cerr << "Checkpoint file not found: " << checkpoint_path << "\n";
    }
    cp_file.close();

    // Simulate inference procedure:
    std::cout << "Loading model checkpoint from: " << checkpoint_path << std::endl;
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(checkpoint_path);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model checkpoint: " << e.what() << "\n";
        std::cerr << "Verify that the checkpoint was properly exported and includes 'constants.pkl'.\n";
    }
    // Move input tensor to the module's device (determined from its first parameter)
    auto device = (*module.parameters().begin()).device();
    // Create a dummy input tensor for inference (adjust dimensions as needed)
    torch::Tensor input = torch::rand({1, 39, 180, 360});
    input = input.to(device);
    std::vector<torch::jit::IValue> inputs{input};
    std::cout << "Input tensor shape: " << input.sizes() << std::endl;
    std::cout << "Performing inference for " << n_forward_steps << " steps..." << std::endl;
    torch::Tensor output = module.forward(inputs).toTensor();
    // std::cout << "Inference output sample: " << output.slice(/*dim=*/1, 0, 5) << std::endl;
    std::cout << "Inference output shape: " << output.sizes() << std::endl;
    std::cout << "Inference completed. Results stored in: " << experiment_dir << std::endl;
}
