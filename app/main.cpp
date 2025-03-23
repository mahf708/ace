#include <iostream>
#include "../inference/inference.hpp"

int main() {
    std::cout << "Starting application..." << std::endl;
    run_inference();
    std::cout << "Inference completed." << std::endl;
    return 0;
}
